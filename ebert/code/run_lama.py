import os

from emb_input_transformers import EmbInputBertModel
from pytorch_transformers import BertForMaskedLM, RobertaForMaskedLM

from utils.util_wikidata import *
from embeddings import load_embedding, MappedEmbedding
from mappers import load_mapper

from tqdm import tqdm

import json
import numpy as np
import argparse
import torch

import pandas

# HOW TO TEST EACH RELATION SET
# {'relation': 'place_of_birth', 'template': '[X] was born in [Y] .'}  , has many many associated "queries" like 
def test(queries, method, model, language_model, 
        model_emb, wiki_emb, mapper, batch_size=4, 
        allowed_vocabulary=None):
    
    template = queries[0]["template"]
    relation = queries[0]["predicate_id"]
    
    assert all([query["template"] == template for query in queries])
    assert all([query["predicate_id"] == relation for query in queries])

    if allowed_vocabulary is None:
        restrict_vocabulary = []
    else:
        restrict_vocabulary = np.array([not i in allowed_vocabulary for i in range(model.config.vocab_size)])

    template = template.replace("[X]", model_emb.tokenizer.unk_token) 
    template = template.replace("[Y]", model_emb.tokenizer.mask_token) 

    template_encoded = model_emb.tokenizer.encode(template, add_special_tokens = True)
    template_tokenized = model_emb.tokenizer.convert_ids_to_tokens(template_encoded)

    template_vectors = [model_emb[token] for token in template_tokenized]   # BERT embeddings for every token in template 
    unk_idx = template_tokenized.index(model_emb.tokenizer.unk_token)
    mask_idx = template_tokenized.index(model_emb.tokenizer.mask_token)
    
    # ??? what is this
    if mask_idx > unk_idx:
        mask_idx -= len(template_tokenized)
    
    pad_vector = model_emb[model_emb.tokenizer.pad_token]
    slash_vector = model_emb["/"]

    batch = []
    template_fillers = []
    replaceable = []
    probs = []
    mean_attention = []

    mapped_titles = {}
    title_vectors = []

    for query in queries:
        replaceable.append(0)
        for title in sorted(list(query["wiki_titles"])):           #ex. 'wiki_titles': ['ENTITY/Khatchig_Mouradian'] 
            if title in wiki_emb:
                replaceable[-1] = 1
                if not title in mapped_titles:
                    mapped_titles[title] = len(mapped_titles)
                    title_vectors.append(wiki_emb[title])           #add wiki_emb vectors of titles to title_vectors
                break
    
    assert len(mapped_titles) == len(title_vectors)

    if len(mapped_titles):
        title_vectors = np.array(title_vectors)
        mapped_title_vectors = mapper.apply(title_vectors)         # THIS IS THE ONLY PLACE THE MAPPER IS used in TEST ( ie, to map wiki embedding of ent to BERT space )

    # FOR DEBUGGING OF ONE INSTANCE SET TO TRUE
    debug = True

    # THIS IS WHERE THE tokenization, attention, input vectors, masking takes place and 
    # then the Embeddings are gathered from EmbInputBERT and the pred is made from the output of that into BertFORMaskedLM 
    for i, query in enumerate(tqdm(queries, desc = method), start=1):
        if debug:
            print("TEST LOOP", i, query["sub_label"], query["wiki_titles"], query["obj_label"], query["masked_sentences"])

        subject_tokenized = model_emb.tokenizer.tokenize(query["sub_label"])         # 'sub_label': 'Khatchig Mouradian', 
        vectors_tokenized = [model_emb[token] for token in subject_tokenized]        # BERT embeddings for all tokens
        vectors_wiki = vectors_slash = vectors_tokenized
        subject_wiki = subject_slash = subject_tokenized
        
        for title in query["wiki_titles"]:
            if title in mapped_titles:
                subject_wiki = [title]
                vectors_wiki = [mapped_title_vectors[mapped_titles[title]]]
                subject_slash = subject_wiki + ["/"] + subject_tokenized
                vectors_slash = vectors_wiki + [slash_vector] + vectors_tokenized

        batch.extend([vectors_tokenized, vectors_wiki, vectors_slash])                    # add Normal BERT version,  EBERT replace,  EBERT concat for current query to batch
        template_fillers.extend([subject_tokenized, subject_wiki, subject_slash])
 
        if len(batch) >= batch_size or i == len(queries):

            if debug:
                # FOR DEBUGGING
                print("Looking at Batch",i )
                for bind, v in enumerate(batch):
                    print(i, bind, len(v), [vv.shape for vv in v])
                # Looking at Batch 2 0%   
                # 2 0 7 [array([-2.34825350e-02,  4.97670695e-02, -2.80051846e-02, -4.51875590e-02, 
                
                print("\nLooking at Template fillers")
                for bind, v in enumerate(template_fillers):
                    print(i, bind, len(v), [vv for vv in v])

            """
            TEST LOOP 1 Khatchig Mouradian ['ENTITY/Khatchig_Mouradian'] Lebanon ['Khatchig Mouradian is a journalist, writer and translator born in [MASK] .']
            TEST LOOP 2 Jacob Henry Studer ['ENTITY/Jacob_H._Studer'] Columbus ['Jacob Henry Studer (26 February 1840 [MASK], Ohio - 2 August 1904 New York City) was a printer, lithographer, painter, and popular ornithologist active in [MASK], Ohio from the 1860s to the 1880s .']

            Looking at Batch 2
            2 0 7 [(768,), (768,), (768,), (768,), (768,), (768,), (768,)]
            2 1 7 [(768,), (768,), (768,), (768,), (768,), (768,), (768,)]
            2 2 7 [(768,), (768,), (768,), (768,), (768,), (768,), (768,)]
            2 3 5 [(768,), (768,), (768,), (768,), (768,)]
            2 4 5 [(768,), (768,), (768,), (768,), (768,)]
            2 5 5 [(768,), (768,), (768,), (768,), (768,)]
            
            Looking at Template fillers
            2 0 7 ['K', '##hat', '##chi', '##g', 'Mo', '##ura', '##dian']
            2 1 7 ['K', '##hat', '##chi', '##g', 'Mo', '##ura', '##dian']
            2 2 7 ['K', '##hat', '##chi', '##g', 'Mo', '##ura', '##dian']
            2 3 5 ['Jacob', 'Henry', 'St', '##ude', '##r']
            2 4 5 ['Jacob', 'Henry', 'St', '##ude', '##r']
            2 5 5 ['Jacob', 'Henry', 'St', '##ude', '##r']

            """


            maxlen = max([len(x) for x in batch]) + len(template_vectors) - 1
            input_vectors = torch.tensor([[pad_vector for _ in range(maxlen)] for _ in batch])
            attention_masks = torch.zeros((len(batch), maxlen))
            
            for batch_i, sample in enumerate(batch):
                template_filled = template_vectors[:unk_idx] + sample + template_vectors[unk_idx+1:]
                input_vectors[batch_i, :len(template_filled)] = torch.tensor(template_filled)
                attention_masks[batch_i, :len(template_filled)] = 1

                if debug:
                    print("Filling in", batch_i, " -> ", len(template_filled), 

            input_vectors = input_vectors.to(device = next(model.parameters()).device)
            attention_masks = attention_masks.to(device = next(model.parameters()).device)
                
            #hidden_states = model(input_ids = input_vectors, attention_mask = attention_masks)[0]

            # WHERE THE MODEL IS ACTUALLY CALLED
            # remember:          model = EmbInputBertModel.from_pretrained(args.modelname, output_attentions = True)
            #      and: language_model = BertForMaskedLM.from_pretrained(args.modelname).cls
            tmp = model(input_ids = input_vectors, attention_mask = attention_masks)

            if debug:
                # FOR DEBUGGING
                print("Model input",i )
                print("Model input_vectors",input_vectors.shape, input_vectors )
                print("Model attention_masks",attention_masks.shape, attention_masks )

                """
                Model input 2
                Model input_vectors torch.Size([6, 14, 768]) tensor([[[ 0.0333,  0.0066, -0.0037,  ...,  0.0043, -0.0085, -0.0164],
                Model attention_masks torch.Size([6, 14]) tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.]])
                """

            hidden_states = tmp[0]
            attentions = torch.stack(tmp[-1], 0).mean(0).mean(1).mean(1).detach().cpu().numpy()
            assert np.allclose(attentions.sum(-1), np.ones((attentions.shape[0],)))
            lm_inputs = torch.zeros_like(hidden_states[:,0])

            for batch_i, _ in enumerate(batch):
                lm_inputs[batch_i] = hidden_states[batch_i][attention_masks[batch_i] == 1][mask_idx]

            # CALL LANGUAGE MODEL ON EMBEDDINGS INPUT
            logits = language_model(lm_inputs)
            logits[:,restrict_vocabulary] = -10000       # THIS MAKES IT SO THAT YOU CAN'T PREDICT WORDS OUTSIDE OF allow_vocabulary
            prob = logits.softmax(-1)
            prob = prob.detach().cpu().numpy()
            
            attn = [[float(a) for a in x] for x in attentions]
            assert len(attn) == prob.shape[0]

            if debug:
                # FOR DEBUGGING
                print("Model ouput")
                print("Model hidden states",hidden_states.shape, hidden_states )
                print("Model attentions",attentions.shape, attentions)

                print("LM MODEL")
                print("LM MODEL inputs", lm_inputs.shape, lm_inputs)
                print("LM MODEL logits", logits.shape)
                print("LM MODEL prob", prob.shape)
                print("LM MODEL attn", len(attn))
                import sys
                sys.exit()

                """
                Model ouput
                Model hidden states torch.Size([6, 14, 768]) tensor([[[ 3.9322e-01,  1.0512e-01, -1.9973e-01,  ...,  6.5817e-03,
                Model attentions (6, 14) [[0.09187021 0.03477917 0.02993263 0.03159837 0.03940959 0.03178956 0.03066862 0.03329325 0.03862337 0.04719618 0.04220517 0.05918208 0.04265609 0.44679582]
                 [0.09187021 0.03477917 0.02993263 0.03159837 0.03940959 0.03178956 0.03066862 0.03329325 0.03862337 0.04719618 0.04220517 0.05918208 0.04265609 0.44679582]
                 [0.09187021 0.03477917 0.02993263 0.03159837 0.03940959 0.03178956 0.03066862 0.03329325 0.03862337 0.04719618 0.04220517 0.05918208 0.04265609 0.44679582]
                 [0.09239926 0.04704424 0.04034141 0.03061972 0.03067828 0.03117144 0.04378817 0.05657936 0.04905342 0.06421196 0.05299358 0.46111923 0.         0.        ]
                 [0.09239926 0.04704424 0.04034141 0.03061972 0.03067828 0.03117144 0.04378817 0.05657936 0.04905342 0.06421196 0.05299358 0.46111923 0.         0.        ]
                 [0.09239926 0.04704424 0.04034141 0.03061972 0.03067828 0.03117144 0.04378817 0.05657936 0.04905342 0.06421196 0.05299358 0.46111923 0.         0.        ]]
                
                LM MODEL
                LM MODEL inputs torch.Size([6, 768]) tensor([[ 0.3372,  0.2180, -0.5007,  ..., -0.1304, -0.0791,  0.1832],
                        [ 0.3372,  0.2180, -0.5007,  ..., -0.1304, -0.0791,  0.1832],
                        [ 0.3372,  0.2180, -0.5007,  ..., -0.1304, -0.0791,  0.1832],
                        [ 0.0967, -0.1118, -0.1813,  ..., -0.2628, -0.2489,  0.0942],
                        [ 0.0967, -0.1118, -0.1813,  ..., -0.2628, -0.2489,  0.0942],
                        [ 0.0967, -0.1118, -0.1813,  ..., -0.2628, -0.2489,  0.0942]],
                       device='cuda:0', grad_fn=<CopySlices>)
                LM MODEL logits torch.Size([6, 28996])
                LM MODEL prob (6, 28996)
                LM MODEL attn 6


                """


            mean_attention.extend(attn)
            probs.extend(prob)
            batch.clear()


    # GET ENCODING OF TRUE ANSWER from obj_label and formulate PREDICTION DICT
    encoded = [model_emb.tokenizer.encode(query["obj_label"], add_special_tokens = False) for query in queries]
    assert all([len(x) == 1 for x in encoded])
    gold_answers = [x[0] for x in encoded]

    assert len(gold_answers) == len(replaceable) == len(queries)
    assert 3 * len(gold_answers) == len(template_fillers) == len(probs)
    assert len(probs) == len(mean_attention)

    for i, query in enumerate(queries):
        query_bert = " ".join(template_tokenized[:unk_idx] + template_fillers[3*i] + template_tokenized[unk_idx+1:])
        query_wiki = " ".join(template_tokenized[:unk_idx] + template_fillers[3*i+1] + template_tokenized[unk_idx+1:])
        query_slash = " ".join(template_tokenized[:unk_idx] + template_fillers[3*i+2] + template_tokenized[unk_idx+1:])

        assert (not "query_bert" in query) or query_bert == query["query_bert"]
        assert (not "query_wiki" in query) or query_wiki == query["query_wiki"]
        assert (not "query_slash" in query) or query_slash == query["query_slash"]
        
        query["query_bert"] = query_bert
        query["query_wiki"] = query_wiki
        query["query_slash"] = query_slash

        prob = {}
        prob["bert"] = probs[3*i]
        prob[method + "_replace"] = probs[3*i+1]
        prob[method + "_concat"] = probs[3*i+2]
        
        prob[method + "_ens_avg"] = (prob["bert"] + prob[method + "_replace"]) / 2
        prob[method + "_ens_max"] = prob["bert"] if prob["bert"].max() > prob[method + "_replace"].max() else prob[method + "_replace"]

        query["attn:bert"] = mean_attention[3*i][:len(query_bert)]
        query["attn:" + method + "_replace"] = mean_attention[3*i+1][:len(query_wiki)]
        query["attn:" + method + "_concat"] = mean_attention[3*i+2][:len(query_slash)]
        
        for key in prob:
            ranking = np.argsort(prob[key])[::-1]
            top10 = tuple(model_emb.tokenizer.convert_ids_to_tokens(ranking[:10]))
            top10_prob = tuple([float(p) for p in prob[key][ranking[:10]]])

            assert (not f"top10_prob:{key}" in query) or query[f"top10_prob:{key}"] == top10_prob
            assert (not f"top10:{key}" in query) or query[f"top10:{key}"] == top10
            query[f"top10:{key}"] = top10
            query[f"top10_prob:{key}"] = top10_prob
            
            assert (not "replaceable" in query) or query["replaceable"] == replaceable[i]
            query["replaceable"] = replaceable[i]

            gold_rank = int(np.where(ranking == gold_answers[i])[0][0])+1
            assert (not f"gold_rank:{key}" in query) or query[f"gold_rank:{key}"] == gold_rank
            query[f"gold_rank:{key}"] = gold_rank
            
            gold_prob = round(float(prob[key][gold_answers[i]]), 5)
            assert (not f"gold_prob:{key}" in query) or query[f"gold_prob:{key}"] == gold_prob
            query[f"gold_prob:{key}"] = gold_prob

        if i == 0:
            print("TEST", i)
            print("QUERY", query)
            print("PROB", prob)
            import sys
            sys.exit()
            """
            QUERY { 'pred': '/people/person/place_of_birth', 'sub': '/m/047fprg', 'obj': '/m/04hqz', 'sub_w': None, 
               'sub_label': 'Khatchig Mouradian', 'sub_aliases': [], 'obj_w': 'Q822', 
               'obj_label': 'Lebanon', 'obj_aliases': ['Lebanese Republic', 'lb', 'Republic of Lebanon', '🇱🇧'], 'uuid': '9c68053f-0e3a-4db1-ab37-4e12d89b26d8',
        'masked_sentences': ['Khatchig Mouradian is a journalist, writer and translator born in [MASK] .'], 'wiki_titles': ['ENTITY/Khatchig_Mouradian'], 
            'predicate_id': 'place_of_birth', 'template': '[X] was born in [Y] .', 
              'query_bert': '[CLS] K ##hat ##chi ##g Mo ##ura ##dian was born in [MASK] . [SEP]', 
              'query_wiki': '[CLS] K ##hat ##chi ##g Mo ##ura ##dian was born in [MASK] . [SEP]', 
             'query_slash': '[CLS] K ##hat ##chi ##g Mo ##ura ##dian was born in [MASK] . [SEP]', 
               'attn:bert': [0.09187021106481552, 0.03477916866540909, ..., 0.44679582118988037], 
     'attn:linear_replace': [0.09187021106481552, 0.03477916866540909, ..., 0.44679582118988037], 
      'attn:linear_concat': [0.09187021106481552, 0.03477916866540909, ..., 0.44679582118988037], 
              'top10:bert': ('Tehran', 'Mumbai', 'Iran', 'India', 'Karachi', 'Dhaka', 'Afghanistan', 'Pakistan', 'Lucknow', 'Delhi'), 
         'top10_prob:bert': (0.10941494256258011, 0.08273160457611084, ..., 0.028987953439354897), 
             'replaceable': 0, 
          'gold_rank:bert': 50, 
          'gold_prob:bert': 0.00284, 
    'top10:linear_replace': ('Tehran', 'Mumbai', 'Iran', 'India', 'Karachi', 'Dhaka', 'Afghanistan', 'Pakistan', 'Lucknow', 'Delhi'), 
'top10_prob:linear_replace': (0.10941494256258011, 0.08273160457611084, ..., 0.028987953439354897), 
 'gold_rank:linear_replace': 50, 
 'gold_prob:linear_replace': 0.00284, 
      'top10:linear_concat': ('Tehran', 'Mumbai', 'Iran', 'India', 'Karachi', 'Dhaka', 'Afghanistan', 'Pakistan', 'Lucknow', 'Delhi'), 
 'top10_prob:linear_concat': (0.10941494256258011, 0.08273160457611084, .., 0.028987953439354897), 
  'gold_rank:linear_concat': 50, 
  'gold_prob:linear_concat': 0.00284, 
     'top10:linear_ens_avg': ('Tehran', 'Mumbai', 'Iran', 'India', 'Karachi', 'Dhaka', 'Afghanistan', 'Pakistan', 'Lucknow', 'Delhi'), 
'top10_prob:linear_ens_avg': (0.10941494256258011, 0.08273160457611084, .., 0.028987953439354897), 
 'gold_rank:linear_ens_avg': 50, 
 'gold_prob:linear_ens_avg': 0.00284, 
     'top10:linear_ens_max': ('Tehran', 'Mumbai', 'Iran', 'India', 'Karachi', 'Dhaka', 'Afghanistan', 'Pakistan', 'Lucknow', 'Delhi'), 
'top10_prob:linear_ens_max': (0.10941494256258011, 0.08273160457611084, .., 0.028987953439354897), 
 'gold_rank:linear_ens_max': 50, 
 'gold_prob:linear_ens_max': 0.00284}

            PROB {'bert': array([0., 0.], ), 'linear_replace': array([0., ..., 0.] ), 'linear_concat': array([0., 0.], ), 'linear_ens_avg': array([0., 0.], ), 'linear_ens_max': array([0., ., 0.], )}
            """



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type = str, default = "../data/LAMA/data")
    parser.add_argument("--methods", type = str, nargs = "+", default = ("linear",))
    parser.add_argument("--modelname", type = str, default = "bert-base-cased")
    parser.add_argument("--wikiname", type = str, default = "wikipedia2vec-base-cased")
    parser.add_argument("--device", type = str, default = "cuda")
    parser.add_argument("--uhn", action = "store_true", default = False)
    parser.add_argument("--infer_entity", action = "store_true", default = False)
    parser.add_argument("--allowed_vocabulary", type = str, default = "../resources/common_vocab_cased.txt")
    parser.add_argument("--output_dir", type = str, default = "../outputs/LAMA")
    return parser.parse_args()


if __name__ == "__main__":
    # (e-bert) diego@macrodeep:~/adv_comp_viz21/ebert-master/code$ 
    # python3 run_lama.py 
    #    --wikiname wikipedia2vec-base-cased 
    #    --modelname bert-base-cased 
    #    --data_dir ../data/LAMA/data 
    #    --output_dir ../outputs/LAMA 
    #    --infer_entity

    args = parse_args()
    
    patterns = [\
            {"relation": "place_of_birth", "template": "[X] was born in [Y] ."},
            {"relation": "date_of_birth", "template": "[X] (born [Y])."},
            {"relation": "place_of_death", "template": "[X] died in [Y] ."}]
    
    with open(os.path.join(args.data_dir, "relations.jsonl")) as handle:
        patterns.extend([json.loads(line) for line in handle])

    #print(patterns)   #49 Total
    # [{'relation': 'place_of_birth', 'template': '[X] was born in [Y] .'}, 
    #  {'relation': 'date_of_birth', 'template': '[X] (born [Y]).'}, 
    #  {'relation': 'place_of_death', 'template': '[X] died in [Y] .'}, 
    #  {'relation': 'P19', 'template': '[X] was born in [Y] .', 'label': 'place of birth', 'description': 'most specific known (e.g. city instead of country, or hospital instead of city) birth location of a person, animal or fictional character', 'type': 'N-1'}, 

    all_queries = []

    # LOAD WIKI EMBEDDINGs and BERT MODEL EMBEDDINGS
    wiki_emb = load_embedding(args.wikiname)
    model_emb = load_embedding(args.modelname)

    print(len(wiki_emb.dict_cache), len(wiki_emb.model.dictionary), "Wikipedia Embeddings length", args.wikiname)   # 2, 5068233 Wikipedia Embeddings length wikipedia2vec-base-cased
    print(len(model_emb.vocab), "BERT Embeddings length", args.modelname)   #28996 BERT Embeddings length bert-base-cased

    allowed_vocabulary = None
    # THIS IS A LIST OF WORDS WHICH MAY BE [MASK] ed.   Any queries whose [MASK] is not in allowed_vocab are removed
    if args.allowed_vocabulary:                                     #default = ../resources/common_vocab_cased.txt    #20970
        with open(args.allowed_vocabulary) as handle:
            lines = [line.strip() for line in handle]
        encoded = [model_emb.tokenizer.encode(token, add_special_tokens = False) for token in lines]
        assert all([len(x) == 1 for x in encoded])
        allowed_vocabulary = set([x[0] for x in encoded if len(x) == 1])

    # LOAD Embedding Input Model ( which looks like just BERT ) and BERT LM
    model = EmbInputBertModel.from_pretrained(args.modelname, output_attentions = True)
    language_model = BertForMaskedLM.from_pretrained(args.modelname).cls

    model = model.to(device = args.device)
    language_model = language_model.to(device = args.device)

    model.eval()
    language_model.eval()
    
    # LOAD linear mapper by default
    mappers = {method: load_mapper(f"{args.wikiname}.{args.modelname}.{method}") for method in args.methods}
    
    for pattern in tqdm(patterns):
        relation, template = pattern["relation"], pattern["template"]
        
        uhn_suffix = "_UHN" if args.uhn else ""
        
        if relation.startswith("P"):
            path = os.path.join(args.data_dir, f"TREx{uhn_suffix}/{relation}.jsonl")
        else:
            path = os.path.join(args.data_dir, f"Google_RE{uhn_suffix}/{relation}_test.jsonl")
        
        if not os.path.exists(path):
            continue
        
        with open(path) as handle:
            queries = [json.loads(line) for line in handle]        

        print(path, len(queries))  #../data/LAMA/data/Google_RE/place_of_birth_test.jsonl 2937
        #print( queries[0:100])
        #  1 EXAMPLE
        # {'pred': '/people/person/place_of_birth', 
        #  'sub': '/m/047fprg', 
        #  'obj': '/m/04hqz', 
        #  'evidences': [{'url': 'http://en.wikipedia.org/wiki/Khatchig_Mouradian', 'snippet': 'Khatchig Mouradian is a journalist, writer and translator born in Lebanon. He was one of the junior editors of the Lebanese-Armenian daily newspaper Aztag from 2000 to 2007, when he moved to Boston and became the editor of the Armenian Weekly. Mouradian holds a B.S. in biology and has studied towards a graduate degree in clinical psychology. He is working towards a PhD in Genocide Studies at Clark University http://www.clarku.edu/departments/holocaust/phd/research.cfm.', 'considered_sentences': ['Khatchig Mouradian is a journalist, writer and translator born in Lebanon .']}], 
        #  'judgments': [{'rater': '17082466750572480596', 'judgment': 'no'}, {'rater': '1014448455121957356', 'judgment': 'yes'}, {'rater': '16169597761094238409', 'judgment': 'yes'}, {'rater': '11595942516201422884', 'judgment': 'yes'}, {'rater': '2050861176556883424', 'judgment': 'yes'}], 
        #  'sub_w': None, 
        #  'sub_label': 'Khatchig Mouradian', 
        #  'sub_aliases': [], 
        #  'obj_w': 'Q822', 
        #  'obj_label': 'Lebanon', 
        #  'obj_aliases': ['Lebanese Republic', 'lb', 'Republic of Lebanon', '🇱🇧'],
        #  'uuid': '9c68053f-0e3a-4db1-ab37-4e12d89b26d8', 
        #  'masked_sentences': ['Khatchig Mouradian is a journalist, writer and translator born in [MASK] .']}     

        if args.infer_entity:
            label2uri = label2wikidata([query["sub_label"] for query in queries], verbose = False)   #see utils.util_wikidata
            all_uris = set()
            for label in label2uri:
                uris = list(label2uri[label])
                label2uri[label] = sorted(list(label2uri[label]), key = lambda x:int(x[1:]))
                all_uris.update(uris)

            uri2title = wikidata2title(all_uris, verbose = False)
            label2title = {}

            for label in label2uri:
                label2title[label] = []
                for uri in label2uri[label]:
                    label2title[label].extend(sorted(list(uri2title[uri])))

            for query in queries:
                query["wiki_titles"] = ["ENTITY/" + x for x in label2title[query["sub_label"]]]

        print(queries[0])  # 'wiki_titles': ['ENTITY/Khatchig_Mouradian']   , obj_label is the true [MASK] value
        assert all([len(model_emb.tokenizer.encode(query["obj_label"], add_special_tokens = False)) == 1 for query in queries])

        if "uncased" in args.modelname:
            for query in queries:
                obj_label = model_emb.tokenizer.tokenize(query["obj_label"])
                assert len(obj_label) == 1
                query["obj_label"] = obj_label[0]

        if allowed_vocabulary:
            for query in queries:
                if not model_emb.tokenizer.encode(query["obj_label"], add_special_tokens = False)[0] in allowed_vocabulary:
                    print(query["obj_label"], "not in allowed vocab")
            queries = [query for query in queries if \
                    model_emb.tokenizer.encode(query["obj_label"], add_special_tokens = False)[0] in allowed_vocabulary]
        
        if not relation.startswith("P"):
            for query in queries:
                query["predicate_id"] = query["pred"].split("/")[-1]
        

        if not args.infer_entity:
            if relation.startswith("P"):
                if "wikipedia2vec" in args.wikiname:
                    mapping = wikidata2title([query["sub_uri"] for query in queries], verbose = False)
                    for query in queries:
                        query["wiki_titles"] = ["ENTITY/" + x for x in mapping[query["sub_uri"]]]
                else:
                    for query in queries:
                        query["wiki_titles"] = [query["sub_uri"]]

            else:
                all_wikidata = [query["sub_w"] for query in queries if query["sub_w"]]
                all_titles = []
                for query in queries:
                    all_titles.extend([title_url2title(evidence["url"]) for evidence in query["evidences"]])

                if "wikipedia2vec" in args.wikiname:
                    mapping = wikidata2title(all_wikidata)
                else:
                    mapping = title2wikidata(all_titles)

                for query in queries:
                    query["wiki_titles"] = set()

                    if query["sub_w"]:
                        if "wikipedia2vec" in args.wikiname:
                            query["wiki_titles"].update(["ENTITY/" + x for x in mapping[query["sub_w"]]])
                        else:
                            query["wiki_titles"].add(query["sub_w"])

                    for title in [title_url2title(evidence["url"]) for evidence in query["evidences"]]:
                        if "wikipedia2vec" in args.wikiname:
                            query["wiki_titles"].add("ENTITY/" + title)
                        else:
                            query["wiki_titles"].update(mapping[title])
                
                    query["wiki_titles"] = sorted(list(query["wiki_titles"]))


        for query in queries:
            query["template"] = template
            if "evidences" in query:
                del query["evidences"]
            if "judgments" in query:
                del query["judgments"]
            if "masked_sentence" in query:
                del query["masked_sentence"]

        # START DOING TEST EVAL FOR EACH RELATION ( cool way to "describe" what you are looking at via tqdm )
        for method in tqdm(args.methods, desc = relation):
            test(queries, method, model, language_model, 
                    model_emb = model_emb, wiki_emb = wiki_emb, 
                    mapper = mappers[method],
                    allowed_vocabulary = allowed_vocabulary)

        inf = "_infer" if args.infer_entity else ""
    
        with open(os.path.join(args.output_dir, f"{relation}.{args.modelname}.{args.wikiname}{inf}.jsonl"), "w") as handle:
            handle.write("\n".join([json.dumps(query) for query in queries]))
    
        all_queries.extend(queries)
    
    with open(os.path.join(args.output_dir, f"all.{args.modelname}.{args.wikiname}{inf}.jsonl"), "w") as handle:
        handle.write("\n".join([json.dumps(query) for query in all_queries]))
        
