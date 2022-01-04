from src.tasks import kvqa_data  
from lxmert.lxmert.src.modeling_frcnn import GeneralizedRCNN
from lxmert.lxmert.src.processing_image import Preprocess
from transformers import LxmertTokenizer
from lxmert.lxmert.src.lxrt.tokenization import BertTokenizer
from lxmert.lxmert.src.lxmert_lrp_ebert import LxmertForQuestionAnswering as LxmertForQuestionAnsweringLRP    
from lxmert.lxmert.src.lxmert_lrp_ebert import LxmertEnhancedForQuestionAnswering as LxmertEnhancedForQuestionAnsweringLRP    #holds EBERT
from lxmert.lxmert.src.lxmert_lrp_ebert import convert_sents_to_features, toks_to_str
from lxmert.lxmert.src.param import args

from tqdm import tqdm
from torch import topk

import lxmert.lxmert.src.vqa_utils as utils
import collections
import faiss
import json
import numpy as np
import os
import random
import time
import torch
from collections import Counter
from torch.utils.data.dataloader import DataLoader

KVQA_VAL_PATH = '/data/diego/adv_comp_viz21/KVQAimgs/' 
KVQA_IMGFEAT_ROOT = '/data/diego/adv_comp_viz21/imgfeat/'
KVQA_URL = '/home/diego/adv_comp_viz21/lxmert/orig_code/lxmert/data/kvqa/'

from src.pretrain.qa_answer_table import load_lxmert_qa_hf   
from src.tasks.kvqa_data import KVQADataset, KVQATorchDataset, KVQAEvaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False, incl_para=False, incl_caption=False, use_lm=None, ent_set=None, use_capt_sep= None) -> DataTuple:
    dset = KVQADataset(splits, incl_para, incl_caption, use_lm, ent_set, use_capt_sep)
    tset = KVQATorchDataset(dset)
    evaluator = KVQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class ModelPert:
    def __init__(self, split_str, incl_caption, use_lm, ent_set, use_capt_sep, save_training, k ):
        self.incl_caption = incl_caption
        self.use_lm = use_lm
        self.ent_set = ent_set
        self.use_capt_sep = use_capt_sep
        self.save_training = save_training
        self.split_str = split_str
        self.k = k

        self.KVQA_VAL_PATH = KVQA_VAL_PATH    #path to IMAGES
        split_num = split_str.split("kvqa")[1]
        self.KVQA_trainval_labs = KVQA_URL + "trainval_label2ans_"+str(split_num)+".json"

        self.kvqa_answers = utils.get_data(self.KVQA_trainval_labs)  #ans id to ans str
        self.ans_str2ans_id = {v:k for k,v in enumerate(self.kvqa_answers)}

        # load models and model components
        self.frcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn_cfg.MODEL.DEVICE = "cuda"
        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg)

        self.image_preprocess = Preprocess(self.frcnn_cfg)
        self.lxmert_tokenizer = BertTokenizer.from_pretrained( "bert-base-uncased", do_lower_case=True)

        if use_lm:
            self.lxmert_vqa = LxmertEnhancedForQuestionAnsweringLRP.from_pretrained("unc-nlp/lxmert-vqa-uncased")
        else:
            self.lxmert_vqa = LxmertForQuestionAnsweringLRP.from_pretrained("unc-nlp/lxmert-vqa-uncased")

        self.lxmert_vqa.resize_num_qa_labels(len(self.kvqa_answers))
        self.load(args.load)
        self.data_tuple = get_data_tuple(split_str, bs=1, shuffle=False, drop_last=False, incl_para = False, incl_caption = self.incl_caption, use_lm = self.use_lm, ent_set = self.ent_set, use_capt_sep = self.use_capt_sep)
        self.lxmert_vqa.to("cuda")

        self.lxmert_vqa.eval()
        self.model = self.lxmert_vqa

        if save_training == False:
            # LOAD APPROPRIATE NUMPY FILE OF TRAIN RESULTS
            # FAISS setup based on https://github.com/diegoolano/biomedical_interpretable_entity_representations  ELC COLAB link
            if ent_set == None:
                ent_set = "oracle"
            if args.tiny:
                ent_set += "_tiny"
            train_embeds_file = "train_embeds_"+split_num+"_"+use_lm+"_"+ent_set+".npy"
            print("Loading ", train_embeds_file)
            train_vecs = np.load(train_embeds_file)
            train_vecs_2d = train_vecs.squeeze()
            #cpu_index = faiss.IndexFlatIP(train_vecs_2d.shape[1])      #for Inner Product

            if args.dataset_wise_norm:
                print("Applying Dataset Wise normalization to training data")
                self.mu = train_vecs_2d.mean(axis=0)
                self.sig = train_vecs_2d.std(axis=0)
                self.eps=.0000001
                train_vecs_2d = (train_vecs_2d - self.mu)/(self.sig + self.eps)

            cpu_index = faiss.IndexFlatL2(train_vecs_2d.shape[1])     
            n_gpu = 1
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            self.gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=co, ngpu=n_gpu)
            print('Adding dataset to index...')
            t0 = time.time()    
            self.gpu_index.add(train_vecs_2d)
            print("Done.  Elapsed:", time.time() - t0)
            if args.tiny:
                split_num += "_tiny"
            self.train_ans = json.load(open("train_ans_"+split_num+".json"))



    def load(self, path):
        state_dict = torch.load("%s.pth" % path)
        lxmert_state_dict = {}
        answer_head_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('lxrt'):
                lxmert_state_dict[key.replace("lxrt_encoder.model.bert.","")] = value
            else:
                answer_head_state_dict[key] = value

        assert(len(state_dict.keys()) ==  ( len(lxmert_state_dict) +  len(answer_head_state_dict)))
        self.lxmert_vqa.lxmert.load_state_dict(lxmert_state_dict)
        self.lxmert_vqa.answer_head.load_state_dict(answer_head_state_dict)


    def get_image_path(self, img_id):
        image_file_path = self.KVQA_VAL_PATH + img_id + '.jpg'
        if not os.path.exists(image_file_path):
            exts = ['.JPG', '.JPEG', '.png', '.PNG', '.jpeg']
            for e in exts:
                image_file_path = self.KVQA_VAL_PATH + img_id + e
                if os.path.exists(image_file_path):
                    break
        return image_file_path

    def get_knn_pred(self, top5, top5_probs, dist, ids, lmb, adict, datum, debug = False):
        if debug:
            print("FROM MODEL:",adict)
            print(dist)  #distances
            print(ids)  #indexes
            print("KNN ANS:", [ self.train_ans[ind] for ind in ids[0]])
            #print("True Ans in Question list:", true_ans in self.ans_str2ans_id)   #gets false answer ( ie, true answer not in softmax! )
            #print(len(self.ans_str2ans_id))  #18334
            #print(self.ans_str2ans_id)   #correctly formated
            """
            FROM MODEL
            {'ents': 1, 'true_ans': 'Oliver Foot', 
             'top5': ['Marcel Marceau', 'Neil Young', 'Jacques Brel', 'Anna Karina', 'Keith Richards'], 
             'top5probs': [-4.6337, -5.62944, -5.79947, -6.00482, -6.05778], 'top1acc': 0, 'top5acc': 0, 
              'sent': 'Who is the person in the image? Oliver Foot Isaac Foot'

            FROM KNN
            D = [[21.189514 23.747742 23.860962 25.699524 26.741638]]
            I = [[1355  985 1913 1733  847]]
            knn ANS: ['Francis Huster', 'Richard Greene', 'Philippe Peythieu', 'Anita Thallaug', 'Michèle Morgan']
            """

        # GENERALIZATION THROUGH MEMORIZATION: NEAREST NEIGHBOR LANGUAGE MODELS.  Urvashi Khandelwal†, et al, ICLR 2020
        # p(y|x) = λ pkNN(y|x) + (1 − λ) pLM(y|x)
        plm_dist_norm = np.exp(top5_probs)/sum(np.exp(top5_probs))
        pknn_dist_norm = np.exp(-1 * dist[0]) / sum(np.exp(-1 * dist[0]))

        keys = top5 + [ self.train_ans[v] for v in ids[0]]
        vals = np.concatenate(((lmb * plm_dist_norm ), ((1- lmb) * pknn_dist_norm)))
        plm_pknn = zip(keys, vals)
        py = Counter()
        for k,v in plm_pknn:
            py.update({k:v})
        fin = py.most_common()
        if debug:
            print("P_lm =", [ v for v in zip(top5, lmb * plm_dist_norm) ])
            print("P_knn =", [ (self.train_ans[v[0]],v[1]) for v in zip(ids[0], (1-lmb) * pknn_dist_norm) ])
            print("plm_pknn =", [ v for v in plm_pknn])
            print("Py = ", fin )

        knntop5 = [ k for k,v in fin ]
        knntop5_probs = [ round(float(v),5) for k,v in fin]
        knntop1_acc = datum["label"].get(knntop5[0], 0)
        knntop5_acc = 1 if list(datum["label"].keys())[0] in knntop5 else 0
        adict['knn'][lmb] = {}
        adict['knn'][lmb]['top5'] = knntop5
        adict['knn'][lmb]['top5probs'] = knntop5_probs
        adict['knn'][lmb]['top1acc'] = knntop1_acc
        adict['knn'][lmb]['top5acc'] =  knntop5_acc

        # Explaining and Improving Model Behavior with k Nearest Neighbor Representations.  Nazneen Rajani, et al. arxiv 2020
        #  Our method deploys backoff to kNN (using CLS) for BERT and RoBERTa on examples with low model confidence without any update to the model parameters. 
        #  they apply dataset-wise batch normalization on the CLS reps (over subset of training examples)  and use temperature scaling instead of interpolating
        zpy = Counter()
        naz = zip([ self.train_ans[v] for v in ids[0]], pknn_dist_norm)
        for k,v in naz:
            zpy.update({k:v})
        finz = zpy.most_common()
        adict['naz'][lmb] = {}
        adict['naz'][lmb]['top5'] = [  k for k,v in finz ]
        adict['naz'][lmb]['top5probs'] = [ round(float(v),5) for k,v in finz ]
        adict['naz'][lmb]['top1acc'] = datum["label"].get(adict['naz'][lmb]['top5'][0], 0)
        adict['naz'][lmb]['top5acc'] =  1 if list(datum["label"].keys())[0] in adict['naz'][lmb]['top5'] else 0
        
        #the above is without temperature scaling
        # https://docs.aws.amazon.com/prescriptive-guidance/latest/ml-quantifying-uncertainty/temp-scaling.html
        # the temperature T usually scales between 1.5 and 3.
        T = 1.5
        pknn_dist_norm_t = np.exp((-1 * dist[0])/T) / sum(np.exp((-1 * dist[0])/T))
        zpy = Counter()
        naz = zip([ self.train_ans[v] for v in ids[0]], pknn_dist_norm_t)
        for k,v in naz:
            zpy.update({k:v})
        fint = zpy.most_common()
        adict['nazt'][lmb] = {}
        adict['nazt'][lmb]['top5'] = [  k for k,v in fint ]
        adict['nazt'][lmb]['top5probs'] = [ round(float(v),5) for k,v in fint ]
        adict['nazt'][lmb]['top1acc'] = datum["label"].get(adict['nazt'][lmb]['top5'][0], 0)
        adict['nazt'][lmb]['top5acc'] =  1 if list(datum["label"].keys())[0] in adict['nazt'][lmb]['top5'] else 0

        if debug:
            print(adict)
            import sys
            sys.exit()
        return adict

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        If self.save_training, save training reps with answers in FAISS key-value store 
        Else Predict the answers to questions in a data split using knn LM interpolation

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        debug = False
        vset, vloader, vevaluator = eval_tuple
        viter_wrapper = (lambda x: tqdm(x, total=len(vloader))) 
        qid2imgid = vset.id2datum  #question_id to data
        quesid2ans = {}
        st = time.time()
        train_reps = []
        train_ans = []
        for i, (ques_id, feats, boxes, sent, ent_spans, target) in viter_wrapper(enumerate(vloader)):
            if True:
                #max_seq_len = -1   #last time when I had batches I used 100.. now use -1 to me no max_seq_len .. see if emb output < tokens due to this
                max_seq_len = 50   #the above was causing a memory error
                train_features, tokens = convert_sents_to_features( sent, ent_spans, max_seq_len, self.lxmert_tokenizer, self.use_lm)
                input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
                input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
                segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
                surface2wiki = []
                
                if debug:
                    print("USE LM=",self.use_lm, ques_id)
                    print("SENTS", len(sent), "Sent 0",sent[0])
                    print("InputIds", input_ids.shape)
                    print("TOKENS",len(tokens), len(tokens[0]))             #32, 18    32 in batch, each sent of len 18
                    print("ENT SPANS",len(ent_spans), len(ent_spans[0]), len(ent_spans[0][0]))  #11 spans each of size 4 for 32 sent in batch
                    print("ENT spans[0][0]", ent_spans[0][0])
                    print("ENT spans[0][1]:", ent_spans[0][1])
                    print("ENT spans[0][2]:", ent_spans[0][2])
                    if len(ent_spans[0]) == 4:
                        print("ENT spans[0][3]:", ent_spans[0][3])
    
                for ii, t in enumerate(tokens):
                    curt = toks_to_str(t)
                    ps = curt.split("[UNK]")
                    out = ""
                    for ind in range(len(ps)):
                        if ind % 2 == 0 or ind == len(ps):
                            out += ps[ind] + "\n"
                        else:
                            out += "[UNK]"+ps[ind]+"[UNK]"
                    if debug: 
                        print("\nExample ",ii) #, curt)
                        print(out)       

                    if len(ent_spans[0]) == 4:
                        wiklist = []
                        for sp in range(11):
                            if ent_spans[sp][0][ii] != '':
                                wiklist.append([ent_spans[sp][0][ii], int(ent_spans[sp][1][ii]), int(ent_spans[sp][2][ii]), ent_spans[sp][3][ii]])
                        #dedup and sort by [1]
                        wiklist_dedup = list(set(tuple(sub) for sub in wiklist))
                        wiklist_dedup.sort(key = lambda x: x[1])
                        wiklist = wiklist_dedup
                        if wiklist != []: 
                            if debug:
                                print(wiklist)
                        surface2wiki.append(wiklist)
                if debug:
                    print("FINAL surface2wiki:", len(surface2wiki), surface2wiki)

                if surface2wiki == []:
                    surface2wiki = None

                self.output = self.lxmert_vqa(
                    input_ids=input_ids.to("cuda"),
                    attention_mask=input_mask.to("cuda"),
                    visual_feats=feats.to("cuda"),
                    visual_pos=boxes.to("cuda"),
                    token_type_ids=segment_ids.to("cuda"),
                    surface2wiki = surface2wiki,
                    return_dict=True,
                    output_attentions=False,
                )
                #output_hidden_states=self.save_training      #for saving of training

                datum = qid2imgid[ques_id[0]]
                true_ans = list(datum['label'].keys())[0]
                sent = datum['sent']
                if self.save_training == False:
                    # do inference over test set using knn

                    if self.output.embedding_output != []:
                        tokens = self.output.embedding_output  #with batches of one this should be fine
                    elif len(tokens) == 1:
                        tokens = tokens[0]
                    self.text_len = len(tokens) if max_seq_len == -1 else max_seq_len  #since batch size of 1

                    answer_dist = self.output.question_answering_score
                    answer_top = answer_dist.argmax()
                    answer_topk_v, answer_topk_i = topk(answer_dist, self.k)
                    answer = self.kvqa_answers[answer_top]
                    top5 = [self.kvqa_answers[ind] for ind in answer_topk_i.data.view(-1)]
                    top5_probs = [ round(float(pr),5) for pr in answer_topk_v.data.view(-1)]
                    top1_acc = datum["label"].get(answer, 0)
                    top5_acc = 1 if list(datum["label"].keys())[0] in top5 else 0
                    if self.use_lm == "ebert": 
                        entslen = len(surface2wiki[0])
                    else:
                        entslen = 0
                    
                    adict = {'ents': entslen, 'true_ans': true_ans,'top5': top5, 'top5probs': top5_probs, 
                             'top1acc':top1_acc, 'top5acc': top5_acc, 'sent': sent, 'tokens': tokens,
                             'knn':{}, 'naz':{}, 'nazt':{}} 
                    v = self.output.pooled_output.cpu().detach().numpy()

                    if args.dataset_wise_norm:
                        v = (v - self.mu)/(self.sig + self.eps)

                    dist, ids = self.gpu_index.search(v, k=self.k)

                    # should be using dev set to tune lambda
                    #for lmb in [.01, .05, .1, .2, .5, .75, .8, .85, .9, .95, .99]:
                    #for lmb in [.947, .948, .949, .9495, .95, .9505, .951, .952, .953 ]:
                    #lmbs = .95
                    for lmb in [.9, .905, .91, .915, .92, .925, .93, .935, .94, .945,  .95, .955, .96, .965, .97, .975, .98, .985, .99]:
                        adict = self.get_knn_pred(top5, top5_probs, dist, ids, lmb, adict, datum, debug)

                    # TODO FIGURE OUT WHAT IS BEST LAYER TO USE AND BEST METHOD FOR INTERPOLATION BETWEEN THE TWO METHODS

                    # TMRW FIGURE OUT FROM HERE (from paper code or QA paper code ) AND READ ON PROMPT TUNING !
                    quesid2ans[ques_id[0]] = adict
                else:
                    #Gather vision/language cross representations
                    """
                    txt_rep =  self.output.language_hidden_states
                    img_rep =  self.output.vision_hidden_states  
                    #LXMERT IS 9 5 5 ( language / cross / vision ) 
                    print("TXT rep",type(txt_rep), len(txt_rep), [ v.shape for v in txt_rep ])  #<class 'tuple'> 14 [torch.Size([1, 16, 768]), torch.Size([1, 16, 768]), ... ]  9 + 5
                    print("IMG rep",type(img_rep), len(img_rep), [ v.shape for v in img_rep ])  #<class 'tuple'> 10 [torch.Size([1, 36, 768]), torch.Size([1, 36, 768]), ... ]  5 + 5
                    print(dir(self.output))  #LxmertForQuestionAnsweringOutput
                    tout = self.output.language_output.squeeze()
                    vout = self.output.vision_output
                    pout = self.output.pooled_output
                    print(type(tout), tout.shape)
                    for i in range(len(txt_rep)):
                        c = txt_rep[i].squeeze()
                        print(i, type(c), c.shape, c[0].argmax(), self.kvqa_answers[c[0].argmax()])  ### txt_rep[15] == tout[0]
                        c = tout[i]
                        print(i, type(c), c.shape, c.argmax(), self.kvqa_answers[c.argmax()])
                    """
                    #train_reps.append(self.output.question_answering_score.cpu().detach())  # saving softmax is probably too sparse so go back to dense rep
                    # include true ans number here as well so you can get rep and ans during inference
                    val = self.output.pooled_output.squeeze().cpu().detach().numpy() 
                    train_reps.append(val)
                    train_ans.append(true_ans)


        print("Done. Elapsed:", time.time() - st)

        if self.save_training == False:
            t1acc = round(np.mean([ quesid2ans[v]['top1acc'] for v in quesid2ans]),5)
            t5acc = round(np.mean([ quesid2ans[v]['top5acc'] for v in quesid2ans]),5)
            print("top1/top5 Acc using LM model", t1acc, t5acc)
            #for lmb in [.01, .05, .1, .2, .5, .75, .8, .85, .9, .95, .99]:
            #for lmb in [.947, .948, .949, .9495, .95, .9505, .951, .952, .953 ]:
            #for lmb in [.95]:
            for lmb in [.9, .905, .91, .915, .92, .925, .93, .935, .94, .945,  .95, .955, .96, .965, .97, .975, .98, .985, .99]:
                knn_t1acc = round(np.mean([ quesid2ans[v]['knn'][lmb]['top1acc'] for v in quesid2ans]),5)
                knn_t5acc = round(np.mean([ quesid2ans[v]['knn'][lmb]['top5acc'] for v in quesid2ans]),5)
                print("top1/top5 Acc using kNN model with lambda=", lmb, knn_t1acc, knn_t5acc)
                naz_t1acc = round(np.mean([ quesid2ans[v]['naz'][lmb]['top1acc'] for v in quesid2ans]),5)
                naz_t5acc = round(np.mean([ quesid2ans[v]['naz'][lmb]['top5acc'] for v in quesid2ans]),5)
                print("top1/top5 Acc using naz model with lambda=", lmb, naz_t1acc, naz_t5acc)
                naz_t1acc = round(np.mean([ quesid2ans[v]['nazt'][lmb]['top1acc'] for v in quesid2ans]),5)
                naz_t5acc = round(np.mean([ quesid2ans[v]['nazt'][lmb]['top5acc'] for v in quesid2ans]),5)
                print("top1/top5 Acc using nazt model with lambda=", lmb, naz_t1acc, naz_t5acc)

            if args.dataset_wise_norm:
                dump = dump.replace(".json","_datawise_normed.json")

            print("Save results to ",dump)
            with open(dump, 'w') as outfile:
                json.dump(quesid2ans, outfile)
        else:
            train_embeddings = np.stack(train_reps)
            split_num = self.split_str.split("kvqa")[1]
            ent_set = self.ent_set
            if self.ent_set == None:
                ent_set = "oracle"
            if args.tiny:
                ent_set += "_tiny"
            train_embeds_file = "train_embeds_"+split_num+"_"+self.use_lm+"_"+ent_set
            print("SAVING ", train_embeds_file, train_embeddings.shape)    #SAVING  train_embeds_1_ebert_oracle.py (565, 1, 18335)  .. 51 secs using softmax
            np.save(train_embeds_file, train_embeddings)
            if args.tiny:
                split_num += "_tiny"
            train_ans_f = "train_ans_"+split_num+".json"
            print("SAVING ", train_ans_f)
            with open(train_ans_f,"w") as fp:
                json.dump(train_ans, fp)
             

def main(args):
    
    #TODOS: 
    #1. make sure this works with Training data for split 0 ( and saves numpy file )
    #2. make sure it works with testing data and loading appropriate numpy file (pass in --save_training flag )
    if args.test is not None:
        split = args.test + args.split_num
    else:
        split  = args.train + "_kvqa" + args.split_num   

    save_training = False
    if args.save_training:
        save_training = True

    print("LOADING ", split, "AND Save Training=", save_training, "WITH K", args.k)
    model_pert = ModelPert(split, args.incl_caption, args.use_lm, args.ent_set, args.use_capt_sep, save_training, int(args.k))     #expecting commaseparate 0,1,2,3  etc

    # if save_training, predict will save train reps with answers to FAISS 
    # otherwise it will do inference using model and faiss for knn LM stuff
    predictions = model_pert.predict(model_pert.data_tuple, dump=args.pred_out)


if __name__ == "__main__":
    main(args)
