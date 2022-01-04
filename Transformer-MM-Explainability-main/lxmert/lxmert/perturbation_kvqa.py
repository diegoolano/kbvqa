#from lxmert.lxmert.src.tasks import vqa_data  #
from src.tasks import kvqa_data  

from lxmert.lxmert.src.modeling_frcnn import GeneralizedRCNN
import lxmert.lxmert.src.vqa_utils as utils
from lxmert.lxmert.src.processing_image import Preprocess

from transformers import LxmertTokenizer
from lxmert.lxmert.src.lxrt.tokenization import BertTokenizer

#from lxmert.lxmert.src.huggingface_lxmert import LxmertForQuestionAnswering
from lxmert.lxmert.src.lxmert_lrp_ebert import LxmertForQuestionAnswering as LxmertForQuestionAnsweringLRP    
from lxmert.lxmert.src.lxmert_lrp_ebert import LxmertEnhancedForQuestionAnswering as LxmertEnhancedForQuestionAnsweringLRP    #holds EBERT
from lxmert.lxmert.src.lxmert_lrp_ebert import convert_sents_to_features, toks_to_str

from tqdm import tqdm
from lxmert.lxmert.src.ExplanationGenerator import GeneratorOurs, GeneratorBaselines, GeneratorOursAblationNoAggregation
from torch import topk
from lxmert.lxmert.src.param import args

import collections
import json
import os
import random
import time
import torch
from collections import Counter
from torch.utils.data.dataloader import DataLoader

KVQA_VAL_PATH = '/data/diego/adv_comp_viz21/KVQAimgs/' 
KVQA_IMGFEAT_ROOT = '/data/diego/adv_comp_viz21/imgfeat/'
KVQA_URL = '/home/diego/adv_comp_viz21/lxmert/orig_code/lxmert/data/kvqa/'

from src.pretrain.qa_answer_table import load_lxmert_qa_hf   #should this be src or lxmert.lxmert.src

#from src.tasks.kvqa_model import KVQAModel
from src.tasks.kvqa_data import KVQADataset, KVQATorchDataset, KVQAEvaluator

from collections import Counter
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

# ADD use_lm
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
    def __init__(self, split_str, incl_caption, use_lm, ent_set, use_capt_sep, test=None, use_lrp=False):
        self.use_lm = use_lm
        self.ent_set = ent_set
        self.use_capt_sep = use_capt_sep
        self.incl_caption = incl_caption

        self.KVQA_VAL_PATH = KVQA_VAL_PATH    #path to IMAGES
        split_num = split_str.split("kvqa")[1]
        self.KVQA_trainval_labs = KVQA_URL + "trainval_label2ans_"+str(split_num)+".json"

        self.kvqa_answers = utils.get_data(self.KVQA_trainval_labs)
        #print("VQA answers:", type(self.kvqa_answers))

        # load models and model components
        self.frcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn_cfg.MODEL.DEVICE = "cuda"
        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg)

        self.image_preprocess = Preprocess(self.frcnn_cfg)
        #self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.lxmert_tokenizer = BertTokenizer.from_pretrained( "bert-base-uncased", do_lower_case=True)

        if use_lrp:
            # TODO.. I didn't base things on lxmert-vqa-uncased so how to deal with this??  
            if use_lm:
                self.lxmert_vqa = LxmertEnhancedForQuestionAnsweringLRP.from_pretrained("unc-nlp/lxmert-vqa-uncased")
            else:
                self.lxmert_vqa = LxmertForQuestionAnsweringLRP.from_pretrained("unc-nlp/lxmert-vqa-uncased")

            self.lxmert_vqa.resize_num_qa_labels(len(self.kvqa_answers))

            if test == None:
                #self.kvqa_dataset = kvqa_data.KVQADataset(splits=split_str, incl_para=False, incl_caption=incl_caption, use_lm=use_lm, debug=False)   #TODO handle rest
                self.data_tuple = get_data_tuple(split_str, bs=1, shuffle=False, drop_last=False, incl_para = False, incl_caption = self.incl_caption, use_lm = self.use_lm)
                load_lxmert_qa_hf("/home/diego/adv_comp_viz21/lxmert/orig_code/lxmert/snap/pretrained/model", self.lxmert_vqa, label2ans=self.kvqa_dataset.label2ans)  #do i need this?
                self.lxmert_vqa.to("cuda")
            else:
                # loading non-finetuned
                self.load(args.load)
                self.data_tuple = get_data_tuple(split_str, bs=1, shuffle=False, drop_last=False, incl_para = False, incl_caption = self.incl_caption, use_lm = self.use_lm, ent_set = self.ent_set, use_capt_sep = self.use_capt_sep)
                self.lxmert_vqa.to("cuda")
        else:
            # never gets used for now
            self.lxmert_vqa = None #LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased").to("cuda")

        self.lxmert_vqa.eval()
        self.model = self.lxmert_vqa

        self.pert_steps = [0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
        self.pert_acc = [0] * len(self.pert_steps)


    def load(self, path):
        #print("!!! Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        #print("Need to load keys: ", state_dict.keys()) 
        #print("\n\nInto Code expecting: LXMERT keys:", self.lxmert_vqa.lxmert.state_dict().keys())
        #print("\n\n AND LXMERT ANSWER HEAD: ", self.lxmert_vqa.answer_head.state_dict().keys())
        #print("Lens: ", len(state_dict.keys()), " = ", len(self.lxmert_vqa.lxmert.state_dict().keys()) ,"+", len(self.lxmert_vqa.answer_head.state_dict().keys()))

        #parse state_dict to load correct keys for lxmert_vqa.lxmert vs lxmert_vqa.answer_head
        lxmert_state_dict = {}
        answer_head_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('lxrt'):
                lxmert_state_dict[key.replace("lxrt_encoder.model.bert.","")] = value
            else:
                answer_head_state_dict[key] = value

        #print(len(state_dict.keys()), " = ", len(lxmert_state_dict),"+", len(answer_head_state_dict))
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


    def predict(self, eval_tuple: DataTuple, expl_ours=None, baselines=None, dump=None):
        """
        Predict the answers to questions in a data split.

        :param expl: ExplainerGenerator class
        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        #self.model.eval()
        vset, vloader, vevaluator = eval_tuple
        #viter_wrapper = (lambda x: tqdm(x, total=len(vloader))) if args.tqdm else (lambda x: x)
        viter_wrapper = (lambda x: tqdm(x, total=len(vloader))) 
        #print("VSET label size", len(vset.label2ans), vset.label2ans[0], type(vset.label2ans[0]))  #18383 

        qid2imgid = vset.id2datum  #question_id to data
        #print("Len qid2imgid: ",len(qid2imgid))          #18697 using vset.data

        debug = False
        quesid2ans = {}
        ans_not_found = 0
        top1_right = 0
        top5_right = 0
        st = time.time()
        #these boxes have been normalized already
        for i, (ques_id, feats, boxes, sent, ent_spans, target) in viter_wrapper(enumerate(vloader)):
            if True:
                # prior code
                # logit = self.model(feats, boxes, sent, ent_spans)  #function expects ent spans no matter what now
              
                #kvqa_model
                #x = self.lxrt_encoder(sent, ent_spans, (feat, pos))  #from lxrt.entry import LXRTEncoder   
                #logit = self.logit_fc(x)
                #score, label = logit.max(1)

                # NOW similar to forward() in lxrt_encoder
                max_seq_len = -1   #last time when I had batches I used 100.. now use -1 to me no max_seq_len .. see if emb output < tokens due to this
                #max_seq_len = 100   # this leads to an error so not this way.. try to do this in code

                #with -1, we get
                # SENT ['Who is the person in the image? Hartmut Nassauer.']
                # TOKENS [['[CLS]', 'who', 'is', 'the', 'person', 'in', 'the', 'image', '?', '[UNK]', 'hart', '##mut', 'nassau', '##er', '[UNK]', '.', '[SEP]']] 1
                # EMB OUTPUT ['[CLS]', 'who', 'is', 'the', 'person', 'in', 'the', 'image', '?', '.', '[SEP]'] 11

                #with 100 we get ...

                train_features, tokens = convert_sents_to_features( sent, ent_spans, max_seq_len, self.lxmert_tokenizer, self.use_lm)

                input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
                input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
                segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
               
                # lxmert_lrp_ebert lxmert_vqa forward pass expects the following
                #def forward( self, input_ids=None, visual_feats=None, visual_pos=None, attention_mask=None, visual_attention_mask=None, token_type_ids=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None,

                surface2wiki = []
                
                if debug:
                    print("USE LM=",self.use_lm, ques_id)
                    print("SENTS", len(sent), "Sent 0",sent[0])
                    print("InputIds", input_ids.shape)
                    print("TOKENS",len(tokens), len(tokens[0]))             #32, 18    32 in batch, each sent of len 18
                    print("ENT SPANS",len(ent_spans), len(ent_spans[0]), len(ent_spans[0][0]))  #11 spans each of size 4 for 32 sent in batch
                    #print("Ent spans[0]", ent_spans[0:10][0:4][0])                   #sent 0 in batch
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

                # instead of returning tokens, try to return fin_tok!
                if self.use_lm == "ebert": 
                    if debug:
                        print("LOOKING AT OUTPUT for EBERT")
                        print(self.output)  #is embedding_output here
                        print("SENT", sent)
                        print("TOKENS", tokens, len(tokens))
                        print("EMB OUTPUT",self.output.embedding_output, len(self.output.embedding_output))   #these are ebert enhaned tokens
                       
                img_id = ques_id[0].split("_")[0]

                if self.output.embedding_output != []:
                    tokens = self.output.embedding_output  #with batches of one this should be fine
                elif len(tokens) == 1:
                    tokens = tokens[0]
                self.text_len = len(tokens) if max_seq_len == -1 else max_seq_len  #since batch size of 1
                self.image_boxes_len = feats.shape[1]  

                if debug:
                    print(ques_id, sent, input_ids.shape, self.text_len, self.image_boxes_len, feats.shape , boxes.shape)
                item = None

                # TODO instead of just hard coding to "ours_no_lrp" also do a couple others
                methods = {"ours_no_lrp":{}, 'transformer_att':{}}
                for method_name in methods:   #can we run this multiple times wihtout needing to re-call model.forward ?
                    if method_name == 'transformer_att':
                        R_t_t, R_t_i = baselines.generate_transformer_attr(item, index=None, method_name="transformer_attr", model = self.lxmert_vqa, output = self.output.question_answering_score )   
                    else:
                        use_lrp = False
                        R_t_t, R_t_i = expl_ours.generate_ours(item, use_lrp=use_lrp, model = self.lxmert_vqa, output = self.output.question_answering_score)  
    
                    cam_image = R_t_i[0]  #connections between [CLS] token and each image token
                    cam_text = R_t_t[0]   #influence of tokens on themselves (self attention)
    
                    cam_image = (cam_image - cam_image.min()) / (cam_image.max() - cam_image.min())
                    cam_text = (cam_text - cam_text.min()) / (cam_text.max() - cam_text.min())
                    text_expl = [round(float(v),4) for v in cam_text]
                    image_expl = [round(float(v),4) for v in cam_image]
                    methods[method_name] = {"text_expl":text_expl, "image_expl":image_expl}


                answer_dist = self.output.question_answering_score
                answer_top = answer_dist.argmax()
                answer_topk_v, answer_topk_i = topk(answer_dist, 5)
                answer = self.kvqa_answers[answer_top]

                datum = qid2imgid[ques_id[0]]
                top5 = [self.kvqa_answers[ind] for ind in answer_topk_i.data.view(-1)]
                top5_probs = [ round(float(pr),5) for pr in answer_topk_v.data.view(-1)]
                top1_acc = datum["label"].get(answer, 0)
                top5_acc = 1 if list(datum["label"].keys())[0] in top5 else 0

                #entslen = len(datum['NamedEntities'])
                if self.use_lm == "ebert": 
                    if debug:
                        print("USING EBERT WITH ENTS:",surface2wiki, len(surface2wiki), len(surface2wiki[0]))
                    entslen = len(surface2wiki[0])
                else:
                    entslen = 0
                true_ans = list(datum['label'].keys())[0]
                sent = datum['sent']
                save_boxes = [ [round(float(a),5) for a in v] for v in boxes.squeeze() ]
                
                #quesid2ans[ques_id[0]] = {'ents': entslen, 'true_ans': true_ans,'top5': top5, 'top5probs': top5_probs, 'top1acc':top1_acc, 'top5acc': top5_acc, 'sent': sent, 'text_expl': text_expl, 'image_expl': image_expl, 'bboxes': save_boxes, 'tokens': tokens} 
                quesid2ans[ques_id[0]] = {'ents': entslen, 'true_ans': true_ans,'top5': top5, 'top5probs': top5_probs, 'top1acc':top1_acc, 'top5acc': top5_acc, 'sent': sent, 'bboxes': save_boxes, 'tokens': tokens, 'expl_methods': methods} 

                top1_right += top1_acc
                top5_right += top5_acc

                curr_acc_result = [ round(top1_right/(i+1) * 100, 2), round(top5_right/(i+1) * 100, 2), top1_right, top5_right, i+1 ]  
                #if (i+1) % 100 == 0:
                #    print("top1/top5 Acc/Raw: {}".format(curr_acc_result))

                #print(ques_id, sent, cam_image.shape, cam_text.shape, [round(float(v),5) for v in cam_image], [round(float(v),5) for v in cam_text])
                # ['14251_0'] Who is the person in the image? Jan Klusák in Karlovy Vary (2009) torch.Size([36]) torch.Size([24])


        print("Done. Elapsed:", time.time() - st)
        print("top1/top5 Acc/Raw: {}".format(curr_acc_result))
        print("Save results to ",dump)
        with open(dump, 'w') as outfile:
            json.dump(quesid2ans, outfile)

# addability to pass through name of file out
# now test three types ( plain, capt, ebert )
# now add explain back

def main(args):
    
    if args.test is not None:
        split = args.test + args.split_num
    else:
        split  = args.train + "_kvqa" + args.split_num   
    #print("Looking at split", split)

    model_pert = ModelPert(split, args.incl_caption, args.use_lm, args.ent_set, args.use_capt_sep, args.test, use_lrp=True)     #expecting commaseparate 0,1,2,3  etc
    ours = GeneratorOurs(model_pert)
    baselines = GeneratorBaselines(model_pert)

    predictions = model_pert.predict(model_pert.data_tuple, ours, baselines, dump=args.pred_out)

    #baselines = GeneratorBaselines(model_pert)
    #oursNoAggAblation = GeneratorOursAblationNoAggregation(model_pert)

    # method_name = args.method
    """
    kvqa_dataset = kvqa_data.KVQADataset(splits=split, incl_para=False, incl_caption=args.incl_caption, use_lm=args.use_lm, debug=False)   
    kvqa_answers = utils.get_data(model_pert.KVQA_trainval_labs)    #they don't use TorchDataset class , see how ExplainationGenerator works
    items = kvqa_dataset.data

    random.seed(1234)
    r = list(range(len(items)))
    random.shuffle(r)
    if args.num_samples != 0:
        pert_samples_indices = r[:args.num_samples]
    else:
        pert_samples_indices = r

    print("Total Examples in Data:",len(items), " and Evaluating:",len(pert_samples_indices))
        
    #iterator = tqdm([kvqa_dataset.data[i] for i in pert_samples_indices])
    iterator = [kvqa_dataset.data[i] for i in pert_samples_indices]

    test_type = "positive" if args.is_positive_pert else "negative"
    modality = "text" if args.is_text_pert else "image"
    print("running {0} pert test for {1} modality with method {2}".format(test_type, modality, args.method))

    top1_right = 0
    top5_right = 0
    st = time.time()
    for index, item in enumerate(iterator):
        #AS LONG AS WE CAN CHANGE ITEM input to have what we need for EBERT this should be fine
        if method_name == 'transformer_att':
            R_t_t, R_t_i = baselines.generate_transformer_attr(item)
        elif method_name == 'attn_gradcam':
            R_t_t, R_t_i = baselines.generate_attn_gradcam(item)
        elif method_name == 'partial_lrp':
            R_t_t, R_t_i = baselines.generate_partial_lrp(item)
        elif method_name == 'raw_attn':
            R_t_t, R_t_i = baselines.generate_raw_attn(item)
        elif method_name == 'rollout':
            R_t_t, R_t_i = baselines.generate_rollout(item)
        elif method_name == "ours_with_lrp_no_normalization":
            R_t_t, R_t_i = ours.generate_ours(item, normalize_self_attention=False)
        elif method_name == "ours_no_lrp":
            R_t_t, R_t_i = ours.generate_ours(item, use_lrp=False)
        elif method_name == "ours_no_lrp_no_norm":
            R_t_t, R_t_i = ours.generate_ours(item, use_lrp=False, normalize_self_attention=False)
        elif method_name == "ours_with_lrp":
            R_t_t, R_t_i = ours.generate_ours(item, use_lrp=True)
        elif method_name == "ablation_no_self_in_10":
            R_t_t, R_t_i = ours.generate_ours(item, use_lrp=False, apply_self_in_rule_10=False)
        elif method_name == "ablation_no_aggregation":
            R_t_t, R_t_i = oursNoAggAblation.generate_ours_no_agg(item, use_lrp=False, normalize_self_attention=False)
        else:
            print("Please enter a valid method name")
            return

        #qa_output = model_pert.output  #if you run explanation method it does forward already
        qa_output = model_pert.forward(item)
        answer_dist = qa_output.question_answering_score
        answer_top = answer_dist.argmax()
        answer_topk_v, answer_topk_i = topk(answer_dist, 5)
        answer = kvqa_answers[answer_top]
        #print("main answer (manually v1): ",answer_top, answer, 'vs item label', item["label"])
        top5 = [kvqa_answers[ind] for ind in answer_topk_i.data.view(-1)]
        top5_probs = [ float(pr) for pr in answer_topk_v.data.view(-1)]
        top1_acc = item["label"].get(answer, 0)
        top5_acc = 1 if list(item["label"].keys())[0] in top5 else 0
        #print("Top 1 Correct?", accuracy)
        print(item['img_id'], ",",list(item["label"].keys())[0], ",",top5, ",", top5_probs, ",",top1_acc, ",", top5_acc)   
        top1_right += top1_acc
        top5_right += top5_acc

        cam_image = R_t_i[0]
        cam_text = R_t_t[0]
        cam_image = (cam_image - cam_image.min()) / (cam_image.max() - cam_image.min())
        cam_text = (cam_text - cam_text.min()) / (cam_text.max() - cam_text.min())

        # TODO HERE INSTEAD OF or IN ADDITION TO pertubation, save out cam_image and cam_text values ( we'd want to test the difference between using EBERT vs not )
        #if args.is_text_pert:
        #    curr_pert_result = model_pert.perturbation_text(item, cam_image, cam_text, args.is_positive_pert)
        #else:
        #    curr_pert_result = model_pert.perturbation_image(item, cam_image, cam_text, args.is_positive_pert)
        #curr_pert_result = [round(res / (index+1) * 100, 2) for res in curr_pert_result]
        #iterator.set_description("Acc: {}".format(curr_pert_result))

        #curr_acc_result = [ round(top1_right/(index+1) * 100, 2), round(top5_right/(index+1) * 100, 2), top1_right, top5_right, index+1 ]  
        #if (index+1) % 100 == 0:
            #print("top1/top5 Acc/Raw: {}".format(curr_acc_result))

    #print("DONE. Elapsed time", time.time() - st)
    #print("FINAL top1/top5 Acc/Raw: {}".format(curr_acc_result))
    """

if __name__ == "__main__":
    main(args)
