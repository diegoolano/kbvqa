from lxmert.lxmert.src.modeling_frcnn import GeneralizedRCNN
import lxmert.lxmert.src.vqa_utils as utils
from lxmert.lxmert.src.processing_image import Preprocess

from transformers import LxmertTokenizer
from lxmert.lxmert.src.lxrt.tokenization import BertTokenizer

from lxmert.lxmert.src.lxmert_lrp_ebert import LxmertForQuestionAnswering as LxmertForQuestionAnsweringLRP    
from lxmert.lxmert.src.lxmert_lrp_ebert import LxmertEnhancedForQuestionAnswering as LxmertEnhancedForQuestionAnsweringLRP    #holds EBERT
from lxmert.lxmert.src.lxmert_lrp_ebert import convert_sents_to_features, toks_to_str  #updated for okvqa

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

OKVQA_VAL_PATH = '/data/diego/adv_comp_viz21/okvqa/img_data/' 
OKVQA_URL = '/home/diego/adv_comp_viz21/lxmert/orig_code/lxmert/data/okvqa/'

from src.pretrain.qa_answer_table import load_lxmert_qa_hf   #should this be src or lxmert.lxmert.src

from src.tasks.okvqa_data import OKVQADataset, OKVQATorchDataset, OKVQAEvaluator   
#from src.tasks import kvqa_data  

from collections import Counter
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

# ADD use_lm and ent_set
def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False, use_lm=None, ent_set=None) -> DataTuple:
    dset = OKVQADataset(splits, use_lm, ent_set)
    tset = OKVQATorchDataset(dset)
    evaluator = OKVQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class ModelPert:
    def __init__(self, split_str, use_lm, ent_set, test=None, use_lrp=False):
        self.use_lm = use_lm
        self.ent_set = ent_set

        self.KVQA_VAL_PATH = OKVQA_VAL_PATH    #path to IMAGES
        self.KVQA_trainval_labs = OKVQA_URL + "train_label2ans.json"

        self.kvqa_answers = utils.get_data(self.KVQA_trainval_labs)
        print("VQA answers:", type(self.kvqa_answers), len(self.kvqa_answers))
        #import sys
        #sys.exit()

        # load models and model components
        self.frcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn_cfg.MODEL.DEVICE = "cuda"
        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg)

        self.image_preprocess = Preprocess(self.frcnn_cfg)
        self.lxmert_tokenizer = BertTokenizer.from_pretrained( "bert-base-uncased", do_lower_case=True)

        if use_lrp:
            # TODO.. I didn't base things on lxmert-vqa-uncased so how to deal with this??  
            if use_lm:
                self.lxmert_vqa = LxmertEnhancedForQuestionAnsweringLRP.from_pretrained("unc-nlp/lxmert-vqa-uncased")
            else:
                self.lxmert_vqa = LxmertForQuestionAnsweringLRP.from_pretrained("unc-nlp/lxmert-vqa-uncased")

            self.lxmert_vqa.resize_num_qa_labels(len(self.kvqa_answers))

            if test == None:
                #TODO look
                self.data_tuple = get_data_tuple(split_str, bs=1, shuffle=False, drop_last=False, use_lm = self.use_lm)
                load_lxmert_qa_hf("/home/diego/adv_comp_viz21/lxmert/orig_code/lxmert/snap/pretrained/model", self.lxmert_vqa, label2ans=self.kvqa_dataset.label2ans)  #do i need this?
                self.lxmert_vqa.to("cuda")
            else:
                # loading non-finetuned
                self.load(args.load)
                self.data_tuple = get_data_tuple(split_str, bs=1, shuffle=False, drop_last=False, use_lm = self.use_lm, ent_set = self.ent_set)
                self.lxmert_vqa.to("cuda")
        else:
            # never gets used for now
            self.lxmert_vqa = None #LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased").to("cuda")

        self.lxmert_vqa.eval()
        self.model = self.lxmert_vqa

        self.pert_steps = [0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
        self.pert_acc = [0] * len(self.pert_steps)


    def load(self, path):
        print("!!! Load model from %s" % path)
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

    def predict(self, eval_tuple: DataTuple, expl_ours=None, dump=None):
        """
        Predict the answers to questions in a data split.

        :param expl: ExplainerGenerator class
        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        debug = True
        vset, vloader, vevaluator = eval_tuple
        #viter_wrapper = (lambda x: tqdm(x, total=len(vloader))) if args.tqdm else (lambda x: x)
        viter_wrapper = (lambda x: tqdm(x, total=len(vloader))) 

        qid2imgid = vset.id2datum  #question_id to data

        quesid2ans = {}
        ans_not_found = 0
        top1_right = 0
        top5_right = 0
        st = time.time()
        #these boxes have been normalized already
        for i, (ques_id, feats, boxes, sent, ent_spans, target) in viter_wrapper(enumerate(vloader)):
            if True:
                #max_seq_len = -1   # last time when I had batches I used 100.. now use -1 to me no max_seq_len
                max_seq_len = -1    # this gave 37% whereas normal way gives 43 <-- most likely the bug of acc is because of label stuff/eval I'm guessing
                #max_seq_len = 20   # trying this way like prior code does leads to bug on size throughout explainability generator though.

                train_features, tokens = convert_sents_to_features( sent, ent_spans, max_seq_len, self.lxmert_tokenizer, self.use_lm)
                input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
                input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
                segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

                surface2wiki = []
                
                if debug:
                    print("LXRT encoder forward")
                    print("USE LM=",self.use_lm)
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
                #self.text_len = len(tokens[0])  #since batch size of 1
                #if self.use_lm == "ebert": 
                if True:
                    if debug:
                        print("LOOKING AT OUTPUT for EBERT")
                        print(self.output)  #is embedding_output here
                        print("SENT", sent)
                        print("TOKENS", tokens, len(tokens))
                        print("EMB OUTPUT",self.output.embedding_output, len(self.output.embedding_output))   #these are ebert enhaned tokens

                if self.output.embedding_output != []:
                    tokens = self.output.embedding_output  #with batches of one this should be fine
                elif len(tokens) == 1:
                    tokens = tokens[0]

               
                #self.text_len = len(tokens[0]) if max_seq_len == -1 else max_seq_len  #since batch size of 1
                #self.text_len = self.output.embedding_output.size(1) if max_seq_len == -1 else max_seq_len  #since batch size of 1
                self.text_len = len(tokens) if max_seq_len == -1 else max_seq_len  #since batch size of 1
                self.image_boxes_len = feats.shape[1]  #is this correct?   does it really only select which ROI to look at for LXMERT?

                print(ques_id, sent, input_ids.shape, self.text_len, self.image_boxes_len, feats.shape , boxes.shape)
                # OKVQA with max_seq_len = 20
                # ['2971475'] ['What sport can you use this for?'] torch.Size([1, 20]) 10 36 torch.Size([1, 36, 2048]) torch.Size([1, 36, 4])

                item = None
                R_t_t, R_t_i = expl_ours.generate_ours(item, use_lrp=False, model = self.lxmert_vqa, output = self.output.question_answering_score)  
                cam_image = R_t_i[0]
                cam_text = R_t_t[0]
                cam_image = (cam_image - cam_image.min()) / (cam_image.max() - cam_image.min())
                cam_text = (cam_text - cam_text.min()) / (cam_text.max() - cam_text.min())

                answer_dist = self.output.question_answering_score
                answer_top = answer_dist.argmax()
                answer_topk_v, answer_topk_i = topk(answer_dist, 5)
                answer = self.kvqa_answers[answer_top]

                datum = qid2imgid[ques_id[0]]
                top5 = [self.kvqa_answers[ind] for ind in answer_topk_i.data.view(-1)]
                top5_probs = [ round(float(pr),5) for pr in answer_topk_v.data.view(-1)]
                top1_acc = datum["label"].get(answer, 0)
                #top5_acc = 1 if list(datum["label"].keys())[0] in top5 else 0    #this is incorrect as it only searches top answer in dict
                # find highest scoring item in top5 if any
                top5_acc = 0
                for ans_str in list(datum["label"].keys()):
                    if ans_str in top5:
                        ans_score = datum["label"].get(ans_str)
                        if ans_score > top5_acc:
                            top5_acc = ans_score

                #entslen = len(datum['NamedEntities'])
                if self.use_lm == "ebert": 
                    entslen = len([t for t in tokens if t == '[UNK]'])/2
                    #TODO get entslen via ent_spans or surface2wiki?
                    print("USING EBERT WITH ENTS:",surface2wiki, len(surface2wiki), len(surface2wiki[0]))
                    entslen = len(surface2wiki[0])
                else:
                    entslen = 0
  
                true_ans = list(datum['label'].keys())[0]
                sent = datum['sent']

                #save out ques_id[0] cam_image and cam_text explanations
                save_boxes = [ [round(float(a),5) for a in v] for v in boxes.squeeze() ]
                
                text_expl = [round(float(v),4) for v in cam_text]
                image_expl = [round(float(v),4) for v in cam_image]
                quesid2ans[ques_id[0]] = {'ents': entslen, 'true_ans': true_ans,'top5': top5, 'top5probs': top5_probs, 'top1acc':top1_acc, 'top5acc': top5_acc, 'sent': sent, 'text_expl': text_expl, 'image_expl': image_expl, 'bboxes': save_boxes, 'tokens': tokens} 

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


def main(args):
    if args.test is not None:
        split = args.test
    else:
        split  = args.train 
    print("Looking at split", split)

    model_pert = ModelPert(split, args.use_lm, args.ent_set, args.test, use_lrp=True)     
    ours = GeneratorOurs(model_pert)
    predictions = model_pert.predict(model_pert.data_tuple, ours, dump=args.pred_out)

    method_name = args.method

if __name__ == "__main__":
    main(args)
