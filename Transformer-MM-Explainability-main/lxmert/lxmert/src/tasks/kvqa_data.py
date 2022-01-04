# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from src.param import args
from src.utils import load_obj_tsv

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
#TINY_IMG_NUM = 51
TINY_IMG_NUM = 1051
FAST_IMG_NUM = 5000

# The path to data and image features.
KVQA_DATA_ROOT = '/home/adv_comp_viz21/lxmert/orig_code/lxmert/data/kvqa/'    #changed from prior data/kvqa
KVQA_IMGFEAT_ROOT = '/data/diego/adv_comp_viz21/imgfeat/'

class KVQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    A KVQA data example

	{'NamedEntities': ['Hal De Forrest', 'Estelle Taylor', 'Jack Dempsey'], 
	 'imgPath': 'KVQAimgs/39346.jpg', 
	 'ParaQuestions': ['Who is in the left side?', 'Who is in the right side?', 'Who is in the center?', 'Who is to the left side of Estelle Taylor?', 'Who is to the right side of Estelle Taylor?', 'Who among the folks in the picture is the eldest?', 'What is the age difference between the center and the rightmost person?', 'How many folks in the picture took birth after the end of World War II?', 'Who among the folks in the picture at any point wedded to Estelle Taylor?', 'How many actors are there in the picture?', 'Do all the folks in the picture have a common occupation?'], 
	 'Qids': ['Q5640736', 'Q439655', 'Q313686'], 
	 'Questions': ['Who is in the left?', 'Who is in the right?', 'Who is in the center?', 'Who is to the left of Estelle Taylor?', 'Who is to the right of Estelle Taylor?', 'Who among the people in the image is the eldest?', 'What is the age gap between the center and the rightmost person?', 'How many people in the image were born after the end of World War II?', 'Who among the people in the image ever married to Estelle Taylor?', 'How many actors are there in the image?', 'Do all the people in the image have a common occupation?'], 
	 'split': [2, 3, 1, 1, 2], 
	 'wikiCap': 'Hal De Forrest (left) and Jack Dempsey with his wife, actress Estelle Taylor.', 
	 'Answers': ['Hal De Forrest', 'Jack Dempsey', 'Estelle Taylor', 'Jack Dempsey', 'Hal De Forrest', 'Person in the left', 1, 0, 'person in the right', 2, 'No'], 
	 'Type of Question': [['spatial', '1-hop'], ['spatial', '1-hop'], ['spatial', '1-hop'], ['spatial', '1-hop'], ['spatial', '1-hop'], ['1-hop subtraction', 'comparison', 'Multi-Entity'], ['multi-hop', 'counting', 'Multi-Entity', 'Multi-Relation'], ['1-hop', 'Multi-Entity'], ['1-hop', 'counting', 'Multi-Entity'], ['1-hop', 'intersection', 'boolean', 'Multi-Entity']]
	 }

    """
    def __init__(self, splits: str, incl_para: bool, incl_caption: bool, use_lm: None, ent_set: None, use_capt_sep: None, debug=False):
        #split is train_kvqa0, valid_kvqa0 or test_kvqa0  where 0 in [0:4].   assuming split is single string for now
        #incl_para: include paraphrase questions as well?
        #incl_caption: include image caption in front of question/paraphrase question?

        debug = False
        abs_path = "/home/diego/adv_comp_viz21/lxmert/orig_code/lxmert/"
        self.name = splits
        self.splits = splits.split(',')
        self.use_lm = use_lm

        # Loading datasets
        self.data = []
        cur_labels = []
        self.ent_span_len = 4  #force 4 now
        self.ent_set = ent_set

        if use_lm == "ebert":
            if ent_set == None:
                entity_spans_dict = json.load(open(abs_path + "data/kvqa/new_kvqa_q_caps_ents0502.json"))
            elif ent_set == "oracle_links":
                entity_spans_dict = json.load(open(abs_path + "data/kvqa/kvqa_oracle_ent_spans_with_wikis0923.json"))  #this is the above but with 4 spans and force in wikiemb, but no noisy adds
            elif ent_set == "oracle_noisy":
                entity_spans_dict = json.load(open(abs_path + "data/kvqa/kvqa_oracle_noisy_ent_spans_with_wikis0923.json"))
            elif ent_set == "sep13_3":
                #entity_spans_dict = json.load(open(abs_path + "data/kvqa/kvqa_ent_spans_sept13_3corb.json"))  #corb is 4 span
                entity_spans_dict = json.load(open(abs_path + "data/kvqa/kvqa_ent_spans_sept13_3cor.json"))  #cor is 3 span
            elif ent_set == "sept13_fewkb":
                entity_spans_dict = json.load(open(abs_path + "data/kvqa/kvqa_ent_spans_sept13_fewkbcor.json"))
            elif ent_set == "sept13_fewkb_links":
                entity_spans_dict = json.load(open(abs_path + "data/kvqa/kvqa_sep13_few_links_ent_spans0926.json"))
            elif ent_set == "yasu0":
                entity_spans_dict = json.load(open(abs_path + "data/kvqa/kvqa0_ent_spans_yasu0.json"))
            elif ent_set == "yasu1":
                entity_spans_dict = json.load(open(abs_path + "data/kvqa/kvqa0_ent_spans_yasu1.json"))
            elif ent_set == "yasu2":
                entity_spans_dict = json.load(open(abs_path + "data/kvqa/kvqa_ent_spans_yasu2_0916.json"))
            elif ent_set == "yasu2links":
                entity_spans_dict = json.load(open(abs_path + "data/kvqa/yasu_links_ent_spans_with_wikis0926.json"))
            elif ent_set == "yasu2noisy":
                entity_spans_dict = json.load(open(abs_path + "data/kvqa/yasu_noisy_ent_spans_with_wikis0926a.json"))
            else:
                print("Ent set passed that is not supported:", ent_set)
                import sys
                sys.exit()

            ks = list(entity_spans_dict.keys())
            if debug:
                print(ks[0], entity_spans_dict[ks[0]], len(entity_spans_dict[ks[0]]['ents_found'][0]))

                print("Use Ent Set: ", ent_set)
                print("0501post first ten ent info")
                for i in range(10):
                    print(i,ks[i],entity_spans_dict[ks[i]])

        for split in self.splits:
            if debug:
                print(split, abs_path+"data/kvqa/"+split+".json")

            cur_split = json.load(open(abs_path+"data/kvqa/%s.json" % split))

            #go through each question in each image and add
            #question_id = image_id + _ + qnum in list
            #question (with/out caption), answer, image, entities, qids, is_para, includes caption
            for r in cur_split:
                image_id = str(r['imgPath'].split("/")[1].split(".")[0])
                for i,q in enumerate(r['Questions']):
                    d = {"question_id": image_id+"_"+str(i), "img_id": image_id, "imgPath": r['imgPath'], "Qids": r["Qids"], 'NamedEntities': r["NamedEntities"], "is_para": False, "label": {str(r["Answers"][i]): 1}}

                    if use_lm == "ebert":
                        #load sent from entity_spans_dict and include entity_spans!
                        cur_ent_spans = entity_spans_dict[image_id]
                        if use_capt_sep:
                            d['sent'] = q + " [SEP] " + cur_ent_spans['wikiCap_new']      # 'Francis Condon in the early 20th Century'
                        else:
                            d['sent'] = q + " " + cur_ent_spans['wikiCap_new']      # 'Francis Condon in the early 20th Century'
                        d['ents_found'] = cur_ent_spans['ents_found']           # [['Francis Condon', 0, 14]]

                        #if image_id+"_"+str(i) == "24742_0":
                        #print("ERROR CHECKING:", d)    ... HERE sent includes wikiCap_new and a period at the end. is that the issue?
                        # {'question_id': '24742_0', 'img_id': '24742', 'imgPath': 'KVQAimgs/24742.jpg', 'Qids': ['Q301358'], 'NamedEntities': ['Hartmut Nassauer'], 'is_para': False, 'label': {'Hartmut Nassauer': 1}, 'sent': 'Who is the person in the image? Hartmut Nassauer.', 'ents_found': [['Hartmut Nassauer', 0, 16]]}
                    
                    else:
                        if use_capt_sep:
                            d['sent'] = q + " [SEP] " + r['wikiCap'] if incl_caption else q
                        else:
                            d['sent'] = q + " " + r['wikiCap'] if incl_caption else q
                    self.data.append(d)
                    cur_labels.append(r["Answers"][i])

                if incl_para:
                    # EBERT experiments don't include paraphrase questions so this defaults to nonebert for now
                    for i,q in enumerate(r['ParaQuestions']):
                        d = {"question_id": image_id+"_"+str(i + len(r['Questions'])), "img_id": image_id, "imgPath": r['imgPath'], "Qids": r["Qids"], 'NamedEntitites': r["NamedEntities"], "is_para": True, "label": {str(r["Answers"][i]): 1}}
                        d['sent'] = q + " " + r['wikiCap'] if incl_caption else q
                        self.data.append(d)
                    
        if debug:
            print("Load %d data from split(s) %s." % (len(self.data), self.name))
            print("LEN data", len(self.data))
            print(self.data[0])

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        #Create These Two files for the current split number (0-4)  :  Make sure to do this using both the Train and Val data for the split
        split_num = self.splits[0].split("kvqa")[1]
        split_type = self.splits[0].split("_kvqa")[0]
        trainval_label2ans_f = abs_path + "data/kvqa/trainval_label2ans_"+str(split_num)+".json"
        trainval_ans2label_f = abs_path + "data/kvqa/trainval_ans2label_"+str(split_num)+".json"

        get_split = "train_kvqa"+str(split_num)
        if split_type == "train":
            get_split = "valid_kvqa"+str(split_num)

        if debug:
            print("LOADING trainval ", split, get_split)    
        # DON'T ALLOW dynamic creationg of trainval because it messes up the original trainval_label2ans and we need to use the existing one anyways)
        # ie, only create if that trainval split ans2lab doesn't exist
        trainval_split_exists = os.path.exists(trainval_label2ans_f)
        #if split_type == "train":
        if not trainval_split_exists:
            get_data_split = json.load(open(abs_path + "data/kvqa/%s.json" % get_split))
            for r in get_data_split:
                for i,q in enumerate(r['Questions']):
                    cur_labels.append(r["Answers"][i])
    
            labels2ans = list(set([str(v) for v in cur_labels]))
            with open(trainval_label2ans_f, 'w') as json_file:
                json_file.write('[' + ',\n'.join(json.dumps(v) for v in labels2ans) + ']')
    
            ans2labels = { v: i for i,v in enumerate(labels2ans) }
            with open(trainval_ans2label_f, 'w') as json_file:
                json_file.write('{' + ',\n'.join(json.dumps(str(v)) + ': '+ str(ans2labels[v]) for v in ans2labels ) + '}')

        # Answers
        self.label2ans = json.load(open(trainval_label2ans_f))     # [ list of all possible "ans strings" ]
        self.ans2label = json.load(open(trainval_ans2label_f))     # { "ans string": index_num in above list, .. }

        if debug:
            #if split_type == "train":    
            if not trainval_split_exists:
                print(len(self.label2ans), len(labels2ans))  # 15234 15234
                print(len(self.ans2label), len(ans2labels))  # 15218 15234
                assert len(self.ans2label) == len(self.label2ans)
            else:
                print(len(self.label2ans))  
                print(len(self.ans2label))  
            
        #XXX in generate_json_splits also create a "tiny" file for debugging  .. not used
        #XXX: make sure you are using answer and label correctly!  Label is a string and Answer is index of answer string (ie, one hot encodeded )

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class KVQATorchDataset(Dataset):
    def __init__(self, dataset: KVQADataset):
        super().__init__()
        self.raw_dataset = dataset
        self.ent_span_len = dataset.ent_span_len
        debug = False
        if debug:
            print("TorchDataset:", len(dataset), dataset.splits)   #TorchDataset: 9009 ['train']
            print("Dataset.data", len(dataset.data))
            print("Ent span lengths:", self.ent_span_len)

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:  
            topk = FAST_IMG_NUM
        else:
            topk = None

        # Loading detection features to img_data
        img_data = []
        for split in dataset.splits:
            # Minival is 5K images, which is used in evaluating KVQA/LXMERT-pre-training.
            # It is saved as the top 5K features in val2014_***.tsv
            load_topk = 5000 if (split == 'minival' and topk is None) else topk
            img_data.extend(load_obj_tsv(
                os.path.join(KVQA_IMGFEAT_ROOT, 'kvqa_imgs_obj36.tsv'),
                topk=load_topk))

        # D: If tiny is set it will only grab the first 51 images and 
        # then only add questions / paraquestions with those 51 img_ids

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        if debug:
            print("Use %d data in torch dataset" % (len(self.data)))
            print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]
        #print(" IN TORCHDATA SET __getitem__")
        #print("Use LM", self.raw_dataset.use_lm)
        #print(item, datum)

        img_id = datum['img_id']
        ques_id = datum['question_id']

        # if use_lm == "ebert" we need to send spans too!
        max_ents = 20  #5 
        if self.raw_dataset.use_lm == "ebert":
           #ques = {'sent': datum['sent'], 'ent_spans': datum['ents_found']}    # 'Francis Condon in the early 20th Century',  [['Francis Condon', 0, 14]]
           ques = datum['sent']
           ent_spans = datum['ents_found']     # this is a list that can be of size 5 max 

           if self.raw_dataset.ent_set in ['sep13_3',None,'yasu0','yasu2','yasu1','yasu2links']:       # force these to be of size 4 IMPORTANT 09/18
               ent_spans = [ [e_spans[0],e_spans[1],e_spans[2],''] for e_spans in ent_spans ]
           n = len(ent_spans)
           if n < max_ents:
               for _ in range(max_ents - n ):
                   #if self.ent_span_len == 4:
                   ent_spans.append(['',-1,-1,''])   #IMPORTANT ( number of elements must equal number provided by ent span json file, either 3 or 4
                   #else:
                   #    ent_spans.append(['',-1,-1])   #IMPORTANT
           else:
               #print("Truncating ent set of len",n," to size of ", max_ents, "for question", ques_id, "with ents:", ent_spans)
               ent_spans = ent_spans[0:max_ents]
        else:
           ques = datum['sent']
           ent_spans = [['',-1,-1,''] for _ in range(max_ents)]   

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            missed = []
            missed_target = []
            for ans, score in label.items():
                if ans not in self.raw_dataset.ans2label:
                    #print("ERROR answer:", ans, type(ans), "with score", score, "not found in in self.raw_dataset.ans2label")
                    ans = str(ans)
                    missed.append(ans)

                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
                else:
                    #print("didn't find",ans,"in target",len(target))
                    missed_target.append(ans)
 
                    
            #if len(missed) > 0:
            #    print("Instances not found in ans2label", missed,  len(missed), "out of", len(label), " (",round(len(missed)/len(label),4),"%)")

            #print("Instances not found in target", len(missed_target), "out of", len(label), " (",round(len(missed_target)/len(label),4),"%)")
            ret = ques_id, feats, boxes, ques, ent_spans, target
            return ret 
        else:
            ret = ques_id, feats, boxes, ques, ent_spans
            return ret 


class KVQAEvaluator:
    def __init__(self, dataset: KVQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        print("In KVQAEvaluator", len(self.dataset.id2datum),"dataset size")
        print("question id, prediction, true label, correct?")
        cur = 0
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if cur < 10:
                print("QID",quesid, "TRUTH",label, "Pred",ans,"CORRECT?",ans in label)    #is label a dict{} ?
            cur += 1
            if ans in label:
                score += label[ans]

        print("DONE WITH EVALUATOR.EVALUATE, ", score, "/", len(quesid2ans), "=", score/len(quesid2ans))
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the KVQA online evaluation.
        KVQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': str(ques_id),
                    'answer': str(ans)
                })
            try:
              json.dump(result, f, indent=4, sort_keys=True)
            except Exception as e:
              f.write('[' + ',\n'.join(json.dumps(v) for v in result) + ']')
