# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features 
OKVQA_DATA_ROOT = '/home/diego/adv_comp_viz21/lxmert/orig_code/lxmert/data/okvqa/'     #mscoco_train2014_annotations.json  mscoco_val2014_annotations.json  OpenEnded_mscoco_train2014_questions.json  OpenEnded_mscoco_val2014_questions.json
OKVQA_IMGFEAT_ROOT = '/data/diego/adv_comp_viz21/imgfeat/'

class OKVQADataset:
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

    A OKVQA data example
       train_ants['question_types']
               'one': 'Vehicles and Transportation',
               'two': 'Brands, Companies and Products'
               'three': 'Objects, Material and Clothing',
               'four': 'Sports and Recreation',
               'five': 'Cooking and Food',
               'six': 'Geography, History, Language and Culture',
               'seven': 'People and Everyday life',
               'eight': 'Plants and Animals',
               'nine': 'Science and Technology',
               'ten': 'Weather and Climate',
               'other': 'Other',

      train_qs['questions'][0]
       {'image_id': 51606, 'question': 'What is the hairstyle of the blond called?', 'question_id': 516065}

      train_ants['annotations'][0]
        {'image_id': 51606, 
         'answer_type': 'other', 
         'question_type': 'four',     <--- 'Sports and Recreation'
         'question_id': 516065, 
         'answers': [ {'answer_id': 1,  'raw_answer': 'pony tail', 'answer_confidence': 'yes', 'answer': 'pony tail'}, 
                    {'answer_id': 2,  'raw_answer': 'pony tail', 'answer_confidence': 'yes', 'answer': 'pony tail'}, 
                    {'answer_id': 3,  'raw_answer': 'pony tail', 'answer_confidence': 'yes', 'answer': 'pony tail'}, 
                    {'answer_id': 4,  'raw_answer': 'pony tail', 'answer_confidence': 'yes', 'answer': 'pony tail'}, 
                    {'answer_id': 5,  'raw_answer': 'pony tail', 'answer_confidence': 'yes', 'answer': 'pony tail'}, 
                    {'answer_id': 6,  'raw_answer': 'pony tail', 'answer_confidence': 'yes', 'answer': 'pony tail'}, 
                    {'answer_id': 7,  'raw_answer': 'braid', 'answer_confidence': 'yes', 'answer': 'braid'}, 
                    {'answer_id': 8,  'raw_answer': 'braid', 'answer_confidence': 'yes', 'answer': 'braid'}, 
                    {'answer_id': 9,  'raw_answer': 'ponytail', 'answer_confidence': 'yes', 'answer': 'ponytail'}, 
                    {'answer_id': 10, 'raw_answer': 'ponytail', 'answer_confidence': 'yes', 'answer': 'ponytail'}], 
         'confidence': 3}

    """
    def __init__(self, splits: str, use_lm: None, ent_set: None, debug=True):
        #split is train or val (which is test)
        # QUESTIONS: OpenEnded_mscoco_train2014_questions.json  OpenEnded_mscoco_val2014_questions.json
        # ANNOTATIONS: mscoco_train2014_annotations.json  mscoco_val2014_annotations.json  

        # if tv in split like [train_tv, val_tv, test_tv]  use train_qs_okvqa.json / train_ans_okvqa.json or val_ or test_

        debug = False
        abs_path = "/home/diego/adv_comp_viz21/lxmert/orig_code/lxmert/"
        self.name = splits
        self.splits = splits.split(',')
        self.use_lm = use_lm

        # Loading datasets
        self.data = []
        cur_labels = []
        self.ent_span_len = 4
        self.ent_set = ent_set

        if use_lm == "ebert":
            if ent_set == None:
                #default
                entity_spans_dict = json.load(open(abs_path + "data/okvqa/okvqa_ent_spans_may06_excluded.json"))    #fixed bug, and exclude common things 10 ents max
            elif ent_set == "new":
                #entity_spans_dict = json.load(open(abs_path + "data/okvqa/okvqa_ent_spans_sep04_excluded.json"))    # too few, 3 
                entity_spans_dict = json.load(open(abs_path + "data/okvqa/okvqa_ent_spans_sept07_fewk.json"))        # see InspectResults-OKVQA, 4
                # i technically never tried sep4b
            elif ent_set == "new_b":
                entity_spans_dict = json.load(open(abs_path + "data/okvqa/okvqa_ent_spans_sept07_fewk_b.json"))        # see InspectResults-OKVQA, 4
            elif ent_set == "new_c":
                entity_spans_dict = json.load(open(abs_path + "data/okvqa/okvqa_ent_spans_sept07_fewk_cc.json"))        # see InspectResults-OKVQA, 4
            elif ent_set == '4k':
                entity_spans_dict = json.load(open(abs_path + "data/okvqa/okvqa_ent_spans_sept08_cleaned.json"))                #sept 8
            elif ent_set == '2p5k':
                entity_spans_dict = json.load(open(abs_path + "data/okvqa/okvqa_ent_spans_sept08_cleaned_filtered.json")) #sept 8  also 4

          
        
            #self.ent_span_len = len(entity_spans_dict[0][0])
            print("Use Ent Set: ", ent_set)
            print("TYPE", type(entity_spans_dict))
            print(len(entity_spans_dict))
            print(len(entity_spans_dict[0]))
            print(len(entity_spans_dict[0][0]))
            #ks = list(entity_spans_dict.keys())    #currently its not indexed by question_id, but we probably should do that TODO
            if debug:
                print("0503 first few ent spans info out of ")  #there are 14054 qs total
                for i in [0,1,2,3,4,69]:
                    #print(i,ks[i],entity_spans_dict[ks[i]])
                    print(i,entity_spans_dict[i])

        # load datasplits to use 
        for split in self.splits:
            if "tv" in split:
                qfile = OKVQA_DATA_ROOT + split.split("_")[0] + "_qs_okvqa.json"
                afile = OKVQA_DATA_ROOT + split.split("_")[0] + "_ans_okvqa.json"
            else:
                qfile = OKVQA_DATA_ROOT + "OpenEnded_mscoco_" + split + "2014_questions.json"
                afile = OKVQA_DATA_ROOT + "mscoco_" + split + "2014_annotations.json"

            cur_split = json.load(open(qfile))
            cur_split_ans = json.load(open(afile))
            if debug:
                print("LOADED ", qfile , "AND", afile, " with ",len(cur_split["questions"]),"examples")
                for i in [0,1,2,3,4,69]:
                    #print(i,ks[i],entity_spans_dict[ks[i]])
                    if use_lm == "ebert":
                        print(i,cur_split['questions'][i], entity_spans_dict[i])
                    else:
                        print(i,cur_split['questions'][i])

            #go through each question in each image and add
            for i, r in enumerate(cur_split['questions']):
                image_id = str(r["image_id"])
                question_id = str(r["question_id"])

                if "_qs_okvqa" in qfile:
                    img_dir = "train2014/" if "test" not in split else "val2014/"
                else:
                    img_dir = "train2014/" if "train" in split else "val2014/"

                t = "COCO_train2014_" if img_dir == "train2014/" else "COCO_val2014_"
                front = "0" * (12 - len(image_id))  
                image_path = img_dir + t + front + image_id + ".jpg"
                d = {"sent": r['question'], "question_id": question_id, "img_id": image_id, "imgPath": image_path, "label":{}} #, "label": {str(r["Answers"][i]): 1}}

                if use_lm == "ebert":
                    if "train" in split:
                        cur_ent_spans = entity_spans_dict[i]   #TODO: this needs to be based train/val/test
                    else: 
                        #train is of size 9009, so we need this offset for test/val sets
                        cur_ent_spans = entity_spans_dict[i+9009]   #TODO: better verify this !

                    d['ents_found'] = cur_ent_spans

                for ans_dict in cur_split_ans['annotations'][i]['answers']:
                    ans = ans_dict['answer']
                    if ans not in d["label"]:
                        d["label"][ans] = 0
                    d["label"][ans] += 1
                    cur_labels.append(ans)
         
                #normalize label answer scores ( between 0 and 1 ... divide by 3 and max 1 )
                sum_labs = sum([d["label"][a] for a in d["label"]])
                for a in d["label"]:
                    d["label"][a] = min(d["label"][a]/3,1)
                    
                self.data.append(d)
                    
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        if debug:
            print("LEN data", len(self.data))
            print(self.data[0])

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        #Create These Two files  ( Does this need to include TEST answers as well ... yes it does.. answer "vine" exists in test set and not train )
        train_label2ans_f = abs_path + "data/okvqa/train_label2ans.json"
        train_ans2label_f = abs_path + "data/okvqa/train_ans2label.json"

        #if split == "train":
        trainval_split_exists = os.path.exists(train_label2ans_f)
        if not trainval_split_exists:
            #prior method which unfairly added test answers.  makes tiny difference ( < 1% )
            """
            print("CREATING train/test label2ans and ans2label maps")
            tqfile = OKVQA_DATA_ROOT + "test_qs_okvqa.json"
            tafile = OKVQA_DATA_ROOT + "test_ans_okvqa.json"
            tcur_split = json.load(open(tqfile))
            tcur_split_ans = json.load(open(tafile))

            for i, r in enumerate(tcur_split['questions']):
                image_id = str(r["image_id"])
                question_id = str(r["question_id"])
                img_dir = "val2014/"
                t = "COCO_val2014_"
                front = "0" * (12 - len(image_id))  
                image_path = img_dir + t + front + image_id + ".jpg"
                d = {"sent": r['question'], "question_id": question_id, "img_id": image_id, "imgPath": image_path, "label":{}} #, "label": {str(r["Answers"][i]): 1}}
                for ans_dict in tcur_split_ans['annotations'][i]['answers']:
                    ans = ans_dict['answer']
                    if ans not in d["label"]:
                        d["label"][ans] = 0
                    d["label"][ans] += 1
                    cur_labels.append(ans)
            """
         
            labels2ans = list(set([str(v) for v in cur_labels]))
            with open(train_label2ans_f, 'w') as json_file:
                json_file.write('[' + ',\n'.join(json.dumps(v) for v in labels2ans) + ']')
    
            ans2labels = { v: i for i,v in enumerate(labels2ans) }
            with open(train_ans2label_f, 'w') as json_file:
                json_file.write('{' + ',\n'.join(json.dumps(str(v)) + ': '+ str(ans2labels[v]) for v in ans2labels ) + '}')

        # Answers
        self.label2ans = json.load(open(train_label2ans_f))     # [ list of all possible "ans strings" ]    # Label = index num,   Ans= String
        self.ans2label = json.load(open(train_ans2label_f))     # { "ans string": index_num in above list, .. }

        if debug:
            print(len(self.label2ans))
            print(len(self.ans2label))

        assert len(self.ans2label) == len(self.label2ans)

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
class OKVQATorchDataset(Dataset):
    def __init__(self, dataset: OKVQADataset):
        super().__init__()
        self.raw_dataset = dataset
        self.ent_span_len = dataset.ent_span_len
        debug = True
        if debug:
            print("TorchDataset:", len(dataset), dataset.splits)   #TorchDataset: 9009 ['train']
            print("Dataset.data", len(dataset.data))
            print("Ent span lengths:", self.ent_span_len)

        if args.tiny and dataset.splits[0] != "val_tv":
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
            load_topk = 5000 if (split == 'minival' and topk is None) else topk      #TODO:  fix this for val_tv  
            img_data.extend(load_obj_tsv(
                os.path.join(OKVQA_IMGFEAT_ROOT, 'okvqa_imgs_obj36.tsv'),
                topk=load_topk))

        if debug:
            print("Img data len",len(img_data))

        # D: If tiny is set it will only grab the first 51 images and 
        # then only add questions / paraquestions with those 51 img_ids

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum    

        if debug:
            print("len img_datum: ", len(self.imgid2img))
            #print(self.imgid2img.keys())

        # Only kept the data with loaded image features
        self.data = []
        for i, datum in enumerate(self.raw_dataset.data):
            if i in [0,1,2,3,4,69] and debug:
                print(i, datum)
                print(list(self.imgid2img.keys())[0:4])
                print(datum['imgPath'].split("/")[1].split(".")[0], datum['imgPath'].split("/")[1].split(".")[0] in self.imgid2img)
            #if datum['img_id'] in self.imgid2img:
            if datum['imgPath'].split("/")[1].split(".")[0] in self.imgid2img:
                self.data.append(datum)

        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        #img_id = datum['img_id']
        img_id = datum['imgPath'].split("/")[1].split(".")[0]
        ques_id = datum['question_id']

        # if use_lm == "ebert" we need to send spans too!
        if self.raw_dataset.use_lm == "ebert":
           max_ents = 11
           #ques = {'sent': datum['sent'], 'ent_spans': datum['ents_found']}    # 'Francis Condon in the early 20th Century',  [['Francis Condon', 0, 14]]
           ques = datum['sent']
           ent_spans = datum['ents_found']     
           # force 3 span entity sets to be four
           if self.raw_dataset.ent_set in [None]:       
               ent_spans = [ [e_spans[0],e_spans[1],e_spans[2],''] for e_spans in ent_spans ]

           n = len(ent_spans)
           if (max_ents - n) < 0:
               print("MAX ENTS - N < 0", img_id)
               import sys
               sys.exit()
           for _ in range(max_ents - n ):
               #if self.ent_span_len == 4:
               ent_spans.append(['',-1,-1,''])   #IMPORTANT ( number of elements must equal number provided by ent span json file, either 3 or 4
               #else:
               #    ent_spans.append(['',-1,-1])   #IMPORTANT
        else:
           max_ents = 11
           ques = datum['sent']
           #if self.ent_span_len == 4:
           ent_spans = [['',-1,-1,''] for _ in range(max_ents)]   #assuming max spans is 5 like kvqa!   #IMPORTANT
           #else:
           #    ent_spans = [['',-1,-1] for _ in range(max_ents)]   #assuming max spans is 5 like kvqa!   #IMPORTANT

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
            for ans, score in label.items():
                if ans not in self.raw_dataset.ans2label:
                    #print("ERROR answer:", ans, type(ans), "with score", score, "not found in in self.raw_dataset.ans2label")
                    ans = str(ans)
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
                else:
                    print("Didn't find ",ans," in ans2label")
                    #target[self.raw_dataset.ans2label[ans]] = 0   target label is in test, but not train/val so there is no label to give a zero to ( they are zeros already anyways! )
            return ques_id, feats, boxes, ques, ent_spans, target
        else:
            return ques_id, feats, boxes, ques, ent_spans


class OKVQAEvaluator:
    def __init__(self, dataset: OKVQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        debug = True
        score = 0.
        outofrange = []
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if debug:
                print("In Evaluate: QID", quesid, "ANS:", ans,type(ans))
                print( len(self.dataset.label2ans) > ans if type(ans) == np.int64 else '', label, ans in self.dataset.ans2label)    #maybe predict is returning wrong thing!

            if ans in label:
                if debug:
                    print("1. Correct!",quesid, ans, label[ans])  #label[ans] is a number, but really it should be a 1 or the divisor sign should be sum of num_answers per question!
                score += label[ans]
            elif type(ans) == int or type(ans) == np.int64:
                ans = int(ans)
                #if debug:
                    #print(type(self.dataset.label2ans), len(self.dataset.label2ans), label)  # self.dataset.label2ans[ans])  #this is what is predicted

                if len(self.dataset.label2ans) > ans:
                    if debug:
                        print("Prediction", self.dataset.label2ans[ans], "in True label set?", label, self.dataset.label2ans[ans] in label) 
                  
                    if self.dataset.label2ans[ans] in label:
                        if debug:
                            print("2. Correct!",quesid, self.dataset.label2ans[ans], label[self.dataset.label2ans[ans]])  
                        score += label[self.dataset.label2ans[ans]]
                else:
                    if debug:
                        print("ERROR size of label2ans < ans? for ", quesid)  
                    outofrange.append([quesid, ans])
                    #do i need to adjust/account size for validation somehow?  
                    #Yes, the val size is only 1895 and ans are base on 9895 size ( yes, for val_tv you shouldn't recreate anything, but rather use existing )

        # IS THIS EVALUATE CORRECT FOR vqa style answer?  Correct
        print("Eval Score", score, "/", len(quesid2ans) , "=", round(score/len(quesid2ans),4))
        print("Num Out of Range", len(outofrange))
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


