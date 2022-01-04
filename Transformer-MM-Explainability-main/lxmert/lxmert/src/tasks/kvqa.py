# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections
import time

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.kvqa_model import KVQAModel
from tasks.kvqa_data import KVQADataset, KVQATorchDataset, KVQAEvaluator

from collections import Counter

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

# ADD use_lm
def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False, incl_para=False, incl_caption=False, use_lm=None) -> DataTuple:
    dset = KVQADataset(splits, incl_para, incl_caption, use_lm)
    tset = KVQATorchDataset(dset)
    evaluator = KVQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class KVQA:
    def __init__(self):
        # Datasets
        args.train = args.train + args.split_num   

        print("Load training", args.train)
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, 
            shuffle=True, drop_last=True,
            incl_para = args.incl_para, 
            incl_caption = args.incl_caption,
            use_lm = args.use_lm
        )
        if args.valid != "":
            args.valid = args.valid + args.split_num   
            print("Load valid", args.valid)
            self.valid_tuple = get_data_tuple(
                args.valid, bs=32,
                shuffle=False, drop_last=False,
                incl_para = args.incl_para,  
                incl_caption = args.incl_caption,
                use_lm = args.use_lm
            )
            # before was 1024, but it crashed my GPU
        else:
            self.valid_tuple = None
        
        # Model
        print("Load KVQAModel with maxlen", args.max_len, "and use_lm", args.use_lm)
        self.model = KVQAModel(self.train_tuple.dataset.num_answers, args.max_len, args.use_lm)

        print("Train label2ans size", type(self.train_tuple.dataset.label2ans), len(self.train_tuple.dataset.label2ans))  #18383
        if args.valid != "":
            print("Val label2ans size", type(self.train_tuple.dataset.label2ans), len(self.valid_tuple.dataset.label2ans))    #18383

        #check for equality of both list and dicts if valid_tuple is set
        if args.valid != "":
            test_list1 = self.train_tuple.dataset.label2ans
            test_list2 = self.valid_tuple.dataset.label2ans
            if len(test_list1)== len(test_list2) and len(test_list1) == sum([1 for i, j in zip(test_list1, test_list2) if i == j]):
                print ("Both label2ans lists are identical")
            else :
                print ("The label2ans lists are not identical!")
                print("Train ",self.train_tuple.dataset.label2ans[0:10])
                print("Valid ",self.valid_tuple.dataset.label2ans[0:10])
                import sys
                sys.exit()

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)

        if args.load_lxmert_qa is not None:     #for finetune etc this is Not NONE ... uses self.model.encoder  and load_lxmert_qa path --> loaded_state_dict = torch.load("%s_LXRT.pth" % path) )
            print("Load LXMERT QA weights")
            load_lxmert_qa(args.load_lxmert_qa, self.model, label2ans=self.train_tuple.dataset.label2ans)

            # D: so this is the model snapshot weights that have been pre-trained for us ( this takes days on multiple GPUs )
            # so we can either 
            #   (a) pre-train using ebert or 
            #   (b) more likely load the snapshot which is uncased and 
            #        learn a mapping via EBERT pretraining that gives the mapping from wikipediavec ents to the LXMERT BERT embedding space
            #        this needs to be outside of here ( ie, in EBERT code before getting to here, we'll learn the mapping and then incorporate that mapping here )  <-- What we did
             
        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            #for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):
            for i, (ques_id, feats, boxes, sent, ent_spans, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent, ent_spans)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    #print("train",qid,l,ans,label)
                    #train qid:10_2,  l:2967 , ans:No ,  tensor([2967, 2967, .... ] )   #the label is 
                    try:
                        quesid2ans[qid.item()] = ans
                    except Exception as e:
                        quesid2ans[qid] = ans
          
            tans = Counter([quesid2ans[q] for q in quesid2ans])
            print("Calling training evaluator with ", type(quesid2ans), len(quesid2ans), tans)
            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                print("CALLING VALIDATION on eval_tuple", eval_tuple)
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        print("set Eval on!", type(self.model))
        self.model.eval()
        vset, vloader, vevaluator = eval_tuple
        viter_wrapper = (lambda x: tqdm(x, total=len(vloader))) if args.tqdm else (lambda x: x)
        print("VSET label size", len(vset.label2ans), vset.label2ans[0], type(vset.label2ans[0]))   #why is this so low

        """
        vkeys = list(vset.label2ans.keys())
        for i in range(10):
            print(i, vkeys[i], vset.label2ans[vkeys[i]])
        """

        quesid2ans = {}
        ans_not_found = 0
        for i, (ques_id, feats, boxes, sent, ent_spans, target) in viter_wrapper(enumerate(vloader)):
            #for i, datum_tuple in enumerate(vloader):

            #ques_id, feats, boxes, sent, ent_spans = datum_tuple[:5]   # Avoid seeing ground truth
            #print("ALL PREDS",i, "sentences",len(sent), sent[0:5], "feats",len(feats), "boxes", len(boxes), "ent_spans:", len(ent_spans), len(ent_spans[0]))#, type(datum_tuple),len(datum_tuple))
            with torch.no_grad():
                #are sent and ent_spans of the right type here ( they are lists and not tensors when they are passed through , which i think is fine)
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent, ent_spans)  #function expects ent spans no matter what now
                # CHECK TMRW::: Is Ent Spans the issue?
                #print("LOGIT", logit.shape, logit)
                score, label = logit.max(1)
                cur = 0
                notfound = 0
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    #at test time l is the answer to use
                    
                    if l < len(vset.label2ans):
                        
                        true_labs = target[cur].nonzero()
                        ans = vset.label2ans[l]
                        if cur < 10:
                            if len(true_labs) == 1:
                                if vset.label2ans[l] == vset.label2ans[int(true_labs)]:
                                    print("FOUND true_labs", qid, l, type(l), "pred", vset.label2ans[l], "true", true_labs, vset.label2ans[int(true_labs)])
                            else:
                                print("true_labs not found", true_labs, "for qid", qid, l, type(l), "pred", vset.label2ans[l])   
                                #maybe I should let test rewrite the trainval json file?
                    else:
                        notfound += 1
                        print(cur, "ANSWER NOT FOUND", qid, l, target[cur])
                        ans = l

                    cur += 1
                    quesid2ans[qid] = ans
  
                    """

                    try:
                        # For Test Predictions don't worry about answer index lookup and just compare against true value
                        #ans = l    #this was giving all 359
                        #print("pred1", qid, qid.item(), l, ans)
                        quesid2ans[qid.item()] = ans   #instea of l
                    except Exception as e:
                        #pred2 13338_4 10391 Sunny Lane
                        if qid == "13338_4":
                            print("pred2", qid, l, ans, target[-1])
                        #quesid2ans[qid] = ans  #instead of l
                        quesid2ans[qid] = ans
                    """

        if dump is not None:
            vevaluator.dump_result(quesid2ans, dump)
        vans = Counter([quesid2ans[q] for q in quesid2ans])
        print("PRED QIDS",vans)
        print("Total", cur, "Not Found", notfound, " ( ",round(notfound/cur,4),"%)")
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        print("IN EVALUATE() with eval tuple",eval_tuple)   #check this versus okvqa
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, ent_spans, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                try:
                    quesid2ans[qid.item()] = ans
                except Exception as e:
                    # TODO: this is either due to question_id being a string and not float in my case OR due to the trainval_labels  files only have train info
                    #print("Error on qid:",qid,type(qid),"with ans",ans,type(ans), e)
                    #print("oracle score",qid,l,ans,qid)
                    quesid2ans[qid] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("!!! Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    st = time.time()

    # Build Class
    kvqa = KVQA()

    # Load KVQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        # only for loading non pre-trained models
        kvqa.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       
        # Always loading all data in test   
        # ALSO (bs = 950?  ) <--- very large batch sizes
        if 'test' in args.test:
            args.test = args.test + args.split_num
            test_file_name = "test_predict"+args.split_num + ".json"
            print("Testing", test_file_name)
            test_tuple = get_data_tuple(args.test, bs=950, shuffle=False, drop_last=False, incl_para = args.incl_para, incl_caption = args.incl_caption, use_lm = args.use_lm)
            print("Test tuple", test_tuple)
            #test if test_tuple same as train_tuple label2ans
            test_list1 = kvqa.train_tuple.dataset.label2ans
            test_list2 = test_tuple.dataset.label2ans
            if len(test_list1)== len(test_list2) and len(test_list1) == sum([1 for i, j in zip(test_list1, test_list2) if i == j]):
                print ("TEST/TRAIN: Both label2ans lists are identical")
            else :
                print ("The label2ans lists are not identical!")
                print("Train ",kvqa.train_tuple.dataset.label2ans[0:10])
                print("Valid ",test_tuple.dataset.label2ans[0:10])
                import sys
                sys.exit()

            result = kvqa.evaluate( 
                test_tuple, 
                dump=os.path.join(args.output, test_file_name)                      
            )
            print(result)
        elif 'val' in args.test:    
            # D: I Don't think we'll be using this
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            minival_file_name = 'minival_predict'+args.split_num+'.json'
            print("Val Testing", minival_file_name)
            result = kvqa.evaluate(
                get_data_tuple('minival', bs=950,
                               shuffle=False, drop_last=False,
                               incl_para = args.incl_para, 
                               incl_caption = args.incl_caption),
                dump=os.path.join(args.output, minival_file_name)
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', kvqa.train_tuple.dataset.splits)
        if kvqa.valid_tuple is not None:
            print('Splits in Valid data:', kvqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (kvqa.oracle_score(kvqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        kvqa.train(kvqa.train_tuple, kvqa.valid_tuple)

    print("Done.  Elapsed time:", time.time() - st )


