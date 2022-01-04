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
from tasks.okvqa_model import OKVQAModel
from tasks.okvqa_data import OKVQADataset, OKVQATorchDataset, OKVQAEvaluator

from collections import Counter

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False, use_lm=None, ent_set=None) -> DataTuple:
    print("USE LM:", use_lm)
    dset = OKVQADataset(splits, use_lm, ent_set)
    tset = OKVQATorchDataset(dset)
    evaluator = OKVQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class OKVQA:
    def __init__(self):
        # Datasets
        print("Load training", args.train)
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, 
            shuffle=True, drop_last=True, 
            use_lm = args.use_lm, ent_set = args.ent_set
        )
        if args.valid != "":
            print("Load valid", args.valid)
            self.valid_tuple = get_data_tuple(
                args.valid, bs=32,
                shuffle=False, drop_last=False,
                use_lm = args.use_lm, ent_set = args.ent_set
            )
        else:
            self.valid_tuple = None
        
        # Model
        print("Load OKVQAModel")
        self.model = OKVQAModel(self.train_tuple.dataset.num_answers, args.max_len, args.use_lm)

        print("Train label2ans size", type(self.train_tuple.dataset.label2ans), len(self.train_tuple.dataset.label2ans))  
        if args.valid != "":
            print("Val label2ans size", type(self.train_tuple.dataset.label2ans), len(self.valid_tuple.dataset.label2ans))    

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

        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        
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
        print("In train")
        print(loader)
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, ent_spans, target) in iter_wrapper(enumerate(loader)):

                print("Made it through")
                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent, ent_spans)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)

                # here just to check the target is being handled correctly
                debug_qid = '3973535'
                if debug_qid in ques_id:
                    nd = [n for n,v in enumerate(ques_id) if v == debug_qid][0] 
                    print("FOR", i, "batch of size", len(ques_id), ques_id[nd], sent[nd], target[nd] )
                    print("TARGET",[(ind, dset.label2ans[ind]) for ind,val in enumerate(target[nd]) if val != 0])     #the issue might be that you are remaking train_label2ans, etc on tiny data
                    print("LOGITs",[(ind, dset.label2ans[ind]) for ind,val in enumerate(logit[nd]) if val > 0.01])
                    print("logit size(1)", logit.size(1), "and loss prior ", loss )
 
                loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    try:
                        quesid2ans[qid.item()] = ans
                    except Exception as e:
                        quesid2ans[qid] = ans
          
            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")
    
                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

                if valid_score == 0:
                    import sys
                    sys.exit()
                     

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

        debug = False
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        ans_not_found = 0
        for i, datum_tuple in enumerate(loader):
            #ques_id, feats, boxes, sent = datum_tuple[:4]   # Avoid seeing ground truth
            ques_id, feats, boxes, sent, ent_spans, target = datum_tuple   
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent, ent_spans)
                score, label = logit.max(1)
                curi = 0
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    try:
                        ans = dset.label2ans[l]
                        if debug:
                            print("Predict",qid,l,ans, ques_id[curi], sent[curi])
                            print("\t",[(ind, v, dset.label2ans[ind]) for ind,v in enumerate(datum_tuple[4][curi]) if v > .01])    
                        quesid2ans[qid.item()] = ans
                    except Exception as e:
                        quesid2ans[qid] = l
                    curi += 1
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)

        vans = Counter([quesid2ans[q] for q in quesid2ans])
        print("PRED QIDS",vans)

        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        dset, loader, evaluator = eval_tuple
        print("In Evaluated on ", len(dset))
        quesid2ans = self.predict(eval_tuple, dump)
        print("Quesid2ans", len(quesid2ans), quesid2ans) 
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
                    quesid2ans[qid] = ans
        #print("Oracle score:", quesid2ans)  #gives most likely answer out of possible however many
        print(quesid2ans[list(quesid2ans.keys())[0]])
        ret =  evaluator.evaluate(quesid2ans)
        return ret

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    st = time.time()

    # Build Class
    okvqa = OKVQA()

    # Load OKVQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        okvqa.load(args.load)

    # Test 
    if args.test is not None:
        args.fast = args.tiny = False       
        # Always loading all data in test   
        # ALSO (bs = 950?  ) <--- very large batch sizes
        if 'val' in args.test or 'test' in args.test:  
            args.test = args.test 
            test_file_name = "test_predict.json"
            test_tuple = get_data_tuple(args.test, bs=950, shuffle=False, drop_last=False, use_lm=args.use_lm, ent_set=args.ent_set)
            print("Testing ", test_file_name, test_tuple.dataset)
            result = okvqa.evaluate(
                test_tuple, 
                dump=os.path.join(args.output, test_file_name)                      
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', okvqa.train_tuple.dataset.splits)
        if okvqa.valid_tuple is not None:
            print('Splits in Valid data:', okvqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (okvqa.oracle_score(okvqa.valid_tuple) * 100))  
        else:
            print("DO NOT USE VALIDATION")
        okvqa.train(okvqa.train_tuple, okvqa.valid_tuple)

    print("Done.  Elapsed time:", time.time() - st )
