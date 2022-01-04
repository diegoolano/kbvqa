# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>

class KVQAModel(nn.Module):
    def __init__(self, num_answers, max_kvqa_length=20, use_lm=None):
        super().__init__()
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=max_kvqa_length,
            use_lm=use_lm                         #use_lm is a flag for us to pass EBERT or EAE into LXMERT
        )
        hid_dim = self.lxrt_encoder.dim
        
        # KVQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    #def forward(self, feat, pos, sent):
    def forward(self, feat, pos, sent, ent_spans):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param ent_spans: (num_ents,[name,start,end],b)  
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """

        debug = False
        if debug:
            print("In KVQA MODEL forward")
            print("SENT BATCH", type(sent), len(sent), sent[0])   #list 32, Who is the person in the image? Daniela Bianchi in Requiem per un agente segreto (1966)
            print("ENT SPANS", type(ent_spans), len(ent_spans), type(ent_spans[0]), len(ent_spans[0]), len(ent_spans[0][0]), ent_spans[0][1].shape, ent_spans[0][2].shape)   # 5 x 3 x 32
            print("\t all for first sentence ", [ (e,v,ent_spans[e][v][0]) for e in range(5) for v in range(3)])
            """
            [(0, 0, 'Daniela Bianchi'), (0, 1, tensor(0)), (0, 2, tensor(15)), 
             (1, 0, ''), (1, 1, tensor(-1)), (1, 2, tensor(-1)), 
             (2, 0, ''), (2, 1, tensor(-1)), (2, 2, tensor(-1)), 
             (3, 0, ''), (3, 1, tensor(-1)), (3, 2, tensor(-1)), 
             (4, 0, ''), (4, 1, tensor(-1)), (4, 2, tensor(-1))]
            """

        x = self.lxrt_encoder(sent, ent_spans, (feat, pos))
        logit = self.logit_fc(x)

        return logit


