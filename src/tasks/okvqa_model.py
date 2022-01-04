# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>

class OKVQAModel(nn.Module):
    def __init__(self, num_answers, max_kvqa_length=50, use_lm=None):
        super().__init__()
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=max_kvqa_length,
            use_lm=use_lm                         #flag to pass EBERT or EAE into LXMERT
        )
        hid_dim = self.lxrt_encoder.dim
        
        # OKVQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent, ent_spans):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param ent_spans: (b,) Type -- list of [ ent_str, char star, char end, Wiki name ]
        :return: (b, num_answer) The logit of each answers.
        """
        print("Calling OKVQA MODEL forward with ", sent, ent_spans)

        x = self.lxrt_encoder(sent, ent_spans, (feat, pos))
        logit = self.logit_fc(x)

        return logit


