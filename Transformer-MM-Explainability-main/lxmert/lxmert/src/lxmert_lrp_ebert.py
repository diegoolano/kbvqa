# coding=utf-8
# Copyright 2018 Hao Tan, Mohit Bansal, and the HuggingFace team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LXMERT model with option to enhance entities with E-BERT"""

import math
import os
import warnings
import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss
#from lxmert.lxmert.src.layers import *
import lxmert.lxmert.src.layers as LAY
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
#from transformers.configuration_lxmert import LxmertConfig
from transformers.models.lxmert.configuration_lxmert import LxmertConfig

from lxmert.lxmert.src.lxrt.tokenization import BertTokenizer
import numpy as np

from operator import itemgetter
from fuzzywuzzy import fuzz  #fuzzy search for entity matches
from collections import namedtuple

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LxmertConfig"
_TOKENIZER_FOR_DOC = "LxmertTokenizer"

LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "unc-nlp/lxmert-base-uncased",
]

ACT2FN = {
    "relu": LAY.ReLU,
    "tanh": LAY.Tanh,
    "gelu": LAY.GELU,
}


@dataclass
class LxmertModelOutput(ModelOutput):
    """
    Lxmert's outputs that contain the last hidden states, pooled outputs, and attention probabilities for the language,
    visual, and, cross-modality encoders. (note: the visual encoder in Lxmert is referred to as the "relation-ship"
    encoder")


    Args:
        language_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the language encoder.
        vision_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the visual encoder.
        pooled_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification, CLS, token) further processed
            by a Linear layer and a Tanh activation function. The Linear
        language_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        language_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        vision_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    language_output: Optional[torch.FloatTensor] = None
    vision_output: Optional[torch.FloatTensor] = None
    pooled_output: Optional[torch.FloatTensor] = None
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    language_attentions: Optional[Tuple[torch.FloatTensor]] = None
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    #NEWLY ADDED
    embedding_output: Optional[torch.FloatTensor] = None


@dataclass
class LxmertForQuestionAnsweringOutput(ModelOutput):
    """
    Output type of :class:`~transformers.LxmertForQuestionAnswering`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.k.
        question_answering_score: (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, n_qa_answers)`, `optional`):
            Prediction scores of question answering objective (classification).
        language_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        language_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        vision_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    question_answering_score: Optional[torch.FloatTensor] = None
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    language_attentions: Optional[Tuple[torch.FloatTensor]] = None
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    #NEWLY ADDED
    embedding_output: Optional[torch.FloatTensor] = None


@dataclass
class LxmertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.LxmertForPreTraining`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cross_relationship_score: (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the textual matching objective (classification) head (scores of True/False
            continuation before SoftMax).
        question_answering_score: (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, n_qa_answers)`):
            Prediction scores of question answering objective (classification).
        language_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        language_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        vision_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.

    """

    loss: [torch.FloatTensor] = None
    prediction_logits: Optional[torch.FloatTensor] = None
    cross_relationship_score: Optional[torch.FloatTensor] = None
    question_answering_score: Optional[torch.FloatTensor] = None
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    language_attentions: Optional[Tuple[torch.FloatTensor]] = None
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


def load_tf_weights_in_lxmert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
                n
                in [
                    "adam_v",
                    "adam_m",
                    "AdamWeightDecayOptimizer",
                    "AdamWeightDecayOptimizer_1",
                    "global_step",
                ]
                for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


class LxmertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, use_lm=None):
        super().__init__()

        #TODO: another way to do things is to just pass things through input_embeds ( and make input_ids NONE).  a little cleaner, but less self contained

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LAY.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = LAY.Dropout(config.hidden_dropout_prob)

        self.add1 = LAY.Add()
        self.add2 = LAY.Add()

    def forward(self, input_ids, token_type_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        seq_length = input_shape[1]

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.add1([token_type_embeddings, position_embeddings])
        embeddings = self.add2([embeddings, inputs_embeds])
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def relprop(self, cam, **kwargs):
        cam = self.dropout.relprop(cam, **kwargs)
        cam = self.LayerNorm.relprop(cam, **kwargs)

        # [inputs_embeds, position_embeddings, token_type_embeddings]
        (cam) = self.add2.relprop(cam, **kwargs)

        return cam


class LxmertAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim = config.hidden_size
        self.query = LAY.Linear(config.hidden_size, self.head_size)
        self.key = LAY.Linear(ctx_dim, self.head_size)
        self.value = LAY.Linear(ctx_dim, self.head_size)

        self.dropout = LAY.Dropout(config.attention_probs_dropout_prob)

        self.matmul1 = LAY.MatMul()
        self.matmul2 = LAY.MatMul()
        self.softmax = LAY.Softmax(dim=-1)
        self.add = LAY.Add()
        self.mul = LAY.Mul()
        self.head_mask = None
        self.attention_mask = None
        self.clone = LAY.Clone()

        self.attn = None
        self.attn_gradients = None
        self.attn_cam = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def get_attn_cam(self):
        return self.attn_cam

    def save_attn_cam(self, attn_cam):
        self.attn_cam = attn_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_relprop(self, x):
        return x.permute(0, 2, 1, 3).flatten(2)

    def forward(self, hidden_states, context, attention_mask=None, output_attentions=False):
        key, value = self.clone(context, 2)
        mixed_query_layer = self.query(hidden_states)
        # mixed_key_layer = self.key(context)
        # mixed_value_layer = self.value(context)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = self.matmul1([query_layer, key_layer.transpose(-1, -2)])
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = self.add([attention_scores, attention_mask])

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

        self.save_attn(attention_probs)
        attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = self.matmul2([attention_probs, value_layer])
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    def relprop(self, cam, **kwargs):
        # Assume output_attentions == False
        cam = self.transpose_for_scores(cam)

        # [attention_probs, value_layer]
        (cam1, cam2) = self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam2 /= 2

        self.save_attn_cam(cam1)

        cam1 = self.dropout.relprop(cam1, **kwargs)

        cam1 = self.softmax.relprop(cam1, **kwargs)

        if self.attention_mask is not None:
            # [attention_scores, attention_mask]
            (cam1, _) = self.add.relprop(cam1, **kwargs)

        # [query_layer, key_layer.transpose(-1, -2)]
        (cam1_1, cam1_2) = self.matmul1.relprop(cam1, **kwargs)
        cam1_1 /= 2
        cam1_2 /= 2

        # query
        cam1_1 = self.transpose_for_scores_relprop(cam1_1)
        cam1_1 = self.query.relprop(cam1_1, **kwargs)

        # key
        cam1_2 = self.transpose_for_scores_relprop(cam1_2.transpose(-1, -2))
        cam1_2 = self.key.relprop(cam1_2, **kwargs)

        # value
        cam2 = self.transpose_for_scores_relprop(cam2)
        cam2 = self.value.relprop(cam2, **kwargs)

        cam = self.clone.relprop((cam1_2, cam2), **kwargs)

        # returning two cams- one for the hidden state and one for the context
        return (cam1_1, cam)


class LxmertAttentionOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = LAY.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LAY.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = LAY.Dropout(config.hidden_dropout_prob)
        self.add = LAY.Add()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        add = self.add([hidden_states, input_tensor])
        hidden_states = self.LayerNorm(add)
        return hidden_states

    def relprop(self, cam, **kwargs):
        cam = self.LayerNorm.relprop(cam, **kwargs)
        # [hidden_states, input_tensor]
        (cam1, cam2) = self.add.relprop(cam, **kwargs)
        cam1 = self.dropout.relprop(cam1, **kwargs)
        cam1 = self.dense.relprop(cam1, **kwargs)

        return (cam1, cam2)


class LxmertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = LxmertAttention(config)
        self.output = LxmertAttentionOutput(config)
        self.clone = LAY.Clone()

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None, output_attentions=False):
        inp1, inp2 = self.clone(input_tensor, 2)
        output = self.att(inp1, ctx_tensor, ctx_att_mask, output_attentions=output_attentions)
        if output_attentions:
            attention_probs = output[1]
        attention_output = self.output(output[0], inp2)
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs

    def relprop(self, cam, **kwargs):
        cam_output, cam_inp2 = self.output.relprop(cam, **kwargs)
        cam_inp1, cam_ctx = self.att.relprop(cam_output, **kwargs)
        cam_inp = self.clone.relprop((cam_inp1, cam_inp2), **kwargs)

        return (cam_inp, cam_ctx)


class LxmertSelfAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = LxmertAttention(config)
        self.output = LxmertAttentionOutput(config)
        self.clone = LAY.Clone()

    def forward(self, input_tensor, attention_mask, output_attentions=False):
        inp1, inp2, inp3 = self.clone(input_tensor, 3)
        # Self attention attends to itself, thus keys and queries are the same (input_tensor).
        output = self.self(
            inp1,
            inp2,
            attention_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            attention_probs = output[1]
        attention_output = self.output(output[0], inp3)
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs

    def relprop(self, cam, **kwargs):
        cam_output, cam_inp3 = self.output.relprop(cam, **kwargs)
        cam_inp1, cam_inp2 = self.self.relprop(cam_output, **kwargs)
        cam_inp = self.clone.relprop((cam_inp1, cam_inp2, cam_inp3), **kwargs)

        return cam_inp


class LxmertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = LAY.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

    def relprop(self, cam, **kwargs):
        cam = self.intermediate_act_fn.relprop(cam, **kwargs)
        cam = self.dense.relprop(cam, **kwargs)
        return cam


class LxmertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = LAY.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LAY.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = LAY.Dropout(config.hidden_dropout_prob)
        self.add = LAY.Add()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        add = self.add([hidden_states, input_tensor])
        hidden_states = self.LayerNorm(add)
        return hidden_states

    def relprop(self, cam, **kwargs):
        cam = self.LayerNorm.relprop(cam, **kwargs)
        # [hidden_states, input_tensor]
        (cam1, cam2)= self.add.relprop(cam, **kwargs)
        cam1 = self.dropout.relprop(cam1, **kwargs)
        cam1 = self.dense.relprop(cam1, **kwargs)
        return (cam1, cam2)


class LxmertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = LxmertSelfAttentionLayer(config)
        self.intermediate = LxmertIntermediate(config)
        self.output = LxmertOutput(config)
        self.clone = LAY.Clone()

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        outputs = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output = outputs[0]
        ao1, ao2 = self.clone(attention_output, 2)
        intermediate_output = self.intermediate(ao1)
        layer_output = self.output(intermediate_output, ao2)
        outputs = (layer_output,) + outputs[1:]  # add attentions if we output them
        return outputs

    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.output.relprop(cam, **kwargs)
        cam1 = self.intermediate.relprop(cam1, **kwargs)
        cam = self.clone.relprop((cam1, cam2), **kwargs)
        cam = self.attention.relprop(cam, **kwargs)
        return cam


class LxmertXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # The cross-attention Layer
        self.visual_attention = LxmertCrossAttentionLayer(config)

        # Self-attention Layers
        self.lang_self_att = LxmertSelfAttentionLayer(config)
        self.visn_self_att = LxmertSelfAttentionLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = LxmertIntermediate(config)
        self.lang_output = LxmertOutput(config)
        self.visn_inter = LxmertIntermediate(config)
        self.visn_output = LxmertOutput(config)

        self.clone1 = LAY.Clone()
        self.clone2 = LAY.Clone()
        self.clone3 = LAY.Clone()
        self.clone4 = LAY.Clone()

    def cross_att(
            self,
            lang_input,
            lang_attention_mask,
            visual_input,
            visual_attention_mask,
            output_x_attentions=False,
    ):
        lang_input1, lang_input2 = self.clone1(lang_input, 2)
        visual_input1, visual_input2 = self.clone2(visual_input, 2)
        if not hasattr(self, 'visual_attention_copy'):
            self.visual_attention_copy = copy.deepcopy(self.visual_attention)
        # Cross Attention
        lang_att_output = self.visual_attention(
            lang_input1,
            visual_input1,
            ctx_att_mask=visual_attention_mask,
            output_attentions=output_x_attentions,
        )
        visual_att_output = self.visual_attention_copy(
            visual_input2,
            lang_input2,
            ctx_att_mask=lang_attention_mask,
            output_attentions=False,
        )
        return lang_att_output, visual_att_output

    def relprop_cross(self, cam, **kwargs):
        cam_lang, cam_vis = cam
        cam_vis2, cam_lang2 = self.visual_attention_copy.relprop(cam_vis, **kwargs)
        cam_lang1, cam_vis1 = self.visual_attention.relprop(cam_lang, **kwargs)
        cam_vis = self.clone2.relprop((cam_vis1, cam_vis2), **kwargs)
        cam_lang = self.clone1.relprop((cam_lang1, cam_lang2), **kwargs)
        return cam_lang, cam_vis


    def self_att(self, lang_input, lang_attention_mask, visual_input, visual_attention_mask):
        # Self Attention
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask, output_attentions=False)
        visual_att_output = self.visn_self_att(visual_input, visual_attention_mask, output_attentions=False)
        return lang_att_output[0], visual_att_output[0]

    def relprop_self(self, cam, **kwargs):
        cam_lang, cam_vis = cam
        cam_vis = self.visn_self_att.relprop(cam_vis, **kwargs)
        cam_lang = self.lang_self_att.relprop(cam_lang, **kwargs)
        return cam_lang, cam_vis

    def output_fc(self, lang_input, visual_input):
        lang_input1, lang_input2 = self.clone3(lang_input, 2)
        visual_input1, visual_input2 = self.clone4(visual_input, 2)
        # FC layers
        lang_inter_output = self.lang_inter(lang_input1)
        visual_inter_output = self.visn_inter(visual_input1)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input2)
        visual_output = self.visn_output(visual_inter_output, visual_input2)

        return lang_output, visual_output

    def relprop_output(self, cam, **kwargs):
        cam_lang, cam_vis = cam
        cam_vis_inter, cam_vis2 = self.visn_output.relprop(cam_vis, **kwargs)
        cam_lang_inter, cam_lang2 = self.lang_output.relprop(cam_lang, **kwargs)
        cam_vis1 = self.visn_inter.relprop(cam_vis_inter, **kwargs)
        cam_lang1 = self.lang_inter.relprop(cam_lang_inter, **kwargs)
        cam_vis = self.clone4.relprop((cam_vis1, cam_vis2), **kwargs)
        cam_lang = self.clone3.relprop((cam_lang1, cam_lang2), **kwargs)
        return cam_lang, cam_vis

    def forward(
            self,
            lang_feats,
            lang_attention_mask,
            visual_feats,
            visual_attention_mask,
            output_attentions=False,
    ):
        lang_att_output, visual_att_output = self.cross_att(
            lang_input=lang_feats,
            lang_attention_mask=lang_attention_mask,
            visual_input=visual_feats,
            visual_attention_mask=visual_attention_mask,
            output_x_attentions=output_attentions,
        )
        attention_probs = lang_att_output[1:]
        lang_att_output, visual_att_output = self.self_att(
            lang_att_output[0],
            lang_attention_mask,
            visual_att_output[0],
            visual_attention_mask,
        )

        lang_output, visual_output = self.output_fc(lang_att_output, visual_att_output)
        return (
            (
                lang_output,
                visual_output,
                attention_probs[0],
            )
            if output_attentions
            else (lang_output, visual_output)
        )

    def relprop(self, cam, **kwargs):
        cam_lang, cam_vis = cam
        cam_lang, cam_vis = self.relprop_output((cam_lang, cam_vis), **kwargs)
        cam_lang, cam_vis = self.relprop_self((cam_lang, cam_vis), **kwargs)
        cam_lang, cam_vis = self.relprop_cross((cam_lang, cam_vis), **kwargs)
        return cam_lang, cam_vis

class LxmertVisualFeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        feat_dim = config.visual_feat_dim
        pos_dim = config.visual_pos_dim

        # Object feature encoding
        self.visn_fc = LAY.Linear(feat_dim, config.hidden_size)
        self.visn_layer_norm = LAY.LayerNorm(config.hidden_size, eps=1e-12)

        # Box position encoding
        self.box_fc = LAY.Linear(pos_dim, config.hidden_size)
        self.box_layer_norm = LAY.LayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = LAY.Dropout(config.hidden_dropout_prob)

    def forward(self, visual_feats, visual_pos):
        x = self.visn_fc(visual_feats)
        x = self.visn_layer_norm(x)
        y = self.box_fc(visual_pos)
        y = self.box_layer_norm(y)
        output = (x + y) / 2

        output = self.dropout(output)
        return output

    def relprop(self, cam, **kwargs):
        cam = self.dropout.relprop(cam, **kwargs)
        cam = self.visn_layer_norm.relprop(cam, **kwargs)
        cam = self.visn_fc.relprop(cam, **kwargs)
        return cam

class LxmertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Obj-level image embedding layer
        self.visn_fc = LxmertVisualFeatureEncoder(config)
        self.config = config

        # Number of layers
        self.num_l_layers = config.l_layers
        self.num_x_layers = config.x_layers
        self.num_r_layers = config.r_layers

        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        self.layer = nn.ModuleList([LxmertLayer(config) for _ in range(self.num_l_layers)])
        self.x_layers = nn.ModuleList([LxmertXLayer(config) for _ in range(self.num_x_layers)])
        self.r_layers = nn.ModuleList([LxmertLayer(config) for _ in range(self.num_r_layers)])

    def forward(
            self,
            lang_feats,
            lang_attention_mask,
            visual_feats,
            visual_pos,
            visual_attention_mask=None,
            output_attentions=None,
    ):

        vision_hidden_states = ()
        language_hidden_states = ()
        vision_attentions = () if output_attentions or self.config.output_attentions else None
        language_attentions = () if output_attentions or self.config.output_attentions else None
        cross_encoder_attentions = () if output_attentions or self.config.output_attentions else None

        visual_feats = self.visn_fc(visual_feats, visual_pos)

        # Run language layers
        for layer_module in self.layer:
            l_outputs = layer_module(lang_feats, lang_attention_mask, output_attentions=output_attentions)
            lang_feats = l_outputs[0]
            language_hidden_states = language_hidden_states + (lang_feats,)
            if language_attentions is not None:
                language_attentions = language_attentions + (l_outputs[1],)

        # Run relational layers
        for layer_module in self.r_layers:
            v_outputs = layer_module(visual_feats, visual_attention_mask, output_attentions=output_attentions)
            visual_feats = v_outputs[0]
            vision_hidden_states = vision_hidden_states + (visual_feats,)
            if vision_attentions is not None:
                vision_attentions = vision_attentions + (v_outputs[1],)

        # Run cross-modality layers
        for layer_module in self.x_layers:
            x_outputs = layer_module(
                lang_feats,
                lang_attention_mask,
                visual_feats,
                visual_attention_mask,
                output_attentions=output_attentions,
            )
            lang_feats, visual_feats = x_outputs[:2]
            vision_hidden_states = vision_hidden_states + (visual_feats,)
            language_hidden_states = language_hidden_states + (lang_feats,)
            if cross_encoder_attentions is not None:
                cross_encoder_attentions = cross_encoder_attentions + (x_outputs[2],)
        visual_encoder_outputs = (
            vision_hidden_states,
            vision_attentions if output_attentions else None,
        )
        lang_encoder_outputs = (
            language_hidden_states,
            language_attentions if output_attentions else None,
        )
        return (
            visual_encoder_outputs,
            lang_encoder_outputs,
            cross_encoder_attentions if output_attentions else None,
        )

    def relprop(self, cam, **kwargs):
        cam_lang, cam_vis = cam
        for layer_module in reversed(self.x_layers):
            cam_lang, cam_vis = layer_module.relprop((cam_lang, cam_vis), **kwargs)

        for layer_module in reversed(self.r_layers):
            cam_vis = layer_module.relprop(cam_vis, **kwargs)

        for layer_module in reversed(self.layer):
            cam_lang = layer_module.relprop(cam_lang, **kwargs)
        return cam_lang, cam_vis


class LxmertPooler(nn.Module):
    def __init__(self, config):
        super(LxmertPooler, self).__init__()
        self.dense = LAY.Linear(config.hidden_size, config.hidden_size)
        self.activation = LAY.Tanh()

        self.pool = LAY.IndexSelect()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # first_token_tensor = hidden_states[:, 0]
        first_token_tensor = self.pool(hidden_states, 1, torch.tensor(0, device=hidden_states.device))
        first_token_tensor = first_token_tensor.squeeze(1)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

    def relprop(self, cam, **kwargs):
        cam = self.activation.relprop(cam, **kwargs)
        cam = self.dense.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        cam = self.pool.relprop(cam, **kwargs)

        return cam


class LxmertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(LxmertPredictionHeadTransform, self).__init__()
        self.dense = LAY.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = LAY.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

    def relprop(self, cam, **kwargs):
        cam = self.LayerNorm.relprop(cam, **kwargs)
        cam = self.transform_act_fn.relprop(cam, **kwargs)
        cam = self.dense.relprop(cam, **kwargs)
        return cam


class LxmertLMPredictionHead(nn.Module):
    def __init__(self, config, lxmert_model_embedding_weights):
        super(LxmertLMPredictionHead, self).__init__()
        self.transform = LxmertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = Linear(
            lxmert_model_embedding_weights.size(1),
            lxmert_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = lxmert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(lxmert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

    def relprop(self, cam, **kwargs):
        cam = self.decoder.relprop(cam, **kwargs)
        cam = self.transform.relprop(cam, **kwargs)
        return cam


class LxmertVisualAnswerHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        hid_dim = config.hidden_size
        #self.logit_fc = nn.Sequential(
        self.logit_fc = LAY.Sequential(
            LAY.Linear(hid_dim, hid_dim * 2),
            LAY.GELU(),
            LAY.LayerNorm(hid_dim * 2, eps=1e-12),
            LAY.Linear(hid_dim * 2, num_labels),
        )

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)

    def relprop(self, cam, **kwargs):
        for m in reversed(self.logit_fc._modules.values()):
            #print(m, type(m))
            cam = m.relprop(cam, **kwargs)
        return cam


class LxmertVisualObjHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = LxmertPredictionHeadTransform(config)
        # Decide the use of visual losses
        visual_losses = {}
        if config.visual_obj_loss:
            visual_losses["obj"] = {"shape": (-1,), "num": config.num_object_labels}
        if config.visual_attr_loss:
            visual_losses["attr"] = {"shape": (-1,), "num": config.num_attr_labels}
        if config.visual_obj_loss:
            visual_losses["feat"] = {
                "shape": (-1, config.visual_feat_dim),
                "num": config.visual_feat_dim,
            }
        self.visual_losses = visual_losses

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder_dict = nn.ModuleDict(
            {key: nn.Linear(config.hidden_size, self.visual_losses[key]["num"]) for key in self.visual_losses}
        )

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for key in self.visual_losses:
            output[key] = self.decoder_dict[key](hidden_states)
        return output

    def relprop(self, cam, **kwargs):
        return self.transform.relprop(cam, **kwargs)


class LxmertPreTrainingHeads(nn.Module):
    def __init__(self, config, lxmert_model_embedding_weights):
        super(LxmertPreTrainingHeads, self).__init__()
        self.predictions = LxmertLMPredictionHead(config, lxmert_model_embedding_weights)
        self.seq_relationship = LAY.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

    def relprop(self, cam, **kwargs):
        cam_seq, cam_pooled = cam
        cam_pooled = self.seq_relationship.relprop(cam_pooled, **kwargs)
        cam_seq = self.predictions.relprop(cam_seq, **kwargs)
        return cam_seq, cam_pooled


class LxmertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LxmertConfig
    load_tf_weights = load_tf_weights_in_lxmert
    base_model_prefix = "lxmert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


LXMERT_START_DOCSTRING = r"""

    The LXMERT model was proposed in `LXMERT: Learning Cross-Modality Encoder Representations from Transformers
    <https://arxiv.org/abs/1908.07490>`__ by Hao Tan and Mohit Bansal. It's a vision and language transformer model,
    pretrained on a variety of multi-modal datasets comprising of GQA, VQAv2.0, MCSCOCO captions, and Visual genome,
    using a combination of masked language modeling, region of interest feature regression, cross entropy loss for
    question answering attribute prediction, and object tag prediction.

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.LxmertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

LXMERT_INPUTS_DOCSTRING = r"""

    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.LxmertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        visual_feats: (:obj:`torch.FloatTensor` of shape :obj:՝(batch_size, num_visual_features, visual_feat_dim)՝):
            This input represents visual features. They ROI pooled object features from bounding boxes using a
            faster-RCNN model)

            These are currently not provided by the transformers library.
        visual_pos: (:obj:`torch.FloatTensor` of shape :obj:՝(batch_size, num_visual_features, visual_pos_dim)՝):
            This input represents spacial features corresponding to their relative (via index) visual features. The
            pre-trained LXMERT model expects these spacial features to be normalized bounding boxes on a scale of 0 to
            1.

            These are currently not provided by the transformers library.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        visual_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`__
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Lxmert Model transformer outputting raw hidden-states without any specific head on top.",
    LXMERT_START_DOCSTRING,
)
class LxmertModel(LxmertPreTrainedModel):
    def __init__(self, config, enhanced_ents=None):
        super().__init__(config)

        # entity enhanced reps ( with concat slash, off by default )
        self.enhanced_ents = enhanced_ents if enhanced_ents not in [None,{}] else None
        if self.enhanced_ents is None:
            self.embeddings = LxmertEmbeddings(config)
        else:
            self.embeddings = EBertEmbeddings(config, enhanced_ents)
            
        self.encoder = LxmertEncoder(config)
        self.pooler = LxmertPooler(config)
        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="unc-nlp/lxmert-base-uncased",
        output_type=LxmertModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            visual_feats=None,
            visual_pos=None,
            attention_mask=None,
            visual_attention_mask=None,
            token_type_ids=None,
            surface2wiki = None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        debug = False

        # Positional Word Embeddings
        fin_toks = []
        if self.enhanced_ents is not None:
            token_type_ids = None
            embedding_output, fin_toks = self.embeddings(input_ids, token_type_ids, inputs_embeds, surface2wiki)

            if debug:
                #after this the size of the embeddings dim is not equal to that of input_ids so we need to make other things size of embedding_output!
                print("Embedding output/Input ids/Token Type ids sizes", embedding_output.size(),  input_ids.size())

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            if self.enhanced_ents == None:
                input_shape = input_ids.size()
            else:
                input_shape = embedding_output[:,:,0].size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if debug:
            print("Using EBERT",self.enhanced_ents,".  Input shape: ", input_shape)

        assert visual_feats is not None, "`visual_feats` cannot be `None`"
        assert visual_pos is not None, "`visual_pos` cannot be `None`"

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        else:
            #make sure attenion_mask matches input_shape
            if attention_mask.size() != input_shape:

                if debug:
                    print("Rehsape attention mask from", attention_mask.size(), attention_mask )
                    print("to input shape", input_shape)
                attention_mask =  torch.ones(input_shape, device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if debug:
            print("Attention Mask / Token Type Ids:", attention_mask.size(), token_type_ids.size())   #these should also be 1,33

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Process the visual attention mask
        if visual_attention_mask is not None:
            extended_visual_attention_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_visual_attention_mask = extended_visual_attention_mask.to(dtype=self.dtype)
            extended_visual_attention_mask = (1.0 - extended_visual_attention_mask) * -10000.0
        else:
            extended_visual_attention_mask = None

        if self.enhanced_ents == None:
            # Positional Word Embeddings
            embedding_output = self.embeddings(input_ids, token_type_ids, inputs_embeds)

        if debug:
            print("Extended attention mask: ", extended_attention_mask.size())   #incorrect Extended attention mask:  torch.Size([1, 1, 1, 35])

        # Run Lxmert encoder
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            visual_attention_mask=extended_visual_attention_mask,
            output_attentions=output_attentions,
        )

        if debug:
            print("Post Encoder", [ z for z in encoder_outputs ])

        visual_encoder_outputs, lang_encoder_outputs = encoder_outputs[:2]
        vision_hidden_states = visual_encoder_outputs[0]
        language_hidden_states = lang_encoder_outputs[0]

        all_attentions = ()
        if output_attentions:
            language_attentions = lang_encoder_outputs[1]
            vision_attentions = visual_encoder_outputs[1]
            cross_encoder_attentions = encoder_outputs[2]
            all_attentions = (
                language_attentions,
                vision_attentions,
                cross_encoder_attentions,
            )

        hidden_states = (language_hidden_states, vision_hidden_states) if output_hidden_states else ()

        visual_output = vision_hidden_states[-1]
        lang_output = language_hidden_states[-1]

        if debug:        
            print("Lang output to pooler: ", lang_output.size())

        pooled_output = self.pooler(lang_output)

        if not return_dict:
            return (lang_output, visual_output, pooled_output) + hidden_states + all_attentions

        #return embedding_output here so we can show tokens to highlight ( only need for explaination generation)
        return LxmertModelOutput(
            pooled_output=pooled_output,
            language_output=lang_output,
            vision_output=visual_output,
            language_hidden_states=language_hidden_states if output_hidden_states else None,
            vision_hidden_states=vision_hidden_states if output_hidden_states else None,
            language_attentions=language_attentions if output_attentions else None,
            vision_attentions=vision_attentions if output_attentions else None,
            cross_encoder_attentions=cross_encoder_attentions if output_attentions else None,
            embedding_output=fin_toks
        )
        #embedding_output=embedding_output

    def relprop(self, cam, **kwargs):
        cam_lang, cam_vis = cam
        cam_lang = self.pooler.relprop(cam_lang, **kwargs)
        cam_lang, cam_vis = self.encoder.relprop((cam_lang, cam_vis), **kwargs)
        return cam_lang, cam_vis



@add_start_docstrings(
    """Lxmert Model with a specified pretraining head on top. """,
    LXMERT_START_DOCSTRING,
)
class LxmertForPreTraining(LxmertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Configuration
        self.config = config
        self.num_qa_labels = config.num_qa_labels
        self.visual_loss_normalizer = config.visual_loss_normalizer

        # Use of pretraining tasks
        self.task_mask_lm = config.task_mask_lm
        self.task_obj_predict = config.task_obj_predict
        self.task_matched = config.task_matched
        self.task_qa = config.task_qa

        # Lxmert backbone
        self.lxmert = LxmertModel(config)

        # Pre-training heads
        self.cls = LxmertPreTrainingHeads(config, self.lxmert.embeddings.word_embeddings.weight)
        if self.task_obj_predict:
            self.obj_predict_head = LxmertVisualObjHead(config)
        if self.task_qa:
            self.answer_head = LxmertVisualAnswerHead(config, self.num_qa_labels)

        # Weight initialization
        self.init_weights()

        # Loss functions
        self.loss_fcts = {
            "l2": SmoothL1Loss(reduction="none"),
            "visual_ce": CrossEntropyLoss(reduction="none"),
            "ce": CrossEntropyLoss(),
        }

        visual_losses = {}
        if config.visual_obj_loss:
            visual_losses["obj"] = {
                "shape": (-1,),
                "num": config.num_object_labels,
                "loss": "visual_ce",
            }
        if config.visual_attr_loss:
            visual_losses["attr"] = {
                "shape": (-1,),
                "num": config.num_attr_labels,
                "loss": "visual_ce",
            }
        if config.visual_obj_loss:
            visual_losses["feat"] = {
                "shape": (-1, config.visual_feat_dim),
                "num": config.visual_feat_dim,
                "loss": "l2",
            }
        self.visual_losses = visual_losses

    def resize_num_qa_labels(self, num_labels):
        """
        Build a resized question answering linear layer Module from a provided new linear layer. Increasing the size
        will add newly initialized weights. Reducing the size will remove weights from the end

        Args:
            num_labels (:obj:`int`, `optional`):
                New number of labels in the linear layer weight matrix. Increasing the size will add newly initialized
                weights at the end. Reducing the size will remove weights from the end. If not provided or :obj:`None`,
                just returns a pointer to the qa labels :obj:`torch.nn.Linear`` module of the model without doing
                anything.

        Return:
            :obj:`torch.nn.Linear`: Pointer to the resized Linear layer or the old Linear layer
        """

        cur_qa_logit_layer = self.get_qa_logit_layer()
        if num_labels is None or cur_qa_logit_layer is None:
            return
        new_qa_logit_layer = self._resize_qa_labels(num_labels)
        self.config.num_qa_labels = num_labels
        self.num_qa_labels = num_labels

        return new_qa_logit_layer

    def _resize_qa_labels(self, num_labels):
        cur_qa_logit_layer = self.get_qa_logit_layer()
        new_qa_logit_layer = self._get_resized_qa_labels(cur_qa_logit_layer, num_labels)
        self._set_qa_logit_layer(new_qa_logit_layer)
        return self.get_qa_logit_layer()

    def get_qa_logit_layer(self) -> nn.Module:
        """
        Returns the the linear layer that produces question answering logits.

        Returns:
            :obj:`nn.Module`: A torch module mapping the question answering prediction hidden states or :obj:`None` if
            LXMERT does not have a visual answering head.
        """
        if hasattr(self, "answer_head"):
            return self.answer_head.logit_fc[-1]

    def _set_qa_logit_layer(self, qa_logit_layer):
        self.answer_head.logit_fc[-1] = qa_logit_layer

    def _get_resized_qa_labels(self, cur_qa_logit_layer, num_labels):

        if num_labels is None:
            return cur_qa_logit_layer

        cur_qa_labels, hidden_dim = cur_qa_logit_layer.weight.size()
        if cur_qa_labels == num_labels:
            return cur_qa_logit_layer

        # Build new linear output
        if getattr(cur_qa_logit_layer, "bias", None) is not None:
            new_qa_logit_layer = nn.Linear(hidden_dim, num_labels)
        else:
            new_qa_logit_layer = nn.Linear(hidden_dim, num_labels, bias=False)

        new_qa_logit_layer.to(cur_qa_logit_layer.weight.device)

        # initialize all new labels
        self._init_weights(new_qa_logit_layer)

        # Copy labels from the previous weights
        num_labels_to_copy = min(cur_qa_labels, num_labels)
        new_qa_logit_layer.weight.data[:num_labels_to_copy, :] = cur_qa_logit_layer.weight.data[:num_labels_to_copy, :]
        if getattr(cur_qa_logit_layer, "bias", None) is not None:
            new_qa_logit_layer.bias.data[:num_labels_to_copy] = cur_qa_logit_layer.bias.data[:num_labels_to_copy]

        return new_qa_logit_layer

    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=LxmertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids=None,
            visual_feats=None,
            visual_pos=None,
            attention_mask=None,
            visual_attention_mask=None,
            token_type_ids=None,
            inputs_embeds=None,
            labels=None,
            obj_labels=None,
            matched_label=None,
            ans=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        r"""
        labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        obj_labels: (``Dict[Str: Tuple[Torch.FloatTensor, Torch.FloatTensor]]``, `optional`):
            each key is named after each one of the visual losses and each element of the tuple is of the shape
            ``(batch_size, num_features)`` and ``(batch_size, num_features, visual_feature_dim)`` for each the label id
            and the label score respectively
        matched_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the whether or not the text input matches the image (classification) loss. Input
            should be a sequence pair (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:

            - 0 indicates that the sentence does not match the image,
            - 1 indicates that the sentence does match the image.
        ans: (``Torch.Tensor`` of shape ``(batch_size)``, `optional`):
            a one hot representation hof the correct answer `optional`

        Returns:
        """

        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        lxmert_output = self.lxmert(
            input_ids=input_ids,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            visual_attention_mask=visual_attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        lang_output, visual_output, pooled_output = (
            lxmert_output[0],
            lxmert_output[1],
            lxmert_output[2],
        )
        lang_prediction_scores, cross_relationship_score = self.cls(lang_output, pooled_output)
        if self.task_qa:
            answer_score = self.answer_head(pooled_output)
        else:
            answer_score = pooled_output[0][0]

        total_loss = (
            None
            if (labels is None and matched_label is None and obj_labels is None and ans is None)
            else torch.tensor(0.0, device=device)
        )
        if labels is not None and self.task_mask_lm:
            masked_lm_loss = self.loss_fcts["ce"](
                lang_prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
            total_loss += masked_lm_loss
        if matched_label is not None and self.task_matched:
            matched_loss = self.loss_fcts["ce"](cross_relationship_score.view(-1, 2), matched_label.view(-1))
            total_loss += matched_loss
        if obj_labels is not None and self.task_obj_predict:
            total_visual_loss = torch.tensor(0.0, device=input_ids.device)
            visual_prediction_scores_dict = self.obj_predict_head(visual_output)
            for key, key_info in self.visual_losses.items():
                label, mask_conf = obj_labels[key]
                output_dim = key_info["num"]
                loss_fct_name = key_info["loss"]
                label_shape = key_info["shape"]
                weight = self.visual_loss_normalizer
                visual_loss_fct = self.loss_fcts[loss_fct_name]
                visual_prediction_scores = visual_prediction_scores_dict[key]
                visual_loss = visual_loss_fct(
                    visual_prediction_scores.view(-1, output_dim),
                    label.view(*label_shape),
                )
                if visual_loss.dim() > 1:  # Regression Losses
                    visual_loss = visual_loss.mean(1)
                visual_loss = (visual_loss * mask_conf.view(-1)).mean() * weight
                total_visual_loss += visual_loss
            total_loss += total_visual_loss
        if ans is not None and self.task_qa:
            answer_loss = self.loss_fcts["ce"](answer_score.view(-1, self.num_qa_labels), ans.view(-1))
            total_loss += answer_loss

        if not return_dict:
            output = (
                         lang_prediction_scores,
                         cross_relationship_score,
                         answer_score,
                     ) + lxmert_output[3:]
            return ((total_loss,) + output) if total_loss is not None else output

        return LxmertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=lang_prediction_scores,
            cross_relationship_score=cross_relationship_score,
            question_answering_score=answer_score,
            language_hidden_states=lxmert_output.language_hidden_states,
            vision_hidden_states=lxmert_output.vision_hidden_states,
            language_attentions=lxmert_output.language_attentions,
            vision_attentions=lxmert_output.vision_attentions,
            cross_encoder_attentions=lxmert_output.cross_encoder_attentions,
        )



@add_start_docstrings(
    """Lxmert Model with a visual-answering head on top for downstream QA tasks""",
    LXMERT_START_DOCSTRING,
)
class LxmertForQuestionAnswering(LxmertPreTrainedModel):
    def __init__(self, config, use_lm=None):
        #if self.use_lm not in globals(): 
        #     print("use lm not in globals")
        self.use_lm = None

        # HERE make class to hold both mapper and wiki_emb and pass this to LxmertModel
        self.enhanced_ents = {}
        if use_lm != None:
             #print("Set self.use_lm to ", use_lm)
             self.use_lm = use_lm     
             if use_lm == "ebert":
                #load linear WikipediaVec to LXMERT pretrained BERT mapper !
                # force linear map
                mapper = load_mapper("wikipedia2vec-base-cased.lxmert-bert-base-uncased.linear")
                wiki_emb = load_wiki_embeddings()
                self.enhanced_ents['mapper'] = mapper
                self.enhanced_ents['wiki_emb'] = wiki_emb
       
        super().__init__(config)
        #print("Calling LxmertForQuestionAnsweringLRP")

        # Configuration
        self.config = config    #from transformers.configuration_lxmert import LxmertConfig
        self.num_qa_labels = config.num_qa_labels
        self.visual_loss_normalizer = config.visual_loss_normalizer

        #print("Config in LxmertQA", self.config)
        #print("QA labels",  config.num_qa_labels  )  
        #by default its 3129 and not the 18335 from KVQA so we just resize

        # Lxmert backbone
        self.lxmert = LxmertModel(config, self.enhanced_ents)

        self.answer_head = LxmertVisualAnswerHead(config, self.num_qa_labels)

        # Weight initialization
        self.init_weights()

        # Loss function
        self.loss = CrossEntropyLoss()

    def resize_num_qa_labels(self, num_labels):
        """
        Build a resized question answering linear layer Module from a provided new linear layer. Increasing the size
        will add newly initialized weights. Reducing the size will remove weights from the end

        Args:
            num_labels (:obj:`int`, `optional`):
                New number of labels in the linear layer weight matrix. Increasing the size will add newly initialized
                weights at the end. Reducing the size will remove weights from the end. If not provided or :obj:`None`,
                just returns a pointer to the qa labels :obj:`torch.nn.Linear`` module of the model without doing
                anything.

        Return:
            :obj:`torch.nn.Linear`: Pointer to the resized Linear layer or the old Linear layer
        """

        #print("Calling resize num qa with", num_labels)
        cur_qa_logit_layer = self.get_qa_logit_layer()
        if num_labels is None or cur_qa_logit_layer is None:
            return
        new_qa_logit_layer = self._resize_qa_labels(num_labels)
        self.config.num_qa_labels = num_labels
        self.num_qa_labels = num_labels

        return new_qa_logit_layer

    def _resize_qa_labels(self, num_labels):
        cur_qa_logit_layer = self.get_qa_logit_layer()
        new_qa_logit_layer = self._get_resized_qa_labels(cur_qa_logit_layer, num_labels)
        self._set_qa_logit_layer(new_qa_logit_layer)
        return self.get_qa_logit_layer()

    def get_qa_logit_layer(self) -> nn.Module:
        """
        Returns the the linear layer that produces question answering logits

        Returns:
            :obj:`nn.Module`: A torch module mapping the question answering prediction hidden states. :obj:`None`: A
            NoneType object if Lxmert does not have the visual answering head.
        """

        if hasattr(self, "answer_head"):
            return self.answer_head.logit_fc[-1]

    def _set_qa_logit_layer(self, qa_logit_layer):
        self.answer_head.logit_fc[-1] = qa_logit_layer

    def _get_resized_qa_labels(self, cur_qa_logit_layer, num_labels):

        if num_labels is None:
            return cur_qa_logit_layer

        cur_qa_labels, hidden_dim = cur_qa_logit_layer.weight.size()
        if cur_qa_labels == num_labels:
            return cur_qa_logit_layer

        # Build new linear output ( is this getting used? )
        if getattr(cur_qa_logit_layer, "bias", None) is not None:
            new_qa_logit_layer = LAY.Linear(hidden_dim, num_labels)
        else:
            new_qa_logit_layer = LAY.Linear(hidden_dim, num_labels, bias=False)

        new_qa_logit_layer.to(cur_qa_logit_layer.weight.device)

        # initialize all new labels
        self._init_weights(new_qa_logit_layer)

        # Copy labels from the previous weights
        num_labels_to_copy = min(cur_qa_labels, num_labels)
        new_qa_logit_layer.weight.data[:num_labels_to_copy, :] = cur_qa_logit_layer.weight.data[:num_labels_to_copy, :]
        if getattr(cur_qa_logit_layer, "bias", None) is not None:
            new_qa_logit_layer.bias.data[:num_labels_to_copy] = cur_qa_logit_layer.bias.data[:num_labels_to_copy]

        return new_qa_logit_layer

    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="unc-nlp/lxmert-base-uncased",
        output_type=LxmertForQuestionAnsweringOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            visual_feats=None,
            visual_pos=None,
            attention_mask=None,
            visual_attention_mask=None,
            token_type_ids=None,
            surface2wiki=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels: (``Torch.Tensor`` of shape ``(batch_size)``, `optional`):
            A one-hot representation of the correct answer

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        lxmert_output = self.lxmert(
            input_ids=input_ids,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            visual_attention_mask=visual_attention_mask,
            surface2wiki = surface2wiki,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        pooled_output = lxmert_output[2]
        answer_score = self.answer_head(pooled_output)
        loss = None
        if labels is not None:
            loss = self.loss(answer_score.view(-1, self.num_qa_labels), labels.view(-1))

        if not return_dict:
            output = (answer_score,) + lxmert_output[3:]
            return (loss,) + output if loss is not None else output

        self.vis_shape = lxmert_output.vision_output.shape

        """
        return LxmertForQuestionAnsweringOutput(
            loss=loss,
            question_answering_score=answer_score,
            language_hidden_states=lxmert_output.language_hidden_states,
            vision_hidden_states=lxmert_output.vision_hidden_states,
            language_attentions=lxmert_output.language_attentions,
            vision_attentions=lxmert_output.vision_attentions,
            cross_encoder_attentions=lxmert_output.cross_encoder_attentions,
            embedding_output=lxmert_output.embedding_output
        )
        """
        #if you need to pass lxmert_output.language_output, lxmert_output.vision_output   and lxmert_output.pooled_output  for use in knn experiment!
        dictionary = {"loss":loss,
            "pooled_output" : pooled_output,
            "language_output" : lxmert_output.language_output,
            "vision_output" : lxmert_output.vision_output,
            "question_answering_score" : answer_score,
            "language_hidden_states" : lxmert_output.language_hidden_states,
            "vision_hidden_states" : lxmert_output.vision_hidden_states,
            "language_attentions" : lxmert_output.language_attentions,
            "vision_attentions" : lxmert_output.vision_attentions,
            "cross_encoder_attentions" : lxmert_output.cross_encoder_attentions,
            "embedding_output" : lxmert_output.embedding_output}
        return namedtuple("LxmertKnnOutput", dictionary.keys())(*dictionary.values())
        


    def relprop(self, cam, **kwargs):
        cam_lang = self.answer_head.relprop(cam, **kwargs)
        cam_vis = torch.zeros(self.vis_shape).to(cam_lang.device)
        cam_lang, cam_vis = self.lxmert.relprop((cam_lang, cam_vis), **kwargs)
        return cam_lang, cam_vis


### NEW CLASSES, ETC below ( along with some minor changes to existing code base above )

@add_start_docstrings(
    """Lxmert Model with Enhanced Entity Representations and a visual-answering head on top for downstream QA tasks""",
    LXMERT_START_DOCSTRING,
)
class LxmertEnhancedForQuestionAnswering(LxmertForQuestionAnswering):
    def __init__(self, config):
        self.use_lm = "ebert"  #hardcoded for now
        #print(config)
        #print(type(config))
        #print(dir(config))
        #config["use_lm"] = self.use_lm  #TypeError: 'LxmertConfig' object does not support item assignment
        #config.update({"use_lm": self.use_lm})
        super().__init__(config, self.use_lm)   #if need be maybe pass use_lm through here?
        #print("Calling LxmertForQuestionAnsweringLRP Enhanced")

def load_mapper(name):
    return LinearMapper.load(name)

class Mapper:
    pass

class LinearMapper(Mapper):
    def train(self, x, y, w=None, verbose=0):
        if not w is None:
            w_sqrt = np.expand_dims(np.sqrt(w), -1)
            x *= w_sqrt
            y *= w_sqrt

        self.model = np.linalg.lstsq(x, y, rcond=None)[0]

    def apply(self, x, verbose=0):
        return x.dot(self.model)

    def save(self, path):
        if not path.endswith(".npy"):
            path += ".npy"
        np.save(path, self.model)

    @classmethod
    def load(cls, path):
        obj = cls()
        if not path.endswith(".npy"):
            path += ".npy"

        if not os.path.exists(path):
            path = os.path.join("/home/diego/adv_comp_viz21/lxmert/orig_code/lxmert_gen/ebert/mappers/", path)

        obj.model = np.load(path)
        return obj

def load_wiki_embeddings():
    return Wikipedia2VecEmbedding(path='wikipedia2vec-base-cased')


class Embedding:
    def __getitem__(self, word_or_words):
        if isinstance(word_or_words, str):
            if not word_or_words in self:
                raise Exception("Embedding does not contain", word_or_words)
            return self.getvector(word_or_words)
        
        for word in word_or_words:
            if not word in self:
                raise Exception("Embedding does not contain", word)
        
        return self.getvectors(word_or_words)
    
    @property
    def vocab(self):
        return self.get_vocab()

    @property
    def all_embeddings(self):
        return self[self.vocab]


class Wikipedia2VecEmbedding(Embedding):
    #note dependency on wikipedia2vec
    def __init__(self, path, prefix = "", do_cache_dict = True, do_lower_case = False):
        #def __init__(self, path, prefix = "ENTITY/", do_cache_dict = True, do_lower_case = False):
        from wikipedia2vec import Wikipedia2Vec, Dictionary
        DATA_RESOURCE_DIR = "/data/diego/adv_comp_viz21/ebert-master/resources/"
        if os.path.exists(os.path.join(DATA_RESOURCE_DIR, "wikipedia2vec", path)):
            #print("Loading ", os.path.join(DATA_RESOURCE_DIR, "wikipedia2vec", path))
            self.model = Wikipedia2Vec.load(os.path.join(DATA_RESOURCE_DIR, "wikipedia2vec", path))
        else:
            raise Exception()

        self.dict_cache = None
        if do_cache_dict:
            self.dict_cache = {}

        self.prefix = prefix
        self.do_lower_case = do_lower_case

        #assert self.prefix + "San_Francisco" in self
        #assert self.prefix + "St_Linus" in self

    def _preprocess_word(self, word):
        if word.startswith(self.prefix):
            word = " ".join(word[len(self.prefix):].split("_"))
        if self.do_lower_case:
            word = word.lower()
        return word
    
    def index(self, word):
        prepr_word = self._preprocess_word(word)

        if (not self.dict_cache is None) and prepr_word in self.dict_cache:
            return self.dict_cache[prepr_word]

        if word.startswith(self.prefix):
            ret = self.model.dictionary.get_entity(prepr_word)
        else:
            ret = self.model.dictionary.get_word(prepr_word)

        if not self.dict_cache is None:
            self.dict_cache[prepr_word] = ret
        
        return ret

    def __contains__(self, word):  
        return self.index(word) is not None

    def getvector(self, word):
        if word.startswith(self.prefix):
            return self.model.get_vector(self.index(word))
        return self.model.get_vector(self.index(word))
    
    @property
    def all_special_tokens(self):
        return []

    def getvectors(self, words):
        return np.stack([self.getvector(word) for word in words], 0)


def toks_to_str(toks):
    ret = toks[0]
    for i, t in enumerate(toks):
        if i > 0:
            if t.startswith("##"):
                ret += t.replace("##","")
            elif t in [".",",","?","!",";","'",'"']:
                ret += t
            else:
                ret += " " + t
    return ret

class EBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config, enhanced_ents):
        super(EBertEmbeddings, self).__init__()
        #print("Loading EBertEmbeddings")
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LAY.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = LAY.Dropout(config.hidden_dropout_prob)
   
        self.mapper = enhanced_ents['mapper']
        self.wiki_emb = enhanced_ents['wiki_emb']
        
        #self.tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

        #this doesn't have
        self.add1 = LAY.Add()
        self.add2 = LAY.Add()

        # Using the Lxmert Bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained( "bert-base-uncased", do_lower_case=True)

    def get_enhanced_embedding( self, input_ids, r, slash_emb, cur_ex, cur_ex_str, cur_ex_toks, cur_ents, unk_locs, tokenizer, debug, debug_end, debug_final):
        # METHOD FOR USE WITH SURFACEWIKI derived ent spans
        cur_embedding = []
        fin_toks = []
        i = 0  #necessary in case len(unk_locs)/2 == 0 ( in which case i would be undefined )

        if debug_end == -1:   #disable since it occurs at end anyways
            print("\nSTRING:" ,cur_ex_str)
            print("Cur Toks",len(cur_ex_toks), cur_ex_toks)
            print("CUR ENTS:", [ [a[0],a[1],a[2],len(a[3])] for a in cur_ents])  #[[9, 15, 'Daniela Bianchi', array([ 2.07436346e-02, -2.98745185e-02,  ... ],...]
            print("UNK_LOCS len /2", int(len(unk_locs) / 2), "CUR ENTS:", len(cur_ents), "\n") 

        is_handled = []
        u_i = -1    
        for i in range(int(len(unk_locs) / 2)):
            if i == 0:
                begin_ids = cur_ex[:unk_locs[i]]   #[tensor([ 101, 2040, 2003, 1996, 2711, 1999, 1996, 3746, 1029], device='cuda:0')]

                try:
                    begin_emb = self.word_embeddings(begin_ids)
                except Exception as e:
                    begin_emb = self.word_embeddings(torch.tensor(begin_ids, dtype=torch.long).cuda())

                if debug:
                    print("BEGIN IDS:",begin_ids, len(begin_ids), tokenizer.convert_ids_to_tokens([int(b) for b in begin_ids]))
                    print("Begin Emb", begin_emb.shape, begin_emb[0][0:5])
                cur_embedding.append(begin_emb)  #get word embeddings for everything prior to first UNK
                toks_added = tokenizer.convert_ids_to_tokens([int(b) for b in begin_ids])
                fin_toks.extend(toks_added)
                #is_handled.append(i)
                if debug_end:
                    print("BEGIN toks add", i, unk_locs[i],  len(toks_added), toks_added, '[UNK]' in toks_added, is_handled) 

            # add WikiMapped embedding + slash embedding + Bert embedding of ent
            # if entity not found in wikipedia vec ( don't construct ebert / .. just leave normal )

            if len(cur_ents) > i:         
                if len(cur_ents[i]) >= 4:
                    if cur_ents[i][3] != []:
                        # cur EBERT entity comes before current word loc  or  this i has already been handled
                        #u_i = is_handled[-1] if len(is_handled) > 0 else 0
                        u_i = len(is_handled) if len(is_handled) > 0 else 0
                        if cur_ents[i][0] < unk_locs[u_i]+1: #  or ( i > 0 and i in is_handled):
                            #add EBERT entity and slash
                            ebert_emb = torch.tensor(cur_ents[i][3], dtype=torch.long).unsqueeze(0).cuda()  
                            cur_embedding.append(ebert_emb)
                            cur_embedding.append(slash_emb)
                            fin_toks.extend([ "<ebert>"+cur_ents[i][2]+"</ebert>", "/" ])
                            if debug_end:
                                print("A. EBERT toks add", i, cur_ents[i][0], "to", cur_ents[i][1], 2, [ "<ebert>"+cur_ents[i][2]+"</ebert>", "/" ], is_handled) 
    
                            if debug:
                                print("EBERT ENT",i, cur_ents[i][2], type(ebert_emb), type(slash_emb), cur_ex[(cur_ents[i][0]+1):(cur_ents[i][1]-1)]) 
                                print("EBERT ENT Emb",i, ebert_emb.shape, [e[0:5] for e in ebert_emb ])

                            #then add BERT based entity rep without UNKs
                            bert_ent_ids = cur_ex[(unk_locs[u_i]+1):(unk_locs[u_i+1])]
                            bert_ent_emb = self.word_embeddings(torch.tensor(bert_ent_ids, dtype=torch.long).cuda())
                            if debug:
                                print("BERT ENT IDS:",u_i, bert_ent_ids, len(bert_ent_ids), tokenizer.convert_ids_to_tokens([int(b) for b in bert_ent_ids]))
                                print("BERT ENT Emb",u_i, bert_ent_emb.shape, [ b[0:5] for b in bert_ent_emb ])
                                
                            cur_embedding.append(bert_ent_emb)
                            toks_added = tokenizer.convert_ids_to_tokens([int(b) for b in bert_ent_ids])
                            fin_toks.extend(toks_added)
                            is_handled.append(i)
                            if debug_end:
                                print("A. BERT ENT toks add", i, (unk_locs[i]+1),"to", (unk_locs[i+1]), len(toks_added), toks_added, '[UNK]' in toks_added, is_handled) 
                        else:
                            next_ent_ind = cur_ents[i][0]
                            cur_start_ind = unk_locs[u_i]+1
                            orig_i = u_i
                            while cur_start_ind < next_ent_ind:
                                #add BERT based entity rep, etc until next EBERT ent ( we are trying to prevent EBERT ents from jumping the gun essentially)
                                bert_ent_ids = cur_ex[(unk_locs[u_i]+1):(unk_locs[u_i+1])]
                                bert_ent_emb = self.word_embeddings(torch.tensor(bert_ent_ids, dtype=torch.long).cuda())
                                if debug:
                                    print("BERT ENT IDS:",u_i, bert_ent_ids, len(bert_ent_ids), tokenizer.convert_ids_to_tokens([int(b) for b in bert_ent_ids]))
                                    print("BERT ENT Emb",u_i, bert_ent_emb.shape, [ b[0:5] for b in bert_ent_emb ])
                                    
                                cur_embedding.append(bert_ent_emb)
                                toks_added = tokenizer.convert_ids_to_tokens([int(b) for b in bert_ent_ids])
                                fin_toks.extend(toks_added)
                                is_handled.append(u_i)
                                #is_handled.append(orig_i)
                                if debug_end:
                                    print("B. BERT ENT toks add", u_i, (unk_locs[u_i]+1),"to", (unk_locs[u_i+1]), len(toks_added), toks_added, '[UNK]' in toks_added, is_handled) 
                                u_i += 1
                                cur_start_ind = unk_locs[u_i]+1

                            #add EBERT entity and slash
                            #i = orig_i
                            ebert_emb = torch.tensor(cur_ents[i][3], dtype=torch.long).unsqueeze(0).cuda()  
                            cur_embedding.append(ebert_emb)
                            cur_embedding.append(slash_emb)
                            fin_toks.extend([ "<ebert>"+cur_ents[i][2]+"</ebert>", "/" ])
                            if debug_end:
                                print("B. EBERT toks add", i, cur_ents[i][0], "to", cur_ents[i][1], 2, [ "<ebert>"+cur_ents[i][2]+"</ebert>", "/" ], is_handled) 
    
                            if debug:
                                print("EBERT ENT",i, cur_ents[i][2], type(ebert_emb), type(slash_emb), cur_ex[(cur_ents[i][0]+1):(cur_ents[i][1]-1)]) 
                                print("EBERT ENT Emb",i, ebert_emb.shape, [e[0:5] for e in ebert_emb ])

                            if (i + 1) == len(cur_ents):
                                # add final BERT rep of entity
                                bert_ent_ids = cur_ex[(unk_locs[u_i]+1):(unk_locs[u_i+1])]
                                bert_ent_emb = self.word_embeddings(torch.tensor(bert_ent_ids, dtype=torch.long).cuda())
                                cur_embedding.append(bert_ent_emb)
                                toks_added = tokenizer.convert_ids_to_tokens([int(b) for b in bert_ent_ids])
                                fin_toks.extend(toks_added)
                                is_handled.append(u_i)
                                if debug:
                                    print("BERT ENT IDS:",u_i, bert_ent_ids, len(bert_ent_ids), tokenizer.convert_ids_to_tokens([int(b) for b in bert_ent_ids]))
                                    print("BERT ENT Emb",u_i, bert_ent_emb.shape, [ b[0:5] for b in bert_ent_emb ])

                                if debug_end:
                                    print("C. BERT ENT toks add", u_i, (unk_locs[u_i]+1),"to", (unk_locs[u_i+1]), len(toks_added), toks_added, '[UNK]' in toks_added, is_handled) 
                             
                else:
                    print("!!! Cur ents[i] poorly formated? has no ind 3", i, cur_ents[i], " out all cur_ents", cur_ents)
            else:
                print("!!! Cur ents is < i ",i,len(cur_ents)," .. all cur_ents", cur_ents)
        
        # after last ent add remaining embeddings
        if u_i != -1:
            i = u_i   # FIX END

        if i < len(unk_locs):
            if len(cur_ex) > (unk_locs[i]+1):
                #end_ids = cur_ex[(unk_locs[i]+1):]
                end_ids = cur_ex[(unk_locs[i+1]+1):]
                if len(end_ids) > 0: 

                    # don't allow any UNKS to filter through
                    end_ids = end_ids[end_ids!=100]
                    end_embs = self.word_embeddings(torch.tensor(end_ids, dtype=torch.long).cuda())
                    cur_embedding.append(end_embs)             
                    toks_added = tokenizer.convert_ids_to_tokens([int(e) for e in end_ids])   #probably need to force '[UNK]' out if found here
                    fin_toks.extend(toks_added)
                    if debug_end:
                        print("END IDS ent toks add", i, (unk_locs[i+1]+1) , "to", len(cur_ex), len(toks_added), toks_added, '[UNK]' in toks_added) 
                else: 
                    if debug:
                        print("end ids was empty", end_ids, "for cur_ex", cur_ex_toks)   

                if debug:
                    print("END IDS:",i, end_ids, len(end_ids), tokenizer.convert_ids_to_tokens([int(e) for e in end_ids]))
                    print("END Emb",i, end_embs.shape, end_embs[0][0:5])
                    print("CurEmbedding sizes",type(cur_embedding), [( type(c), c.shape) for c in cur_embedding])   
            else:
                if debug:
                    print("UNK is last tok in", i, cur_ex_toks, "with UNKS at", unk_locs, "or UNKS == []")

                if cur_embedding == []:
                    end_embs = self.word_embeddings(input_ids[r])  #default to no UNKs
                    cur_embedding.append(end_embs)
                    toks_added = tokenizer.convert_ids_to_tokens([int(e) for e in input_ids[r]])     #Possible here to need to remove '[UNK]'
                    fin_toks.extend(toks_added)  #??
                    if debug_end:
                        print("POST last ent UNK is last toks add", i, len(toks_added), toks_added, '[UNK]' in toks_added) 
        else:
            if debug:
                print("i >= len(unk_locs).  UNK is last tok in", i, cur_ex_toks, "with UNKS at", unk_locs, cur_embedding, "or UNKS == []")
            if cur_embedding == []:
                end_embs = self.word_embeddings(input_ids[r])  #default to no UNKs  TODO check if this needs to be [input_ids[r]]
                cur_embedding.append(end_embs)
                toks_added = tokenizer.convert_ids_to_tokens([int(e) for e in input_ids[r]])
                fin_toks.extend(toks_added)  
                if debug_end:
                    print("POST i > len(unk_locs) toks add", i, len(toks_added), toks_added, '[UNK]' in toks_added) 

        return cur_embedding, fin_toks
    
    def remove_unks(self, x, debug=False):
        newx = x
        if debug:
            print("remove UNKS (100) from ", x)
        while 100 in newx:
            unklocs = [i for i in range(x.size(0)) if x[i] == 100 ]
            unkslen = len(unklocs)
            if debug:
                print("UNKS AT", unklocs, "handling first 2 of", unkslen)
            newx = torch.cat((x[0:unklocs[0]], x[(unklocs[0]+1):unklocs[1]], x[(unklocs[1]+1):]), axis=0)
            x = newx
            if debug:
                print("POST", newx, 100 in newx)
        return x

    def forward(self, input_ids, token_type_ids=None, inputs_embeds=None, surface2wiki=None):

        # add Wikipedia Mapper embeddings and slash if mapper not None!
        # 1) in entry.py convert_sents_to_features()  we add UNK tags around entities at token level
        # 2) (a)here get WikipediaVec emb of UNK spans, 
        #    (b)map each one to BERT uncased space and 
        #    (c)add them to embedding output followed by slah
        # 3) add position/tokentypes/etc after
      
        debug = False ## <--- SEE TO TRUE FOR DETAILED DEV / ERROR PURPOSES or FALSE to hide
        debug_end = False  #to show more concise debugging
        debug_final = False #to show just final enhanced output with respect to input str
        if debug:
            print("in Bert Embeddings with surface2wiki", surface2wiki)
       
        if debug:
            print('1', 'Yitzhak Ben - Zvi' in self.wiki_emb)  #False
            print('2', 'Yitzhak Ben-Zvi' in self.wiki_emb)    #True
            print('3', 'Yitzhak_Ben-Zvi' in self.wiki_emb)    #True

        # return either enhanced tokens or nothing ( and if nothing use tokens returned when getting input_ids 
        fin_toks = []
        if self.mapper is not None: 
            slash_tok_id = torch.tensor(self.tokenizer.convert_tokens_to_ids("/"), dtype=torch.long).cuda()
            slash_emb = self.word_embeddings(slash_tok_id)

            word_embeddings = []
            if debug:
                print("In BERT EMBEDDINGS with") 
                print("InputIDS",input_ids.shape, input_ids[0])
                print("Mapper",self.mapper)
                print("WikiEmb",self.wiki_emb)   
                print("tokenizer",self.tokenizer)   #do i need to do something about this?

            for r in range(input_ids.shape[0]):
                # find UNK token 100 in tensors to get list of ents
                cur_ex = input_ids[r]
                cur_ex_toks = self.tokenizer.convert_ids_to_tokens([int(t) for t in cur_ex])

                try:
                    cur_ex_str = toks_to_str(cur_ex_toks)
                except Exception as e:
                    print("Couldn't make string from ", r, cur_ex_toks)
                    print(cur_ex)
                    import sys
                    sys.exit()

                unk_locs = (cur_ex == 100).nonzero(as_tuple=True)[0]
                try:
                    assert(unk_locs.shape[0] % 2 == 0)
                except Exception as ex:
                    if debug:
                        print("!!!! UNK LOCS is not a factor of 2! for cur_ex_toks", cur_ex_toks)
                        print("UNK LOCS=", unk_locs, " so default to not showing any entities for this sentence")
                    bert_embs = self.word_embeddings(input_ids[r])  #default to no UNKs
                    word_embeddings.append(bert_embs)   
                    continue

                unk_locs = [ int(u) for u in unk_locs ]
 
                if debug:
                    print("Cur Ex",cur_ex)  #tensor([101, 2040, ..])
                    print("Cur Toks",cur_ex_toks)
                    print("Cur Str",cur_ex_str)
                    print("Unk Locations",unk_locs)

                cur_ents = []
                offset = 0
                for i in range(int(len(unk_locs) / 2)):
                   start_ent_ind, end_ent_ind = unk_locs[i*2], unk_locs[(i*2)+1] +1
                   cur_ex_ent = cur_ex[start_ent_ind:end_ent_ind]  #[100] id id id [100] 
                   cond1 = surface2wiki == None
                   cond2 = ( surface2wiki[r] == [] )
                   cond3 = False
                   if not cond1:
                       if not cond2:
                           #if any of the below are true, use brute force prior method
                           #print("CMP ", len(cur_ents), "and", len(surface2wiki[r]), r, i)
                           #print(cur_ents, surface2wiki[r])

                           if len(cur_ents) > (len(surface2wiki[r]) - 1):
                               cond3 = True # this should never happen
                           elif len(surface2wiki[r][len(cur_ents)]) > 4:
                               cond3 = True
                           else:
                               cond3 = surface2wiki[r][len(cur_ents)][3] == ''   

                   if cond1 or cond2 or cond3:
                       #convert ids to word and use WikiEmb on it and then mapper on that 
                       ent_title_toks =  cur_ex_toks[(start_ent_ind+1):(end_ent_ind-1)]   #without UNKS
                       
                       try:
                           ent_title =  toks_to_str(ent_title_toks)
                           if debug:
                               print("Without surface2wiki Checking if ",ent_title," in self.wiki_emb")
                           ent_title_orig = ent_title
                           if ent_title not in self.wiki_emb:
                               ent_title = ent_title.title()
                               if ent_title not in self.wiki_emb:
                                   ent_title = ent_title_orig.capitalize()
                                   if ent_title not in self.wiki_emb and ent_title_orig.endswith("s") and ent_title_orig[0:-1].title() in self.wiki_emb:
                                       ent_title = ent_title_orig[0:-1].title()
                       except Exception as ex:
                           if debug:
                               print("Couldn't find WikiEnt for", ent_title_toks, "from start/end indices",start_ent_ind+1, end_ent_ind, "in cur_ex_tokens",cur_ex_toks)
                               print(ex)
                           continue
                           
                   else:
                       #NEW use surface2wiki to get current ent_title to use!
                       ent_title = surface2wiki[r][len(cur_ents)][3] 
                       if debug:
                           print("SURFACE2WIKI: ADD ENT TITLE", r, surface2wiki[r][len(cur_ents)], "--->", ent_title)
                       
                   if debug:
                       print("Looking at span from ",start_ent_ind,"to", end_ent_ind, ent_title)

                   ##add wiki_emb vectors to ent title), 
                   #NOTE should be doing this at batch level, but variable num ents makes it tricky so skip for now
                   try:
                       ent_vectors = np.array(self.wiki_emb[ent_title])         #hmmm... Exception: ('Embedding does not contain', 'Yitzhak Ben - Zvi')  <-- HERE  
                       if debug:
                           print("FOUND entity in wiki_emb", ent_title )    # FOUND entity ,  <-- TODO Debug when/why this is happening!
                       mapped_title_vectors = self.mapper.apply(ent_vectors)     
                   except Exception as ex:
                       if debug:
                           print("NOT FOUND Entity in wiki_emb", ent_title, ex)   #if we are missing alot we might need to retrain WikipediaVecs !
                       mapped_title_vectors = []
                       if " - " in ent_title or "' " in ent_title:
                           ent_title = ent_title.replace(" - ","-").replace("' ","'")
                           if ent_title in self.wiki_emb:
                               if debug:
                                   print(" BUT THEN FOUND entity in wiki_emb", ent_title )    # FOUND entity ,  <-- TODO Debug when/why this is happening!
                               ent_vectors = np.array(self.wiki_emb[ent_title])
                               mapped_title_vectors = self.mapper.apply(ent_vectors)     
                      
                       
                   cur_ents.append([start_ent_ind, end_ent_ind, ent_title, mapped_title_vectors]) 

                #Now that we have Mapped Wikients, add them and slash embs to cur_ex
                if debug:
                    print("CUR ENTS:", [ [a[0],a[1],a[2],len(a[3])] for a in cur_ents])  #[[9, 15, 'Daniela Bianchi', array([ 2.07436346e-02, -2.98745185e-02,  ... ],...]
                    print("UNK_LOCS len /2", int(len(unk_locs) / 2), "CUR ENTS:", len(cur_ents)) 

                ##pre-check existence of Cur Ents if they all have len(a[3]) == 3 act as if the surface2wiki[r] == []
                all_have_no_wik_ents_matched = True
                for a in cur_ents:
                    if len(a[3]) != 0:
                        all_have_no_wik_ents_matched = False

                if surface2wiki == None: # or ( surface2wiki[r] == [] ):
                    print("TODO get prior method here for non 4span and test or just make any 3 span a 4 span in debug which is probably easier")
                    print(r,cur_ex_str,  surface2wiki[r])  #this was dying when not ent spans were passed .. check to see other method works on this case!
                    print("\nSTRING:" ,cur_ex_str)
                    print("Cur Toks",len(cur_ex_toks), cur_ex_toks)
                    print("CUR ENTS:", [ [a[0],a[1],a[2],len(a[3])] for a in cur_ents])  
                    print("EXITING")
                    import sys
                    sys.exit()
                else:
                    #print("Get enhanced embeddings using surface2wiki (ie, ent spans and WikiTitle passed in ")
                    if surface2wiki[r] == [] or all_have_no_wik_ents_matched :
                        if debug:
                            print("IN SURFACE2WIKI == [] with input_ids[r]:", input_ids[r], len(input_ids), type(input_ids))
                            #remove UNKS from input_ids[r] if they exist
                            print(input_ids.size())
                        fin_toks = cur_ex_toks
                        if 100 in input_ids[r]:
                            tmp_vec = self.remove_unks(input_ids[r], debug=debug_final)
                            #for training, pad tmp_vect so it has correct size of existing tensor
                            if input_ids.size(0) == 1:
                                del input_ids
                                input_ids = torch.tensor(tmp_vec ).unsqueeze(0).cuda()
                            else:
                                input_ids[r] = torch.nn.functional.pad(tmp_vec, pad=(0,input_ids[r].size(0) - tmp_vec.size(0)))
                            if debug:
                                print("After removing UNKS, input_ids[r] now: ", input_ids[r])
                            fin_toks = [ t for t in cur_ex_toks if t != '[UNK]']

                        cur_embedding = self.word_embeddings(input_ids[r])
                        #print(type(cur_embedding), cur_embedding.size())
                        tcur_embedding = cur_embedding
                  
                    else:
                        cur_embedding, fin_toks = self.get_enhanced_embedding( input_ids, r, slash_emb, cur_ex, cur_ex_str, cur_ex_toks, cur_ents, unk_locs, self.tokenizer, debug, debug_end, debug_final)
                        tcur_embedding = torch.cat(cur_embedding)
                        #this resizing section is unnecessary because we have batch size of one during explanation generation !
                        """
                        #before adding to word embeddings make sure its the same size as cur_ex_toks ( which has a max len so lots of padding at end)
                        if tcur_embedding.size(0) > len(cur_ex_toks):
                            cur_len = tcur_embedding.size(0)
                            tcur_embedding = tcur_embedding[:len(cur_ex_toks),]
                            print("Too big..  Old Len", cur_len, " and now: ", tcur_embedding.size(0))

                        elif tcur_embedding.size(0) < len(cur_ex_toks):
                            pad_emb = tcur_embedding[-1,:]
                            cur_len = tcur_embedding.size(0)
                            diff = len(cur_ex_toks) - cur_len
                            pad_emb_np = pad_emb.cpu().detach().numpy()
                            pad_embs_np = np.tile(pad_emb_np ,(diff,1))
                            print(tcur_embedding.size(), "vs", len(cur_ex_toks), "and ", type(pad_emb), pad_emb_np.shape)
                            pad_embs = torch.tensor(pad_embs_np).cuda()
                            #print( pad_embs.size()) 
                            # torch.Size([44, 768]) vs 50 and  <class 'torch.Tensor'>
                            tcur_embedding = torch.cat([tcur_embedding, pad_embs], axis=0)
                            #print("Too small.. Old len", cur_len, " and now: ",tcur_embedding.size(0))
                        """
                
                if debug or debug_end or debug_final:
                    print("\nSTRING:",r, cur_ex_str)
                    print("Cur Toks",len(cur_ex_toks), cur_ex_toks)
                    print("CUR ENTS:", [ [a[0],a[1],a[2],len(a[3])] for a in cur_ents])  
                    print("\nFinal EBERT concat embedding: ", tcur_embedding.shape)
                    print("ENHANCED:", len(fin_toks), fin_toks)
                    print("**************************************\n")

                    if r < -1: # or surface2wiki[r] == []:  #just for debug purposes works on sentences without ent spans see d_003
                        import sys
                        sys.exit()
    
                # VERIFY no entity UNK tags exist! ie is embedding for Token ID 100 in any of the items in cur_embedding? 
                #tcur_embedding = torch.cat(cur_embedding)
                assert(100 not in tcur_embedding)
                assert("[UNK]" not in fin_toks)

                if debug:
                    print("Final EBERT concat embedding: ", tcur_embedding.shape)

                word_embeddings.append(tcur_embedding)   
           

            # pad according to max len over 1st dimension
            if debug:
                print("Input_ids", input_ids.shape)
                print("Word Embeddings sizes: ")
                for ind, w in enumerate(word_embeddings):   #all ending vals are 0s
                    print(ind, w.shape, w[-10:][0:10])

            words_embeddings = torch.stack([ w[:word_embeddings[0].size(0)] for w in word_embeddings])   #

            if debug_final:
                print("Final Word Embeddings:", words_embeddings.shape)
                print("Input_ids ", input_ids.shape)

            word_emb_2d = words_embeddings[:,:,0]
            seq_length = words_embeddings.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=word_emb_2d.device)  
            position_ids = position_ids.unsqueeze(0).expand_as(word_emb_2d)
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(word_emb_2d, dtype=torch.int64)
               
            if debug:
                print(input_ids.size(1), words_embeddings.size(1), words_embeddings.shape, word_emb_2d.shape)  #21 21 torch.Size([1, 21, 768]) torch.Size([1, 21, 768])
                print("position ", position_ids.shape, position_ids.dtype)
                print("tokentypes:", token_type_ids.shape, token_type_ids.dtype)

            position_embeddings = self.position_embeddings(position_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)  #mostly likely need to 

    
            if debug:
                print("EBERT Dims Words", words_embeddings.shape, "Positions: ", position_embeddings.shape, "TokenTypes: ", token_type_embeddings.shape)
                #Dims Words torch.Size([1, 33, 768]) Positions:  torch.Size([1, 33, 768]) TokenTypes:  torch.Size([1, 33, 768])

        else:
            words_embeddings = self.word_embeddings(input_ids)

            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
    
            position_embeddings = self.position_embeddings(position_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
    
            if debug:
                print("Non EBERT Dims", words_embeddings.shape, "Positions: ", position_embeddings.shape, "TokenTypes: ", token_type_embeddings.shape)

        #Dims Words torch.Size([127, 32, 768]) Positions:  torch.Size([32, 127, 768]) TokenTypes:  torch.Size([32, 100, 768])
        #THIS IS WHAT WE'VE BEEN USING BUT MAYBE WE HAVE TO USE add1 and add2 for LRP?
        #embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.add1([token_type_embeddings, position_embeddings])
        embeddings = self.add2([embeddings, words_embeddings])   #as opposed to 2nd being inputs_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, fin_toks

    #this is new as of 9/27
    def relprop(self, cam, **kwargs):
        cam = self.dropout.relprop(cam, **kwargs)
        cam = self.LayerNorm.relprop(cam, **kwargs)

        # [inputs_embeds, position_embeddings, token_type_embeddings]
        (cam) = self.add2.relprop(cam, **kwargs)

        return cam


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

def brute_search(ents, input_str, debug=False):
    #need to completely redo index numbers for ents
    qchar_loc = input_str.index("?")
    for i, e in enumerate(ents):
        if e[0] in input_str[qchar_loc:]:
            ent_start_ind = input_str.index(e[0], qchar_loc)  #only consider positions past ?
        else:
            #fuzzy match
            found = False      
            eps = e[0].split(" ")
            sent_ps = input_str[(qchar_loc+1):].split(" ")
            if debug:
                print("Didn't find exact match so try fuzzy match for",eps,"in", sent_ps)
            for ep in eps:
                for sp in sent_ps:
                    ratio = fuzz.ratio(ep,sp)
                    if debug:
                        print(ratio,ep,sp)
                    if ratio > 75 and not found:
                        if debug:
                            print("Fuzzy found ",ep,sp,input_str,e)            
                            ent_start_ind = input_str.index(sp, qchar_loc) 
                            ents[i][0] = sp           #change entity to reflect mention ( TODO: might need to change this )
                            found = True
            if not found:
                return ents
        if ent_start_ind > qchar_loc :
            ent_start_ind -= qchar_loc + 2           #cause of initial data error
            ent_len = len(ents[i][0])
            ent_end_ind = ent_start_ind + ent_len
            ents[i][1] = ent_start_ind
            ents[i][2] = ent_end_ind  
    ents_sorted = sorted(ents, key=itemgetter(1))
    return ents_sorted

def all_matches_found(ents, qchar_loc,input_str, try_fix=True, init_off=2):  
    debug = False
    matches_found = True
    for e in ents:
        if e[0] != '':
            true = e[0]
            entstart = qchar_loc+e[1]+init_off    #this is the character location of start
            entend = qchar_loc+e[2]+init_off      #this is the character location of end 
            found = input_str[entstart:entend]   
            if found != true:
                if debug:
                    print("MISMATCH FOUND:  True",true,"Found",found, true == found, qchar_loc, entstart, entend)
                    print("SENT ", input_str )
                    print("ENTS ", ents )
                matches_found = False
    
    if try_fix != False:
        if not matches_found: 
            # can we fix it
            if ents[0][1] < -1 and not try_fix=="2nd":
                if debug:
                    print("PRE-FIX:", ents)
                all_same_pos = all(e[1] == ents[0][1] for e in ents)
                if all_same_pos:
                  #to fix ent6
                  ents = brute_search(ents, input_str)
                  if debug:
                      print("FIXING ents all same post fix:", ents)
                  matches_found, ents = all_matches_found(ents, qchar_loc, input_str, try_fix="2nd")
                else:
                  #to fix ents4 and ents5
                  offset = -1 * ents[0][1]
                  ents[0][1] = 0
                  ents[0][2] += offset
                  for e in range(1,5):
                      if ents[e][0] != '':
                          ents[e][1] += offset - 1
                          ents[e][2] += offset - 1
                  if debug:
                      print("FIXING: ents post fix:", ents)
                  matches_found, ents = all_matches_found(ents, qchar_loc, input_str, try_fix="2nd")
            else:
                # brute search as last resort
                ents = brute_search(ents, input_str)
                if debug:
                    print("New ents post fix:7", ents, "try_fix=",try_fix)
                matches_found, ents = all_matches_found(ents, qchar_loc, input_str, try_fix=False)
          
    return matches_found, ents

def return_okvqa_tokens_with_unks_and_ent_set(input_str, ents, tokenizer, debug=True):

    str_toks = tokenizer.tokenize(input_str)
    qchar_loc = 0
    qchar_tok_loc = 0
    try_fix = True    #default
    init_offset = 0   #as opposed to past 2 which works for KVQA
    init_len = len(str_toks)
  
    #0 pre-check ents are matched and if not rematch them (TODO)! 
    pre_ents = ents
    matches_found, ents = all_matches_found(ents, qchar_loc,input_str, try_fix, init_offset)
    if not matches_found:
        if debug:
            print("!!!!!  BAD !!!! MATCHES NOT FOUND FOR:",input_str,ents)
            print("Pre Ents:", pre_ents)
        return str_toks, [['', -1, -1], ['', -1, -1], ['', -1, -1], ['', -1, -1], ['', -1, -1]]   #return empty ent set of correct size

    #1. remove any ents that appear within the bounds of a prior ent another ( assuming ents are order by start)
    ent_ranges = [[e[1],e[2]]  for e in ents]
    to_delete = []
    for i in range(1,len(ent_ranges)):
        cur_start, cur_end = ent_ranges[i]
        for prior in range(0,i):
            prior_start, prior_end = ent_ranges[prior]
            cond1 = (cur_start >= prior_start and cur_end < prior_end)
            cond2 = (cur_start > prior_start and cur_end <= prior_end) 
            cond3 = (cur_start == prior_start and cur_end == prior_end)
            cond4 = (cur_start == -1  and cur_end == -1)
            if (cond1 or cond2 or cond3) and not cond4:
                if debug:
                    print("Delete Entity ",i, ents[i], "which occurs in span of entity",ents[prior])

                if i not in to_delete:
                    to_delete.append(i)
                    break
    to_delete.reverse()
    for v in to_delete:
        del ents[v]
  
    if debug:
        print("\nINPUT str:",input_str)
        print("Tokens",init_len, str_toks)
        print("Ents: ", ents)
  
    #2. find locations of ents
    ents_to_replace = []
    for e in ents:
        if e[0] != '':
          true = e[0]
          entstart = qchar_loc+e[1]+init_offset    #this is the character location of start
          entend = qchar_loc+e[2]+init_offset      #this is the character location of end 
          found = input_str[entstart:entend]            
          new_str_tok_front = tokenizer.tokenize(input_str[:entstart])       
          new_str_tok_mid = tokenizer.tokenize(input_str[entstart:entend])
          new_str_tok_end = tokenizer.tokenize(input_str[entend:])
          str_toks = new_str_tok_front + new_str_tok_mid + new_str_tok_end
          if debug:
              print("\nTRUE:",true, "FOUND",found, "MATCH", true == found, "( start:",entstart,", end:",entend,")" )
              print("\t", new_str_tok_mid)
          ents_to_replace.append(new_str_tok_mid)
  
    #3. finally add unks to ents
    last_tok_ind = -1   
    for e in ents_to_replace:
        # find start and end token inds to replace, and verify sublist is correct
        start_e, end_e = e[0], e[-1]
        start_e_ind, end_e_ind = -1, -1
        if debug:
            print("\nReplace",e)
    
        for i, tok in enumerate(str_toks):
            if i > last_tok_ind and i > qchar_tok_loc:
                if start_e_ind == -1 or ( start_e_ind != -1 and end_e_ind == -1 ):
                    if len(e) == 1 and tok == start_e:
                        start_e_ind = i
                    elif tok == start_e and str_toks[i+1] == e[1]:
                        start_e_ind = i
          
                    if start_e_ind != -1 and debug:
                        if len(str_toks) < i+1:
                            if debug:
                                print("For e[0]",e[0], " found index",start_e_ind, tok, str_toks[i+1])
                        else:
                            if debug:
                                print("For e[0]",e[0], " found index at end",start_e_ind, tok)
          
                if start_e_ind != -1 and end_e_ind == -1:
                    if len(e) == 1 and tok == end_e and (e == str_toks[start_e_ind:(i + 1)]):            
                        end_e_ind = i + 1
                        last_tok_ind = end_e_ind
                    elif tok == end_e and str_toks[i-1] == e[len(e)-2] and (e == str_toks[start_e_ind:(i + 1)]):
                        end_e_ind = i + 1
                        last_tok_ind = end_e_ind
          
                        if end_e_ind != -1 and debug:
                            print("For e[1]",e[1], " found index",end_e_ind, str_toks[i-1], tok)
   
        #make sure you found the ent! if not, ???
        try:
            assert(e == str_toks[start_e_ind:end_e_ind])
            str_toks = str_toks[:start_e_ind] + ['[UNK]'] +  str_toks[start_e_ind:end_e_ind] + ['[UNK]'] + str_toks[end_e_ind:]
        except Exception as ex:
            if debug:
                print(ex,"issues adding UNK so don't for",e)
            str_toks = str_toks

    val_ents = sum([1 if e[0] != '' else 0 for e in ents])
    if debug:
        print("Initial Length", init_len)
        print("Number of valid ents", val_ents * 2)
        print("Length after adding UNKS for ents", len(str_toks))
        print("At End StrToks:", str_toks)     
  
    try:
        assert((init_len +(val_ents*2) )== len(str_toks))
    except Exception as ex:
        if debug:
            print("ERROR init len",init_len,  " + val_ents*2 ", val_ents*2, "!=", len(str_toks) )
    

    
    return str_toks, ents

def return_tokens_with_unks_and_ent_set(input_str, ents, tokenizer, debug=True):

    str_toks = tokenizer.tokenize(input_str)
    qchar_loc = input_str.index("?")
    qchar_tok_loc = str_toks.index("?")
    init_len = len(str_toks)
  
    #0 pre-check ents are matched and if not rematch them (TODO)! 
    pre_ents = ents
    #called with init_offset 2 by default
    matches_found, ents = all_matches_found(ents, qchar_loc,input_str) 
    if not matches_found:
        if debug:
            print("!!!!!  BAD !!!! MATCHES NOT FOUND FOR:",input_str,ents)
            print("Pre Ents:", pre_ents)
        return str_toks, [['', -1, -1], ['', -1, -1], ['', -1, -1], ['', -1, -1], ['', -1, -1]]   #return empty ent set of correct size

    #1. remove any ents that appear within the bounds of a prior ent another ( assuming ents are order by start)
    ent_ranges = [[e[1],e[2]]  for e in ents]
    to_delete = []
    for i in range(1,len(ent_ranges)):
        cur_start, cur_end = ent_ranges[i]
        for prior in range(0,i):
            prior_start, prior_end = ent_ranges[prior]
            cond1 = (cur_start >= prior_start and cur_end < prior_end)
            cond2 = (cur_start > prior_start and cur_end <= prior_end) 
            cond3 = (cur_start == prior_start and cur_end == prior_end)
            if cond1 or cond2 or cond3:
                if debug:
                    print("Delete Entity ",i, ents[i], "which occurs in span of entity",ents[prior])

                if i not in to_delete:
                    to_delete.append(i)
                    break
    to_delete.reverse()
    for v in to_delete:
        del ents[v]
  
    if debug:
        print("\nINPUT str:",input_str)
        print("Tokens",init_len, str_toks)
        print("Ents: ", ents)
  
    #2. find locations of ents
    ents_to_replace = []
    for e in ents:
        if e[0] != '':
          true = e[0]
          entstart = qchar_loc+e[1]+2    #this is the character location of start
          entend = qchar_loc+e[2]+2      #this is the character location of end 
          found = input_str[entstart:entend]            
          new_str_tok_front = tokenizer.tokenize(input_str[:entstart])       
          new_str_tok_mid = tokenizer.tokenize(input_str[entstart:entend])
          new_str_tok_end = tokenizer.tokenize(input_str[entend:])
          str_toks = new_str_tok_front + new_str_tok_mid + new_str_tok_end
          if debug:
              print("\nTRUE:",true, "FOUND",found, "MATCH", true == found, "( start:",entstart,", end:",entend,")" )
              print("\t", new_str_tok_mid)
          ents_to_replace.append(new_str_tok_mid)
  
    #3. finally add unks to ents
    last_tok_ind = -1   
    for e in ents_to_replace:
        # find start and end token inds to replace, and verify sublist is correct
        start_e, end_e = e[0], e[-1]
        start_e_ind, end_e_ind = -1, -1
        if debug:
            print("\nReplace",e)
    
        for i, tok in enumerate(str_toks):
            if i > last_tok_ind and i > qchar_tok_loc:
                if start_e_ind == -1 or ( start_e_ind != -1 and end_e_ind == -1 ):
                    if len(e) == 1 and tok == start_e:
                        start_e_ind = i
                    elif tok == start_e and str_toks[i+1] == e[1]:
                        start_e_ind = i
          
                    if start_e_ind != -1 and debug:
                        if len(str_toks) < i+1:
                            if debug:
                                print("For e[0]",e[0], " found index",start_e_ind, tok, str_toks[i+1])
                        else:
                            if debug:
                                print("For e[0]",e[0], " found index at end",start_e_ind, tok)
          
                if start_e_ind != -1 and end_e_ind == -1:
                    if len(e) == 1 and tok == end_e and (e == str_toks[start_e_ind:(i + 1)]):            
                        end_e_ind = i + 1
                        last_tok_ind = end_e_ind
                    elif tok == end_e and str_toks[i-1] == e[len(e)-2] and (e == str_toks[start_e_ind:(i + 1)]):
                        end_e_ind = i + 1
                        last_tok_ind = end_e_ind
          
                        if end_e_ind != -1 and debug:
                            print("For e[1]",e[1], " found index",end_e_ind, str_toks[i-1], tok)
   
        #make sure you found the ent! if not, ???
        try:
            assert(e == str_toks[start_e_ind:end_e_ind])
            str_toks = str_toks[:start_e_ind] + ['[UNK]'] +  str_toks[start_e_ind:end_e_ind] + ['[UNK]'] + str_toks[end_e_ind:]
        except Exception as ex:
            if debug:
                print(ex,"issues adding UNK so don't for",e)
            str_toks = str_toks

    val_ents = sum([1 if e[0] != '' else 0 for e in ents])
    if debug:
        print("Initial Length", init_len)
        print("Number of valid ents", val_ents * 2)
        print("Length after adding UNKS for ents", len(str_toks))
        print("At End StrToks:", str_toks)     
  
    try:
        assert((init_len +(val_ents*2) )== len(str_toks))
    except Exception as ex:
        if debug:
            print("ERROR init len",init_len,  " + val_ents*2 ", val_ents*2, "!=", len(str_toks) )
    

    
    return str_toks, ents


def toks_to_str(toks):
    ret = toks[0]
    for i, t in enumerate(toks):
        if i > 0:
            if t.startswith("##"):
                ret += t.replace("##","")
            elif t in [".",",","?","!",";","'",'"']:
                ret += t
            else:
                ret += " " + t
    return ret

def convert_sents_to_features(sents, ent_spans, max_seq_length=-1, tokenizer=None, use_lm = None):
    """Loads a data file into a list of `InputBatch`s."""

    # if use_lm == "ebert" add ebert concat format here!

    # We didn't have entity spans originally so we considered using KnowBert to get them? look at ebert AIDA code .. see ebert/prepare.sh .. pretty complicated...
    # Instead we thought to try BLINK elq ?  github.com/facebookreseach/BLINK/tree/master/elq ( but it turns out you need over 100MB GPU )
    # Finally we got spans for KVQA programatically.. see kvqa/KVQA_data_lookup.ipynb ( spacy wiki linker isn't full functional either )
    
    # kvqa gives 10 ent_spans, while okvqa is 11.  for now we use that to call different return_tokens_with_unks methods

    debug = False
    features, all_tokens  = [], []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())
        if use_lm == "ebert":
            #Add entity span tags to tokens_a

            #ent_spans  # [['Francis Condon', 0, 14]]
            #sent       # 'Francis Condon in the early 20th Century'

            if len(ent_spans[0]) == 4:
                #if we pass in a 4th value, it is the wiki span to use and 0 is the span to find
                cur_ents = [ [ent_spans[e][0][i], int(ent_spans[e][1][i]), int(ent_spans[e][2][i]), ent_spans[e][3][i]] for e in range(len(ent_spans))]  #[name, start, end] for each ent in sentence
            else:
                #prior 3 value version where surface form is assumed to be wiki title
                cur_ents = [ [ent_spans[e][0][i], int(ent_spans[e][1][i]), int(ent_spans[e][2][i])] for e in range(len(ent_spans))]  #[name, start, end] for each ent in sentence
            # sort by start indexes 
            cur_ents_sorted = sorted(cur_ents, key=itemgetter(1))

            if len(cur_ents_sorted) == 11: 
                #if len(ent_spans[0]) == 4: 
                # call OKVQA specific func
                if debug:
                    print("CALLING RETURN OKVQA")
                tokens_a, ents = return_okvqa_tokens_with_unks_and_ent_set(sent, cur_ents_sorted, tokenizer, debug)
            else:
                if debug:
                    print("CALLING RETURN KVQA")
                tokens_a, ents = return_tokens_with_unks_and_ent_set(sent, cur_ents_sorted, tokenizer, debug)
            #if i > 4:
            #    import sys
            #    sys.exit()
  
            # TODO: the tokenizer does lowercase which is fine except that the Wikivec Mapper expects Cased ! <-- need to load a cased tokenizer

            # NOW verify here and then add changes to modeling.py to use mapper and add slash!


        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2 and max_seq_length != -1:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        
        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        if max_seq_length == -1: 
            max_seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        all_tokens.append(tokens)
    return features, all_tokens
