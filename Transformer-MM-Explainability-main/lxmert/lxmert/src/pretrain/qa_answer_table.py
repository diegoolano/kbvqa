# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import torch


class AnswerTable:
    ANS_CONVERT = {
        "a man": "man",
        "the man": "man",
        "a woman": "woman",
        "the woman": "woman",
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
        'grey': 'gray',
    }

    def __init__(self, dsets=None):
        self.all_ans = json.load(open("data/lxmert/all_ans.json"))
        if dsets is not None:
            dsets = set(dsets)
            # If the answer is used in the dsets
            self.anss = [ans['ans'] for ans in self.all_ans if
                         len(set(ans['dsets']) & dsets) > 0]
        else:
            self.anss = [ans['ans'] for ans in self.all_ans]
        self.ans_set = set(self.anss)

        self._id2ans_map = self.anss
        self._ans2id_map = {ans: ans_id for ans_id, ans in enumerate(self.anss)}

        assert len(self._id2ans_map) == len(self._ans2id_map)
        for ans_id, ans in enumerate(self._id2ans_map):
            assert self._ans2id_map[ans] == ans_id

    def convert_ans(self, ans):
        if len(ans) == 0:
            return ""
        ans = ans.lower()
        if ans[-1] == '.':
            ans = ans[:-1].strip()
        if ans.startswith("a "):
            ans = ans[2:].strip()
        if ans.startswith("an "):
            ans = ans[3:].strip()
        if ans.startswith("the "):
            ans = ans[4:].strip()
        if ans in self.ANS_CONVERT:
            ans = self.ANS_CONVERT[ans]
        return ans

    def ans2id(self, ans):
        return self._ans2id_map[ans]

    def id2ans(self, ans_id):
        return self._id2ans_map[ans_id]

    def ans2id_map(self):
        return self._ans2id_map.copy()

    def id2ans_map(self):
        return self._id2ans_map.copy()

    def used(self, ans):
        return ans in self.ans_set

    def all_answers(self):
        return self.anss.copy()

    @property
    def num_answers(self):
        return len(self.anss)


def load_lxmert_qa(path, model, label2ans):
    """
    Load model weights from LXMERT pre-training.
    The answers in the fine-tuned QA task (indicated by label2ans)
        would also be properly initialized with LXMERT pre-trained
        QA heads.

    :param path: Path to LXMERT snapshot.
    :param model: LXRT model instance.
    :param label2ans: The label2ans dict of fine-tuned QA datasets, like
        {0: 'cat', 1: 'dog', ...}
    :return:
    """
    #print("Load QA pre-trained LXMERT from %s " % path)
    loaded_state_dict = torch.load("%s_LXRT.pth" % path)
    model_state_dict = model.state_dict()

    #print("model state dict: ", model_state_dict.keys())

    # Handle Multi-GPU pre-training --> Single GPU fine-tuning
    for key in list(loaded_state_dict.keys()):
        loaded_state_dict[key.replace("module.", '')] = loaded_state_dict.pop(key)

    # Isolate bert model
    bert_state_dict = {}
    for key, value in loaded_state_dict.items():
        if key.startswith('bert.'):
            bert_state_dict[key] = value

    # Isolate answer head
    answer_state_dict = {}
    for key, value in loaded_state_dict.items():
        if key.startswith("answer_head."):
            answer_state_dict[key.replace('answer_head.', '')] = value

    # Do surgery on answer state dict
    ans_weight = answer_state_dict['logit_fc.3.weight']
    ans_bias = answer_state_dict['logit_fc.3.bias']
    import copy
    new_answer_weight = copy.deepcopy(model_state_dict['logit_fc.3.weight'])
    new_answer_bias = copy.deepcopy(model_state_dict['logit_fc.3.bias'])
    answer_table = AnswerTable()
    loaded = 0
    unload = 0
    if type(label2ans) is list:
        label2ans = {label: ans for label, ans in enumerate(label2ans)}
    for label, ans in label2ans.items():
        new_ans = answer_table.convert_ans(ans)
        if answer_table.used(new_ans):
            ans_id_9500 = answer_table.ans2id(new_ans)
            new_answer_weight[label] = ans_weight[ans_id_9500]
            new_answer_bias[label] = ans_bias[ans_id_9500]
            loaded += 1
        else:
            new_answer_weight[label] = 0.
            new_answer_bias[label] = 0.
            unload += 1
    #print("Loaded %d answers from LXRTQA pre-training and %d not" % (loaded, unload))
    #print()
    answer_state_dict['logit_fc.3.weight'] = new_answer_weight
    answer_state_dict['logit_fc.3.bias'] = new_answer_bias

    # Load Bert Weights
    bert_model_keys = set(model.lxrt_encoder.model.state_dict().keys())
    bert_loaded_keys = set(bert_state_dict.keys())
    assert len(bert_model_keys - bert_loaded_keys) == 0
    model.lxrt_encoder.model.load_state_dict(bert_state_dict, strict=False)

    # Load Answer Logic FC Weights
    model_keys = set(model.state_dict().keys())
    ans_loaded_keys = set(answer_state_dict.keys())
    assert len(ans_loaded_keys - model_keys) == 0

    model.load_state_dict(answer_state_dict, strict=False)


def load_lxmert_qa_hf(path, model, label2ans):
    """
    Load model weights from LXMERT pre-training.
    The answers in the fine-tuned QA task (indicated by label2ans)
        would also be properly initialized with LXMERT pre-trained
        QA heads.

    :param path: Path to LXMERT snapshot.
    :param model: LXRT model instance.
    :param label2ans: The label2ans dict of fine-tuned QA datasets, like
        {0: 'cat', 1: 'dog', ...}
    :return:
    """
    #print("Load QA pre-trained LXMERT from %s " % path)
    loaded_state_dict = torch.load("%s_LXRT.pth" % path)
    model_state_dict = model.state_dict()

    #print(model_state_dict.keys())

    # Handle Multi-GPU pre-training --> Single GPU fine-tuning
    for key in list(loaded_state_dict.keys()):
        loaded_state_dict[key.replace("module.", '')] = loaded_state_dict.pop(key)

    # Isolate bert model
    bert_state_dict = {}
    for key, value in loaded_state_dict.items():
        if key.startswith('bert.'):
            bert_state_dict[key] = value

    # Isolate answer head
    answer_state_dict = {}
    for key, value in loaded_state_dict.items():
        if key.startswith("answer_head."):
            answer_state_dict[key] = value

    # Do surgery on answer state dict
    ans_weight = answer_state_dict['answer_head.logit_fc.3.weight']
    ans_bias = answer_state_dict['answer_head.logit_fc.3.bias']
    import copy
    new_answer_weight = copy.deepcopy(model_state_dict['answer_head.logit_fc.3.weight'])
    new_answer_bias = copy.deepcopy(model_state_dict['answer_head.logit_fc.3.bias'])
    answer_table = AnswerTable()
    loaded = 0
    unload = 0
    if type(label2ans) is list:
        label2ans = {label: ans for label, ans in enumerate(label2ans)}
    for label, ans in label2ans.items():
        new_ans = answer_table.convert_ans(ans)
        if answer_table.used(new_ans):
            ans_id_9500 = answer_table.ans2id(new_ans)
            new_answer_weight[label] = ans_weight[ans_id_9500]
            new_answer_bias[label] = ans_bias[ans_id_9500]
            loaded += 1
        else:
            new_answer_weight[label] = 0.
            new_answer_bias[label] = 0.
            unload += 1
    #print("Loaded %d answers from LXRTQA pre-training and %d not" % (loaded, unload))
    #print()
    answer_state_dict['answer_head.logit_fc.3.weight'] = new_answer_weight
    answer_state_dict['answer_head.logit_fc.3.bias'] = new_answer_bias

    # Load Bert Weights
    #print(dir(model))
    # 'add_memory_hooks', 'add_module', 'adjust_logits_during_generation', 'answer_head', 'apply', 'authorized_missing_keys', 'authorized_unexpected_keys', 'base_model', 'base_model_prefix', 'beam_sample', 'beam_search', 'bfloat16', 'buffers', 'children', 'config', 'config_class', 'cpu', 'cuda', 'device', 'double', 'dtype', 'dummy_inputs', 'dump_patches', 'estimate_tokens', 'eval', 'extra_repr', 'float', 'floating_point_ops', 'forward', 'from_pretrained', 'generate', 'get_extended_attention_mask', 'get_head_mask', 'get_input_embeddings', 'get_output_embeddings', 'get_qa_logit_layer', 'greedy_search', 'half', 'init_weights', 'invert_attention_mask', 'keys_to_never_save', 'load_state_dict', 'load_tf_weights', 'loss', 'lxmert', 'modules', 'name_or_path', 'named_buffers', 'named_children', 'named_modules', 'named_parameters', 'num_parameters', 'num_qa_labels', 'parameters', 'prepare_inputs_for_generation', 'prune_heads', 'register_backward_hook', 'register_buffer', 'register_forward_hook', 'register_forward_pre_hook', 'register_parameter', 'relprop', 'requires_grad_', 'reset_memory_hooks_state', 'resize_num_qa_labels', 'resize_token_embeddings', 'sample', 'save_pretrained', 'set_input_embeddings', 'share_memory', 'state_dict', 'tie_weights', 'to', 'train', 'training', 'type', 'visual_loss_normalizer', 'zero_grad']

    #bert_model_keys = set(model.lxrt_encoder.model.state_dict().keys())
    #print(dir(model.lxmert.encoder))
    #'add_module', 'apply', 'bfloat16', 'buffers', 'children', 'config', 'cpu', 'cuda', 'double', 'dump_patches', 'eval', 'extra_repr', 'float', 'forward', 'half', 'layer', 'load_state_dict', 'modules', 'named_buffers', 'named_children', 'named_modules', 'named_parameters', 'num_l_layers', 'num_r_layers', 'num_x_layers', 'parameters', 'r_layers', 'register_backward_hook', 'register_buffer', 'register_forward_hook', 'register_forward_pre_hook', 'register_parameter', 'relprop', 'requires_grad_', 'share_memory', 'state_dict', 'to', 'train', 'training', 'type', 'visn_fc', 'x_layers', 'zero_grad']

    bert_model_keys = set(model.lxmert.encoder.state_dict().keys())  #442
    bert_loaded_keys = set(bert_state_dict.keys())                   #449
    #print("Diff = ", len(bert_model_keys),  len(bert_loaded_keys), len(bert_model_keys) - len(bert_loaded_keys), len(bert_model_keys - bert_loaded_keys))

    bert_ll = set([l.replace("bert.encoder.","") for l in list(bert_loaded_keys)])
    #print("LL Diff = ", len(bert_model_keys),  len(bert_ll), len(bert_model_keys) - len(bert_ll), len(bert_model_keys - bert_ll), bert_model_keys - bert_ll)
    #print(len(set(sorted(list(bert_model_keys))) - set(sorted(list(bert_loaded_keys)))), "should be zero" )  
    #print(set(sorted(list(bert_model_keys))) - set(sorted(list(bert_loaded_keys))), "<-- how they differ")  
   
    #is this a function of answer_head stuff.?
    #print("BERT_MODEL_KEYS", sorted(list(bert_model_keys)))
    #print("\n\n\n")
    #print("BERT_LOADED_KEYS", sorted(list(bert_loaded_keys)))


    bert_state_dict2 = {}
    for key, value in bert_state_dict.items():
        bert_state_dict2[key.replace("bert.encoder.","")] = value
    bert_loaded_keys = set(bert_state_dict2.keys()) 
    assert len(bert_model_keys - bert_loaded_keys) == 0      # Diff =  442, 449, -7, 442

    #model.lxrt_encoder.model.load_state_dict(bert_state_dict, strict=False)
    model.lxmert.encoder.load_state_dict(bert_state_dict2, strict=False)

    # Load Answer Logic FC Weights
    model_keys = set(model.state_dict().keys())
    ans_loaded_keys = set(answer_state_dict.keys())
    assert len(ans_loaded_keys - model_keys) == 0

    model.load_state_dict(answer_state_dict, strict=False)


