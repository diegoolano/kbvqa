# coding=utf-8
# Copyright 2019 project LXRT.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import os

import torch
import torch.nn as nn

from lxrt.tokenization import BertTokenizer
from lxrt.modeling import LXRTFeatureExtraction as VisualBertForLXRFeature, VISUAL_CONFIG
from operator import itemgetter
from fuzzywuzzy import fuzz  #fuzzy search for entity matches


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
    # this qchar_loc and qchar_tok_loc Assumes entities are after ?
    # which is true in KVQA case, but not OKVQA case
    # this means i probably need to retrain ebert OKVQA

    qchar_loc = input_str.index("?")
    qchar_tok_loc = str_toks.index("?")

    init_offset = 2
    try_fix = True    #default
    
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
    
        #make sure you found the ent! if not, ???
        try:
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

def convert_sents_to_features(sents, ent_spans, max_seq_length, tokenizer, use_lm = None):
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

            #Prior was 
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
            if debug:
                print("PRE",i, tokens_a, cur_ents_sorted)

            if len(cur_ents_sorted) == 11: 
                # call OKVQA specific func
                if debug:
                    print("CALLING RETURN on OKVQA ENT SPAN LEN")
                tokens_a, ents = return_okvqa_tokens_with_unks_and_ent_set(sent, cur_ents_sorted, tokenizer, debug)
            else:
                if debug:
                    print("CALLING RETURN on KVQA ENT SPAN LEN")
                tokens_a, ents = return_tokens_with_unks_and_ent_set(sent, cur_ents_sorted, tokenizer, debug)

            if debug:
                print("POST",i, tokens_a, ents)
  
            # TODO: the tokenizer does lowercase which is fine except that the Wikivec Mapper expects Cased ! <-- need to load a cased tokenizer

            # NOW verify here and then add changes to modeling.py to use mapper and add slash!


        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        
        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
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


def set_visual_config(args):
    VISUAL_CONFIG.l_layers = args.llayers
    VISUAL_CONFIG.x_layers = args.xlayers
    VISUAL_CONFIG.r_layers = args.rlayers


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
            path = os.path.join("ebert/mappers/", path)

        obj.model = np.load(path)
        return obj

def load_wiki_embeddings():
    #kwargs = {key: kwargs[key] for key in kwargs if key in ("do_lower_case", "prefix")}
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
        #def __init__(self, path, prefix = "ENTITY/", do_cache_dict = True, do_lower_case = False):   why no ENTITY/ ?
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

class LXRTEncoder(nn.Module):
    def __init__(self, args, max_seq_length, use_lm=None, mode='x'):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.use_lm = use_lm
        set_visual_config(args)

        self.mapper = None
        self.wiki_emb = None
        if use_lm != None:
            if use_lm == "ebert":
                #load linear WikipediaVec to LXMERT pretrained BERT mapper !
                # force linear map
                #print("LOAD EBERT")
                self.mapper = load_mapper("wikipedia2vec-base-cased.lxmert-bert-base-uncased.linear")
                self.wiki_emb = load_wiki_embeddings()

        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained( "bert-base-uncased", do_lower_case=True)
        #self.tokenizer_cased = BertTokenizer.from_pretrained( "bert-base-cased", do_lower_case=False)   #TODO add to make WikiEmbs still cased

        # Build LXRT Model
        self.model = VisualBertForLXRFeature.from_pretrained(
            "bert-base-uncased",
            mode=mode,
            mapper=self.mapper,
            wiki_emb=self.wiki_emb,
            tokenizer=self.tokenizer
        )

        if args.from_scratch:
            print("initializing all the weights")
            self.model.apply(self.model.init_bert_weights)

    def multi_gpu(self):
        self.model = nn.DataParallel(self.model)

    @property
    def dim(self):
        return 768

    def forward(self, sents, ent_spans, feats, visual_attention_mask=None):
        train_features, tokens = convert_sents_to_features(
            sents, ent_spans, self.max_seq_length, self.tokenizer, self.use_lm)

        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()
        
        surface2wiki = []
        """
        if len(ent_spans[0]) == 4:
            # Wiki ent to link to is provided so need map
            cur_b = []
            for b, v in ent_spans:
                if v[0] != ''  and v[3] !='':
                    cur_b.append[v]
        """
            
        debug = False
        if debug:
            print("LXRT encoder forward")
            print("USE LM=",self.use_lm)
            print("SENTS", len(sents), "Sent 0",sents[0])
            print("InputIds", input_ids.shape)
            print("TOKENS",len(tokens), len(tokens[0]))             #32, 18    32 in batch, each sent of len 18
            print("ENT SPANS",len(ent_spans), len(ent_spans[0]), len(ent_spans[0][0]))  #11 spans each of size 4 for 32 sent in batch
            #print("Ent spans[0]", ent_spans[0:10][0:4][0])                   #sent 0 in batch
            print("ENT spans[0][0]", ent_spans[0][0])
            print("ENT spans[0][1]:", ent_spans[0][1])
            print("ENT spans[0][2]:", ent_spans[0][2])
            if len(ent_spans[0]) == 4:
                print("ENT spans[0][3]:", ent_spans[0][3])

        #for sp in range(11):
        #    print(sp, [ent_spans[sp][0][0], ent_spans[sp][1][0], ent_spans[sp][2][0], ent_spans[sp][3][0]])
        for i, t in enumerate(tokens):
            curt = toks_to_str(t)
            ps = curt.split("[UNK]")
            out = ""
            for ind in range(len(ps)):
                if ind % 2 == 0 or ind == len(ps):
                    out += ps[ind] + "\n"
                else:
                    out += "[UNK]"+ps[ind]+"[UNK]"

            if debug:
                print("\nExample ",i) #, curt)
                print(out)       

            if len(ent_spans[0]) == 4:
                wiklist = []
                for sp in range(11):
                    if ent_spans[sp][0][i] != '':
                        #print("\t",i, sp, [ent_spans[sp][0][i], ent_spans[sp][1][i], ent_spans[sp][2][i], ent_spans[sp][3][i]])
                        wiklist.append([ent_spans[sp][0][i], int(ent_spans[sp][1][i]), int(ent_spans[sp][2][i]), ent_spans[sp][3][i]])
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

        # so input_ids is a tensor of batch size x max token len   ( add slash and wikivec embedding in BertEmbedding within this model() call )
        output = self.model(input_ids, segment_ids, input_mask, visual_feats=feats, visual_attention_mask=visual_attention_mask, surface2wiki = surface2wiki)

        return output

    def save(self, path):
        torch.save(self.model.state_dict(),
                   os.path.join("%s_LXRT.pth" % path))

    def load(self, path):
        # Load state_dict from snapshot file
        print("Load LXMERT pre-trained model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.model.load_state_dict(state_dict, strict=False)




