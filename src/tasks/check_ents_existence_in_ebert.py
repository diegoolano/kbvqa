import json
import time
from lxrt.entry import load_wiki_embeddings
from operator import itemgetter
from fuzzywuzzy import fuzz

#call from lxmert/src/
#PYTHONPATH=$PYTHONPATH:./src python tasks/check_ents_existence_in_ebert.py

#load data/kvqa/datasets.json
with open('../data/kvqa/dataset.json', 'r') as fp:
  kvqa_data = json.load(fp)

#get list of unique entities ( QID and name and image ids they appear in )
st = time.time()
print("Get Unique KVQA ents")
kvqa_ents = {}
img2qid = {}
for img_id in kvqa_data:
  cur = kvqa_data[img_id]
  for i, ent in enumerate(cur['NamedEntities']): 
    qid = cur['Qids'][i]
    if qid not in kvqa_ents:
      kvqa_ents[qid] = {'name':ent, 'images':[]}
    if img_id not in kvqa_ents[qid]['images']:
      kvqa_ents[qid]['images'].append(img_id)    
  img2qid[img_id] = [cur['Qids'], cur['NamedEntities']]

print("Elapsed", time.time() - st)
print("Found ", len(kvqa_ents),"ents")

ent_qids = list(kvqa_ents.keys())
for i in range(10):
  print(i, ent_qids[i], kvqa_ents[ent_qids[i]])

"""
Get Unique KVQA ents
Found  18880 ents
0 Q1441444 {'name': 'Francis Condon', 'images': ['21717']}
1 Q331838 {'name': 'Michael Steele', 'images': ['24345', '44974']}
2 Q1109829 {'name': 'Liu Mingkang', 'images': ['30218']}
3 Q4798060 {'name': 'Arthur Bourchier', 'images': ['18778']}
4 Q57405 {'name': 'Susilo Bambang Yudhoyono', 'images': ['45472', '29953', '29952', '29951']}
5 Q57675 {'name': 'Najib Razak', 'images': ['45472', '27849', '45471', '41973', '27850', '27848']}
6 Q275469 {'name': 'Elsie de Wolfe', 'images': ['14657', '43184', '14656']}
7 Q189164 {'name': 'Feodor Chaliapin', 'images': ['43793', '17334', '17335']}
8 Q529647 {'name': 'Paolo Tornaghi', 'images': ['43793']}
9 Q979387 {'name': 'Chaz Bono', 'images': ['15509']}
"""

#load wikiembeddings
st = time.time()
print("\nLoad Wikiembeddings")
wiki_embs = load_wiki_embeddings()
print("Elapsed", time.time() - st)

#for each unique entitiy see if it is in wikiembeddings or not
st = time.time()
print("\nLooking if KVQA ents are in  Wikiembeddings")
found, total = 0, 0
not_found = []
for qid in kvqa_ents:
  cur = kvqa_ents[qid]
  in_wiki = False
  if cur['name'] in wiki_embs:
    in_wiki = True
    found += 1
  else:
    not_found.append(cur['name'])
  kvqa_ents[qid]['in_wiki'] = in_wiki
  total += 1

print("Elapsed", time.time() - st)
print("FOR KVQA ents")
#print("Not Found List:", sorted(not_found))
print("\nFound ", found, "out of", total, " (",round(found/total,4),"%)")
print("Missed", len(not_found), " (",round(len(not_found)/total,4),"%)")
# Exact Match: Found  17320 out of 18880  ( 0.9174 %)
# Missed 1560  ( 0.0826 %)

#show number and save list of entities (qid / name ) and whether its in WikipediaVec
"""
with open("../data/kvqa/kvqa_ents.json","w") as f:
  json.dump(kvqa_ents, f)
"""

#next, go through data/kvqa/new_kvqa_q_caps_ents0423.json  and see which ents are in WikipediaVec / KVQA unique entities ( possibly trade out for next step )
print("Looking at QCAP entities found")
with open('../data/kvqa/new_kvqa_q_caps_ents0423.json', 'r') as fp:
  kvqa_qcap_data = json.load(fp)

qc_img_ids = list(kvqa_qcap_data.keys())
for i in range(10):
  print(i, qc_img_ids[i], kvqa_qcap_data[qc_img_ids[i]])

"""
0 21717 {'wikiCap_new': 'Francis Condon in the early 20th Century', 'ents_found': [['Francis Condon', 0, 14]]}
1 24345 {'wikiCap_new': 'Michael Steele in June 2016.', 'ents_found': [['Michael Steele', 0, 14]]}
2 30218 {'wikiCap_new': 'Liu Mingkang', 'ents_found': [['Liu Mingkang', 0, 12]]}
3 18778 {'wikiCap_new': 'Bourchier as Macbeth, 1910', 'ents_found': [['Bourchier', 0, 9]]}
4 45472 {'wikiCap_new': 'Najib with President Susilo Bambang Yudhoyono in Putrajaya on 18 December 2012.', 'ents_found': [['Susilo Bambang Yudhoyono', 21, 45], ['Najib', 0, 5]]}
5 14657 {'wikiCap_new': 'Elsie de Wolfe, James Hazen Hyde Ball, January 31, 1905', 'ents_found': [['Elsie de Wolfe', 0, 14]]}
6 43793 {'wikiCap_new': 'Chaliapin and Tornagi', 'ents_found': [['Chaliapin', 0, 9], ['Tornaghi', 14, 21]]}
7 15509 {'wikiCap_new': 'Bono at the 2012 GLAAD Awards', 'ents_found': [['Bono', 0, 4]]}
8 22511 {'wikiCap_new': 'Henry Labouchère', 'ents_found': [['Henry Labouchère', 0, 16]]}
9 30296 {'wikiCap_new': 'Dietrich Klagges', 'ents_found': [['Dietrich Klagges', 0, 16]]}
"""

#for each unique entitiy see if it is in wikiembeddings or not
st = time.time()
print("\nLooking if KVQA QCAP ents are in  Wikiembeddings")
cfound, ctotal = 0, 0
cnot_found = []
for img_id in kvqa_qcap_data:
  cur = kvqa_qcap_data[img_id]
  in_wiki = False
  for ent_i, ent in enumerate(cur['ents_found']):
    if ent[0] in wiki_embs:
      in_wiki = True
      cfound += 1
    else:
      cnot_found.append(ent[0])
  kvqa_qcap_data[img_id]['ents_found'][ent_i].append(in_wiki)
  ctotal += 1

print("Elapsed", time.time() - st)
print("FOR KVQA CaP DATA")
#print("Not Found List:", sorted(cnot_found))
print("\nFound ", cfound, "out of", total, " (",round(cfound/ctotal,4),"%)")
print("Missed", len(cnot_found), " (",round(len(cnot_found)/ctotal,4),"%)")

#Exact Match Found  19595 out of 18880  ( 0.7965 %)
#Missed 9335  ( 0.3794 %)


def brute_search(ents, input_str):
  #need to completely redo index numbers for ents
  qchar_loc = 0 
  print("CALLED BRUTE SEARCH WITH", ents)
  print("Inputstr",input_str,input_str[(qchar_loc+1):])
  for i, e in enumerate(ents):  
    if e[0] in input_str:
      ent_start_ind = input_str.index(e[0], qchar_loc)  
      print("Found",e)    
    else:
      print("Not Found so trying fuzzy",e)    
      #fuzzy match
      found = False      
      eps = e[0].split(" ")
      sent_ps = input_str[(qchar_loc+1):].split(" ")
      print("Fuzzy Looking for",eps,"in", sent_ps)
      for ep in eps:
        for sp in sent_ps:
          ratio = fuzz.ratio(ep,sp)
          print(ratio,ep,sp)
          if ratio > 75 and not found:
            print("Fuzzy found ",ep,sp,input_str,e)            
            ent_start_ind = input_str.index(sp, qchar_loc) 
            ents[i][0] = sp           
            found = True
      if not found:
        return ents

    #if ent_start_ind > -1 :
    if ent_start_ind > qchar_loc:
      print("found start of",e[0],"at ",ent_start_ind, qchar_loc, len(e[0]))        
      ent_start_ind -= qchar_loc + 2           #cause of initial data error
      ent_len = len(e[0])
      ent_end_ind = ent_start_ind + ent_len
      ents[i][1] = ent_start_ind
      ents[i][2] = ent_end_ind  
      print(input_str[(qchar_loc+ent_start_ind+2):(qchar_loc+ent_end_ind+2)])
  ents_sorted = sorted(ents, key=itemgetter(1))
  return ents_sorted

def remove_dupes_or_overlapping_ents(ents, debug=True):
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
          print("\ncur_start/cur_end",cur_start,cur_end,"and prior_start/prior_end", prior_start, prior_end)
          print("ent_ranges",ent_ranges)
          print("Prior", prior,"i", i,"len(ents)",len(ents),ents)
          print("Delete Entity ", ents[i], "which occurs in span of entity",ents[prior])
        #del ents[i]
        if i not in to_delete:
          to_delete.append(i)
          break
      elif cur_start == prior_start and cur_end > prior_end:   #add this to main code ?
        if debug:
          print("\ncur_start/cur_end",cur_start,cur_end,"and prior_start/prior_end", prior_start, prior_end)
          print("ent_ranges",ent_ranges)
          print("Prior", prior,"i", i,"len(ents)",len(ents),ents)
          print("Delete Prior Entity ", ents[i-1], "which occurs in span of entity",ents[i])
        
        if i not in to_delete:
          to_delete.append(i-1)
          break


  if to_delete != []:
    if debug:
      print("\nTo delete",[v for v in to_delete])
    to_delete.reverse()
    for v in to_delete:
      del ents[v]
 
  return ents

def inter_search(ind,e,curwikicap,cur_ents,pre="",debug=True):
  if e[0] in cur_ents[ind]:
    if debug:
      print(pre+"replace ",e[0], "with", cur_ents[ind]) 
    rep_with = cur_ents[ind]
    found = 1
  else:
    found = 0
    for cind, ce in enumerate(cur_ents):
      if e[0] in ce:
        if debug:
          print(pre+"creplace ",e[0], "with", cur_ents[cind])
        rep_with = cur_ents[cind]
        found = 1
    if not found:
      print(pre+"not replace,",e[1],e[2],curwikicap[e[1]:e[2]],ind,cur_ents[ind]) 
      rep_with = ""
  return rep_with, found

#go through new data and substitute text spans in wikiCap_new with true entity if a partial hit is found (and update ent pos) and save out as new 0501 file
total = 0
total_wrong_num = 0
total_changed = 0
nopes = 0
debug = False
for i, img_id in enumerate(kvqa_qcap_data):
  cap_ent2true_ent = {}
  curcap = kvqa_qcap_data[img_id]
  cur_qids, cur_ents = img2qid[img_id]

  newcap = curcap
  priorlen = len(curcap['ents_found'])
  priorcap_ents = sorted(curcap['ents_found'], key=itemgetter(1))
  curcap_ents = remove_dupes_or_overlapping_ents(priorcap_ents, debug=False)
  
  cur_not_eq = False
  if len(curcap_ents) != priorlen:
    cur_not_eq = True
    total += 1
    """
    print("\n",i,curcap)
    print(cur_qids, cur_ents)
    print("CurCapEnts",curcap_ents)
    """ 
    #call brute force ... didn't work well for these instances
    #cur_ents_stend = [[c,0,0] for c in cur_ents]
    #curcap_ents = brute_search(cur_ents_stend, curcap['wikiCap_new'])
    #curcap_ents = brute_search(priorcap_ents, curcap['wikiCap_new'])
    #print("After Brute Search: CurCapEnts now", curcap_ents)
   
 
  if len(curcap_ents) != len(cur_ents):
    total_wrong_num += 1

  changes = 0
  for ind, e in enumerate(curcap_ents):  #this assumes curcap_ents are in order by position in string
    if e[0] not in cur_ents: 
      curwikicap = curcap['wikiCap_new']
      if debug:
        print("\n",i, ind,e,"not exactly found in", cur_ents, "Cap ents:", curcap_ents) 
        print("\t",curwikicap)
        
      if curwikicap[e[1]:e[2]] == e[0]:
        rep_with, found = inter_search(ind,e,curwikicap,cur_ents,pre="",debug=debug)
      else:
        ratio = fuzz.ratio(curwikicap[e[1]:e[2]],e[0])
        #print("Fuzzy Ratio", e[1],e[2], curwikicap[e[1]:e[2]], e[0], ratio)
        if ratio > 75:
          rep_with, found = inter_search(ind,e,curwikicap,cur_ents,pre="f",debug=debug)
        else:
          print(i, img_id, "NOPE,","start/end",e[1],e[2],"wiki[start:end]",curwikicap[e[1]:e[2]],"ind,cur_ents[ind]",ind,cur_ents[ind],"CAP:",curwikicap, curcap_ents, cur_ents) 
          nopes += 1
          rep_with, found = "", 0

      if found:
         if changes != 1:
            changes = 1
            total_changed += 1

         prior_e0, prior_e1, prior_e2, = e[0],e[1],e[2]
         curcap_ents[ind][0] = rep_with

         #change start_nd, end pos for curcap_ents[e] depending and those pos for those that follow it.  changewikicap as well
         #HANDLE ones where you need to change Start
         prior_end = curcap_ents[ind][2]
         curcap_ents[ind][2] = curcap_ents[ind][1] + len(rep_with)
         cur_end = curcap_ents[ind][2]
         diff = cur_end - prior_end
         for rind, re in enumerate(curcap_ents):         
           #print("Updating start/end for other entities", cur_end, re)
           if re[0] != "" and re[1] > prior_end:
             curcap_ents[rind][1] += diff
             curcap_ents[rind][2] += diff
  
         #print(rep_with, prior_e0, curwikicap[e[1]:e[2]])
         if prior_e0 in curcap['wikiCap_new']:
           curcap['wikiCap_new'] = curcap['wikiCap_new'].replace(prior_e0,rep_with)   
         else:
           priorwiki_ent = curwikicap[prior_e1:prior_e2]
           if priorwiki_ent in curcap['wikiCap_new']:
             curcap['wikiCap_new'] = curcap['wikiCap_new'].replace(priorwiki_ent,rep_with)   #for fuzzy things
           else:
             print("ERROR NO REPLACEMENT FOUND")
             import sys
             sys.exit()

         #if (ind + 1) == len(curcap_ents):
         #only update order after you've processed all ents
         #curcap['ents_found'] = sorted(curcap['ents_found'], key=itemgetter(1)) 
         if debug:
           print("\tAfter changes: curcap_ents", curcap_ents, curcap)

  if cur_not_eq or changes == 1:
    #make sure ents are in order
    curcap['ents_found'] = sorted(curcap['ents_found'], key=itemgetter(1)) 
    print(i, img_id, "After changes:", curcap)
    kvqa_qcap_data[img_id] = curcap
    #print(i, "After changes: curcap_ents", curcap_ents, curcap)
  
  if i < -1:
    import sys
    sys.exit()

print(total)             #78 had ents that needed to be deleted
print(total_wrong_num)   #also 78 weird
print(total_changed)     #12531  
print(nopes)             #89

#ok so fix examples where cap_ents != cur_ents  ... but before then go through and see if all wiki_ents are accounted for with exact match and if not fix/add them ( this should handle 68 too!)
# whats up with these 78?
# save new file and try model with it instead of 0423 version

#NOPES SHOULD BE LARGELY FIXED BY CURRENT CODEa.. not sure about 78.. just try it

with open('../data/kvqa/new_kvqa_q_caps_ents0501.json', 'w') as fp:
  json.dump(kvqa_qcap_data, fp)

print("WROTE OUT NEW KVQA Q CAPS DATA.. still a little noisy, but should be usable")

#NEXT run numbers for ebert with newdata 0501
#save all to github tonight at latests
