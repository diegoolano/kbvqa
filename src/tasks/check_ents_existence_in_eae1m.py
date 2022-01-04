import json
import time

kvqa_dataset_f = "../../data/kvqa/dataset.json"
eae1m_vocab_f = "/data/diego/adv_comp_viz21/en_wiki_20190301_json_chunks_add_mentions_filtered_1m/entity_vocab_1m_titles.txt"  # or entity_vocab_1m.txt is qids
eae1m_vocab_ids_f = "/data/diego/adv_comp_viz21/en_wiki_20190301_json_chunks_add_mentions_filtered_1m/entity_vocab_1m.txt"  
wiki_id2_qid_f = "/data/diego/adv_comp_viz21/en_wiki_20190301_json_chunks_add_mentions_filtered_1m/category_cache_5m.json"
kvqa_in_eae1_f = "/data/diego/adv_comp_viz21/en_wiki_20190301_json_chunks_add_mentions_filtered_1m/kvqa_ents_eae_using_qidcache_split0_trainval_list.json"


#load data/kvqa/datasets.json
with open(kvqa_dataset_f, 'r') as fp:
  kvqa_data = json.load(fp)

#get list of unique entities ( QID and name and image ids they appear in )
print("Get Unique KVQA ents")
kvqa_ents = {}
img2qid = {}
for img_id in kvqa_data:
  cur = kvqa_data[img_id]
  cur_split = cur['split']
  if cur_split[0] in [1,2]:
    #only get entities that occur in training or val set for Split 0
    for i, ent in enumerate(cur['NamedEntities']): 
      qid = cur['Qids'][i]
      if qid not in kvqa_ents:
        kvqa_ents[qid] = {'name':ent, 'images':[]}
      if img_id not in kvqa_ents[qid]['images']:
        kvqa_ents[qid]['images'].append(img_id)    
    img2qid[img_id] = [cur['Qids'], cur['NamedEntities']]

print("Found ", len(kvqa_ents),"ents")

ent_qids = list(kvqa_ents.keys())
for i in range(10):
  print(i, ent_qids[i], kvqa_ents[ent_qids[i]])

st = time.time()
print("Read 1M vocab list")
#go through Yasu list of 1 million wiki ents and see % of KVQA that are there or not ( save things to github )
with open(eae1m_vocab_f, "r") as fp:
  eae1vocab_titles = {s.strip():i for i,s in enumerate(fp.read().splitlines())}   #title: index
eae_ind2title = {eae1vocab_titles[v]:v for v in eae1vocab_titles}                 #index: title

with open(eae1m_vocab_ids_f, "r") as fp:
  eae1vocab_ids = {s.strip():i for i,s in enumerate(fp.read().splitlines())}
eae_ind2id = {eae1vocab_ids[v]:v for v in eae1vocab_ids}

wiki2qids = []
with open(wiki_id2_qid_f, "r") as fp:
  for line in fp.read().splitlines():
    wiki2qids.append(json.loads(line))

#{"qid": "Q1", "y_url": "https://en.wikipedia.org/wiki/Universe", "y_title": "Universe", "y_page_id": "31880", "y_category": ["astronomical dynamical systems", "environments", "main topic articles", "physical cosmology", "universe"]}
for i in range(10):
  print(i,wiki2qids[i])

pageid2qid = {}
qid2pageid = {}
for i in range(len(wiki2qids)):
  pageid2qid[wiki2qids[i]['y_page_id']] = wiki2qids[i]['qid']
  qid2pageid[wiki2qids[i]['qid']] = wiki2qids[i]['y_page_id']
  
print("Elapsed", time.time() - st)

titles = list(eae1vocab_titles.keys())
indtitles = list(eae_ind2title.keys())
ids = list(eae1vocab_ids.keys())
indids = list(eae_ind2id.keys())
for i in range(10):
  print(i,titles[i],indtitles[i], ids[i], indids[i])


# now see how many kvqa ents appear in 1m vocab list  THESE ARE NOT QIDs ( ie Germany's Wikidata ID is Q183 
st = time.time()
debug = False
found = 0
all_out = []
for qid in kvqa_ents:
  cur = kvqa_ents[qid]
  name = cur['name']
  if qid in qid2pageid:
    pageid = qid2pageid[qid]
  else:
    pageid = "-1"

  all_out.append([ qid, name, pageid in eae1vocab_ids, pageid])
  """
  if pageid in eae1vocab_ids:
    loc = eae1vocab_ids[pageid]
    print("1 qid",qid,"name",name,"pageid",pageid,"loc",loc,"title",eae_ind2id[loc]) #, eae1vocab_titles[loc])
    if loc in eae_ind2title:
      title = eae_ind2title[loc]
    elif int(loc) in eae_ind2title:
      title = eae_ind2title[loc]
    else:
      print("For ",cur, "with qid:",qid, "name", name, "pageid",pageid,"  pageid in eae1vocabs is true and returns ",loc)
      import sys 
      sys.exit()

    if debug:
      print("For qid",qid,"found ", title)
    kvqa_ents[qid]['1M'] = [1,qid,title,pageid]
    found +=1

  elif name in eae1vocab_titles:
    loc = eae1vocab_titles[name]
    print("2 qid",qid,"name",name,"list index",loc, "new wiki_id",eae_ind2id[loc])
    new_wiki_id = eae1vocab_ids[eae_ind2id[loc]]
    if debug:
      print("For qid",qid,"didn't find match via id but did via name",name,"at wiki_id:",new_wiki_id)
    kvqa_ents[qid]['1M'] = [1, qid,name, loc]
    found +=1
  else:
    if debug:
      print("For wiki_id",wiki_id,"didn't find match via id or name ",name)
    kvqa_ents[qid]['1M'] = [0, qid,name, -1]
   """

#total = len(kvqa_ents)
#print("Found ",found,"kvqa ents in 1M top wiki ents list out of total",total," (",round(found/total,4),"%)")
#print("Elapsed", time.time() - st)
# Found  12446 kvqa ents in 1M top wiki ents list out of total 18880  ( 0.6592 %)  ( JUST USING STRING MATCH )

# Found  12473 kvqa ents in 1M top wiki ents list out of total 18880  ( 0.6606 %)

with open(kvqa_in_eae1_f, 'w') as fp:
  #json.dump(kvqa_ents, fp)
  json.dump(all_out, fp)
