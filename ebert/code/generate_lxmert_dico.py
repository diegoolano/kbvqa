from param import args
#print(args)
import pickle
pickle.dump( args, open( "lxmert_args.pkl", "wb" ) )
import sys
sys.exit()

from lxrt.entry import LXRTEncoder     #has dependencies on lxrt.tokenization, lxrt.modeling   ( so copy over params.py lxrt/entry , lxrt/modeling, lxrt/tokenization to ebert/code )

print("Load Encoder Code")
lxrt_encoder = LXRTEncoder( args, max_seq_length=50)
print("Load Encoder Pre Trained Weights")
lxrt_encoder.load("model")    ## copied lxmert/snap/pretrained/model_LXRT.pth  here
model, tokenizer =  lxrt_encoder.model.bert, lxrt_encoder.tokenizer

print(model)
print(tokenizer)

print("GENERATE LXMERT RESOURCES FILE! ../resources/lxmert.all.dico.txt")
with open("../resources/lxmert.all.dico.txt","w") as f:    
  for i, v in enumerate(tokenizer.vocab):
    f.write(v+"\n")

print("Done")
