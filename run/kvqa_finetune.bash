# The name of this experiment.
name=$2_$3

# Save logs and models under snap/vqa; make backup.
output=snap/kvqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
# D:  Do we need to change our generate process to make validation smaller?  
#     here they train on both train and nominival ( which is the validation set, minus a chunk "minival" that they actually use for validatoin )

# --train train,nominival --valid minival  \

# $1  -  GPU to use
# $2  -  name of sepxeriment
# $3  -  split number ( 0 - 4 ) to train / validate on  
# $4  -  rest of params to pass to kvqa.py

# SAMPLE CALL to do quick development ( not using captions, not using paraphrase )
# bash run/kvqa_finetune.bash 3 kvqa_lxr955_tiny --tiny

# SAMPLE CALL to run full fine tuning on GPU 3 for split 0  using captions and paraphrase  ( see src/params.py to see params definition )
# bash run/kvqa_finetune.bash 3 kvqa_lxr955_split0_captions_paraphase 0 --incl_caption --incl_para

# bash run/kvqa_finetune.bash 3 kvqa_lxr955_split0_captions_paraphase 0 --incl_caption --incl_para --use_lm ebert --max_len 100    #to change max len sequence size

# by default runs 4 epochs ( using orig ent spans )

CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/kvqa.py \
    --train train_kvqa --valid valid_kvqa  \
    --split_num $3 \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERTQA /home/diego/adv_comp_viz21/lxmert/orig_code/lxmert/snap/pretrained/model \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 6 \
    --tqdm --output $output ${@:4}
