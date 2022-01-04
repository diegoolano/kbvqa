# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/okvqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
# D:  Do we need to change our generate process to make validation smaller?  
#     here they train on both train and nominival ( which is the validation set, minus a chunk "minival" that they actually use for validatoin )

# --train train,nominival --valid minival  \

# $1  -  GPU to use
# $2  -  name of sepxeriment
# $3  -  rest of params to pass to okvqa.py

# SAMPLE CALL to do quick development ( not using captions, not using paraphrase )
# bash run/okvqa_finetune.bash 3 okvqa_lxr955_tiny --tiny

# SAMPLE CALL to run full fine tuning on GPU 3 
# bash run/okvqa_finetune.bash 3 okvqa_lxr955_split0

# WHAT TO DO ABOUT VALIDATION

#--train train_kvqa --valid valid_kvqa  \

#epochs was 8, now do 20, was 11, now 15

CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/okvqa.py \
    --train train --valid ""\
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERTQA /home/diego/adv_comp_viz21/lxmert/orig_code/lxmert/snap/pretrained/model \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 15 \
    --tqdm --output $output ${@:3}
