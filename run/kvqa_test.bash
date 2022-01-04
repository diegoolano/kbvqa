# The name of this experiment ( assuming we have split number in name!  )
name=$2_$3

# Save logs and models under snap/vqa; make backup.
output=snap/kvqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# $1  -  GPU to use
# $2  -  name of experiment
# $3  -  split number ( 0 - 4 ) to train / validate on  
# $4  -  rest of params to pass to kvqa.py

# SAMPLE CALL on GPU 3 for split 0
# bash run/kvqa_test.bash 3 kvqa_lxr955_results 0 --test test --load snap/kvqa/kvqa_lxr955/BEST   (add --incl_caption --incl_para if needed )
   

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/kvqa.py \
    --tiny --train train_kvqa --valid ""  \
    --split_num $3 \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 4 \
    --tqdm --output $output ${@:4}
    

#--tiny --train train --valid ""  \
