# The name of this experiment ( assuming we have split number in name!  )
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/okvqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# $1  -  GPU to use
# $2  -  name of sepxeriment
# $3  -  rest of params to pass to kvqa.py

# SAMPLE CALL on GPU 3 using model checkpoint finetuned
# bash run/okvqa_test.bash 3 okvqa_lxr955_results --test test --load snap/kvqa/okvqa_lxr955/BEST   

# don't use -load to just test out of the box?
   

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/okvqa.py \
    --tiny --train train_tv --valid ""  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 4 \
    --tqdm --output $output ${@:3}
