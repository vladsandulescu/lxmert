# The name of this experiment.
name=$2

# Save logs and models under snap/hm; make backup.
output=snap/hm/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/hm.py \
    --train train --valid dev  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERT snap/pretrained/model \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 20 \
    --tqdm --output $output ${@:3}