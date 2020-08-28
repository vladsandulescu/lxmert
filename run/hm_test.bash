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
    --tiny --train train --valid ""  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 4 \
    --data_root ./../../data/ \
    --imgfeat_root ./../../data/imgfeat/ \
    --tqdm --output $output ${@:3}
