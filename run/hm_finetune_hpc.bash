# The name of this experiment.
name=$2

# Save logs and models under snap/hm; make backup.
output=/playground/hmm/lib/lxmert/snap/hm/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/hm.py \
    --train train,dev --valid dev  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERT /playground/hmm/lib/lxmert/snap/pretrained/model \
    --batchSize 64 --optim bert --lr 2e-5 --epochs 5 \
    --data_root /playground/hmm/data/ \
    --imgfeat_root /playground/hmm/data/imgfeat/ \
    --tqdm --output $output ${@:3}
