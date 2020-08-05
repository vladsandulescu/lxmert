# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv

from sklearn.metrics import roc_auc_score

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.
HM_DATA_ROOT = '../../data/'
HM_IMGFEAT_ROOT = '../../data/imgfeat/'
SPLIT2NAME = {
    'train': 'train',
    'dev': 'dev',
    'test': 'test',
}


def parse_img_id(x):
    # this will break when other fields are also len 4 but you do want an int
    if len(x) < 4:
        return int(x)
    elif len(x) == 4:
        return "0" + str(x)
    else:
        return str(x)


class HMDataset:
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend([json.loads(line, parse_int=parse_img_id)
                              for line in open("%s%s.jsonl" % (HM_DATA_ROOT, split)).
                             read().splitlines()])
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['id']: datum
            for datum in self.data
        }

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class HMTorchDataset(Dataset):
    def __init__(self, dataset: HMDataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        # Loading detection features to img_data
        img_data = []
        for split in dataset.splits:
            load_topk = 5000 if (split == 'minival' and topk is None) else topk
            img_data.extend(load_obj_tsv(
                os.path.join(HM_IMGFEAT_ROOT, '%s_d2_36-36_batch.tsv' % (SPLIT2NAME[split])),
                topk=load_topk))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['id']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # Text
        text = datum['text']

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            return img_id, feats, boxes, text, label
        else:
            return img_id, feats, boxes, text


class HMEvaluator:
    def __init__(self, dataset: HMDataset):
        self.dataset = dataset

    def evaluate(self, imgid2label: dict):
        acc_score = 0.
        targets = []
        pred_probs = []
        for img_id, (pred_proba, pred_label) in imgid2label.items():
            datum = self.dataset.id2datum[img_id]
            target = datum['label']
            targets.append(target)
            pred_probs.append(pred_proba)
            if pred_label == target:
                acc_score += 1
        acc = acc_score / len(imgid2label)
        auroc = roc_auc_score(np.array(targets), np.array(pred_probs))
        return acc, auroc

    def dump_result(self, imgid2label: dict, path):
        """
        Dump results to a json file, which could be submitted to the HM online evaluation.
        HM json file submission requirement:
            results = [result]
            result = {
                "img_id": str,
                "proba": float
                "label": int
            }

        :param imgid2label: dict of img_id --> (proba, label)
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            f.write("id,proba,label\n")
            for img_id, (proba, label) in imgid2label.items():
                f.write(f'{img_id},{str(np.round(proba, 4))},{label}\n')
