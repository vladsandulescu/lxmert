# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from tasks.hm_model import HMModel
from tasks.hm_data import HMDataset, HMTorchDataset, HMEvaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_data_tuple(data_root: str, imgfeat_root: str, splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = HMDataset(data_root, imgfeat_root, splits)
    tset = HMTorchDataset(dset)
    evaluator = HMEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class HM:
    def __init__(self):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.data_root, args.imgfeat_root,
            args.train, bs=args.batch_size, shuffle=True, drop_last=False
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.data_root, args.imgfeat_root,
                args.valid, bs=1024,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        
        # Model
        self.model = HMModel()

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.mce_loss = nn.CrossEntropyLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            imgid2label = {}
            for i, (img_id, feats, boxes, text, target) in iter_wrapper(enumerate(loader)):
                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, text)
                loss = self.mce_loss(logit, target) * logit.size(0)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                probs = F.softmax(logit, dim=1)
                proba = probs[:, 1]
                label = probs.argmax(dim=1)
                for id, p, l in zip(img_id,
                                    proba.detach().cpu().numpy(),
                                    label.detach().cpu().numpy()):
                    imgid2label[id] = (p, l)

            train_acc, _ = evaluator.evaluate(imgid2label)
            log_str = "\nEpoch %d: Train Acc - %0.4f\n" % (epoch, train_acc)

            if self.valid_tuple is not None:  # Do Validation
                valid_acc_score, valid_auroc_score = self.evaluate(eval_tuple)
                if valid_auroc_score > best_valid:
                    best_valid = valid_auroc_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid Acc - %0.4f, Valid AUROC - %0.4f, Best AUROC - %0.4f\n" % \
                           (epoch, valid_acc_score, valid_auroc_score, best_valid)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        imgid2label = {}
        for i, datum_tuple in enumerate(loader):
            img_id, feats, boxes, text = datum_tuple[:4]   # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, text)
                probs = F.softmax(logit, dim=1)
                proba = probs[:, 1]
                label = probs.argmax(dim=1)
                for id, p, l in zip(img_id, proba.cpu().numpy(), label.cpu().numpy()):
                    imgid2label[id] = (p, l)
        if dump is not None:
            evaluator.dump_result(imgid2label, dump)
        return imgid2label

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        imgid2label = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(imgid2label)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        imgid2label = {}
        for i, (img_id, feats, boxes, text, target) in enumerate(loader):
            label = target
            for id, l in zip(img_id, label.cpu().numpy()):
                imgid2label[id] = (1, l)
        return evaluator.evaluate(imgid2label)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    # Build Class
    hm = HM()

    # Load HM model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        hm.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            hm.predict(
                get_data_tuple(
                    args.data_root, args.imgfeat_root,
                    args.test, bs=1000, shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.csv')
            )
        elif 'val' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = hm.evaluate(
                get_data_tuple(
                    args.data_root, args.imgfeat_root,
                    'minival', bs=950, shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'minival_predict.json')
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', hm.train_tuple.dataset.splits)
        if hm.valid_tuple is not None:
            print('Splits in Valid data:', hm.valid_tuple.dataset.splits)
            val_acc, val_auroc = hm.oracle_score(hm.valid_tuple)
            print("Valid Acc Oracle - %0.4f, Valid AUROC Oracle - %0.4f" % (val_acc, val_auroc))
        else:
            print("DO NOT USE VALIDATION")
        hm.train(hm.train_tuple, hm.valid_tuple)


