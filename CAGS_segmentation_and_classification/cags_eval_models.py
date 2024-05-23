#!/usr/bin/env python3
# Simple model evaluation / ensembling script
# Author: nexuso1

import argparse
import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import glob
import numpy as np
from utils import up, compile_model
import torch
import torchvision.transforms.v2 as v2

from torchvision.transforms.v2 import RandAugment
from cags_dataset import CAGS
from cags_segmentation import prep_fn, prep_fn_test
from functools import partial

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()

parser.add_argument('--i', help='Source folder', default='best', type=str)
parser.add_argument('--n', help='Destination filename prefix', default='cifar', type=str)
parser.add_argument('--suffix', help='Destination filename suffix', default='npy')
parser.add_argument('--batch_size', help='Eval batch size', default=128, type=int)
parser.add_argument('--o', help='Output filename', default='ensemble_probs.txt')

def main(args):
    paths = glob.glob(f'{args.i}/**/{args.n}*.{args.suffix}', recursive=True)
    print(paths)
    total_dev = 0
    total_test = 0
    cags = CAGS()
    dev_ds = cags.dev.transform(prep_fn)
    dev = torch.utils.data.DataLoader(dev_ds, batch_size=args.batch_size)
    test = torch.utils.data.DataLoader(cags.test.transform(prep_fn_test), batch_size=args.batch_size)
    
    for path in paths:
        args.logdir = os.path.dirname(path)
        model = keras.saving.load_model(path, custom_objects={'CAGS.MaskIoUMetric' : CAGS.MaskIoUMetric}, compile=False)
        dev_preds = model.predict(dev)
        iou = CAGS.evaluate_segmentation(cags.dev, dev_preds)
        print(f'Model performance: iou = {iou}')
        with open(os.path.join(args.logdir, "metrics.txt"), 'w') as f:
            f.write(f'IoU : {iou}')

        with open(os.path.join(args.logdir, "dev_preds.npy"), 'wb') as f:
            np.save(f, np.array(dev_preds))
        
        test_preds = model.predict(test, batch_size=args.batch_size)

        total_dev += dev_preds
        total_test += test_preds

    total_dev = total_dev / len(paths)
    iou = CAGS.evaluate_segmentation(cags.dev, total_dev)
    print(f'Ensemble Dev IOU: {iou}')

    total_test = total_test / len(paths)
    with open(os.path.join(args.i, "cags_segmentation_ensemble.txt"), "w", encoding="utf-8") as predictions_file:
        # Predict the masks on the test set
        for mask in total_test:
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file) 

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)