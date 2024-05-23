# CAGS classification model
# Author: nexuso1

#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import ast
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import numpy as np
import torch

from torchvision.transforms import RandAugment
from cags_dataset import CAGS
from utils import compile_model
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout p')
parser.add_argument('--layers', type=str, default='[512, 512, 512]', help='Classifier layer structure')
parser.add_argument('--lr', help='Classifier LR', type=float, default=0.001)
parser.add_argument('--magnitude', help='RandAugment magnitude', type=int, default=9)
parser.add_argument('--fine_tune', action='store_true', help='Do finetuning', default=False)

def dense_norm_activ(in_block, units):
    x = keras.layers.Dense(units)(in_block)
    x = keras.layers.BatchNormalization(-1)(x)
    x = keras.layers.ReLU()(x)

    return x

def create_model(args, input_shape, n_outputs, backbone) -> keras.models.Model:
    inputs = keras.layers.Input(input_shape)
    x = backbone(inputs)
    layer_units = ast.literal_eval(args.layers)
    layers = []
    for i, units in enumerate(layer_units):
        x = dense_norm_activ(x, units)
        if args.dropout > 0 and i < len(layer_units) - 1:
            x = keras.layers.Dropout(args.dropout)(x)

        layers.append(x)

    outputs = keras.layers.Dense(n_outputs, activation='softmax')(x)
    model = keras.models.Model(inputs=inputs, outputs=outputs)

    model.summary()
    return model

def prep_fn(x, aug=None):
    img = x['image']
    img = torch.as_tensor(img, dtype=torch.uint8)
    img = torch.moveaxis(img, -1, 0)
    if aug is not None:
        img = aug(img)
    return torch.moveaxis(img, 0, -1), keras.ops.one_hot(x['label'], len(CAGS.LABELS))

def prep_fn_test(x):
    return torch.as_tensor(x['image'], dtype=torch.uint8)

def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[224, 224, 3]` tensor of `torch.uint8` values in [0-255] range,
    # - "mask", a `[224, 224, 1]` tensor of `torch.float32` values in [0-1] range,
    # - "label", a scalar of the correct class in `range(len(CAGS.LABELS))`.
    cags = CAGS()
    aug = RandAugment(num_ops=3, magnitude=args.magnitude)
    train_ds = cags.train.transform(partial(prep_fn, aug=aug))
    dev_ds = cags.dev.transform(prep_fn)

    train = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(dev_ds, batch_size=args.batch_size)

    # Load the EfficientNetV2-B0 model. It assumes the input images are
    # represented in the [0-255] range. 
    backbone = keras.applications.EfficientNetV2B2(include_top=False, pooling="avg")
    if not args.fine_tune:
        backbone.trainable = False

    chkpt_path = os.path.join(args.logdir, 'classification_model.keras')
    callbacks = [keras.callbacks.ModelCheckpoint(chkpt_path, monitor='val_f1_score', save_best_only=True, verbose=1, mode='max')]

    model = create_model(args, [CAGS.H, CAGS.W, CAGS.C], len(CAGS.LABELS), backbone)
    compile_model(args, model, len(train_ds) // args.batch_size * args.epochs)

    model.fit(train, validation_data=dev, batch_size=args.batch_size, epochs=args.epochs, callbacks=callbacks)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    test = torch.utils.data.DataLoader(cags.test.transform(prep_fn_test), batch_size=args.batch_size)

    # Load the best model
    model = keras.saving.load_model(chkpt_path, compile=True)

    # Best model eval
    loss, acc, f1 = model.evaluate(dev, batch_size = args.batch_size)
    print(f'Best model performance: acc = {acc}, f1 = {f1}')
    with open(os.path.join(args.logdir, "metrics.txt"), 'w') as f:
        f.write(f'Acc: {acc}, F1: {f1}')

    with open(os.path.join(args.logdir, "dev_preds.npy"), 'wb') as f:
        preds = model.predict(dev, batch_size=args.batch_size)
        np.save(f, np.array(preds))

    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        test_probabilities = model.predict(test)

        with open(os.path.join(args.logdir, "cags_classification_test.npy"), "wb") as probs_file:
            np.save(probs_file, np.array(test_probabilities))

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)