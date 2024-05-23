#!/usr/bin/env python3
# ModelNet 3D recognition model
# Authors: Matúš Konig, nexuso1

import argparse
import datetime
import os
import re

import torch.utils
import torch.utils.data

os.environ.setdefault(
    "KERAS_BACKEND", "torch"
)  # Use PyTorch backend unless specified otherwise

import keras
import numpy as np
import torch
from utils import up_3D, down_3D

from modelnet import ModelNet

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=40, type=int, help="Number of epochs.")
parser.add_argument("--modelnet", default=32, type=int, help="ModelNet dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument(
    "--threads", default=0, type=int, help="Maximum number of threads to use."
)

parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--width', default=3, type=int, help='Model width')
parser.add_argument('--depth', default=26, type=int, help='Model depth')


def create_unet(args, depth, width, input_shape, num_classes):
    n = (depth - 4) // 6
    inputs = keras.Input(shape=input_shape)
    x = inputs
    configs = [(16 * width, 3, 3), (32 * width, 3, 3), (64 * width, 3, 3)]
    down_convs = []
    for filters, kernel_size, n in configs:
        x, last_conv = down_3D(x, filters, n, kernel_size=kernel_size)
        down_convs.append(last_conv)

    for i in range(len(configs) -1, -1, -1):
        filters, kernel_size, n = configs[i]
        connected_conv = down_convs[i]
        prev_filters = configs[max(0, i-1)][0]
        x = up_3D(x, connected_conv, filters, prev_filters, n)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    pool = keras.layers.GlobalAveragePooling3D()(x)
    classifier = keras.layers.Dense(num_classes, activation='softmax')(pool)
    model = keras.models.Model(inputs=inputs, outputs=classifier)
    model.summary()
    return model

def create_residual_layer(
    kernel: int,
    filters: int,
    inputs,
    downscale=False,
):
    if downscale:
        residual = keras.layers.Conv3D(
            filters,
            kernel_size=1,
            strides=2,
            kernel_regularizer=keras.regularizers.L2(1e-4),
        )(inputs)
        residual = keras.layers.BatchNormalization()(residual)

    else:
        residual = inputs

    inputs = keras.layers.BatchNormalization()(inputs)
    inputs = keras.layers.ReLU()(inputs)
    inputs = keras.layers.Conv3D(
        filters,
        kernel,
        padding="same",
        strides=1 if not downscale else 2,
        kernel_regularizer=keras.regularizers.L2(1e-4),
    )(inputs)

    inputs = keras.layers.BatchNormalization()(inputs)
    inputs = keras.layers.ReLU()(inputs)
    inputs = keras.layers.Conv3D(
        filters,
        kernel,
        padding="same",
        strides=1,
        kernel_regularizer=keras.regularizers.L2(1e-4),
    )(inputs)

    return inputs + residual


def create_model(modelnet: ModelNet):
    input_shape = [modelnet.D, modelnet.H, modelnet.W, modelnet.C]

    input = hidden = keras.layers.Input(input_shape)
    hidden = create_residual_layer(3, 16, hidden)
    hidden = create_residual_layer(3, 16, hidden)

    hidden = create_residual_layer(3, 32, hidden, downscale=True)
    hidden = create_residual_layer(3, 32, hidden)

    hidden = create_residual_layer(3, 64, hidden, downscale=True)
    hidden = create_residual_layer(3, 64, hidden)

    hidden = keras.layers.GlobalAveragePooling3D()(hidden)
    hidden = keras.layers.Flatten()(hidden)
    hidden = keras.layers.Dropout(0.2)(hidden)

    hidden = keras.layers.Dense(1024, activation="relu")(hidden)
    hidden = keras.layers.Dropout(0.2)(hidden)

    output = keras.layers.Dense(len(modelnet.LABELS), activation="softmax")(hidden)

    model = keras.Model(inputs=input, outputs=output)
    return model


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join(
        "logs",
        "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(
                (
                    "{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v)
                    for k, v in sorted(vars(args).items())
                )
            ),
        ),
    )

    # Load the data
    modelnet = ModelNet(args.modelnet)
    num_classes = len(modelnet.LABELS)
    class_weights = np.zeros(num_classes)
    for i in range(len(modelnet.dev)):
         class_weights[modelnet.dev[i]['label']] += 1

    class_weights = torch.as_tensor(1 - (class_weights / len(modelnet.dev))) # inverse class weighting
    #class_weights = { i : w for i, w in enumerate(class_weights)}

    train = torch.utils.data.DataLoader(
        modelnet.train.transform(lambda data: (data["grid"], keras.ops.one_hot(data["label"], num_classes))),
        shuffle=True,
        batch_size=args.batch_size,
    )
    dev = torch.utils.data.DataLoader(
        modelnet.dev.transform(lambda data: (data["grid"], keras.ops.one_hot(data["label"], num_classes))),
        batch_size=args.batch_size,
    )
    test = torch.utils.data.DataLoader(
        modelnet.test.transform(lambda data: (data["grid"])),
        batch_size=args.batch_size,
    )

    model = create_unet(args, depth=args.depth, width=args.width,
                        input_shape=[modelnet.D, modelnet.H, modelnet.W,modelnet.C],
                        num_classes=num_classes)

    batch_count = len(modelnet.train) / args.batch_size
    decay_steps = args.epochs * batch_count
    optimizer = keras.optimizers.AdamW(
        learning_rate=keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.lr, decay_steps=decay_steps
        )
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalFocalCrossentropy(alpha=class_weights, label_smoothing=0.1),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy"), 
                 keras.metrics.F1Score(average='macro')],
    )

    model.summary()
    chkpt_path = os.path.join(args.logdir, '3d_model.keras')

    # Will automatically save the best model during training w.r.t. a given metric
    callbacks = [keras.callbacks.ModelCheckpoint(chkpt_path, monitor='val_f1_score', save_best_only=True, verbose=1, mode='max')]
    
    model.fit(
        train, batch_size=args.batch_size, epochs=args.epochs, validation_data=dev,
        callbacks=callbacks
    )

    # Load the best model
    model = keras.saving.load_model(chkpt_path, compile=False)

    with open(os.path.join(args.logdir, "dev_preds.npy"), 'wb') as f:
        preds = model.predict(dev, batch_size=args.batch_size)
        np.save(f, np.array(preds))

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(
        os.path.join(args.logdir, "3d_recognition.txt"), "w", encoding="utf-8"
    ) as predictions_file:
        # TODO: Predict the probabilities on the test set
        test_probabilities = model.predict(test)

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
