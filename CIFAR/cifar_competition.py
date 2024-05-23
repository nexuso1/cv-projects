#!/usr/bin/env python3
# CIFAR classification model
# Author: Samuel Fanči (@nexuso1)

import argparse
import datetime
import os
import re
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import numpy as np
import torch

from torchvision.transforms import v2
from torchvision.transforms.functional import rgb_to_grayscale
from cifar10 import CIFAR10

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
parser.add_argument("--scale", default=1, type=int, help='Layer channels multiplier')
parser.add_argument("--aug_prob", default=0.5, type=float, help='Augmentation probability')
parser.add_argument("--mode", type=str, help='CNN architecture to use', default=None)
parser.add_argument("--greyscale", help='Use grayscale images', action='store_true', default=False)
parser.add_argument("--o", help='Output base folder', default='.', type=str)


def se_block(in_block, filters, activation='silu', ratio=4):
    """
    Squeeze-Excitation
    """
    x = keras.layers.GlobalAveragePooling2D('channels_last')(in_block)
    x = keras.layers.Dense(filters//ratio, activation=activation)(x)
    x = keras.layers.Dense(filters, activation='sigmoid')(x)
    return keras.layers.multiply([in_block, x])

def conv_norm_activ(in_block, filters, activation='silu', **kwargs):
    x = keras.layers.Conv2D(filters, activation=None, padding='same',**kwargs)(in_block)
    x = keras.layers.BatchNormalization()(x)
    if activation is not None:
        return keras.layers.Activation(activation)(x)
    else:
        return x

def depthwise_norm_activ(in_block, activation='silu', **kwargs):
    x = keras.layers.DepthwiseConv2D(padding='same', **kwargs)(in_block)
    x = keras.layers.BatchNormalization()(x)
    if activation is not None:
        return keras.layers.Activation(activation)(x)
    else:
        return x
    
def mbconv(in_block, in_filters, out_filters, expand_ratio, **kwargs):
    expanded_filters = in_filters*expand_ratio
    x = conv_norm_activ(in_block, filters=expanded_filters, kernel_size=1)
    x = depthwise_norm_activ(x, **kwargs)
    x = se_block(x, expanded_filters)
    x = conv_norm_activ(x, out_filters, kernel_size=1, activation=None)

    if in_filters == out_filters:
        # Residual connection
        x = keras.layers.Add()([in_block, x])
    
    return x

def dropout_block(in_block, filters, kernel_size=3, strides=1, first=False):
    x = conv_norm_activ(in_block,filters, kernel_size=kernel_size, strides=strides)
    x = keras.layers.SpatialDropout2D(0.2)(x)
    x = conv_norm_activ(x, filters, kernel_size=kernel_size)
    if first:
        downsampled = keras.layers.Conv2D(filters, 1, padding='same', strides=strides)(in_block)
        return keras.layers.Add()([downsampled, x])
    
    return keras.layers.Add()([in_block, x])

def dropout_group(in_block, filters, kernel_size, n, strides):
    x = dropout_block(in_block, filters, kernel_size=kernel_size, first=True, strides=strides)
    for _ in range(n-1):
        x = dropout_block(x, filters, kernel_size)

    return x

def fused_mbconv(in_block, in_filters, out_filters, expand_ratio, **kwargs):
    expanded_filters = in_filters*expand_ratio
    x = conv_norm_activ(in_block, filters=expanded_filters, **kwargs)
    x = se_block(x, expanded_filters)
    x = conv_norm_activ(x, out_filters, kernel_size=1, activation=None)

    if in_filters == out_filters:
        # Residual connection
        x = keras.layers.Add()([in_block, x])
    
    return x

def transpose_cna(in_block, in_filters, activation = 'relu', connected_block = None, **kwargs):

    x = keras.layers.Conv2DTranspose(in_filters, activation=None, padding='same', kernel_size=2, strides=2)(in_block)
    if connected_block is not None:
        x = keras.layers.Concatenate()([connected_block, x])
    x = keras.layers.BatchNormalization()(x)
    if activation is not None:
        return keras.layers.Activation(activation)(x)
    else:
        return x

def up(in_block, connected_block, in_filters, out_filters, n, kernel_size=3):
    # Upscaling
    x = transpose_cna(in_block, in_filters, connected_block=connected_block)
    for _ in range(n-1):
        x = conv_norm_activ(x, out_filters, kernel_size=kernel_size)
    return x

def down(in_block, out_filters, n, **kwargs):
    x = conv_norm_activ(in_block, out_filters, **kwargs)
    for _ in range(n-1):
        x = conv_norm_activ(x, out_filters, **kwargs)
    
    last_conv = x
    pool = keras.layers.AveragePooling2D(2)(x)
    return pool, last_conv

def fusedmb_group(in_block, in_filters, out_filters, expand_ratio, kernel_size = 1, n=1, strides=(1, 1)):
    x = fused_mbconv(in_block, in_filters, out_filters, expand_ratio, kernel_size=kernel_size, strides=strides)
    for _ in range(n-1):
        x = fused_mbconv(x, out_filters, out_filters, expand_ratio, kernel_size=kernel_size)
        x = keras.layers.SpatialDropout2D(0.2)(x)
    return x

def mb_group(in_block, in_filters, out_filters, expand_ratio, kernel_size = 1, n=1, strides=(1, 1)):
    x = mbconv(in_block, in_filters, out_filters, expand_ratio, kernel_size=kernel_size, strides=strides)
    for _ in range(n-1):
        x = mbconv(x, out_filters, out_filters, expand_ratio, kernel_size=kernel_size)
        x = keras.layers.SpatialDropout2D(0.2)(x)
    return x

def create_basic_cnn(args, input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1/255)(inputs)
    mult = args.scale
    conf = [(16, 3, 4, 1), (32, 3, 4, 2), (64, 3, 4, 2)]
    conf = [(filters*mult, k, n, strides) for filters, k, n, strides in conf]
    x = keras.layers.Conv2D(16, (3, 3), padding='same')(x)
    for filters, k, n, strides in conf:
        x = dropout_group(x, filters, k, n, strides)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    pool = keras.layers.GlobalAveragePooling2D('channels_last')(x)
    classifier = keras.layers.Dense(num_classes, activation='softmax')(pool)
    model = keras.models.Model(inputs=inputs, outputs=classifier)
    model.summary()

    return model
def create_unet(args, input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1/255)(inputs)
    mult = args.scale
    configs = [(16 * mult, 3, 3), (32 * mult, 3, 3), (64 * mult, 3, 3)]
    down_convs = []
    for filters, kernel_size, n in configs:
        x, last_conv = down(x, filters, n, kernel_size=kernel_size)
        down_convs.append(last_conv)

    for i in range(len(configs) -1, -1, -1):
        filters, kernel_size, n = configs[i]
        connected_conv = down_convs[i]
        prev_filters = configs[max(0, i-1)][0]
        x = up(x, connected_conv, filters, prev_filters, n)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    pool = keras.layers.GlobalAveragePooling2D('channels_last')(x)
    classifier = keras.layers.Dense(num_classes, activation='softmax')(pool)
    model = keras.models.Model(inputs=inputs, outputs=classifier)
    model.summary()
    return model

def create_enetv2(args, input_shape, num_classes : int) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    scale = keras.layers.Rescaling(1/255)(inputs)
    mult = args.scale
    first_conv = conv_norm_activ(scale, mult*48, kernel_size=3)
    x = fusedmb_group(first_conv, mult*48, mult*48, 1, kernel_size=3, n=1)
    x = fusedmb_group(x, mult*48, mult*64, 4, kernel_size=3, n=2)
    x = fusedmb_group(x, mult*64, mult*128, 4, kernel_size=3, n=4, strides=(2, 2))
    x = fusedmb_group(x, mult*128, mult*160, 4, kernel_size=3, n=4, strides=(2, 2))
    x = conv_norm_activ(x, mult* 160 * 4, kernel_size=1)
    x = keras.layers.GlobalAveragePooling2D('channels_last')(x)
    dropout = keras.layers.Dropout(0.2)(x)
    flatten = keras.layers.Flatten()(dropout)
    classifier = keras.layers.Dense(num_classes, activation='softmax')(flatten)
    model = keras.Model(inputs=inputs, outputs=classifier)
    model.summary()

    return model
def create_enet(args, input_shape, num_classes : int) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    scale = keras.layers.Rescaling(1/255)(inputs)
    mult = args.scale
    first_conv = conv_norm_activ(scale, mult*24, kernel_size=3)
    x = mb_group(first_conv, mult*32, mult*16, 1, kernel_size=3, n=1)
    x = mb_group(x, mult*16, mult*24, 6, kernel_size=3, n=2)
    x = mb_group(x, mult*24, mult*40, 6, kernel_size=3, n=2, strides=(2, 2))
    x = mb_group(x, mult*40, mult*80, 6, kernel_size=3, n=3, strides=(2, 2))
    x = mb_group(x, mult*80, mult*112, 6, kernel_size=3, n=3, strides=(2, 2))
    x = conv_norm_activ(x, mult* 112 * 4, kernel_size=3)
    x = keras.layers.GlobalAveragePooling2D('channels_last')(x)
    dropout = keras.layers.Dropout(0.2)(x)
    flatten = keras.layers.Flatten()(dropout)
    classifier = keras.layers.Dense(num_classes, activation='softmax')(flatten)
    model = keras.Model(inputs=inputs, outputs=classifier)
    model.summary()

    return model

def compile_model(args, model : keras.Model, training_steps):
    schedule = keras.optimizers.schedules.CosineDecay(args.lr, training_steps)
    optim = keras.optimizers.AdamW(learning_rate=schedule)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    metrics = [
        keras.metrics.CategoricalAccuracy(name='accuracy'),
        keras.metrics.F1Score()
    ]

    model.compile(
        optimizer=optim,
        loss=loss, 
        metrics=metrics
    )

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, cifar: CIFAR10.Dataset, size : int = None, augmentation_fn=None, greyscale=False, return_labels=True) -> None:
        self.data = cifar.data['images']
        if size:
            self.data[:size]
        if return_labels:
            self.labels = cifar.data['labels']
        else:
            self.labels = None
        if size:
            self.data[:size]
        self.aug = augmentation_fn
        self.grayscale = greyscale

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[np.ndarray | torch.Tensor, int]:
        img = self.data[index]
        img = torch.as_tensor(img, dtype=torch.uint8)
        if self.grayscale:
            img = grayscale(img)
        if self.aug:
            img = self.aug(img)

        if self.labels is not None:
            return img, keras.ops.one_hot(self.labels[index], len(CIFAR10.LABELS))
        else:
            return img
    
def grayscale(img):
    img = torch.as_tensor(img, dtype=torch.uint8)
    img = torch.moveaxis(img, -1, 0)
    img = rgb_to_grayscale(img)
    return torch.moveaxis(img, 0, -1)

def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join(args.o, "logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    print(args.logdir)

    # Load data
    cifar = CIFAR10()

    transformation = v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomInvert(0.5),
            v2.RandomAffine([-90, 90], translate=(0.25, 0.25), scale=(1, 1.25), shear=(0.3, 0.3)),
            v2.CenterCrop(32)
        ])

    def augmentation_fn(image: np.ndarray) -> torch.Tensor:
        img = torch.as_tensor(image, dtype=torch.uint8)
        img = torch.moveaxis(img, -1, 0)
        e = np.random.random()
        if e < args.aug_prob:
            img = transformation(img)
        return torch.moveaxis(img, 0, -1)

    # TODO: Create the model and train it
    image_shape = [CIFAR10.H, CIFAR10.W, CIFAR10.C]
    if args.greyscale:
        image_shape[-1] = 1
    if args.mode == 'v2':
        model = create_enetv2(args, image_shape, len(CIFAR10.LABELS))
    elif args.mode == 'unet':
        model = create_unet(args, image_shape, len(CIFAR10.LABELS))
    elif args.mode == 'enet':
        model = create_enet(args, image_shape, len(CIFAR10.LABELS))
    else:
        model = create_basic_cnn(args, image_shape, len(CIFAR10.LABELS))
    compile_model(args, model, cifar.train.data['images'].shape[0] // args.batch_size * args.epochs)

    train_ds = TorchDataset(cifar.train, augmentation_fn=augmentation_fn, greyscale=args.greyscale)
    dev_ds = TorchDataset(cifar.dev, greyscale=args.greyscale)

    train = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(dev_ds, batch_size=args.batch_size, shuffle=True)

    model.fit(train, batch_size=args.batch_size, epochs=args.epochs, validation_data = dev)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    test = TorchDataset(cifar.test, greyscale=args.greyscale, return_labels=False)
    loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size)
    probs = np.array(model.predict(loader))
    with open(os.path.join(args.logdir, "cifar_competition_test.npy"), "wb") as predictions_file:
        np.save(predictions_file, probs)

    txt_path = os.path.join(args.logdir, "cifar_competition_test.txt")
    np.savetxt(txt_path, probs, encoding='utf-8')
    # with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
    #     # TODO: Perform the prediction on the test data.
    #     for prob in probs:
    #         predictions_file.write(str(prob))

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
