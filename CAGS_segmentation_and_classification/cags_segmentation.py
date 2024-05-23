#!/usr/bin/env python3
# CAGS Segmenation model
# Author: nexuso1

import argparse
import datetime
import os
import re
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import numpy as np
from utils import up, compile_model
import torch
import torchvision.transforms.v2 as v2

from cags_dataset import CAGS
from functools import partial

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--n", default=3, help='Number of convolutions in the up direction', type=int)
parser.add_argument("--custom_filters", default=False, help='Use custom filter sizes when upsampling', type=bool)
parser.add_argument('--lr', default=0.0001, help='Learning rate', type=float)
parser.add_argument('--ver', default='b3', help='ENet version', type=str)

class RandomerAugment:
    """
    Augmentation based on the RandAugment paper.
    """
    def __init__(self, image_shape) -> None:
        self.crop = v2.CenterCrop([image_shape[0], image_shape[1]])
        self.transforms = [
            (v2.RandomRotation([-90, 90]), True),
            (v2.RandomEqualize(1), False),
            (v2.RandomAutocontrast(1), False),
            (v2.RandomSolarize(0.7, 1), False),
            (v2.RandomPosterize(4, 1), False),
            (v2.RandomVerticalFlip(1), True),
            (v2.RandomHorizontalFlip(1), True),
            (v2.RandomAffine(0, shear=[-30, 30]), True),
            (v2.RandomInvert(1), False),
            (v2.ColorJitter([0.75, 1.5]), False),
            (v2.ColorJitter(contrast=[0.75, 1.5]), False),
            (v2.RandomAdjustSharpness(2, 1), False),
            (v2.RandomAffine(0, translate=[0.33, 0.33]), True)
        ]

    def __call__(self, img, mask):
        idxs = np.random.choice(len(self.transforms), 3)
        transforms = [self.transforms[i] for i in idxs]
        seed = np.random.randint(2147483647)
        # Set seeds so that the transforms are the same for 
        # the image and the mask
        np.random.seed(seed) 
        torch.manual_seed(seed)
        for transform, _ in transforms:
            img = transform(img)

        np.random.seed(seed) 
        torch.manual_seed(seed)
        # Will be the same transform
        for transform, apply_to_mask in transforms:
            if apply_to_mask:
                mask = transform(mask)

        # Crop to original dims
        img = self.crop(img)
        mask = self.crop(mask)

        return img, mask

def aug(img, mask):
    """
    Functional version of RandomerAugment
    """
    original_size = img.shape[0], img.shape[1]
    crop = v2.CenterCrop(original_size)
    transforms = [
        (v2.RandomRotation([-90, 90]), True),
        (v2.RandomEqualize(1), False),
        (v2.RandomAutocontrast(1), False),
        (v2.RandomSolarize(0.7, 1), False),
        (v2.RandomPosterize(4, 1), False),
        (v2.RandomVerticalFlip(1), True),
        (v2.RandomHorizontalFlip(1), True),
        (v2.RandomAffine(0, shear=[-30, 30]), True),
        (v2.RandomInvert(1), False),
        (v2.ColorJitter([0.75, 1.5]), False),
        (v2.ColorJitter(contrast=[0.75, 1.5]), False),
        (v2.RandomAdjustSharpness(2, 1), False),
        (v2.RandomAffine(0, translate=[0.33, 0.33]), True)
    ]

    idxs = np.random.choice(len(transforms), 3)
    transforms = [transforms[i] for i in idxs]
    seed = np.random.randint(2147483647)
    # Set seeds so that the transforms are the same for 
    # the image and the mask
    np.random.seed(seed) 
    torch.manual_seed(seed)
    for transform, _ in transforms:
        img = transform(img)

    np.random.seed(seed) 
    torch.manual_seed(seed)
    # Will be the same transform
    for transform, apply_to_mask in transforms:
        if apply_to_mask:
            mask = transform(mask)

    img = crop(original_size)
    mask = crop(original_size)

    return img, mask

def unet(args, input_shape, backbone):
    """
    Create a U-net based CNN 
    """
    inputs = keras.layers.Input(input_shape)
    backbone_out= backbone(inputs)
    backbone_filters = [layer.shape[-1] for layer in backbone_out]

    # 
    if args.custom_filters:
        up_filters = [640, 320, 160, 80, 40] # From 7x7 to 224x224

    # Connect the layers to the corresponding resolution outputs in the backbone,
    # and use transposed convolution to upsample the image
    x = backbone_out[0]
    for i, layer in enumerate(backbone_out[1:]):
        filters = backbone_filters[i] 
        if args.custom_filters:
            # Use custom filter sizes for upsampling
            next_filters = up_filters[i]
        else:
            # Use the same dimension as in the model
            next_filters = backbone_filters[i]
        x = up(x, layer, filters, next_filters, args.n)

    x = keras.layers.BatchNormalization()(x)

    if args.custom_filters:
        final_filters = up_filters[-1]
        
    else:
        # Filters at the highest resolution are not defined in ENet, so use this default number
        final_filters = 40

    # Last upsampling
    x = up(x, None, next_filters , final_filters, args.n) # 224 x 224

    # Mask output
    mask = keras.layers.Conv2D(1, 1, 1, padding='same', activation='sigmoid')(x) 

    model = keras.models.Model(inputs=inputs, outputs=mask)
    model.summary()

    return model

def prep_fn(x, aug=None):
    """
    Process the data from the dataset to the required Torch format (C, H, W),
    and optionally augment the image using a callable aug
    """
    img, mask = x['image'], x['mask']
    img = torch.as_tensor(img, dtype=torch.uint8)
    mask = torch.as_tensor(mask, dtype=torch.uint8)
    img = torch.moveaxis(img, -1, 0)
    mask = torch.moveaxis(mask, -1, 0)
    if aug is not None:
        img, mask = aug(img, mask)
    return torch.moveaxis(img, 0, -1) , torch.moveaxis(mask, 0, -1)

def compile_model(args, model : keras.Model, training_steps, metrics = None):
    """
    Compile the model using AdamW and Cosine decay
    """
    schedule = keras.optimizers.schedules.CosineDecay(args.lr, training_steps)
    optim = keras.optimizers.AdamW(learning_rate=schedule)
    loss = keras.losses.BinaryCrossentropy(label_smoothing=0.1)
    if metrics is None:
        metrics = [
            CAGS.MaskIoUMetric() # Metric from the competition
        ]

    model.compile(
        optimizer=optim,
        loss=loss, 
        metrics=metrics
    )

def prep_fn_test(x):
    """
    Prep function for test data, could be absorbed to prep_fn
    """
    img = x['image']
    return torch.as_tensor(img, dtype=torch.uint8)

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

    # Load the EfficientNetV2-B0 model. It assumes the input images are
    # represented in the [0-255] range.
    if args.ver == 'b0':
        backbone = keras.applications.EfficientNetV2B0(include_top=False)
    elif args.ver == 'b1':
        backbone = keras.applications.EfficientNetV2B1(include_top=False)
    elif args.ver == 'b2':
        backbone = keras.applications.EfficientNetV2B2(include_top=False)
    elif args.ver == 'b3':
        backbone = keras.applications.EfficientNetV2B3(include_top=False)
        
    # Extract features of different resolution. Assuming 224x224 input images
    # (you can set this explicitly via `input_shape` of the above constructor),
    # the below model returns five outputs with resolution 7x7, 14x14, 28x28, 56x56, 112x112.
    backbone = keras.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(layer).output for layer in [
            "top_activation", "block5e_add", "block3b_add", "block2b_add", "block1a_project_activation"]]
    )

    # Augmentation callable
    aug = RandomerAugment([CAGS.H, CAGS.W, CAGS.C])

    # Transform the train dataset using the augmentation
    train_ds = cags.train.transform(partial(prep_fn, aug=aug))

    # The dev dataset will not be transformed
    dev_ds = cags.dev.transform(prep_fn)

    # Prepare dataloaders for training
    train = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(dev_ds, batch_size=args.batch_size)

    # TODO: Create the model and train it
    chkpt_path = os.path.join(args.logdir, 'segmentation_model.keras')

    # Will automatically save the best model during training w.r.t. a given metric
    callbacks = [keras.callbacks.ModelCheckpoint(chkpt_path, monitor='val_iou', save_best_only=True, verbose=1, mode='max')]

    # TODO: Create the model and train it
    model = unet(args, [CAGS.H, CAGS.W, CAGS.C], backbone)
    compile_model(args, model, len(train_ds) // args.batch_size * args.epochs)

    model.fit(train, validation_data=dev, batch_size=args.batch_size, epochs=args.epochs, callbacks=callbacks)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    test = torch.utils.data.DataLoader(cags.test.transform(prep_fn_test), batch_size=args.batch_size)

    # Load the best model
    model = keras.saving.load_model(chkpt_path, compile=True, custom_objects={'CAGS.MaskIoUMetric' : cags.MaskIoUMetric})
    # Best model eval
    loss, iou = model.evaluate(dev, batch_size = args.batch_size)
    print(f'Best model performance: iou = {iou}')
    with open(os.path.join(args.logdir, "metrics.txt"), 'w') as f:
        f.write(f'IoU : {iou}')

    with open(os.path.join(args.logdir, "dev_preds.npy"), 'wb') as f:
        preds = model.predict(dev, batch_size=args.batch_size)
        np.save(f, np.array(preds))

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the masks on the test set
        test_masks = model.predict(test)

        for mask in test_masks:
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