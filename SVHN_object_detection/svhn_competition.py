#!/usr/bin/env python3
# Authors: Matúš Konig, nexuso1

import argparse
import datetime
import os
import re

import torch.utils
import torch.utils.data
import torch.utils.data.dataloader

os.environ.setdefault(
    "KERAS_BACKEND", "torch"
)  # Use PyTorch backend unless specified otherwise

import keras
import numpy as np
import torch
import torchvision.transforms.v2 as v2

import bboxes_utils
from svhn_dataset import SVHN
from functools import partial
from torchvision.transforms.functional import resize
from torchvision.ops import batched_nms, nms

from utils import up, dropout_group

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=26, type=int, help="Batch size.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument(
    "--threads", default=0, type=int, help="Maximum number of threads to use."
)
parser.add_argument(
    "--n", default=1, type=int, help="Number of conv layers in the output heads"
)
parser.add_argument(
    "--custom_filters",
    default=True,
    action="store_true",
    help="Use custom filter sizes",
)
parser.add_argument("--lr", default=0.0005, type=float, help="Learning rate")
parser.add_argument("--ver", default="b0", help="ENet version", type=str)
parser.add_argument('--eval', default=False, help='Evaluate model with this path', type=str)


class TorchTensorBoardCallback(keras.callbacks.Callback):
    def __init__(self, path):
        self._path = path
        self._writers = {}

    def writer(self, writer):
        if writer not in self._writers:
            import torch.utils.tensorboard

            self._writers[writer] = torch.utils.tensorboard.SummaryWriter(
                os.path.join(self._path, writer)
            )
        return self._writers[writer]

    def add_logs(self, writer, logs, step):
        if logs:
            for key, value in logs.items():
                self.writer(writer).add_scalar(key, value, step)
            self.writer(writer).flush()

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            if isinstance(
                getattr(self.model, "optimizer", None), keras.optimizers.Optimizer
            ):
                logs = logs | {
                    "learning_rate": keras.ops.convert_to_numpy(
                        self.model.optimizer.learning_rate
                    )
                }
            self.add_logs(
                "train",
                {k: v for k, v in logs.items() if not k.startswith("val_")},
                epoch + 1,
            )
            self.add_logs(
                "val",
                {k[4:]: v for k, v in logs.items() if k.startswith("val_")},
                epoch + 1,
            )


class NumericalAccuracyCallback(keras.callbacks.Callback):

    def __init__(self, anchors, num_classes, val_ds, val_data, svhn):
        self.anchors = anchors
        self.svhn = svhn
        self.num_classes = num_classes
        self.val_ds = val_ds
        self.val = val_data
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        eval_model(self.model, self.anchors, self.val, self.val_ds, self.svhn)
        return super().on_epoch_end(epoch, logs)


class RandomerAugment:
    """
    Augmentation based on the RandAugment paper.
    """

    def __init__(self, image_shape) -> None:
        self.crop = v2.CenterCrop([image_shape[0], image_shape[1]])
        self.transforms = [
            (v2.RandomEqualize(1), False),
            (v2.RandomAutocontrast(1), False),
            (v2.RandomSolarize(0.7, 1), False),
            (v2.RandomPosterize(4, 1), False),
            # (v2.RandomVerticalFlip(1), True),
            # (v2.RandomHorizontalFlip(1), True),
            # (v2.RandomAffine(0, shear=[-30, 30]), True),
            (v2.RandomInvert(1), False),
            (v2.ColorJitter([0.75, 1.5]), False),
            (v2.ColorJitter(contrast=[0.75, 1.5]), False),
            (v2.RandomAdjustSharpness(2, 1), False),
            # (v2.RandomAffine(0, translate=[0.33, 0.33]), True)
        ]

    def __call__(self, img):
        idxs = np.random.choice(len(self.transforms), 3)
        transforms = [self.transforms[i] for i in idxs]
        seed = np.random.randint(2147483647)
        # Set seeds so that the transforms are the same for
        # the image and the mask
        np.random.seed(seed)
        torch.manual_seed(seed)
        for transform, _ in transforms:
            img = transform(img)

        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # # Will be the same transform
        # for transform, apply_to_mask in transforms:
        #     if apply_to_mask:
        #         mask = transform(mask)

        # Crop to original dims
        # img = self.crop(img)
        # mask = self.crop(mask)

        return img


# Tak trochu ma naviedol rado, že by sme mali dať 0 vektor ako label ak vyjde background z bboxes_training, a treba pre tieto anchors dať weight 0
# Že keď dostane nejaký anchor label 0, čiže background, z bboxes_training, tak one hot target tohoto anchoru má byť nulový vektor
# Samo Fanči
# A že nestačí mať focal loss, ale treba manuálne dať background anchors weight 0 aby sa na tom nič netrénovalo
def prep_fn(x, anchors, size=[224, 224], ious=0.5, num_classes=11, aug=None):
    # Change the bbox representation to relative
    bboxes = bbox_to_relative(x['image'], x['bboxes'])
    # orig_shape = np.array(x['image'].shape)
    classes, bboxes = bboxes_utils.bboxes_training(anchors, x['classes'], bboxes, iou_threshold=ious)
    weights = keras.ops.where(classes != 0, 1, 0)
    original_dims = x['image'].shape
    # Resize the image to an uniform size
    img = torch.as_tensor(x['image'], dtype=torch.uint8)
    img = torch.moveaxis(img, -1, 0)
    img = resize(img, size)
    if aug:
        img = aug(img)
    img = torch.moveaxis(img, 0, -1)
    return (
        img,
        {
            "classification": keras.ops.one_hot(classes, num_classes),
            "bboxes": bboxes,
            "original_dims": np.array(original_dims),
        },
        {"classification": torch.ones_like(weights), "bboxes": weights},
    )


def prep_fn_test(x, size=[224, 224]):
    # Change the bbox representation to relative
    original_dims = x["image"].shape
    # Resize the image to an uniform size
    img = torch.as_tensor(x["image"], dtype=torch.uint8)
    img = torch.moveaxis(img, -1, 0)
    img = resize(img, size)
    img = torch.moveaxis(img, 0, -1)
    return img, {"original_dims": np.array(original_dims)}


def bbox_to_relative(img, bboxes):
    h, w = img.shape[0], img.shape[1]
    return bboxes / np.array([h, w, h, w])


def relative_to_bbox(original_shape, bboxes):
    h, w = original_shape[..., 0], original_shape[..., 1]
    return bboxes * np.array([h, w, h, w])


def reverse_scaling(img, bboxes, original_shape):
    rescaled = resize(img, original_shape)
    corrected_bb = relative_to_bbox(original_shape, bboxes)

    return rescaled, corrected_bb


def basic_model(backbone, anchors_per_pixel, classes, input_shape=[224, 224, 3]):
    inputs = keras.layers.Input(input_shape)
    backbone_out = backbone(inputs)
    backbone_filters = [layer.shape[-1] for layer in backbone_out]
    up_filters = [320, 160, 80]
    x = backbone_out[0]

    for i, layer in enumerate(backbone_out[1:3]):
        filters = backbone_filters[i]
        if args.custom_filters:
            # Use custom filter sizes for upsampling
            next_filters = up_filters[i]
        else:
            # Use the same dimension as in the model
            next_filters = backbone_filters[i]
        x = up(x, layer, filters, next_filters, args.n)

    cls_features = dropout_group(x, up_filters[-1], kernel_size=3, n=3, strides=1)

    cls_out = keras.layers.Conv2D(
        anchors_per_pixel * classes, kernel_size=1, padding="same"
    )(cls_features)
    cls_out = keras.layers.Reshape((-1, classes))(cls_out)
    cls_out = keras.layers.Activation("softmax", name="classification")(cls_out)

    bbox_features = dropout_group(x, up_filters[-1], kernel_size=3, n=3, strides=1)
    bbox_out = keras.layers.Conv2D(
        anchors_per_pixel * 4, kernel_size=1, padding="same"
    )(bbox_features)
    bbox_out = keras.layers.Reshape((-1, 4), name="bboxes")(bbox_out)

    outputs = {"classification": cls_out, "bboxes": bbox_out}

    output_names = sorted(outputs.keys())
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.output_names = output_names
    model.summary()

    return model


def create_anchors(img_shape, scales, ratios):
    """
    Create anchors for every pixel in the image. Returns a array of (top, left, bottom, right) coordinates of the anchors.

    Adapted from https://d2l.ai/chapter_computer-vision/anchor.html#generating-multiple-anchor-boxes
    """
    w, h = img_shape[-2:]
    anchors_per_pixel = scales.shape[0] + ratios.shape[0] - 1
    # Step size for centers
    x_step, y_step = 1 / w, 1 / h

    # Pixel center
    x_offset, y_offset = 0.5, 0.5

    center_xs = (np.arange(w) + x_offset) * x_step
    center_ys = (np.arange(h) + y_offset) * y_step

    shift_y, shift_x = np.meshgrid(center_ys, center_xs, indexing="ij")
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Widths for every (ratio_1, scale_i)
    widths_1 = scales * np.sqrt(ratios[0])

    # Widths for every (ratio_i, scale_1)
    widths_2 = scales[0] * np.sqrt(ratios[1:])
    widths = np.concatenate([widths_1, widths_2]) * w / h  # Rectangular images

    # Same for heights
    heights_1 = scales / np.sqrt(ratios[0])
    heights_2 = scales[0] / np.sqrt(ratios[1:])

    heights = np.concatenate([heights_1, heights_2])

    # Create offsets of coordinates relative to the center of an anchor, for every
    # anchor. Widths and heights need to be divided by half.
    anchor_offsets = (
        np.tile(np.stack((-widths, -heights, widths, heights)).T, (h * w, 1)) / 2
    )

    out_grid = np.stack([shift_x, shift_y, shift_x, shift_y], axis=1).repeat(
        anchors_per_pixel, axis=0
    )

    # Output shape is (num_anchors * num_pixels, 4)
    return out_grid + anchor_offsets


def compile_model(args, model: keras.Model, training_steps, metrics=None):
    """
    Compile the model using AdamW and Cosine decay
    """
    schedule = keras.optimizers.schedules.CosineDecay(args.lr, training_steps)
    optim = keras.optimizers.AdamW(learning_rate=schedule)
    loss = {
        "classification": keras.losses.CategoricalFocalCrossentropy(
            alpha=[0.75] + [0.25] * 10,
            #   label_smoothing=0.01
        ),
        "bboxes": keras.losses.Huber(),
    }
    if metrics is None:
        metrics = {
            "classification": [keras.metrics.CategoricalAccuracy(name="accuracy")],
            "bboxes": [keras.metrics.MeanSquaredError(name="bbox_mse")],
            # 'bboxes' : keras.metrics.MeanIoU(11, spa)
        }

    model.compile(optimizer=optim, loss=loss, metrics=metrics)


def clip_to_image(img_dims, boxes):
    if boxes.shape[0] == 0:
        return
    boxes[..., SVHN.TOP] = np.maximum(boxes[..., SVHN.TOP], 0)
    boxes[..., SVHN.LEFT] = np.maximum(boxes[..., SVHN.LEFT], 0)
    boxes[..., SVHN.BOTTOM] = np.minimum(boxes[..., SVHN.BOTTOM], img_dims[0])
    boxes[..., SVHN.RIGHT] = np.minimum(boxes[..., SVHN.RIGHT], img_dims[1])

    return boxes


def batch_nms(
    bboxes,
    scores,
    num_anchors,
    batch_size,
    num_classes,
    iou_threshold=0.5,
    score_threshold=0.2,
):
    # Expand the boxes for every class
    boxes = (
        bboxes.unsqueeze(2).expand(batch_size, num_anchors, num_classes, 4).contiguous()
    )

    # Flatten them to shape [N, 4]
    boxes = boxes.view(-1, 4)

    # Flatten the scores to shape [N]
    scores = torch.as_tensor(scores, dtype=torch.float32).view(-1)

    # Prepare indices for batched_nms
    rows = torch.arange(batch_size, dtype=torch.long)[:, None]
    cols = torch.arange(num_classes, dtype=torch.long)[None, :]
    idxs = rows * num_classes + cols
    idxs = idxs.unsqueeze(1).expand(batch_size, num_anchors, num_classes).contiguous()

    # Will have shape [N]
    idxs = idxs.view(-1)

    # Apply thresholding to scores
    mask = scores >= score_threshold
    boxesf = boxes[mask].contiguous()
    scoresf = scores[mask].contiguous()
    idxsf = idxs[mask].contiguous()

    keep = batched_nms(boxesf, scoresf, idxsf, iou_threshold)

    # Keep only the predicted boxes
    boxes = boxesf[keep]
    scores = scoresf[keep]

    # Recover the original labels and indices to the batch
    labels = idxsf[keep] % num_classes
    batch_index = idxsf[keep] // num_classes

    return boxes, scores, labels, batch_index


def apply_nms(preds, anchors, num_classes, iou_threshold=0.5, score_threshold=0.2):
    # num_anchors = preds['classification'].shape[1]
    final_preds = {
        "bboxes": [[] for _ in range(preds["bboxes"].shape[0])],
        "labels": [[] for _ in range(preds["bboxes"].shape[0])],
        "scores": [[] for _ in range(preds["bboxes"].shape[0])],
    }

    for i in range(0, preds["bboxes"].shape[0]):
        # Convert from rcnn format to yxyx
        bboxes = torch.as_tensor(
            bboxes_utils.bboxes_from_fast_rcnn(anchors, preds["bboxes"][i]),
            dtype=torch.float32,
        ).view(-1, 4)
        scores = preds["classification"][i]
        labels = torch.as_tensor(np.argmax(scores, axis=-1))
        scores = torch.as_tensor(np.max(scores, axis=-1))
        # Keep only positive
        mask = labels > 0
        boxes = bboxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        nms_indices = nms(boxes + labels[..., None] * 100, scores, iou_threshold)
        # boxes, scores, labels, relative_indices = batch_nms(bboxes.unsqueeze(0), scores, num_anchors, 1,
        #                                                     num_classes, iou_threshold, score_threshold)

        boxes = boxes[nms_indices]
        scores = scores[nms_indices]
        labels = labels[nms_indices]
        # Throw out low score boxes
        mask = labels > score_threshold
        final_preds["bboxes"][i] = boxes[mask]
        # Correct labels
        final_preds["labels"][i] = (labels - 1)[mask]
        final_preds["scores"][i] = scores[mask]

    return final_preds


def eval_model(model, anchors, dataloader, dataset, svhn):
    nms_preds = nms_predict(model, anchors, dataloader, dataset)
    orderings = [
        torch.argsort(boxes[..., SVHN.LEFT], 0) for boxes in nms_preds["bboxes"]
    ]
    labels = [
        nms_preds["labels"][i][orderings[i]] for i in range(len(nms_preds["labels"]))
    ]
    boxes = [
        nms_preds["bboxes"][i][orderings[i]] for i in range(len(nms_preds["bboxes"]))
    ]
    print("first bbox", boxes[0], boxes[0].shape)
    print("first label", labels[0], labels[0].shape)
    acc = SVHN.evaluate(svhn.dev, list(zip(labels, boxes)))
    print("model nms accuracy", acc)
    return acc


def nms_predict(model, anchors, dataloader, dataset):
    preds = model.predict(dataloader)
    # print(
    #     "non-background predictions: ",
    #     np.sum(np.argmax(preds["classification"][0], -1) != 0),
    # )
    # print(
    #     "background predictions: ",
    #     np.sum(np.argmax(preds["classification"][0], -1) == 0),
    # )

    # print("First bbox example")
    # print(preds["bboxes"][0])
    # print(dataset[0][1]["bboxes"])
    # print((preds["bboxes"][0]).shape)
    # nonzero_mask = np.any(dataset[0][1]["bboxes"], axis=1)
    # print("Nonbackground example")
    # print(preds["bboxes"][0][nonzero_mask])
    # print(dataset[0][1]["bboxes"][nonzero_mask])

    nms_preds = apply_nms(preds, anchors, 11)
    for i in range(len(nms_preds["bboxes"])):
        nms_preds["bboxes"][i] = relative_to_bbox(
            dataset[i][1]["original_dims"], nms_preds["bboxes"][i]
        )
        clip_to_image(dataset[i][1]["original_dims"], nms_preds["bboxes"][i])
    return nms_preds

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

    os.makedirs(args.logdir, exist_ok=True)

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[SIZE, SIZE, 3]` tensor of `torch.uint8` values in [0-255] range,
    # - "classes", a `[num_digits]` vector with classes of image digits,
    # - "bboxes", a `[num_digits, 4]` vector with bounding boxes of image digits.
    svhn = SVHN()
    
    
    # Load the EfficientNetV2-B0 model. It assumes the input images are
    # represented in the [0-255] range.
    if args.ver == "b0":
        backbone = keras.applications.EfficientNetV2B0(include_top=False)
    elif args.ver == "b1":
        backbone = keras.applications.EfficientNetV2B1(include_top=False)
    elif args.ver == "b2":
        backbone = keras.applications.EfficientNetV2B2(include_top=False)
    elif args.ver == "b3":
        backbone = keras.applications.EfficientNetV2B3(include_top=False)

    # Extract features of different resolution. Assuming 224x224 input images
    # (you can set this explicitly via `input_shape` of the above constructor),
    # the below model returns five outputs with resolution 7x7, 14x14, 28x28, 56x56, 112x112.
    backbone = keras.Model(
        inputs=backbone.input,
        outputs=[
            backbone.get_layer(layer).output
            for layer in [
                "top_activation",
                "block5e_add",
                "block3b_add",
                "block2b_add",
                "block1a_project_activation",
            ]
        ],
    )

    # for layer in backbone.layers:
    #     layer.trainable = False

    num_classes = 11
    ratios = np.array([1 / 1, 2 / 1])
    scales = np.array([0.9, 0.75, 0.5])
    anchors_per_position = len(ratios) + len(scales) - 1
    anchors = []
    sizes = [[28, 28]]  # Feature map sizes
    for size in sizes:
        fmap_anchors = create_anchors(np.array(size), scales, ratios)
        anchors.append(fmap_anchors)
        # fmap = np.zeros([224, 224])
        # SVHN.visualize(fmap, np.zeros(fmap_anchors.shape[0]), relative_to_bbox(np.array([224, 224]), fmap_anchors), show=True)

    anchors = np.concatenate(anchors, axis=0)
    print("anchors", anchors.shape)
    # aug = RandomerAugment([224, 224])
    train_ds = svhn.train.transform(partial(prep_fn, anchors=anchors, aug=None))
    dev_ds = svhn.dev.transform(partial(prep_fn, anchors=anchors))
    test_ds = svhn.test.transform(prep_fn_test)
    
    print("image,", train_ds[0][0].shape)
    print("data", train_ds[0][1])
    train = torch.utils.data.DataLoader(train_ds, args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(dev_ds, args.batch_size)
    test = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)
    # for img, classes, fast_anchors, orig_shapes in train:
    #     bboxes = bboxes_utils.bboxes_from_fast_rcnn(anchors, fast_anchors)
    #     vis_mask = classes[0] != 0

    #     SVHN.visualize(img[0], classes[0][vis_mask] - 1, relative_to_bbox(np.array([224, 224]),bboxes[0][vis_mask]), show=True)
    

    if args.eval:
        model = keras.saving.load_model(args.eval, compile=True)
        eval_model(model, anchors, dev, dev_ds, svhn)
        with open(os.path.join(os.path.dirname(args.eval), "dev_preds.npy"), "wb") as f:
            preds = model.predict(dev, batch_size=args.batch_size)
            np.save(f, np.array(preds))

        test_ds = svhn.test.transform(prep_fn_test)
        test = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)

        with open(
            os.path.join(os.path.dirname(args.eval), "svhn_competition.txt"), "w", encoding="utf-8"
        ) as predictions_file:
            # TODO: Predict the digits and their bounding boxes on the test set.
            # Assume that for a single test image we get
            # - `predicted_classes`: a 1D array with the predicted digits,
            # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;
            preds = nms_predict(model, anchors, test, test_ds)
            for predicted_classes, predicted_bboxes in zip(
                preds["labels"], preds["bboxes"]
            ):
                output = []
                for label, bbox in zip(predicted_classes, predicted_bboxes):
                    output += [label.numpy()] + list(bbox.numpy())
                print(*output, file=predictions_file)
                
        return
    
    # TODO: Create the model and train it
    model = basic_model(backbone, anchors_per_position, num_classes)

    chkpt_path = os.path.join(args.logdir, "bbox_model.keras")

    # Will automatically save the best model during training w.r.t. a given metric
    callbacks = [
        keras.callbacks.ModelCheckpoint(chkpt_path, monitor='val_classification_accuracy', save_best_only=True, verbose=1, mode='max'),
        NumericalAccuracyCallback(anchors, num_classes, dev_ds, dev, svhn)
    ]

    compile_model(args, model, len(train_ds) // args.batch_size * args.epochs)

    # # eval_model(model, anchors, dev, dev_ds, svhn)
    # keras.utils.plot_model(model, to_file="svhn_model.png", show_layer_names=True)
    model.fit(
        train,
        validation_data=dev,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)


    # Load the best model
    model = keras.saving.load_model(chkpt_path, compile=True)
    # Best model eval
    model.evaluate(dev, batch_size=args.batch_size)

    with open(os.path.join(args.logdir, "dev_preds.npy"), "wb") as f:
        preds = model.predict(dev, batch_size=args.batch_size)
        np.save(f, np.array(preds))

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(
        os.path.join(args.logdir, "svhn_competition.txt"), "w", encoding="utf-8"
    ) as predictions_file:
        # TODO: Predict the digits and their bounding boxes on the test set.
        # Assume that for a single test image we get
        # - `predicted_classes`: a 1D array with the predicted digits,
        # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;
        preds = nms_predict(model, anchors, test, test_ds)
        for predicted_classes, predicted_bboxes in zip(
            preds["labels"], preds["bboxes"]
        ):
            output = []
            for label, bbox in zip(predicted_classes, predicted_bboxes):
                output += [label.numpy()] + list(bbox.numpy())
            print(*output, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
