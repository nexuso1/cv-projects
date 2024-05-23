#!/usr/bin/env python3
# Supplementary code for the SVHN competition, for creating/using bboxes
# Template created by Milan Straka, https://github.com/foxik, filled out by nexuso1

from typing import Tuple

import numpy as np

# Bounding boxes and anchors are expected to be Numpy/TensorFlow tensors,
# where the last dimension has size 4.


# For bounding boxes in pixel coordinates, the 4 values correspond to:
TOP: int = 0
LEFT: int = 1
BOTTOM: int = 2
RIGHT: int = 3


def bboxes_area(bboxes: np.ndarray) -> np.ndarray:
    """ Compute area of given set of bboxes.

    The computation can be performed either using Numpy or TensorFlow.
    Each bbox is parametrized as a four-tuple (top, left, bottom, right).

    If the bboxes.shape is [..., 4], the output shape is bboxes.shape[:-1].
    """
    return np.maximum(bboxes[..., BOTTOM] - bboxes[..., TOP], 0) \
        * np.maximum(bboxes[..., RIGHT] - bboxes[..., LEFT], 0)


def bboxes_iou(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """ Compute IoU of corresponding pairs from two sets of bboxes `xs` and `ys`.

    The computation can be performed either using Numpy or TensorFlow.
    Each bbox is parametrized as a four-tuple (top, left, bottom, right).

    Note that broadcasting is supported, so passing inputs with
    `xs.shape=[num_xs, 1, 4]` and `ys.shape=[1, num_ys, 4]` produces an output
    with shape `[num_xs, num_ys]`, computing IoU for all pairs of bboxes from
    `xs` and `ys`. Formally, the output shape is `np.broadcast(xs, ys).shape[:-1]`.
    """
    intersections = np.stack([
        np.maximum(xs[..., TOP], ys[..., TOP]),
        np.maximum(xs[..., LEFT], ys[..., LEFT]),
        np.minimum(xs[..., BOTTOM], ys[..., BOTTOM]),
        np.minimum(xs[..., RIGHT], ys[..., RIGHT]),
    ], axis=-1)

    xs_area, ys_area, intersections_area = bboxes_area(xs), bboxes_area(ys), bboxes_area(intersections)

    return intersections_area / (xs_area + ys_area - intersections_area)


def bboxes_to_fast_rcnn(anchors: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    """ Convert `bboxes` to a Fast-R-CNN-like representation relative to `anchors`.

    The `anchors` and `bboxes` are arrays of four-tuples (top, left, bottom, right);
    you can use the TOP, LEFT, BOTTOM, RIGHT constants as indices of the
    respective coordinates.

    The resulting representation of a single bbox is a four-tuple with:
    - (bbox_y_center - anchor_y_center) / anchor_height
    - (bbox_x_center - anchor_x_center) / anchor_width
    - log(bbox_height / anchor_height)
    - log(bbox_width / anchor_width)

    If the `anchors.shape` is `[anchors_len, 4]` and `bboxes.shape` is `[anchors_len, 4]`,
    the output shape is `[anchors_len, 4]`.
    """

    # TODO: Implement according to the docstring.

    anchor_w = anchors[..., RIGHT] - anchors[..., LEFT]
    anchor_h = anchors[..., BOTTOM] - anchors[..., TOP]
    anchor_cx = anchors[..., LEFT] + anchor_w / 2
    anchor_cy = anchors[..., TOP] + anchor_h / 2

    bbox_w = bboxes[..., RIGHT] - bboxes[..., LEFT]
    bbox_h = bboxes[..., BOTTOM] - bboxes[..., TOP]
    bbox_cx = bboxes[..., LEFT] + bbox_w / 2
    bbox_cy = bboxes[..., TOP] + bbox_h / 2

    res = np.stack([
        (bbox_cy - anchor_cy) / anchor_h,
        (bbox_cx - anchor_cx) / anchor_w,
        np.log(bbox_h / anchor_h),
        np.log(bbox_w / anchor_w)
    ], axis=-1)
    
    return res


def bboxes_from_fast_rcnn(anchors: np.ndarray, fast_rcnns: np.ndarray) -> np.ndarray:
    """ Convert Fast-R-CNN-like representation relative to `anchor` to a `bbox`.

    The `anchors.shape` is `[anchors_len, 4]`, `fast_rcnns.shape` is `[anchors_len, 4]`,
    the output shape is `[anchors_len, 4]`.
    """

    # TODO: Implement according to the docstring.
    anchor_w = anchors[..., RIGHT] - anchors[..., LEFT]
    anchor_h = anchors[..., BOTTOM] - anchors[..., TOP]
    anchor_cx = anchors[..., LEFT] + anchor_w / 2
    anchor_cy = anchors[..., TOP] + anchor_h / 2
    
    rcnn_w = np.exp(fast_rcnns[..., RIGHT]) * anchor_w
    rcnn_h = np.exp(fast_rcnns[..., BOTTOM]) * anchor_h
    rcnn_cy = fast_rcnns[..., TOP] * anchor_h + anchor_cy
    rcnn_cx = fast_rcnns[..., LEFT] * anchor_w + anchor_cx

    return np.stack([
        rcnn_cy - rcnn_h / 2,
        rcnn_cx - rcnn_w / 2,
        rcnn_cy + rcnn_h / 2,
        rcnn_cx + rcnn_w / 2
    ], axis=-1)


def bboxes_training(
    anchors: np.ndarray, gold_classes: np.ndarray, gold_bboxes: np.ndarray, iou_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """ Compute training data for object detection.

    Arguments:
    - `anchors` is an array of four-tuples (top, left, bottom, right)
    - `gold_classes` is an array of zero-based classes of the gold objects
    - `gold_bboxes` is an array of four-tuples (top, left, bottom, right)
      of the gold objects
    - `iou_threshold` is a given threshold

    Returns:
    - `anchor_classes` contains for every anchor either 0 for background
      (if no gold object is assigned) or `1 + gold_class` if a gold object
      with `gold_class` is assigned to it
    - `anchor_bboxes` contains for every anchor a four-tuple
      `(center_y, center_x, height, width)` representing the gold bbox of
      a chosen object using parametrization of Fast R-CNN; zeros if no
      gold object was assigned to the anchor

    Algorithm:
    - First, for each gold object, assign it to an anchor with the largest IoU
      (the one with smaller index if there are several). In case several gold
      objects are assigned to a single anchor, use the gold object with smaller
      index.
    - For each unused anchor, find the gold object with the largest IoU
      (again the one with smaller index if there are several), and if the IoU
      is >= iou_threshold, assign the object to the anchor.
    """

    # First, for each gold object, assign it to an anchor with the
    # largest IoU (the one with smaller index if there are several). In case
    # several gold objects are assigned to a single anchor, use the gold object
    # with smaller index.
    
    # For every gold, assign it to an anchor
    ious = bboxes_iou(gold_bboxes.reshape(-1, 1, 4), anchors.reshape(1, -1, 4))
    assigned_anchor_idxs = np.argmax(ious, axis=-1)

    # For every anchor, get its gold
    golds_assigned_to_anchor = {}
    for i, anchor in enumerate(assigned_anchor_idxs):
        if not anchor in golds_assigned_to_anchor:
            golds_assigned_to_anchor[anchor] = i
    
    anchor_classes = np.zeros(shape=anchors.shape[0])
    anchor_bboxes = np.zeros(shape=anchors.shape)
    for i in golds_assigned_to_anchor.keys():
        anchor_classes[i] = 1 + gold_classes[golds_assigned_to_anchor[i]]
        anchor_bboxes[i] = gold_bboxes[golds_assigned_to_anchor[i]]

    # For each unused anchor, find the gold object with the largest IoU
    # (again the one with smaller index if there are several), and if the IoU
    # is >= threshold, assign the object to the anchor.

    # Create mask denoting unused bboxes
    unused_mask = anchor_classes == 0

    # Take the unassigned ious 
    unassigned_ious = np.take_along_axis(ious, np.argwhere(unused_mask).repeat(ious.shape[0], axis=-1).T, -1)
    passed_threshold_mask = np.max(unassigned_ious, axis=0) >= iou_threshold
    if np.any(passed_threshold_mask):
        passed_idx = np.argwhere(unused_mask)[passed_threshold_mask].flatten()
        anchor_classes[passed_idx] = 1 + gold_classes[np.argmax(unassigned_ious, axis=0)[passed_threshold_mask]]
        anchor_bboxes[passed_idx] = gold_bboxes[np.argmax(unassigned_ious, axis=0)[passed_threshold_mask]]

    nonzero = anchor_classes != 0
    anchor_bboxes[nonzero] = bboxes_to_fast_rcnn(anchors[nonzero], anchor_bboxes[nonzero])
    return anchor_classes, anchor_bboxes

