'''
**************************************************************************
||                        SiMa.ai CONFIDENTIAL                          ||
||   Unpublished Copyright (c) 2022-2023 SiMa.ai, All Rights Reserved.  ||
**************************************************************************
 NOTICE:  All information contained herein is, and remains the property of
 SiMa.ai. The intellectual and technical concepts contained herein are 
 proprietary to SiMa and may be covered by U.S. and Foreign Patents, 
 patents in process, and are protected by trade secret or copyright law.

 Dissemination of this information or reproduction of this material is 
 strictly forbidden unless prior written permission is obtained from 
 SiMa.ai.  Access to the source code contained herein is hereby forbidden
 to anyone except current SiMa.ai employees, managers or contractors who 
 have executed Confidentiality and Non-disclosure agreements explicitly 
 covering such access.

 The copyright notice above does not evidence any actual or intended 
 publication or disclosure  of  this source code, which includes information
 that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.

 ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
 DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
 CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE 
 LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
 CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO 
 REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
 SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.                

**************************************************************************
'''


'''
Common functions
'''


'''
Author: SiMa Technologies
'''


import numpy as np
from pathlib import Path
from typing import List, Tuple
import cv2


DIVIDER = '-'*50

# bounding box colors
color_palette = np.array(
                       [[201, 216, 192],
                        [ 13, 137, 110],
                        [248,  51, 188],
                        [211,  49, 216],
                        [ 55, 113, 113],
                        [ 83,  75,  76],
                        [106,   4, 197],
                        [ 22, 188,  49],
                        [242, 163,   4],
                        [139, 149, 128],
                        [101,  33, 134],
                        [255,  30, 186],
                        [129, 253,  52],
                        [ 17, 157,   2],
                        [100, 250, 219],
                        [  4,  86,  22],
                        [243, 186,  25],
                        [  6, 153, 135],
                        [109, 217,  51],
                        [ 68,  46,  49],
                        [ 64, 114, 219],
                        [213, 240, 212],
                        [  4,  26, 218],
                        [ 78, 174, 211],
                        [128, 191, 107],
                        [134,  56, 158],
                        [ 38, 158,  80],
                        [209, 151, 151],
                        [226,  13, 216],
                        [ 35, 199,  10],
                        [233, 158, 230],
                        [113, 203, 253],
                        [ 30, 220,  39],
                        [246,  15,   2],
                        [193,  32, 115],
                        [244, 127,  51],
                        [ 43, 223, 140],
                        [133, 142, 138],
                        [186,   3, 175],
                        [114,  76, 250],
                        [189,  85, 210],
                        [ 42,  80, 151],
                        [ 21, 147, 142],
                        [165, 119, 199],
                        [195,  46,  92],
                        [  8, 189,  51],
                        [ 32,  67,  81],
                        [119, 211,  50],
                        [ 55,  41, 112],
                        [ 18, 178,  34],
                        [  4, 236, 193],
                        [123, 249, 245],
                        [ 73, 232, 170],
                        [193, 149,  18],
                        [205, 207, 142],
                        [214, 107, 118],
                        [130, 130, 165],
                        [ 82, 117, 184],
                        [155, 197,  79],
                        [ 81, 217, 173],
                        [150, 253, 171],
                        [151,  28, 134],
                        [226, 240, 199],
                        [202, 221, 192],
                        [ 44, 248,  45],
                        [134, 101,  71],
                        [ 25, 132, 167],
                        [167,  42,  54],
                        [238, 118,  90],
                        [212,  23, 117],
                        [199, 202, 234],
                        [218, 184, 109],
                        [ 75, 160,  62],
                        [168, 166,  12],
                        [188,  11, 180],
                        [250, 219, 217],
                        [115,  42, 123],
                        [ 31,  22, 201],
                        [157,  52,  39],
                        [147, 174, 195]], dtype=np.uint8)

def list_image_paths(folder: str) -> List[str]:
  """
  Return a sorted list of file paths for image files in `folder` (non-recursive).
  Args:
    folder: Directory containing images.
  Returns:
    List of absolute paths (as strings) to image files.
  """
  IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

  p = Path(folder)

  if not p.exists() or not p.is_dir():
    raise NotADirectoryError(f"Not a directory: {folder}")

  return sorted(
    str(f.resolve())
    for f in p.iterdir()
    if f.is_file() and f.suffix.lower() in IMAGE_EXTS)




def preprocess(image_path: str, target_height: int, target_width: int, transpose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads an image from `image_path` into a NumPy array (BGR, forced 3 channels).
    Pads with black borders to make it square.
    Resizes to (target_height, target_width).
    Converts BGR -> RGB.
    Adds a batch dimension at axis 0 so shape is (1, H, W, C).
    Optionally transposes to NCHW format
    Returns padded, resized image as contiguous np.float32 array (values in [0, 255], not normalized).
    """
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target_height and target_width must be positive integers.")

    # Read image, force 3 channels
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    # img is BGR, shape (H, W, 3)
    h, w = img.shape[:2]
    side = max(h, w)

    # Compute centered padding to make square
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left

    # Pad with black borders
    img_sq = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Choose interpolation based on scaling direction
    interp = cv2.INTER_AREA if side > max(target_height, target_width) else cv2.INTER_LINEAR
    img_resized = cv2.resize(img_sq, (target_width, target_height), interpolation=interp)

    # BGR -> RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Add batch dimension and convert to float32
    out = img_rgb[np.newaxis, ...].astype(np.float32, copy=False)


    if transpose:
        # NHWC -> NCHW (returns a view when possible; no data copy)
        out = np.transpose(out, (0, 3, 1, 2))

    # Ensure contiguous (1, H, W, C), float32
    return img_resized, np.ascontiguousarray(out)



#def get_model_outputs(input_buffer: List):
#    '''
#    reshape model outputs
#    input_buffer is a list of 3 np arrays
#    0 - (1, 80, 80, 85)
#    1 - (1, 40, 40, 85)
#    2 - (1, 20, 20, 85)
#
#    pred is an np array (1, 8400, 85)
#    '''
#    print(len(input_buffer))
#    print(input_buffer[0].shape)
#    print(input_buffer[1].shape)
#    print(input_buffer[2].shape)
#    first_output = np.expand_dims(input_buffer[0], axis=0).reshape(1, -1, 85)  # (1, 80, 80, 85) -> (1, 6400, 85)
#    second_output = np.expand_dims(input_buffer[1], axis=0).reshape(1, -1, 85) # (1, 40, 40, 85) -> (1, 1600, 85)
#    third_output = np.expand_dims(input_buffer[2], axis=0).reshape(1, -1, 85)  # (1, 20, 20, 85) -> (1, 400, 85)
#
#    pred = np.concatenate((first_output, second_output, third_output), axis=1)
#    return pred


def get_model_outputs(input_buffer: List) -> np.ndarray:
    """
    Flattens multi-scale detector outputs into a single (1, N, C) array.
    Accepts each tensor as (H, W, C) or (1, H, W, C).
    """
    outs = []
    for x in input_buffer:
        x = np.asarray(x)

        # Normalize shape: allow (H, W, C) or (1, H, W, C)
        if x.ndim == 4:
            if x.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, got shape {x.shape}")
            x = x[0]  # drop the batch dim -> (H, W, C)
        elif x.ndim != 3:
            raise ValueError(f"Expected (H,W,C) or (1,H,W,C), got shape {x.shape}")

        C = x.shape[-1]  # number of channels (e.g., 85)
        x = np.ascontiguousarray(x).reshape(1, -1, C)  # -> (1, H*W, C)
        outs.append(x)

    pred = np.concatenate(outs, axis=1)  # (1, sum_i H_i*W_i, C)

    return pred


def demo_postprocess(outputs: np.ndarray, img_size, p6=False):
    '''
    YOLOX-style decoder that converts raw network predictions into 
    absolute image-space boxes by adding the cell grid and scaling by the output stride.
    '''
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]
    
    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs



def _nms(boxes, scores, nms_thr):
    '''
    Single class NMS implemented in Numpy.
    '''
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    '''
    Multiclass NMS implemented in Numpy.
    Class-agnostic version.
    '''
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]
    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = _nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets

