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
Run ONNX model

Example Usage:
    python run_onnx.py -m ./yolox_s_opt_no_reshapes.onnx -ti 10 -bd build
'''


'''
Author: SiMa Technologies
'''


import onnx
import onnxruntime as ort
import sys, shutil
import argparse
import numpy as np
from pathlib import Path
import cv2

import utils

DIVIDER = '-'*50


def implement(args):

    '''
    Prepare test data
    '''
    test_data = utils.list_image_paths(args.test_dir)
    num_test_images = min(args.num_test_images, len(test_data))

    print(f'Using {num_test_images} out of {len(test_data)}  test images',flush=True)


    '''
    Make results folder
    '''
    build_dir = Path(args.build_dir).resolve()
    annotated_images = (build_dir / 'onnx_pred').resolve()

    # If the folder exists, delete it first
    if annotated_images.exists():
        if annotated_images.is_dir():
            print(f"Removing existing directory: {annotated_images}", flush=True)
            shutil.rmtree(annotated_images)
        else:
            raise NotADirectoryError(f"Path exists but is not a directory: {annotated_images}",flush=True)

    # Create a clean folder
    annotated_images.mkdir(parents=True, exist_ok=False)
    print(f"Annotated images will be written to {annotated_images}", flush=True)


    '''
    ONNX inference
    '''
    # Load & validate ONNX model
    onnx_model = onnx.load(args.model_path)
    onnx.checker.check_model(onnx_model)

    # Create an ONNX Runtime inference session
    ort_sess = ort.InferenceSession(args.model_path)

 
    # loop over test images
    inputs = dict()
    for i in range(num_test_images):
        print(f"Processing image: {test_data[i]}") 

        filename = Path(test_data[i]).name
        img_bgr, inputs['images'] = utils.preprocess(test_data[i], 640, 640, transpose=True)
    
        # run inference - outputs a list of numpy arrays
        # outputs shapes: 1, 85, 80, 80) (1, 85, 40, 40) (1, 85, 20, 20)
        pred = ort_sess.run(None, inputs)

        # convert to NHWC
        for i,p in enumerate(pred):
            pred[i] = np.transpose(p, axes=[0, 2, 3, 1])

        output = utils.get_model_outputs(pred)
        predictions = utils.demo_postprocess(output, (640, 640))[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.

        dets = utils.multiclass_nms(boxes_xyxy, scores, nms_thr=0.5, score_thr=0.5)

        if dets is not None:
          boxes, scores, labels = dets[:, :4], dets[:, 4], dets[:, 5]

          #filter again
          keep = scores > 0.5
          boxes = boxes[keep]
          scores = scores[keep]
          labels = labels[keep]

          # draw boxes on the image
          for i in range(len(boxes)):
              box = boxes[i]
              #score = scores[i]
              class_id = int(labels[i])
              x1, y1, x2, y2 = box
              color = utils.color_palette[class_id]
              color = tuple(map(int, color))

              cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # write the image out
        file_path= f'{annotated_images}/test_{filename}' 
        ok = cv2.imwrite(file_path, img_bgr)
        if ok:
           print(f'Wrote output image to {file_path}',flush=True)
        else:
           print('ERROR: Image write failed!',flush=True)


 

    return



def run_main():
  
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-bd', '--build_dir',         type=str, default='build', help='Path of build folder. Default is build')
    ap.add_argument('-m',  '--model_path',        default='./yolox_s_opt_no_reshapes.onnx', type=str, help='path to ONNX model')
    ap.add_argument('-td', '--test_dir',          type=str, default='./test_images', help='Path to test images folder. Default is ./test_images')
    ap.add_argument('-ti', '--num_test_images',   type=int, default=10, help='Number of test images. Default is 10')
    args = ap.parse_args()

    # print Python version
    print('\n'+DIVIDER,flush=True)
    print(sys.version,flush=True)
    print(DIVIDER,flush=True)


    implement(args)


if __name__ == '__main__':
    run_main()