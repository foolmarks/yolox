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
Author: Mark Harvey
'''


import os, sys, shutil
import argparse
import numpy as np
from pathlib import Path
import cv2
from typing import List, Tuple

import utils


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf

# Palette-specific imports
from afe.apis.error_handling_variables import enable_verbose_error_messages
from afe.apis.release_v1 import get_model_sdk_version
from afe.core.utils import length_hinted
from afe.apis.model import Model


DIVIDER = '-'*50


def implement(args):

    enable_verbose_error_messages()

    '''
    Make results folder
    '''
    build_dir = Path(args.build_dir).resolve()
    annotated_images = (build_dir / 'accel_pred').resolve()

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
    load quantized model
    '''
    model_path = f'{args.build_dir}/{args.model_name}'
    print(f'Loading {args.model_name} quantized model from {model_path}',flush=True)
    quant_model = Model.load(f'{args.model_name}.sima', model_path)




    '''
    Prepare test data
      - create list of dictionaries
      - Each dictionary key is an input name, value is a preprocessed data sample
    '''
    image_files = utils.list_image_paths(args.test_dir)
    num_test_images = min(args.num_test_images, len(image_files))

    print(f'Using {num_test_images} out of {len(image_files)}  test images',flush=True)

    test_data=[]
    original_images=[]
    for i in range(num_test_images):
        inputs = dict()
        img_bgr, inputs['images'] = utils.preprocess(image_files[i], 640, 640)
        original_images.append(img_bgr)
        test_data.append(inputs)


    '''
    Run in accel mode
    Returns a list of lists of np arrays
    Outer list length = num_test_images
    Inner list lengths = number of model outputs = 3
    Np array shapes = (1, 80, 80, 85) (1, 40, 40, 85) (1, 20, 20, 85)
    '''   
    pred = quant_model.execute_in_accelerator_mode(input_data=length_hinted(num_test_images, test_data),
                                                   devkit=args.hostname,
                                                   username=args.username,
                                                   password=args.password)

    print("Model is executed in accelerator mode.",flush=True)


    '''
    Evaluate results
    '''
    color_palette = np.random.uniform(0, 255, size=(80, 3))
    for i in range(num_test_images):

        print(f"Processing image: {image_files[i]}") 
        filename = Path(image_files[i]).name
        img_bgr = original_images[i]

        output = utils.get_model_outputs(pred[i])
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
              color = color_palette[class_id]
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
    ap.add_argument('-bd', '--build_dir',       type=str, default='build', help='Path of build folder. Default is build')
    ap.add_argument('-m',  '--model_name',      type=str, default='yolox_s_opt_no_reshapes', help='quantized model name')
    ap.add_argument('-td', '--test_dir',        type=str, default='./test_images', help='Path to test images folder. Default is ./test_images')
    ap.add_argument('-ti', '--num_test_images', type=int, default=10, help='Number of test images. Default is 10')
    ap.add_argument('-u',  '--username',        type=str, default='sima', help='Target device user name. Default is sima')
    ap.add_argument('-p',  '--password',        type=str, default='edgeai', help='Target device password. Default is edgeai')
    ap.add_argument('-hn', '--hostname',        type=str, default='192.168.1.29', help='Target device IP address. Default is 192.168.1.29')
    args = ap.parse_args()

    print('\n'+DIVIDER,flush=True)
    print('Model SDK version',get_model_sdk_version())
    print(sys.version,flush=True)
    print(DIVIDER,flush=True)


    implement(args)


if __name__ == '__main__':
    run_main()