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
Quantize, evaluate and compile model
'''


'''
Author: SiMa Technologies 
'''


import onnx
import sys, shutil
import argparse
import numpy as np
import tarfile 
import logging
import cv2
from pathlib import Path
from typing import List, Dict

import dataclasses

import utils


# Palette-specific imports
from afe.load.importers.general_importer import ImporterParams, onnx_source
from afe.apis.defines import default_quantization, gen1_target, gen2_target, CalibrationMethod
from afe.ir.tensor_type import ScalarType
from afe.apis.loaded_net import load_model
from afe.apis.error_handling_variables import enable_verbose_error_messages
from afe.apis.release_v1 import get_model_sdk_version
from afe.core.utils import length_hinted


DIVIDER = '-'*50


def _get_onnx_input_shapes_dtypes(model_path):
    """
    Load an ONNX model and return two dictionaries describing its *true* inputs,
    ignoring any graph initializers (weights/biases).

    Returns:
        shapes_by_input:
            { input_name: (d0, d1, ...) } where each dimension (dn) is:
              - int for fixed sizes,
              - str for symbolic dimensions (e.g., "batch", "N"),
              - None if the dimension is present but unknown,
              - or the entire value can be None if the tensor is rank-unknown.
        dtypes_by_input:
            { input_name: dtype } where:
              - if the ONNX dtype is float32 -> the value is the symbol ScalarType.float32
              - otherwise -> the original NumPy-style dtype string (e.g., 'float16', 'int64')
              - or None if it could not be determined.
    """
    # Parse and sanity-check the model graph structure.
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    # Filter out parameters that appear as graph inputs.
    initializer_names = {init.name for init in model.graph.initializer}

    # Plain dictionaries
    shapes_by_input = {}
    dtypes_by_input = {}

    # Iterate over declared graph inputs
    for vi in model.graph.input:
        if vi.name in initializer_names:
            continue  # not a real runtime input

        # Only handle tensor inputs
        if not vi.type.HasField("tensor_type"):
            continue

        ttype = vi.type.tensor_type

        # ----- dtype -----
        elem_type = ttype.elem_type
        np_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(elem_type, None)

        if np_dtype is None:
            dtypes_by_input[vi.name] = None
        else:
            dtype_name = np_dtype.name  # e.g., 'float32', 'int64'
            if dtype_name == 'float32':
                dtypes_by_input[vi.name] = ScalarType.float32
            else:
                dtypes_by_input[vi.name] = dtype_name
                print(f'Warning - input {vi.name} is not float32')

        # ----- shape -----
        if not ttype.HasField("shape"):
            shapes_by_input[vi.name] = None  # rank-unknown
            continue

        dims_list = []
        for d in ttype.shape.dim:
            if d.HasField("dim_value"):
                dims_list.append(int(d.dim_value))       # fixed dimension
            elif d.HasField("dim_param"):
                dims_list.append(d.dim_param)            # symbolic dimension
            else:
                dims_list.append(None)                   # unknown dimension

        # Store as immutable tuple
        shapes_by_input[vi.name] = tuple(dims_list)

    return shapes_by_input, dtypes_by_input





def _data_prep(folder_path: str, num_images: int, input_shapes_dict: Dict) -> List[Dict]:
  '''
  Prepare data
  '''
  processed_data =[]
  image_list = utils.list_image_paths(folder_path)
  num_images = min(num_images, len(image_list))

  # make a list of dictionaries
  # key = input name, value = pre-processed data
  for input_name in input_shapes_dict.keys():
    inputs = dict()
    for i in range(num_images):
      _, inputs[input_name] = utils.preprocess(image_list[i], input_shapes_dict[input_name][2], input_shapes_dict[input_name][3])
      processed_data.append(inputs)

  return processed_data





def implement(args):

  # enable verbose error messages.
  enable_verbose_error_messages()


  '''
  Make results folder
  '''
  # Derive directory name from model filename (without extension)
  output_model_name = Path(args.model_path).stem

  build_dir = Path(args.build_dir).resolve()
  results_dir = (build_dir / output_model_name).resolve()

  # If the folder exists, delete it first
  if results_dir.exists():
    if results_dir.is_dir():
        print(f"Removing existing directory: {results_dir}", flush=True)
        shutil.rmtree(results_dir)
    else:
        raise NotADirectoryError(f"Path exists but is not a directory: {results_dir}")

  # Create a clean folder
  results_dir.mkdir(parents=True, exist_ok=False)
  print(f"Results will be written to {results_dir}", flush=True)




  
  '''
  Load the floating-point ONNX model
  input types & shapes are dictionaries
  input types dictionary: each key,value pair is an input name (string) and a type
  input shapes dictionary: each key,value pair is an input name (string) and a shape (tuple)
  '''
  input_shapes_dict, input_types_dict = _get_onnx_input_shapes_dtypes(args.model_path)
  print(DIVIDER)
  print('Model Inputs:')
  for name, dims in input_shapes_dict.items():
     print(f'{name}: {dims}')
  print(DIVIDER)
     
  # importer parameters
  importer_params: ImporterParams = onnx_source(model_path=args.model_path,
                                                shape_dict=input_shapes_dict,
                                                dtype_dict=input_types_dict)
  

  # load ONNX floating-point model into SiMa's LoadedNet format
  target = gen2_target if args.generation == 2 else gen1_target
  loaded_net = load_model(importer_params,target=target)
  print(f'Loaded model from {args.model_path}',flush=True)



  '''
  Prepare calibration data
    - create list of dictionaries
    - Each dictionary key is an input name, value is a preprocessed data sample
  '''
  calib_data = _data_prep(args.calib_dir, args.num_calib_images, input_shapes_dict)


  '''
  Quantize
  '''
  print(f'Quantizing with {len(calib_data)} calibration samples',flush=True)

  calibration_method = CalibrationMethod.from_str(args.calib_method)
  quant_config = dataclasses.replace(default_quantization, calibration_method=calibration_method)


  quant_model = loaded_net.quantize(calibration_data=length_hinted(len(calib_data),calib_data),
                                    quantization_config=quant_config,
                                    model_name=output_model_name,
                                    log_level=logging.WARN)

  # optional save of quantized model - saved model can be opened with Netron
  quant_model.save(model_name=output_model_name, output_directory=results_dir)
  print(f'Quantized model saved to {results_dir}/{output_model_name}.sima.json',flush=True)



  '''
  Execute, evaluate quantized model
  '''
  if (args.evaluate):

    annotated_images = (build_dir / 'quant_pred').resolve()

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



    color_palette = np.random.uniform(0, 255, size=(80, 3))

    test_data = utils.list_image_paths(args.test_dir)
    num_images = min(args.num_test_images, len(test_data))

    inputs = dict()
    for input_name in input_shapes_dict.keys():
      for i in range(num_images):      
        print(f"Processing image: {test_data[i]}") 

        filename = Path(test_data[i]).name
        img_bgr, inputs[input_name] = utils.preprocess(test_data[i], input_shapes_dict[input_name][2], input_shapes_dict[input_name][3])

        # output shapes [0]: 1,80,80,85   [1]:1,40,40,85  [2]: 1,20,20,85
        quantized_net_output = quant_model.execute(inputs, fast_mode=True)

        output = utils.get_model_outputs(quantized_net_output)
        predictions = utils.demo_postprocess(output, (640, 640))[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        # boxes_xyxy /= ratio

        #x_scale = FRAME_WIDTH / MODEL_WIDTH
        #y_scale = FRAME_HEIGHT / MODEL_HEIGHT

        #boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]] * x_scale
        #boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]] * y_scale

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


  '''
  Compile
  '''
  print(f'Compiling with batch size set to {args.batch_size}',flush=True)
  quant_model.compile(output_path=results_dir,
                      batch_size=args.batch_size,
                      log_level=logging.WARN)  

  print(f'Wrote compiled model to {results_dir}/{output_model_name}_mpk.tar.gz',flush=True)

  with tarfile.open(f'{results_dir}/{output_model_name}_mpk.tar.gz') as tar:
     tar.extract(f'{output_model_name}_mpk.json' ,f'{results_dir}/benchmark')
     tar.extract(f'{output_model_name}_stage1_mla.elf' ,f'{results_dir}/benchmark')


  return



def run_main():
  
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-bd', '--build_dir',         type=str, default='build', help='Path of build folder. Default is build')
  ap.add_argument('-m',  '--model_path',        type=str, default='yolox_s_opt_no_reshapes.onnx', help='path to FP model')
  ap.add_argument('-b',  '--batch_size',        type=int, default=1, help='requested batch size. Default is 1')
  ap.add_argument('-om', '--output_model_name', type=str, default='yolox_s_opt_no_reshapes', help="Output model name. Default is yolox_s_opt_no_reshapes")
  ap.add_argument('-cd', '--calib_dir',         type=str, default='./calib_images', help='Path to calib images folder. Default is ./calib_images')
  ap.add_argument('-td', '--test_dir',          type=str, default='./test_images', help='Path to test images folder. Default is ./test_images')
  ap.add_argument('-ci', '--num_calib_images',  type=int, default=50, help='Number of calibration images. Default is 50')
  ap.add_argument('-ti', '--num_test_images',   type=int, default=10, help='Number of test images. Default is 10')
  ap.add_argument('-g',  '--generation',        type=int, default=2, choices=[1,2], help='Target device: 1 = DaVinci, 2 = Modalix. Default is 2')
  ap.add_argument('-e',  '--evaluate',          action="store_true", default=False, help="If set, evaluate the quantized model") 
  ap.add_argument('-cm',  '--calib_method',     type=str, default='min_max', choices=['mse','min_max','moving_average','entropy','percentile'], help="Calibration method. Default is min_max") 
  args = ap.parse_args()

  print('\n'+DIVIDER,flush=True)
  print('Model SDK version',get_model_sdk_version())
  print(sys.version,flush=True)
  print(DIVIDER,flush=True)


  implement(args)


if __name__ == '__main__':
    run_main()