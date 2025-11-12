import numpy as np 
import cv2
from python_plugin_template import AggregatorTemplate
from python_plugin_template import SimaaiPythonBuffer, MetaStruct
import gi
from typing import List, Tuple
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GObject', '2.0')
from gi.repository import Gst, GObject, GstBase

"""
To use Metadata fieds from the input buffers:  
Parse the MetaStruct object. It has the following 4 fields:  
class MetaStruct:
    def __init__(self, buffer_name, stream_id, timestamp, frame_id):
        self.buffer_name = buffer_name
        self.stream_id = stream_id
        self.timestamp = timestamp
        self.frame_id = frame_id

"""

plugin_name = "yolox_postproc_overlay"   #define PLUGIN_NAME HERE

out_size = int(1280 * 720 * 1.5)  # outsize of plugin in bytes
class MyPlugin(AggregatorTemplate):
    def __init__(self):
        self.out_size = int(1280 * 720 * 1.5)  # outsize of plugin in bytes
        super(MyPlugin, self).__init__(plugin_name=plugin_name, out_size=out_size)
        self.model_outs = [
            (80, 80, 85),
            (40, 40, 85),
            (20, 20, 85),
        ]
        
        self.color_palette = np.random.uniform(0, 255, size=(80, 3))
        self.FRAME_HEIGHT = 720
        self.FRAME_WIDTH = 1280

        self.MODEL_HEIGHT = 640
        self.MODEL_WIDTH = 640

    
    def get_model_outputs(self, input_buffer):
        start = 0
        model_outputs = []
        # first get box inputs 
        for out_shape in self.model_outs:
            seg_size = np.prod(out_shape)
            arr = input_buffer[start: start+seg_size].reshape(out_shape)
            model_outputs.append(arr)
            start += seg_size

        # YoloX has 3 outputs
        first_output = np.expand_dims(model_outputs[0], axis=0).reshape(1, -1, 85)  # (1, 80, 80, 85) -> (1, 6400, 85)
        second_output = np.expand_dims(model_outputs[1], axis=0).reshape(1, -1, 85) # (1, 40, 40, 85) -> (1, 1600, 85)
        third_output = np.expand_dims(model_outputs[2], axis=0).reshape(1, -1, 85)  # (1, 20, 20, 85) -> (1, 400, 85)

        pred = np.concatenate((first_output, second_output, third_output), axis=1)

        return pred
        
        
    def nms(self,boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
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
    
    
    def multiclass_nms(self, boxes, scores, nms_thr, score_thr, class_agnostic=True):
        """Multiclass NMS implemented in Numpy. Class-agnostic version."""
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self.nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            )
        return dets


    def demo_postprocess(self, outputs, img_size, p6=False):
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

    def run(self, input_buffers: List[SimaaiPythonBuffer], output_buffer: bytes) -> None:
        """
        Define your plugin logic HERE
        Inputs:
        input_buffers List[SimaaiPythonBuffer]: List of input buffers  
        Object of class SimaaiPythonBuffer has three fields:  
        1. metadata MetaStruct Refer to the structure above
        2. data bytes - raw bytes of the incoming buffer  
        3. size int - size of incoming buffer in bytes
        """
        input_shape =(self.MODEL_WIDTH, self.MODEL_HEIGHT)
        # ratio  = min(input_shape[0]/ self.FRAME_WIDTH  , input_shape[1]/self.FRAME_HEIGHT)

        model_buffer  = np.frombuffer(input_buffers[0].data, dtype=np.float32)
        y_size = self.FRAME_HEIGHT * self.FRAME_WIDTH
        y_ = np.frombuffer(input_buffers[1].data[:y_size], dtype=np.uint8).reshape(self.FRAME_HEIGHT, self.FRAME_WIDTH)
        uv_ = np.frombuffer(input_buffers[1].data[y_size:], dtype=np.uint8).reshape(self.FRAME_HEIGHT // 2, self.FRAME_WIDTH // 2, 2)
        output = self.get_model_outputs(model_buffer)
        predictions = self.demo_postprocess(output, (640, 640))[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        # boxes_xyxy /= ratio

        x_scale = self.FRAME_WIDTH / self.MODEL_WIDTH
        y_scale = self.FRAME_HEIGHT / self.MODEL_HEIGHT

        boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]] * x_scale
        boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]] * y_scale

        dets = self.multiclass_nms(boxes_xyxy, scores, nms_thr=0.5, score_thr=0.5)

        y_ = y_.copy()
        uv_ = uv_.copy()

        if dets is not None:
            boxes, scores, labels = dets[:, :4], dets[:, 4], dets[:, 5]
            

            #filter again
            keep = scores > 0.5
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            for i in range(len(boxes)):
                box = boxes[i]
                score = scores[i]
                class_id = int(labels[i])
                x1, y1, x2, y2 = box
                color = self.color_palette[class_id]
                color = tuple(map(int, color))

                cv2.rectangle(y_, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.rectangle(uv_, (int(x1//2), int(y1//2)), (int(x2//2), int(y2//2)), color, 2)

        output_buffer[:self.out_size] = np.concatenate([y_.flatten(), uv_.flatten()]).tobytes()


GObject.type_register(MyPlugin)
__gstelementfactory__ = (plugin_name, Gst.Rank.NONE, MyPlugin)