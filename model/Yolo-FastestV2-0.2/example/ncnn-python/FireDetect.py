# -*- coding: utf-8 -*-
"""
@File    :   FireDetect.py
@Time    :   2022/03/19 05:14:35
@Author  :   lijunyu
@Version :   0.0.1
@Desc    :   None
"""

from ast import Return
import ncnn
import cv2
import numpy as np

class FireDetect:
    def __init__(self, param, model) -> None: 
        self.num_threads = 4
        self.num_anchor = 3
        self.num_category = 1
        self.nms_thresh = 0.25
        self.input_width = 320
        self.input_height = 320
        self.anchor = [20.36,31.76, 54.63,63.03, 82.37,123.16, 134.47,199.35, 177.87,96.39, 255.96,226.69]

        self.mean_vals = [0.0, 0.0, 0.0]
        self.norm_vals = [1.0/255.0, 1.0/255.0, 1.0/255.0]

        self.net = ncnn.Net()
        self.net.load_param(param)
        self.net.load_model(model)
        self.input_names = self.net.input_names()
        self.output_names = self.net.output_names()

    def preHandle():
        Return

    def detect(self, img):
        img_h = img.shape[0]
        img_w = img.shape[1]

        mat_in = ncnn.Mat.from_pixels_resize(
            img,
            ncnn.Mat.PixelType.PIXEL_BGR,
            img.shape[1],
            img.shape[0],
            self.input_width,
            self.input_height,
        )
        mat_in.substract_mean_normalize([], self.norm_vals)
        mat_in.substract_mean_normalize(self.mean_vals, [])

        ex = self.net.create_extractor()
        ex.set_num_threads(self.num_threads)

        ex.input(self.input_names[0], mat_in)

        ret, out_mat1 = ex.extract(self.output_names[0])
        ret, out_mat2 = ex.extract(self.output_names[1])

# for test
if __name__ == '__main__':
    detect = FireDetect('./model/Yolo-FastestV2-0.2/example/ncnn-python/ncnn_model/model-int8.param'
                    , './model/Yolo-FastestV2-0.2/example/ncnn-python/ncnn_model/model-int8.bin')
    img = cv2.imread('D:/Code/FireAndSmokeDetect/dataset/VOC2020/JPEGImages/00000.jpg')
    detect.detect(img)