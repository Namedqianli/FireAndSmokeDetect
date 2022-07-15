# -*- coding: utf-8 -*-
"""
@File    :   FireDetect.py
@Time    :   2022/03/19 05:14:35
@Author  :   lijunyu
@Version :   0.0.1
@Desc    :   None
"""

from ast import Return
from unicodedata import category
import ncnn
import cv2
import numpy as np
from objects import TargetBox

class FireDetect:
    def __init__(self, param, model) -> None: 
        self.num_threads = 4
        self.num_anchor = 3
        self.num_category = 1
        self.nms_thresh = 0.25
        self.thresh = 0.3
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

    def getCategory(self, values, index):
        score = 0.0
        objScore = values[4 * self.num_anchor + index]
        category = 0

        for i in range(self.num_category):
            clsScore = values[4 * self.num_anchor + self.num_anchor + i]
            clsScore *= objScore

            if clsScore > score:
                score = clsScore
                category = i
        
        return score, category

    def preHandle(self, ncnn_out, img_w, img_h, out_c, out_h, out_w):
        scale_w = img_w / self.input_width
        scale_h = img_h / self.input_height

        objs = []
        # output1
        stride = self.input_height / out_h
        out = np.array(ncnn_out)
        print(out.shape)
        out = out.reshape(out_c, out_h * out_w)
        print(out.shape[1])
        for i in range(out.shape[1]):
            for j in range(self.num_anchor):
                values = out[:,j]
                s, c = self.getCategory(values, 0)
                if s > self.thresh:
                    h = i % out_h
                    w = i // out_h
                    bcx = ((values[j * 4 + 0] * 2.0 - 0.5) + w) * stride
                    bcy = ((values[j * 4 + 1] * 2.0 - 0.5) + h) * stride
                    bw = pow((values[j * 4 + 2] * 2.), 2) * self.anchor[(i * self.num_anchor * 2) + j * 2 + 0]
                    bh = pow((values[j * 4 + 3] * 2.), 2) * self.anchor[(i * self.num_anchor * 2) + j * 2 + 1]
                    tmpBox = TargetBox()
                    tmpBox.x1 = (bcx - 0.5 * bw) * scale_w
                    tmpBox.y1 = (bcy - 0.5 * bh) * scale_h
                    tmpBox.x2 = (bcx + 0.5 * bw) * scale_w
                    tmpBox.y2 = (bcy + 0.5 * bh) * scale_h
                    tmpBox.score = s
                    tmpBox.cate = c
                    objs.append(tmpBox)
        return objs

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
        
        objs1 = self.preHandle(out_mat1, img_w, img_h, 18, 20, 20)
        objs2 = self.preHandle(out_mat2, img_w, img_h, 18, 10, 10)
        objs = objs1 + objs2
        print('obj nums',len(objs1))

# for test
if __name__ == '__main__':
    detect = FireDetect('./model/Yolo-FastestV2-0.2/example/ncnn-python/ncnn_model/model-int8.param'
                    , './model/Yolo-FastestV2-0.2/example/ncnn-python/ncnn_model/model-int8.bin')
    img = cv2.imread('D:/Code/FireAndSmokeDetect/dataset/VOC2020/JPEGImages/00000.jpg')
    detect.detect(img)