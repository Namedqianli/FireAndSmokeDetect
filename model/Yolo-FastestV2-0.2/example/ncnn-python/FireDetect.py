# -*- coding: utf-8 -*-
"""
@File    :   FireDetect.py
@Time    :   2022/03/19 05:14:35
@Author  :   lijunyu
@Version :   0.0.1
@Desc    :   None
"""

import ncnn
import cv2
import numpy as np

class FireDetect:
    def __init__(self, param, model) -> None: 
        self.net = ncnn.Net()
        self.net.load_param(param)
        self.net.load_model(model)
        self.input_names = self.net.input_names()
        self.output_names = self.net.output_names()
        self.alloctor = ncnn.PoolAllocator()

    def detect(self, in_mat):
        objs = []
        with self.net.create_extractor() as ex:
            ex.set_num_threads(2)
            ex.set_blob_allocator(self.alloctor)
            ex.set_workspace_allocator(self.alloctor)
            ex.input(self.input_names[0], in_mat)
            ret, out_mat1 = ex.extract(self.output_names[0])
            ret, out_mat2 = ex.extract(self.output_names[1])

# for test
if __name__ == '__main__':
    detect = FireDetect('./ncnn_model/model-opt.param', './ncnn_model/model-opt.bin')
    img = cv2.imread('D:/Code/FireAndSmokeDetect/dataset/VOC2020/JPEGImages/00000.jpg')
    img = cv2.resize(img, (320, 320))
    img_ = img[:,:,::-1].transpose((2,0,1))
    in_mat = ncnn.Mat(img_)
    detect.detect(in_mat)
    cv2.waitKey()