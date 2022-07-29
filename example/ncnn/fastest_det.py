# -*- coding: utf-8 -*-
import ncnn
import cv2
import numpy as np

class FastestDet():
    def __init__(self, param, model, input_width=352, input_height=352) -> None:
        self.input_width=input_width
        self.input_height=input_height
        self.num_threads = 4
        self.thresh = 0.6
        self.net = ncnn.Net()
        self.net.load_param(param)
        self.net.load_model(model)
        self.input_names = self.net.input_names()
        self.output_names = self.net.output_names()

    # sigmoid函数
    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    # tanh函数
    def tanh(self, x):
        return 2. / (1 + np.exp(-2 * x)) - 1

    def preprocess(self, src_img, size):
        output = cv2.resize(src_img,(size[0], size[1]),interpolation=cv2.INTER_AREA)
        output = output.transpose(2,0,1)
        output = output.reshape((1, 3, size[1], size[0])) / 255

        return output.astype('float32')

    def nms(self, dets, thresh=0.4):
        # dets:N*M,N是bbox的个数，M的前4位是对应的（x1,y1,x2,y2），第5位是对应的分数
        # #thresh:0.3,0.5....
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 求每个bbox的面积
        order = scores.argsort()[::-1]  # 对分数进行倒排序
        keep = []  # 用来保存最后留下来的bboxx下标

        while order.size > 0:
            i = order[0]  # 无条件保留每次迭代中置信度最高的bbox
            keep.append(i)

            # 计算置信度最高的bbox和其他剩下bbox之间的交叉区域
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # 计算置信度高的bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            # 求交叉区域的面积占两者（置信度高的bbox和其他bbox）面积和的必烈
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # 保留ovr小于thresh的bbox，进入下一次迭代。
            inds = np.where(ovr <= thresh)[0]

            # 因为ovr中的索引不包括order[0]所以要向后移动一位
            order = order[inds + 1]
        
        output = []
        for i in keep:
            output.append(dets[i].tolist())

        return output

    def detection(self, img):
        H = img.shape[0]
        W = img.shape[1]

        data = self.preprocess(img, [self.input_width, self.input_height])

        data_mat = ncnn.Mat(data)

        ex = self.net.create_extractor()
        ex.set_num_threads(self.num_threads)

        mat_in = ncnn.Mat.from_pixels_resize(
            img,
            ncnn.Mat.PixelType.PIXEL_BGR,
            img.shape[1],
            img.shape[0],
            self.input_width,
            self.input_height,
        )
        mat_in.substract_mean_normalize([0.0, 0.0, 0.0], [1.0/255.0, 1.0/255.0, 1.0/255.0])

        ex.input(self.input_names[0], mat_in)

        ret, out_mat = ex.extract(self.output_names[0])
        feature_map = np.array(out_mat)

        # 输出特征图转置: CHW, HWC
        feature_map = feature_map.transpose(1, 2, 0)
        print(feature_map.shape)
        # 输出特征图的宽高
        feature_map_height = feature_map.shape[0]
        feature_map_width = feature_map.shape[1]

        # 特征图后处理
        pred = []
        for h in range(feature_map_height):
            for w in range(feature_map_width):
                data = feature_map[h][w]

                # 解析检测框置信度
                obj_score, cls_score = data[0], data[5:].max()
                score = (obj_score ** 0.6) * (cls_score ** 0.4)

                # 阈值筛选
                if score > self.thresh:
                    # 检测框类别
                    cls_index = np.argmax(data[5:])
                    # 检测框中心点偏移
                    x_offset, y_offset = self.tanh(data[1]), self.tanh(data[2])
                    # 检测框归一化后的宽高
                    box_width, box_height = self.sigmoid(data[3]), self.sigmoid(data[4])
                    # 检测框归一化后中心点
                    box_cx = (w + x_offset) / feature_map_width
                    box_cy = (h + y_offset) / feature_map_height
                    
                    # cx,cy,w,h => x1, y1, x2, y2
                    x1, y1 = box_cx - 0.5 * box_width, box_cy - 0.5 * box_height
                    x2, y2 = box_cx + 0.5 * box_width, box_cy + 0.5 * box_height
                    x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)

                    pred.append([x1, y1, x2, y2, score, cls_index])
        if len(pred) == 0:
            return []
        return self.nms(np.array(pred))
    
    def draw_frame(self, img, bboxes):
        for b in bboxes:
            print(b)
            obj_score, cls_index = b[4], int(b[5])
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])

            #绘制检测框
            cv2.rectangle(img, (x1,y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
            cv2.putText(img, 'fire', (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
        return img

if __name__ == "__main__":
    detect = FastestDet("./FastestDet-opt.param", "FastestDet-opt.bin")
    img = cv2.imread("./00000.jpg")
    res = detect.detection(img)
    img = detect.draw_frame(img, res)
    cv2.imshow('result', img)
    cv2.waitKey()
    print(res)