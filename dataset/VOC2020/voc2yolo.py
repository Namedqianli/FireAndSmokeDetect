# -*- encoding: utf-8 -*-
'''
@File    :   voc2yolo.py
@Time    :   2022/03/09 20:38:56
@Author  :   lijunyu 
@Desc    :   None
'''

from cProfile import label
import os
import argparse
from turtle import width
from xml.dom import minidom

def parseArgs():
    description = "you should add those parameter"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--xmlpath', type=str, default="D:\\Code\\YOLOX\\datasets\\VOCdevkit\\VOC2022\\Annotations" ,help="the xml files's path")
    parser.add_argument('--imgpath', type=str, default="" ,help="the image's path")
    parser.add_argument('--outputpath', type=str, default="D:\\Code\\YOLOX\\datasets\\VOCdevkit\\VOC2022\\yolo\\" ,help="yolo txt file's output path")
    parser.add_argument('--label', type=str, default="D:\\Code\\YOLOX\\datasets\\VOCdevkit\\VOC2022\\ImageSets\\labels.txt", help="yolo label file path")
    args = parser.parse_args()
    return args

def getFilesNames(path):
    fileNames = []
    for file in os.listdir(path) :
        if os.path.splitext(file)[1] == '.xml':
            fileNames.append(file)
    return fileNames


def vocToYolo(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

if __name__ == '__main__':
    args = parseArgs()
    # read labels
    labels = {}
    with open(args.label, "r") as f:
        readLabels = f.readlines()
        print(len(readLabels))
        for i in range(0, len(readLabels)):
            labels[readLabels[i]] = str(i)
    print(labels)
    # check output dir if not then create
    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)
    # get all xml files from xml path
    xmlFileNames = getFilesNames(args.xmlpath)
    for xml in xmlFileNames:
        # open xml file
        dom = minidom.parse(args.xmlpath + '\\' + xml)
        annotation = dom.documentElement
        # get img size
        sizeNode = annotation.getElementsByTagName('size')[0]
        width = sizeNode.getElementsByTagName('width')[0].firstChild.data
        height = sizeNode.getElementsByTagName('height')[0].firstChild.data
        depth = sizeNode.getElementsByTagName('depth')[0].firstChild.data
        # img size [width, height, depth]
        imgSize = [int(width), int(height), int(depth)]
        objList = []
        for node in annotation.childNodes:
            if node.nodeName == 'object':
                try:
                    objectName = node.getElementsByTagName('name')[0].firstChild.data
                    bndBox = node.getElementsByTagName('bndbox')[0]
                    xMin = bndBox.getElementsByTagName('xmin')[0].firstChild.data
                    yMin = bndBox.getElementsByTagName('ymin')[0].firstChild.data
                    xMax = bndBox.getElementsByTagName('xmax')[0].firstChild.data
                    yMax = bndBox.getElementsByTagName('ymax')[0].firstChild.data
                    vocBox = [int(float(xMin)), int(float(xMax)), int(float(yMin)), int(float(yMax))]
                    x, y, w, h = vocToYolo(imgSize, vocBox)
                    labelId = labels[objectName]
                    obj = labelId + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h)
                    objList.append(obj)
                except IndexError:
                    print('index error', xml)
        print(objList)
        with open(args.outputpath + '\\' + os.path.splitext(xml)[0] + '.txt', "wt") as f: 
            for obj in objList:
                f.write(obj + '\n')
            f.close()