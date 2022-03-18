# -*- coding: utf-8 -*-
"""
@File    :   create_train_dataSet.py
@Time    :   2022/03/18 20:24:47
@Author  :   lijunyu
@Version :   0.0.1
@Desc    :   None
"""

import os
import shutil
import argparse
import random

VAL_DATA_RATE = 0.3

def parseArgs():
    description = "you should add those parameter"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--imgpath', type=str, default="D:\\Code\\FireAndSmokeDetect\\dataset\\VOC2020\\JPEGImages" ,help="the image's path")
    parser.add_argument('--outputpath', type=str, default="D:\\Code\\FireAndSmokeDetect\\model\\data" ,help="yolo txt file's output path")
    parser.add_argument('--label', type=str, default="D:\\Code\\FireAndSmokeDetect\\dataset\\VOC2020\\yolo_label", help="yolo label file path")
    args = parser.parse_args()
    return args

def getFilesNames(path):
    fileNames = []
    for file in os.listdir(path) :
        if os.path.splitext(file)[1] == '.txt':
            fileNames.append(file)
    return fileNames

def clearFloder(path):
    for i in os.listdir(path) :
        file_data = os.path.join(path, i)
        if os.path.isfile(file_data) == True:
            os.remove(file_data)
        else:
            clearFloder(file_data)

if __name__ == '__main__':
    # parseargs
    args = parseArgs()
    # check output dir if not then create
    trainPath = os.path.join(args.outputpath, 'train')
    valPath = os.path.join(args.outputpath, 'val')
    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)
    if not os.path.exists(trainPath):
        os.makedirs(trainPath)
    if not os.path.exists(valPath):
        os.makedirs(valPath)
    # del the train and val floder's old files
    clearFloder(trainPath)
    clearFloder(valPath)
    # create dataset
    txtFlies = getFilesNames(args.label)
    trainTxtPath = os.path.join(args.outputpath, 'train.txt')
    valTxtPath = os.path.join(args.outputpath, 'val.txt')
    trainTxt = open(trainTxtPath, "wt")
    valTxt = open(valTxtPath, "wt")
    random.shuffle(txtFlies)
    valCount = int(len(txtFlies) * 0.3)
    for txt in txtFlies:
        fileName = os.path.splitext(txt)[0]
        if valCount != 0:
            movePath = valPath
            valCount = valCount - 1
            valTxt.write(os.path.join(valPath, fileName + '.jpg') + '\n')
        else:
            movePath = trainPath
            trainTxt.write(os.path.join(trainPath, fileName + '.jpg') + '\n')
        # copy file
        shutil.copyfile(os.path.join(args.label, fileName + '.txt'), 
            os.path.join(movePath, fileName + '.txt'))
        shutil.copyfile(os.path.join(args.imgpath, fileName + '.jpg'), 
            os.path.join(movePath, fileName + '.jpg'))
    valTxt.close()
    trainTxt.close()