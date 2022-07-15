# -*- coding: utf-8 -*-
"""
@File    :   objects.py
@Time    :   2022/03/28 14:39:03
@Author  :   lijunyu
@Version :   0.0.1
@Desc    :   None
"""

from select import select
import numpy as np

class TargetBox:
    def __init__(self) -> None:
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0

        self.cate = 0
        self.score = 0.0

    def getWidth(self) -> float:
        return self.x2 - self.x1

    def getHeight(self) -> float:
        return self.y2 - self.y1

    def getArea(self) -> float:
        return self.getWidth() * self.getHeight()
