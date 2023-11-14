#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   02-softmax.py
@Time    :   2023/11/05 13:09:35
@Author  :   不要葱姜蒜
@Version :   1.0
@Desc    :   None
'''

import numpy as np

def softmax(x):
    '''
    计算softmax函数
    '''
    x = x - np.max(x) # 防止溢出, exp(x)函数可以将输入值转换为非负数，当小于等于0时，输出值为小于等于1.
    x = np.exp(x)
    x = x / np.sum(x)
    return x

if __name__ == "__main__":
    x = np.array([1,2,3,4,5])
    print(softmax(x))