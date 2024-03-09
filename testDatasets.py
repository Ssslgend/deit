#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：deit 
@File    ：testDatasets.py
@Author  ：songliqiang
@Date    ：2024/3/6 16:28 
'''

import torch
import torchvision.transforms as transforms
import torchvision


from torchvision import datasets


def GetDataSet(path):
    # 定义数据预处理的转换器
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为224x224
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像
    ])

    train_Set = datasets.ImageFolder(root=path, transform=transform)
    print(type(train_Set))
    return train_Set







if __name__ == "__main__":
    root = 'F:\datasets01\imagenet2012task3'
    train_set =  GetDataSet(root)
    train_set.class_to_idx
    print(train_set.class_to_idx)
    print(train_set)
    pass