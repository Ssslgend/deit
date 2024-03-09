
import os
import shutil
import torch
# check you have the right version of timm
import timm

# assert timm.__version__ == "0.3.2"
import  torch
import time
import pandas as pd
import torchvision.transforms as transforms
import torchvision
# from tdqm import tdqm
def getdataSet(root):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    )

    data = torchvision.datasets.CIFAR10(root='F:\datasets01\\archive', train=True, download=True,
                               transform=transform_train)

    return  data

def evluate_time(data):
    res_dict = []
    for bs in [32, 64, 128, 256]:
        for num_workers in [0, 2, 4, 8]:
            dl = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=False, num_workers=num_workers,
                                             pin_memory=True)
            start = time.time()

            c = 0
            for x, y in dl:
                c += len(x)
                if c > 1000:
                    break

            end_time = time.time()
            r = {'bs': bs, 'num_workers': num_workers, 'time': end_time - start / c * len(data)
                 }
            print(r)
            res_dict.append(r)

    pd.DataFrame(res_dict).to_csv('F:\datasets01\\archive\\res_dict.csv', index=False)


if __name__ == '__main__':
#     # 定义路径
#     path = 'F:\\datasets01\\imagenet2012task3\\n02102040\\train'
# # 检查路径是否存在
#     print(os.path.exists(path))
# # 获取数据集
#     dataset=getdataSet(path)
#
# # 打印数据集
#     print(dataset)


# now load it with torchhub
#     model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
#     print(model)

    print(torch.cuda.is_available())