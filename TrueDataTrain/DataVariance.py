import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import pickle
from DataSetName import *
import os

"""
This code is for extracting the REAL input data corvariance
Return a 4D-Tensor which has same definition with DynamicMeanField Moledule

              ******width*******
    height    ******************
              ******************
              left to right (j), up to down (i)

"""

def RealDataStructure(data_set, mean = None, var = None):
    # Due to the flexibility of python language, we can use dictionary to implement the switch-case
    savePath = '../' + data_set + '_DATA/'
    if not os.path.exists(savePath):
        os.mkdir(savePath)
        print("Directory " , savePath,  " Created ")
    else:    
        print("Directory " , savePath ,  " already exists, continuing ... ")
    # pre-process the image to standard format
    trans = None
    if mean is not None and var is not None:
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, var)])
    else:
        trans = transforms.Compose([transforms.ToTensor()])
    Dset = eval('dset.' + DATASET.get(data_set))
    train_set = Dset(savePath, train=True, transform=trans, download=True)
    train_data = train_set.data
    NSample = train_data.shape[0]
    Height = train_data.shape[1]
    Width = train_data.shape[2] # [3rd components will be 3 for cifa but disappear for mnist]
    NSize = Height*Width
    NChannel = DATASET_NC.get(data_set)
    MatrixC = torch.zeros(NSize*NChannel,NSize*NChannel)
    CTensor = torch.zeros(NSize,NSize,NChannel,NChannel)
    # Firstly constructing the Big Matrix || Secondly transform the Big Matrix to the 4th-order Tensor
    for i, data in enumerate(train_set, 0):
        if i%(NSample//10) == 0:
            print('Process {:.0%}  finished'.format(i/NSample))
        # data is tuple with one element, data[0]: [C * H * W] and data[1]: target index
        # print(type(data)) 
        Sample =  data[0].view(-1,NSize*NChannel)   # SHALLOW COPY
        C = torch.mm(torch.transpose(Sample,0,1),Sample)
        MatrixC += C
    MatrixC /= NSample
    # Transform Big Matrix to 4D-Tensor
    for cidx in range(NChannel):
            for i in range(Height):
                for j in range(Width):
                    # Cj loop
                    for cidx_ in range(NChannel):
                        for i_ in range(Height):
                            for j_ in range(Width):
                                Ci = cidx*NSize + i*Width + j
                                Cj = cidx_*NSize + i_*Width + j_
                                CTensor[i*Width + j][i_*Width + j_][cidx][cidx_] = MatrixC[Ci][Cj]
    return CTensor
    

#RealDataStructure('MNIST')
RealDataStructure('CIFAR10')

"""
for i, data in enumerate(train_set, 0):
        print(data[i][0])
        # PIL
        img = transforms.ToPILImage()(data[i][0])
        img.show()
        break
"""