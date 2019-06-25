import numpy as np
import sys
import os

def ConvMult(Mat1, Mat2, bias = 0, stride = 1):
    # Mat2 is kernel(small square matrix) and Mat1 is input
    kernel_size = Mat2.shape[0]
    out_size = (Mat1.shape[0] - kernel_size)//stride + 1
    if (Mat1.shape[0] - kernel_size)%stride:
        try:
            sys.exit(0)
        except:
            print('Convolution Matrix stride is not valid!')
            os._exit(0)
    Mat3 = np.zeros((out_size,out_size))
    for i in range(out_size):
        for j in range(out_size):
            mat1_sub = Mat1[stride*i:stride*i+kernel_size,j*stride:j*stride+kernel_size]
            Mat3[i][j] = np.sum(mat1_sub*Mat2 + bias)
    return Mat3
    
"""
# test sample
Mat1 = np.array([(1,2,3,4,5),(2,3,4,5,6),(3,4,5,6,7),(4,5,6,7,8),(5,6,7,8,9)])
Mat2 = np.array([(1,1),(1,1)])
print('Mat1')
print(Mat1)
print('kernel')
print(Mat2)
Mat = ConvMult(Mat1,Mat2)
print('Conv result')
print(Mat)
"""
"""
Mat1 = np.array([(1,2,3,4,5,6),(2,3,4,5,6,7),(3,4,5,6,7,8),(4,5,6,7,8,9),(5,6,7,8,9,0),(6,7,8,9,0,1)])
Mat2 = np.array([(1,1),(1,1)])
print('Mat1')
print(Mat1)
print('kernel')
print(Mat2)
Mat = ConvMult(Mat1,Mat2,2)
print('Conv result')
print(Mat)
"""

