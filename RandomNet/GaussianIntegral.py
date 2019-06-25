import numpy as np
import math

def NormalIntegral1D(func, N = 10000):
    x = np.random.randn(1,N)
    y = func(x)
    integral = y.sum()/N
    return integral

def NormalIntegral2D(func, N = 15000):
    x = np.random.randn(1,N)
    y = np.random.randn(1,N)
    z = func(x,y)
    integral = z.sum()/N
    return integral

# test samples
"""
def f1(x):
    return x*x

def f2(x,y):
    return x*x*y*y

print(NormalIntegral1D(f1))
print(NormalIntegral2D(f2))
"""