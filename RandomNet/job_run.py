from RandomNet import *
import numpy as np
import time


# initialize random network
net_struct = [1,3,3]
net = RandomNet(net_struct,2)
start = time.time()
DIM = RandomNetDim(net, GaussianRandomInputCD, ensemble_num = 7000)
end = time.time()
print('CPU run time: {0} h {1} min {2} seconds'.format((end-start)//60//60,(end-start)//60%60,(end-start)%60%60))

for l in range(len(DIM)):
    print('Layer {0} dimensions: '.format(l))
    for c in range(len(DIM[l])):
        print('Channel {0} dimensions: {1}'.format(c,DIM[l][c][0]))