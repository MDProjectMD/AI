import numpy as np
import matplotlib.pyplot as plt
import sys

# Moledule used for linking the Full Connected network and CNN
# shrink the last layer from CNN to 1D neurons shape  
def LinkCNN2FC(net_cnn):
    # self.h_mean[alpha][c][0] alpha is feature map position and c is channel index \\ see DynamicMeanField.py for the definition of self.mean
    # MUST BE CALLED AFTER CNN's dimensionality being CALCULATED !!!
    C_fc = net_cnn.CBigMat.copy()
    # reshape the self.h_mean to column vector
    num_channel_cnn = net_cnn.struct[-1]
    edge_size_cnn2 = net_cnn.CTensor.shape[0]
    m_fc = np.zeros([edge_size_cnn2*num_channel_cnn,1])
    edge_size_cnn = int(np.sqrt(edge_size_cnn2))
    for c in range(num_channel_cnn):
        for i in range(edge_size_cnn):
            for j in range(edge_size_cnn):
                idx = c*edge_size_cnn2 + i*edge_size_cnn + j
                m_fc[idx][0] = net_cnn.h_mean[i*edge_size_cnn + j][c][0]
    return m_fc,C_fc

