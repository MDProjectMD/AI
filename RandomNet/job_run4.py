from DynamicMeanField import *
from FullConnectedNet import *
from LinkFCCNN import *

# initialize random network
net_struct_cnn = [3,4,4]
net_cnn = DMFNet(net_struct_cnn,3,input_size = 10,detailed_info = True, eigen_check = True)
lengthOfLayers = len(net_struct_cnn)

# feed forward [lengthOfLayers-1] steps
# print out the heat graph of Corvariance if needed by adding the 'PrintCorvarianceHeatMap()' function
for l in range(lengthOfLayers-1):
    net_cnn.UpdateDimensionality()
    #net.PrintCorvarianceHeatMap()
    net_cnn.UpdateDeltaTensor()
    net_cnn.IterateOneLayer()
net_cnn.UpdateDimensionality()
#net.PrintCorvarianceHeatMap()

m_fc,cor_fc = LinkCNN2FC(net_cnn)
net_struct_fc = [m_fc.shape[0],64,10]
net_fc = FCNet(net_struct_fc,input_cor = cor_fc, input_m = m_fc, detailed_info = True)
lengthOfLayers = len(net_struct_fc)

# feed forward [lengthOfLayers-1] steps
for l in range(lengthOfLayers-1):
    net_fc.UpdateDimensionality()
    net_fc.UpdateDeltaTensor()
    net_fc.IterateOneLayer()
net_fc.UpdateDimensionality()

# print out the dimensionality info
net_cnn.PrintDimInfo()
net_fc.PrintDimInfo()
