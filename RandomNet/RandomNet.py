from Convolution import ConvMult
import numpy as np
import sys
import os

class RandomNet:
    # input layer is marked by index 0 and weight/bias/activation lists's first element(index 0) is all labeled by 'layer0'
    # for consistence 
    # if weight/bias are not assigned then it will be controlled by the shared parameters of idd gaussian variables
    # if weight/bias are given they should have the following data structure:
    #                   weight[layer idx][Cout][Cin]    /    bias[layer idx][Cout]  elements are list structure and have one array element
    #                   weight/bias are outer list consists of dependent list elements
    # default parameters
    mu_w = 0
    g_w = 0.8
    mu_b = 0
    sigma_b = 0.1
    def __init__(self, struct, kernel_size, stride = 1, phi = np.tanh,weight=None, bias=None):
        # struct:   list for number of channels in each layer
        # mu and sigma should have the same shape with self.weight
        self.kernel_size = kernel_size
        self.layer_len = len(struct)
        self.Phi = phi
        self.struct = struct
        self.stride = stride
        if weight == None:
            self.weight = []
            self.weight.append('layer0')
            # weight and bias list index start from 1
            for n in range(self.layer_len - 1):
                C_in = struct[n]
                C_out = struct[n+1]
                weightPerLayer = [[[] for j in range(C_in)] for i in range(C_out)]
                self.weight.append(weightPerLayer) 
            self.initialWeight(struct)
        else:
            self.weight = weight
        if bias == None:
            self.bias = []
            self.bias.append('layer0')
            for n in range(self.layer_len - 1):
                C_out = struct[n+1]
                biasPerLayer = [[] for j in range(C_out)]
                self.bias.append(biasPerLayer)
            self.initialBias(struct)
        else:
            self.bias = bias
        
        self.activation = []
        self.activation.append('layer0')

    def initialWeight(self,struct):
        # return weight matrix of each (C_in,C_out) channel pair
        # lidx: layer index     cidx:   channel idx
        for n in range(self.layer_len - 1):
            C_in = struct[n]
            C_out = struct[n+1]
            for i in range(C_out):
                for j in range(C_in):
                    # weight[layer idx][Cout][Cin]
                    self.weight[n+1][i][j].append(np.random.randn(self.kernel_size,self.kernel_size)*(np.sqrt(RandomNet.g_w/(C_in*self.kernel_size*self.kernel_size))))

    def initialBias(self,struct):
        for n in range(self.layer_len - 1):
            C_out = struct[n+1]
            for j in range(C_out):
                self.bias[n+1][j].append(np.random.randn(self.kernel_size,self.kernel_size)*np.sqrt(RandomNet.sigma_b))
    
    def forward(self,x_in): # further modified for multi-channel input
        x_prev_layer = []
        x_current_layer = []
        x_prev_layer.append(x_in)
        
        for n in range(1,self.layer_len):
            for j in range(len(self.bias[n])):
                actiPerChannel = 0
                for i in range(len(x_prev_layer)):
                    actiPerChannel = actiPerChannel + ConvMult(x_prev_layer[i],self.weight[n][j][i][0],self.bias[n][j][0]/len(x_prev_layer),self.stride)
                x_current_layer.append(self.Phi(actiPerChannel)) # add in different channel's data
            self.activation.append(x_current_layer.copy())
            x_prev_layer.clear()
            x_tmp = x_prev_layer 
            x_prev_layer = x_current_layer
            x_current_layer = x_tmp
        
    def getActivationOfLayer(self,lidx):
        # return the list of array activations at layer [lidx]
        if lidx >= self.layer_len:
            try:
                sys.exit(0)
            except:
                print('Layer index exceeds the boundary!')
                os._exit(0)
        return self.activation[lidx].copy()
    
    def getActivationAtLayerChannel(self,lidx,cidx):
        # return the array of activation at layer [lidx] and channel [cidx]
        if lidx >= self.layer_len:
            try:
                sys.exit(0)
            except:
                print('Layer index exceeds the boundary!')
                os._exit(0)
        if cidx >= len(self.activation[lidx]):
            try:
                sys.exit(0)
            except:
                print('Channel index exceeds the boundary!')
                os._exit(0)
        return self.activation[lidx][cidx].copy()

# Gaussian correlation input matrix
"""
Steps:
input vector x [N*N]    Corvariance matrix [N*N,N*N] = <x[k]x[j]> = r_kj = o_kj/N
o_kj is even random number between [-rho,rho] 
reshape as [1,N*N]
            x[k] = sqrt(1-r_kj(j!=k))*t_kk + {sqrt(r_kj)*t_kj where j!=k} = sqrt(1-A1)*t_kk + A2
            x[j] = sqrt(1-r_kj(k!=j))*t_jj + {sqrt(r_kj)*t_kj where k!=j} = sqrt(1-B1)*t_jj + B2
Function input: edge size N | radius rho
"""
def GaussianRandomInput(N = 30, rho = None):
    if rho == None:
        rho = 0.05 * N
    N2 = N*N
    # corvariance matrix [-1,1]
    CorMat = (np.random.rand(N2,N2)-np.random.rand(N2,N2))*rho/N2
    # symmetrization
    CorMat = (CorMat + CorMat.T)/2
    CorMatABS = np.absolute(CorMat)
    
    gaussian_t_mat = np.random.randn(N2,N2)
    # symmetrization
    for j in range(N2-1):
        for i in range(j+1,N2):
            gaussian_t_mat[i][j] = gaussian_t_mat[j][i]
    x = np.zeros([1,N2])
    for k in range(N2):
        # generate each input element
        A1 = 0
        A2 = 0
        for j in range(N2):
            if j is not k:
                A1 += CorMatABS[k][j]
        for j in range(N2):
            if j is not k:
                A2 += np.sqrt(CorMatABS[k][j])*gaussian_t_mat[k][j]
        x[0][k] = np.sqrt(1-A1)*gaussian_t_mat[k][k] + A2
    return x.reshape(N,N)
    
def GaussianRandomInputCD(C, F, N = 30, rho = None):
    # use Cholesky Decomposition Method
    #   Cor = L*L.T and multi correlated gaussian samples are L*z
    if rho == None:
        rho = 0.05 * N
    N2 = N*N
    # corvariance matrix [-1,1]
    CorMat = (np.random.rand(N2,N2)-np.random.rand(N2,N2))*rho/N2#np.sqrt(C*F*F)
    # symmetrization
    CorMat = (CorMat + CorMat.T)/2
    # TEST,TEST_VEC = np.linalg.eig(CorMat)
    CDL = np.linalg.cholesky(CorMat)
    z = np.random.randn(1,N2)
    sample_1D = np.dot(CDL,z)
    return sample_1D.reshape(N,N) 
    


# return the list of dim for each layer and channel
# ensemble_num >= 20
def RandomNetDim(net, GaussianRandomInput, input_size = 30, rho = None, ensemble_num = 10000):
    # each hihj_vec[layer idx][channel idx] is a list consists of one array element
    hihj_vec = [[[] for j in range(net.struct[i])] for i in range(net.layer_len)]
    hi_vec = [[[] for j in range(net.struct[i])] for i in range(net.layer_len)]
    CorMatrix_vec = [[[] for j in range(net.struct[i])] for i in range(net.layer_len)]
    D = [[[] for j in range(net.struct[i])] for i in range(net.layer_len)]
    # perspective field size of each layer
    h_size = [input_size]
    for l in range(1,net.layer_len):
        h_size.append((h_size[-1]-net.kernel_size)//net.stride + 1)
    for l in range(net.layer_len):
        for c in range(net.struct[l]):
            hihj_vec[l][c].append(np.zeros([h_size[l]*h_size[l],h_size[l]*h_size[l]]))
            hi_vec[l][c].append(np.zeros([1,h_size[l]*h_size[l]]))
    for n in range(ensemble_num):
        if n % (ensemble_num//20) == 0:
            print('Process: {0} %'.format(100 * n/ensemble_num))
        print(n)
        input_sample = GaussianRandomInputCD(1,net.kernel_size,input_size,rho) 
        net.forward(input_sample)
        for l in range(1,net.layer_len):
            activation_list = net.getActivationOfLayer(l)
            activation_list = [act.reshape(1,h_size[l]*h_size[l]) for act in activation_list]
            for c in range(net.struct[l]):
                hihj = np.dot(activation_list[c].T,activation_list[c])
                hihj_vec[l][c][0] += hihj
                hi = activation_list[c]
                hi_vec[l][c][0] += hi
        # input as 0th layer activation
        activation_list = [input_sample]
        activation_list = [act.reshape(1,h_size[0]*h_size[0]) for act in activation_list]
        for c in range(net.struct[0]):
            hihj = np.dot(activation_list[c].T,activation_list[c])
            hihj_vec[0][c][0] += hihj
            hi = activation_list[c]
            hi_vec[0][c][0] += hi
    
    for c in range(net.struct[0]):
        hihj_vec[0][c][0] /= ensemble_num
        hi_vec[0][c][0] /= ensemble_num
        CorMatrix_vec[0][c].append(hihj_vec[0][c][0] - np.dot(hi_vec[0][c][0].T,hi_vec[0][c][0]))
        C2 = np.dot(CorMatrix_vec[0][c][0],CorMatrix_vec[0][c][0])
        dim_input = np.square(np.trace(CorMatrix_vec[0][c][0]))/np.trace(C2)/h_size[0]/h_size[0]
        D[0][c].append(dim_input)
    
    for l in range(1,net.layer_len):
        for c in range(net.struct[l]):
            hihj_vec[l][c][0] /= ensemble_num
            hi_vec[l][c][0] /= ensemble_num
            CorMatrix_vec[l][c].append(hihj_vec[l][c][0] - np.dot(hi_vec[l][c][0].T,hi_vec[l][c][0]))
            Cor2 = np.dot(CorMatrix_vec[l][c][0],CorMatrix_vec[l][c][0])
            dim = np.square(np.trace(CorMatrix_vec[l][c][0]))/np.trace(Cor2)/h_size[l]/h_size[l]
            D[l][c].append(dim)
    return D
        








        
