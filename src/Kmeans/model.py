import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from data_generator import Generator
import torch.nn.functional as F
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm


if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    torch.manual_seed(0)

def gmul(input):
    W, x, y = input
    # x is a tensor of size (bs, N, num_features)
    # y is a tensor of size (bs, N, N)
    # W is a tensor of size (bs, N, N, J)
    N = W.size(-2)
    W = W.split(1, 3)
    W = W + (y.unsqueeze(3),)
    W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)
    output = torch.bmm(W, x) # output has size (bs, J*N, num_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
    return output

def bnorm(x, U):
    mx = torch.bmm(U,x)
    subs = x-mx
    subs2 = subs*subs
    vx = torch.bmm(U,subs2)
    out = subs / (vx.clamp(min=1e-10).sqrt() + 1e-5)
    return out

class Gconv(nn.Module):
    def __init__(self, feature_maps, J, last=False):
        super(Gconv, self).__init__()
        self.num_inputs = J*feature_maps[0] # we have J=3 matrixes: identity, mu (average) and "similitude" (learned)
        self.num_outputs = feature_maps[1]
        self.last = last
        self.fc1 = nn.Linear(self.num_inputs, self.num_outputs // 2).type(dtype)
        self.fc2 = nn.Linear(self.num_inputs, self.num_outputs // 2).type(dtype)
        self.beta = nn.Linear(feature_maps[0], feature_maps[0], bias=False).type(dtype)
        self.gamma = nn.Parameter(torch.ones(self.num_outputs).type(dtype))

    def forward(self, input):
        W, x, Y = input
        N = Y.size(-1)
        bs = Y.size(0)
        mask1 = Variable((W.data[:,:,:,-1] > 0).float())
        mask2 = Variable(W.data[:,:,:,0].float().sum(2))
        U = Variable(W.data[:,:,:,-1])
        Ns = Variable(W.data[:,:,:,0].float().sum(2).sum(1).clamp(min=1))
        xB = self.beta(x) * mask2.unsqueeze(2).expand_as(x) # has size (bs,N,R)
        Y = torch.bmm(xB, x.permute(0,2,1)) - (1-mask1)*10000
        Y = F.softmax(Y.permute(1,0,2)).permute(1,0,2)
        #Y = (Y + Y.permute(0,2,1)) / 2
        #Y = Y * mask1
        
        x = gmul((W, x, Y)) # out has size (bs, N, num_inputs)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        if self.last:
            x1 = self.fc1(x)
        else:
            x1 = F.sigmoid(self.fc1(x)) # has size (bs*N, num_outputs // 2)
        x2 = self.fc2(x)
        x = torch.cat((x1, x2), 1)
        x = x.view(*x_size[:-1], self.num_outputs)
        x = bnorm(x, U)
        x = x * self.gamma.unsqueeze(0).unsqueeze(1).expand_as(x)
        
        return W, x, Y

class GNN(nn.Module):
    def __init__(self, num_features, num_layers, J, dim_input=1):
        super(GNN, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.featuremap_in = [dim_input, num_features]
        self.featuremap_mi = [num_features, num_features]
        self.featuremap_end = [num_features, num_features]
        self.layer0 = Gconv(self.featuremap_in, J)
        for i in range(num_layers):
            module = Gconv(self.featuremap_mi, J)
            self.add_module('layer{}'.format(i + 1), module)
        self.layerlast = Gconv(self.featuremap_end, J, last=True)

    def forward(self, input):
        cur = self.layer0(input)
        for i in range(self.num_layers):
            cur = self._modules['layer{}'.format(i+1)](cur)
        W, x, Y = self.layerlast(cur)
        return W, x, Y


class Split_GNN(nn.Module):
    def __init__(self, num_features, num_layers, J, dim_input=1):
        super(Split_GNN, self).__init__()
        self.num_features = num_features
        self.Gnn = GNN(num_features, num_layers, J, dim_input)
        self.linear_last1 = nn.Linear(self.num_features, 1, bias=True).type(dtype)
        self.linear_last2 = nn.Linear(self.num_features, 1, bias=True).type(dtype)

    def forward(self, input):
        W, x, Y = self.Gnn(input)
        bs = input[0].size(0)
        x = x.view(-1,self.num_features)
        prob_scores = self.linear_last1(x).view(bs, -1)
        vol_scores = self.linear_last2(x).view(bs, -1)
        return prob_scores, vol_scores

class Split_BaselineGNN(nn.Module):
    def __init__(self, num_features, num_layers, J, K, dim_input=1):
        super(Split_BaselineGNN, self).__init__()
        self.num_features = num_features
        self.K = K
        self.Gnn = GNN(num_features, num_layers, J, dim_input)
        self.linear_last1 = nn.Linear(self.num_features, 2**K, bias=True).type(dtype)
        self.linear_last2 = nn.Linear(self.num_features, 2**K, bias=True).type(dtype)

    def forward(self, input):
        W, x, Y = self.Gnn(input)
        bs = input[0].size(0)
        x = x.view(-1,self.num_features)
        prob_scores = self.linear_last1(x).view(bs, -1, 2**self.K)
        vol_scores = self.linear_last2(x).view(bs, -1, 2**self.K)
        return prob_scores, vol_scores


if __name__ == '__main__':
    a = 1
    # do nothing
    
    
    
    
    
    
    
    
    
    
