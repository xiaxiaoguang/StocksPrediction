import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math


class DynamicFilterGNN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(DynamicFilterGNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.li = nn.Linear(in_features,out_features)
        self.transform = Parameter(torch.zeros(out_features, in_features))
        self.weight    = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, matrix):
        transformed_adjacency = matrix + self.transform * 0.1
        result_embed = F.linear(input, transformed_adjacency.matmul(self.weight), self.bias) + \
            self.li(input)
        #F.linear(input, transformed_adjacency.matmul(self.weight), self.bias)
        return result_embed

    
    def get_transformed_adjacency(self,matrix):
        transformed_adjacency = matrix + self.transform * 0.1
        return transformed_adjacency
    
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) + ')'

