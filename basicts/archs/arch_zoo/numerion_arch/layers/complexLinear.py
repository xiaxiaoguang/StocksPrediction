
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(out_features, in_features, 2))
        self.bias = nn.Parameter(torch.zeros(out_features, 2))
        scale = 1 / (2 * self.in_features) 
        self.weight.data.uniform_(-scale, scale)
        pattern_idx = torch.tensor([
            [0, 1 ],
            [1, 0 ],
        ], dtype=torch.long)
        
        pattern_sign = torch.tensor([
            [ 1, -1],
            [ 1,  1],
        ], dtype=torch.get_default_dtype())
        
        self.register_buffer('pattern_idx', pattern_idx)   
        self.register_buffer('pattern_sign', pattern_sign) 
        
    def forward(self, x):
        B = x.shape[0] 
        F_in, O = self.out_features, self.in_features
        weight_expanded = self.weight.unsqueeze(2)  
        pattern_idx = self.pattern_idx.unsqueeze(0).unsqueeze(0)  
        weight_gathered = torch.gather(weight_expanded.expand(F_in, O, 2, 2),
                                       dim=-1,
                                       index=pattern_idx.expand(F_in, O, 2, 2))
        weight_matrix = weight_gathered * self.pattern_sign.unsqueeze(0).unsqueeze(0)
        out = torch.einsum("bcfj,ofij->bcoi", x, weight_matrix)
        out = out + self.bias
        return out
