
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange


class OctonionLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(OctonionLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features, 8))
        self.bias = nn.Parameter(torch.zeros(out_features, 8))
        scale = 1 / (8 * self.in_features)
        self.weight.data.uniform_(-scale, scale)

        pattern_idx = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 0, 3, 2, 5, 4, 7, 6],
            [2, 3, 0, 1, 6, 7, 4, 5],
            [3, 2, 1, 0, 7, 6, 5, 4],
            [4, 5, 6, 7, 0, 1, 2, 3],
            [5, 4, 7, 6, 1, 0, 3, 2],
            [6, 7, 4, 5, 2, 3, 0, 1],
            [7, 6, 5, 4, 3, 2, 1, 0]
        ], dtype=torch.long)
        
        pattern_sign = torch.tensor([
            [ 1, -1, -1, -1, -1, -1, -1, -1],
            [ 1,  1, -1,  1, -1,  1,  1, -1],
            [ 1,  1,  1, -1, -1, -1,  1,  1],
            [ 1, -1,  1,  1, -1,  1, -1,  1],
            [ 1,  1,  1,  1,  1, -1, -1, -1],
            [ 1, -1,  1, -1,  1,  1,  1, -1],
            [ 1, -1, -1,  1,  1, -1,  1,  1],
            [ 1,  1, -1, -1,  1,  1, -1,  1]
        ], dtype=torch.get_default_dtype())
        
        self.register_buffer('pattern_idx', pattern_idx)   # shape (8, 8)
        self.register_buffer('pattern_sign', pattern_sign) # shape (8, 8)

        
    def forward(self, x):
        B = x.shape[0]  # Batch size
        F_in, O = self.out_features, self.in_features
        weight_expanded = self.weight.unsqueeze(2)  
        pattern_idx = self.pattern_idx.unsqueeze(0).unsqueeze(0)  
        weight_gathered = torch.gather(weight_expanded.expand(F_in, O, 8, 8),
                                       dim=-1,
                                       index=pattern_idx.expand(F_in, O, 8, 8))
        weight_matrix = weight_gathered * self.pattern_sign.unsqueeze(0).unsqueeze(0)

        if len(x.shape) == 4:
            out = torch.einsum("bcfj,ofij->bcoi", x, weight_matrix)
        elif len(x.shape) == 5:
            out = torch.einsum("bacfj,ofij->bacoi", x, weight_matrix)
        elif len(x.shape) == 3:
            out = torch.einsum("bfj,ofij->boi", x, weight_matrix)

        out = out + self.bias
        return out