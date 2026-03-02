
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange


class SedenionLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(SedenionLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features, 16))
        self.bias = nn.Parameter(torch.zeros(out_features, 16))
        scale = 1 / (16 * self.in_features) 
        self.weight.data.uniform_(-scale, scale)

        pattern_idx = torch.tensor([
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
            [ 1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10, 13, 12, 15, 14],
            [ 2,  3,  0,  1,  6,  7,  4,  5, 10, 11,  8,  9, 14, 15, 12, 13],
            [ 3,  2,  1,  0,  7,  6,  5,  4, 11, 10,  9,  8, 15, 14, 13, 12],
            [ 4,  5,  6,  7,  0,  1,  2,  3, 12, 13, 14, 15,  8,  9, 10, 11],
            [ 5,  4,  7,  6,  1,  0,  3,  2, 13, 12, 15, 14,  9,  8, 11, 10],
            [ 6,  7,  4,  5,  2,  3,  0,  1, 14, 15, 12, 13, 10, 11,  8,  9],
            [ 7,  6,  5,  4,  3,  2,  1,  0, 15, 14, 13, 12, 11, 10,  9,  8],
            [ 8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7],
            [ 9,  8, 11, 10, 13, 12, 15, 14,  1,  0,  3,  2,  5,  4,  7,  6],
            [10, 11,  8,  9, 14, 15, 12, 13,  2,  3,  0,  1,  6,  7,  4,  5],
            [11, 10,  9,  8, 15, 14, 13, 12,  3,  2,  1,  0,  7,  6,  5,  4],
            [12, 13, 14, 15,  8,  9, 10, 11,  4,  5,  6,  7,  0,  1,  2,  3],
            [13, 12, 15, 14,  9,  8, 11, 10,  5,  4,  7,  6,  1,  0,  3,  2],
            [14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  0,  1],
            [15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0]
        ], dtype=torch.long)

        pattern_sign = torch.tensor([
            [ 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [ 1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1,  1],
            [ 1,  1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1],
            [ 1, -1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1],
            [ 1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1,  1],
            [ 1,  1,  1, -1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1, -1],
            [ 1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1, -1],
            [ 1, -1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1, -1],
            [-1, -1,  1, -1,  1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1,  1],
            [-1,  1, -1,  1, -1,  1, -1,  1,  1,  1,  1, -1,  1, -1,  1,  1],
            [-1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1],
            [ 1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1,  1,  1],
            [ 1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1],
            [-1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1],
            [ 1, -1,  1,  1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1,  1,  1],
            [-1,  1, -1,  1,  1, -1,  1, -1, -1, -1,  1,  1, -1,  1,  1,  1]
        ], dtype=torch.get_default_dtype())
        self.register_buffer('pattern_idx', pattern_idx)   # shape (8, 8)
        self.register_buffer('pattern_sign', pattern_sign) # shape (8, 8)
        
    def forward(self, x):
        B = x.shape[0]  # Batch size
        F_in, O = self.out_features, self.in_features
        weight_expanded = self.weight.unsqueeze(2)  
        pattern_idx = self.pattern_idx.unsqueeze(0).unsqueeze(0)  
        weight_gathered = torch.gather(weight_expanded.expand(F_in, O, 16, 16),
                                       dim=-1,
                                       index=pattern_idx.expand(F_in, O, 16, 16))
        weight_matrix = weight_gathered * self.pattern_sign.unsqueeze(0).unsqueeze(0)

        out = torch.einsum("bcfj,ofij->bcoi", x, weight_matrix)
        out = out + self.bias
        return out