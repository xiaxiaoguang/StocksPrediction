import math
import copy
import torch
from typing import Optional
import torch.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer,TransformerDecoder,TransformerDecoderLayer

class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self,src):
        B, N, L, D = src.shape
        src = src * math.sqrt(self.d_model)
        src = src.reshape(B*N, L, D)
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, mask=None)
        output = output.transpose(0, 1).view(B, N, L, D)
        return output

class TransformerLayers2(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        decoder_layers = TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

    def forward(self, tgt, memory):
        B, N, L, D = memory.shape
        memory = memory * math.sqrt(self.d_model)
        memory = memory.view(B*N, L, D)
        memory = memory.transpose(0, 1)

        B, N, L, D = tgt.shape
        tgt = tgt * math.sqrt(self.d_model)
        tgt = tgt.view(B*N, L, D)
        tgt = tgt.transpose(0, 1)


        output = self.transformer_decoder(tgt , memory)
        output = output.transpose(0, 1).view(B, N, L, D)
        return output

class TransformerLayers3(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        decoder_layers = TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
    def forward(self, tgt, memory):
        B, L, D  = memory.shape
        memory = memory * math.sqrt(self.d_model)
        tgt = tgt * math.sqrt(self.d_model)
        memory = memory.transpose(0 , 1)
        tgt    =    tgt.transpose(0 , 1)
        output = self.transformer_decoder(tgt , memory)
        output = output.transpose(0, 1)
        return output
