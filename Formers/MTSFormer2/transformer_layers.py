import math
import copy
import torch
from typing import Optional
import torch.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer,TransformerDecoder,TransformerDecoderLayer


class EasyDecoder(nn.Module):
    def __init__(self,num_token,embed_dim,mid_num):
        super().__init__()

        L1 = nn.ModuleList()
        L1.append(nn.Linear(embed_dim,embed_dim//4))
        L1.append(nn.ReLU())
        L1.append(nn.Dropout(0.5))
        self.L1 = L1
        
        L2 = nn.ModuleList()
        L2.append(nn.Linear(num_token,mid_num))
        L2.append(nn.ReLU())
        L2.append(nn.Dropout(0.5))
        self.L2 = L2
        
        L3 = nn.ModuleList()
        L3.append(nn.Conv2d(mid_num,num_token,(4,3),padding=(0,1),stride=(4,1)))
        L3.append(nn.ReLU())
        self.L3 = L3

        L4 = nn.ModuleList()
        L4.append(nn.Linear(embed_dim//16,embed_dim))
        L4.append(nn.ReLU())
        L4.append(nn.Dropout(0.5))
        self.L4 = L4
    def forward(self,x):
        for i in self.L1:# X is B,N,T,F
            x = i(x)
        x = x.transpose(2,3) # B,N,F,T
        for i in self.L2:
            x = i(x)
        x = x.transpose(1,3) # B,T,F,N
        for i in self.L3:
            x = i(x)
        x = x.transpose(2,3) # B,T,N,F
        for i in self.L4:
            x = i(x)
        x = x.transpose(1,2) # B,N,T,F
        return x

class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self,src):
        B, N, L, D = src.shape
        src = src * math.sqrt(self.d_model)
        src = src.view(B*N, L, D)
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
