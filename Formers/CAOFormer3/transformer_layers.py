import math
import torch
from typing import Optional
import torch.functional as F
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer,TransformerDecoder,TransformerDecoderLayer

class TransformerLayers2(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        decoder_layers = TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

    def forward(self, tgt, memory):
        B, L, D = memory.shape
        memory = memory * math.sqrt(self.d_model)
        memory = memory.transpose(0, 1)

        B, L, D = tgt.shape
        tgt = tgt * math.sqrt(self.d_model)
        tgt = tgt.transpose(0, 1)

        output = self.transformer_decoder(tgt , memory)
        output = output.transpose(0, 1).view(B, L, D)
        return output


class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):
        # 这里就是直接把N和B乘在一起了，torch.nn中直接使用了
        flg = 0

        if len(src.shape) == 4:
            flg = 1
            B, N, L, D = src.shape
            src = src * math.sqrt(self.d_model)
            src = src.view(B*N, L, D)
            src = src.transpose(0, 1)

        output = self.transformer_encoder(src, mask=None)
        if flg == 1 : 
            output = output.transpose(0, 1).view(B, N, L, D)
        return output

