import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Transformer_EncDec import Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import DataEmbedding_inverted
import numpy as np


class CustomConvLayer(nn.Module):
    def __init__(self, input_dim, D, output_dim):
        super(CustomConvLayer, self).__init__()
        self.input_dim = input_dim
        self.D = D
        self.output_dim = output_dim

        # Create a single large weight matrix
        self.weight = nn.Parameter(torch.randn(D, input_dim, output_dim))
        self.bias = nn.Parameter(torch.randn(D, output_dim))

    def forward(self, x):
        B, input_dim, D = x.shape

        assert input_dim == self.input_dim and D == self.D, "Input dimensions must match"

        # Reshape x to (B*D, input_dim)
        x_reshaped = x.permute(0, 2, 1).reshape(B * D, input_dim)
        
        # Expand weights and biases to match the batch size
        weight_expanded = self.weight.view(D, input_dim, self.output_dim).expand(B, -1, -1, -1).\
            reshape(B * D, input_dim, self.output_dim)
        bias_expanded = self.bias.view(D, self.output_dim).expand(B, -1, -1).\
            reshape(B * D, self.output_dim)

        # Perform batched matrix multiplication and add bias
        output_reshaped = torch.bmm(x_reshaped.unsqueeze(1), weight_expanded).squeeze(1) + bias_expanded

        # Reshape the output back to (B, output_dim, D)
        output = output_reshaped.view(B, D, self.output_dim)
        
        return output
    
class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, seq_len , pred_len , d_model , dropout , d_ff , n_heads , e_layers , use_norm , use_new = False, num_nodes = 260):
        super(iTransformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_norm = use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model,
                                                    dropout=dropout)
        # self.enc_embedding = CustomConvLayer(seq_len,num_nodes,d_model)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forecast(self, x_enc):

        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)

        enc_out = self.enc_embedding(x_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, history_data=None,future_data=None,batch_seen=None,epoch=None,train=None):
        dec_out = self.forecast(history_data.squeeze(-1))
        return dec_out[:, -self.pred_len:, :].unsqueeze(-1)  # [B, L, D , 1]