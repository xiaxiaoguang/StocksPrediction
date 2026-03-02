import math
import copy
import torch
from typing import Optional
import torch.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer,TransformerDecoder,TransformerDecoderLayer



class EasyDecoder(nn.Module):
    def __init__(self,embed_dim,final_num):
        super().__init__()
        # step = (final_num-embed_dim)//3
        L1 = nn.ModuleList()
        L1.append(nn.Conv2d(embed_dim,final_num,kernel_size=(3,3),padding=(1,1)))
        L1.append(nn.ReLU())
        L1.append(nn.BatchNorm2d(final_num))
        # L1.append(nn.Conv2d(embed_dim+step,embed_dim+step+step,kernel_size=(5,5),padding=(2,2)))
        # L1.append(nn.ReLU())
        # L1.append(nn.Dropout(0.5))
        # L1.append(nn.Conv2d(embed_dim+step+step,final_num,kernel_size=(3,3),padding=(1,1)))
        # L1.append(nn.ReLU())
        # L1.append(nn.Dropout(0.5))
        L1.append(nn.Conv2d(final_num,embed_dim,kernel_size=(1,1)))
        self.L1 = L1
    def forward(self,x):
        # batch_size,num_nodes,num_patch,_= x.shape
        # x= x.reshape(batch_size*num_nodes,num_patch,-1)
        x = x.transpose(1,3)
        for i in self.L1:# X is B,N,T,F / B,T,N,F
            x = i(x)
        x = x.transpose(1,3)
        # x= x.reshape(batch_size,num_nodes,num_patch,-1)
        return x
    

# class EasyDecoder(nn.Module):
#     def __init__(self,embed_dim,mid_num):
#         super().__init__()

#         L1 = nn.ModuleList()
#         L1.append(nn.Linear(embed_dim,mid_num))
#         L1.append(nn.ReLU())
#         L1.append(nn.Dropout(0.5))
#         self.L1 = L1
        
#         L2 = nn.ModuleList()
#         L2.append(nn.Linear(mid_num,mid_num))
#         L2.append(nn.ReLU())
#         L2.append(nn.Dropout(0.5))
#         self.L2 = L2

#         L3 = nn.ModuleList()
#         L3.append(nn.Linear(mid_num,embed_dim))
#         self.L3 = L3
#     def forward(self,x):
#         # batch_size,num_nodes,num_patch,_= x.shape
#         # x= x.reshape(batch_size*num_nodes,num_patch,-1)
#         for i in self.L1:# X is B,N,T,F / B,T,N,F
#             x = i(x)
#         for i in self.L2:
#             x = i(x)
#         for i in self.L3:
#             x = i(x)
#         # x= x.reshape(batch_size,num_nodes,num_patch,-1)
#         return x
    
# class EasyDecoder(nn.Module):
#     def __init__(self,num_token,embed_dim,mid_num):
#         super().__init__()

#         L1 = nn.ModuleList()
#         L1.append(nn.Linear(embed_dim,embed_dim//4))
#         L1.append(nn.ReLU())
#         L1.append(nn.Dropout(0.5))
#         self.L1 = L1
        
#         # L2 = nn.ModuleList()
#         # L2.append(nn.Linear(mid_num,num_token)) # 下采样
#         # L2.append(nn.Sigmoid()) # 防止全杀了
#         # L2.append(nn.Dropout(0.5))
#         # self.L2 = L2

#         L2_2 = nn.ModuleList()
#         L2_2.append(nn.Linear(num_token,num_token))
#         L2_2.append(nn.ReLU())
#         L2_2.append(nn.Dropout(0.5))
#         self.L2_2 = L2_2
        
#         L3 = nn.ModuleList()
#         L3.append(nn.Conv2d(num_token,num_token,(3,3),padding=(1,1),stride=(1,1)))
#         L3.append(nn.ReLU())
#         L3.append(nn.Dropout(0.5))
#         L3.append(nn.Conv2d(num_token,num_token,(3,3),padding=(1,1),stride=(1,1)))
#         self.L3 = L3

#         L4 = nn.ModuleList()
#         L4.append(nn.Linear(embed_dim//2,embed_dim))
#         L4.append(nn.ReLU())
#         L4.append(nn.Dropout(0.5))

#         self.L4 = L4
#     def forward(self,x):
#         for i in self.L1:# X is B,N,T,F  Y is B,T*4,N,F
#             x = i(x)
#             # y = i(y)

#         x = x.transpose(2,3) # B,N,F/4,T torch.Size([1, 500, 50, 64])
#         # y = y.permute(0,2,3,1) 
#         # for i in self.L2:
#         #     y = i(y) # B,N,F/4,T
        
#         for i in self.L2_2:
#             # y = i(y)
#             x = i(x)

#         x = x.transpose(1,3) # B,T*4,F/4,N torch.Size([1, 500, 64, 200])
#         for i in self.L3:
#             x = i(x)
 
#         # y = y.transpose(2,3) #  B,N,T,F/4
#         x = x.permute(0,3,1,2) # B,N,T,F/4 torch.Size([1, 500, 200, 64])
#         # x = torch.cat((x,y),dim=-1) # B,N,T,F/2

#         for i in self.L4:
#             x = i(x)
#         x = x.transpose(1,2) # B,N,T,F
#         return x

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
        memory = memory.reshape(B*N, L, D)
        memory = memory.transpose(0, 1)

        B, N, L, D = tgt.shape
        tgt = tgt * math.sqrt(self.d_model)
        tgt = tgt.reshape(B*N, L, D)
        tgt = tgt.transpose(0, 1)

        output = self.transformer_decoder(tgt , memory)
        output = output.transpose(0, 1).view(B, N, L, D)
        return output
