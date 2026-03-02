import torch.nn as nn
from mamba_ssm import Mamba
from .model2 import RMSNorm
from einops import rearrange

class mamba3(nn.Module):
    def __init__(self,
                 input_len,pred_len,input_dim,embed_dim,
                   d_state, d_conv, expand, n_layers):
        super(mamba3, self).__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.embedding = nn.Linear(input_dim,embed_dim)
        self.act = nn.SiLU()
        self.INPUT_LEN = input_len
        self.mamba_layers = nn.ModuleList([
            Mamba(
                    d_model=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
            ) for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(embed_dim)
        self.norm_b = nn.BatchNorm1d(embed_dim)
        # self.outLinear = nn.Conv2d(input_len, pred_len , kernel_size=(1,1))
        self.outLinear = nn.Linear(embed_dim,input_dim)

    def forward(self,  history_data=None, future_data=None , batch_seen=None , epoch=None , train=None):
        B,T,N,D = history_data.shape
        embedded = rearrange(self.act(self.embedding(history_data)),"B T N D -> (B N) T D")
        
        for i in range(self.n_layers):
            embedded = embedded + self.mamba_layers[i](self.norm_f(embedded))
            hid = embedded

        embedded = self.norm_b(rearrange(embedded,"B T D -> B D T")).transpose(1,2)
        # output2 = self.outLinear2(embedded)
        output=None
        if T == self.INPUT_LEN:
            embedded = rearrange(embedded,"(B N) T E -> B T N E", B=B, N=N)
            output = self.outLinear(embedded)
        return output,hid