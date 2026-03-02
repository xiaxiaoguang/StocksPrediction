import torch.nn as nn
from .model2 import ResidualBlock,RMSNorm,ModelArgs
from einops import rearrange
import torch

class mamba(nn.Module):
    def __init__(self,
                 input_len,pred_len,input_dim,embed_dim,
                   d_state, d_conv, expand, n_layers):
        super(mamba, self).__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.embedding = nn.Linear(input_dim,embed_dim)
        self.mamba_layers = nn.ModuleList([
            ResidualBlock(
                ModelArgs(
                    d_model=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    n_layer=n_layers,
                    vocab_size=1,
                )   
            ) for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(embed_dim)
        self.conv = nn.Conv2d(input_len, pred_len , kernel_size=(1,embed_dim))

    def forward(self,  history_data=None, future_data=None , batch_seen=None , epoch=None , train=None , ConL = False):
        B,T,N,D = history_data.shape

        embedded = rearrange(self.embedding(history_data),"B T N D -> (B N) T D")
        
        hid_sta = None
        for i in range(self.n_layers):
            embedded = self.mamba_layers[i](embedded)
            if hid_sta == None:
                hid_sta = embedded.unsqueeze(0)
            else :
                hid_sta = torch.cat([hid_sta,embedded.unsqueeze(0)],dim=0)

        if ConL == True:
            return hid_sta
        
        embedded = self.norm_f(embedded)
        embedded = rearrange(embedded,"(B N) T E -> B T N E", B=B, N=N)
        output = self.conv(embedded)
        return output,hid_sta