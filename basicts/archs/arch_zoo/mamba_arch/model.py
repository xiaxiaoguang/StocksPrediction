import torch.nn as nn
from mamba_ssm import Mamba
from .model2 import RMSNorm
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
            Mamba(
                    d_model=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
            ) for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(embed_dim)
        self.prob = (2 ** nn.Parameter(torch.arange(1, 12, dtype=torch.float32,device='cuda')))
        self.prob /= self.prob.sum()
        self.conv = nn.Conv2d(input_len, pred_len , kernel_size=(1,embed_dim))

    def get_probability_vectors(self,base): 
        # Base 测试得到的结果是模型希望他越小越好，也就是前面的位置相对来说模型预测比较成功
        # 那么我们就知道肯定要越大越好！
        indices = torch.arange(1, 12, dtype=torch.float32, device='cuda')
        self.probs = base ** indices
        self.probs =  self.probs /  self.probs.sum()
        return self.probs
    

    def forward(self,  history_data=None, future_data=None , batch_seen=None , epoch=None , train=None , ConL = False):
        B,T,N,D = history_data.shape
        embedded = rearrange(self.embedding(history_data),"B T N D -> (B N) T D")
        
        hid_sta = None
        for i in range(self.n_layers):
            embedded = embedded + self.mamba_layers[i](self.norm_f(embedded))
            if hid_sta == None:
                hid_sta = embedded.unsqueeze(0)
            else :
                hid_sta = torch.cat([hid_sta,embedded.unsqueeze(0)],dim=0)

        if ConL == True:
            return hid_sta
        
        embedded = self.norm_f(embedded)
        embedded = rearrange(embedded,"(B N) T E -> B T N E", B=B, N=N)
        output = self.conv(embedded)

        return output , hid_sta