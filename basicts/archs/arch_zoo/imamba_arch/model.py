import torch.nn as nn
import torch
from mamba_ssm import Mamba

class imamba(nn.Module):
    def __init__(self,num_len , pred_len , embed_dim,
                   d_state, d_conv, expand, n_layers):
        super(imamba, self).__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.linear1 = nn.Linear(num_len,embed_dim)
        self.mamba1_layers = nn.ModuleList([
            Mamba(
                d_model=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(n_layers)
        ])
        self.mamba2_layers = nn.ModuleList([
            Mamba(
                d_model=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(n_layers)
        ])
        self.linear2 = nn.Linear(embed_dim, pred_len)

    def forward(self, history_data=None,future_data=None,batch_seen=None,epoch=None,train=None,ConL=False):
        embeded = self.linear1(history_data.squeeze(-1).permute(0, 2, 1).contiguous())
        for i in range(self.n_layers):
            result1 = self.mamba1_layers[i](embeded)
            embeded = torch.flip(embeded,[1])
            result2 = self.mamba2_layers[i](embeded)
            result2 = torch.flip(result2,[1])
            embeded = nn.SiLU()(result1 + result2)
            hid = embeded.unsqueeze(0)
        if ConL:
            return hid
        output = self.linear2(embeded).transpose(1,2).contiguous()  # [batch_size, node_num * step_size, 1]
        return output.unsqueeze(-1)
