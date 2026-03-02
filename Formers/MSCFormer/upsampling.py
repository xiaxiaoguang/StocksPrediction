from torch import nn
import torch

class Upsampling(nn.Module):
    """Unit Multiscale time series."""

    def __init__(self, patch,embed_dim,num_decoder,needLinear=True):
        super().__init__()
        self.patch = [1] * num_decoder
        for i in range(1,num_decoder):
            self.patch[i] = self.patch[i-1]*patch[i]
        self.embed_dim = embed_dim
        self.num_decoder = num_decoder
        self.nL = needLinear
        self.FL = nn.Linear(embed_dim,patch[0])

    def forward(self, all_states):
        output = [0] * self.num_decoder
        for i in range(self.num_decoder):
            B,N,T,D = all_states[i].shape
            output[i]=all_states[i].repeat(1,1,self.patch[i],1)
        if self.nL is True:
            ret = torch.sum(torch.stack(output),dim=0)        
            ret = self.FL(ret)
        else :
            ret = torch.mean(torch.stack(output),dim=0)
        return ret