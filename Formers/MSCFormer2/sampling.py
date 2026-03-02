from torch import nn
import torch

class Upsummary(nn.Module):
    def __init__(self, patch ,num_decoder ,embed_dim ) -> None:
        super().__init__()
        self.patch = [patch[0]] * num_decoder
        for i in range(1,num_decoder):
            self.patch[i] = self.patch[i-1]*patch[i]
        self.embed_dim = embed_dim
        self.num_decoder = num_decoder
        self.FL = nn.Linear(embed_dim,1)
    def forward(self,states):
        output = [0] * self.num_decoder
        for i in range(self.num_decoder):
            B,N,T,D = states[i].shape
            output[i]=states[i].repeat(1,1,self.patch[i],1)
        ret = torch.mean(torch.stack(output),dim=0)
        ret = self.FL(ret)
        return ret        

class Upsampling(nn.Module):
    """Unit Multiscale time series."""
    def __init__(self, patch,num_decoder,out_channel) -> None:
        super().__init__()
        self.patch=patch
        in_channel = 1
        self.out_channel = out_channel
        self.samplelayer = [0]*num_decoder
        self.device = "cuda:0"
        for i in range(num_decoder):
            self.samplelayer[i] = nn.ModuleList()
            self.samplelayer[i].append(nn.Conv2d(in_channel,out_channel[i]*self.patch[i],
                                                 kernel_size=(1,1),device=self.device))
            self.samplelayer[i].append(nn.SiLU())
            self.samplelayer[i].append(nn.Conv2d(out_channel[i]*self.patch[i],self.patch[i],
                                                 kernel_size=(1,1),device=self.device))
    
    def forward(self, states, i):
        B,N,T,D = states.shape
        states = states.reshape(B*N,1,T,D)
        for layer in self.samplelayer[i]:
            states=layer(states)

        states = states.reshape(B,N,T*self.patch[i],D)
        return states
    
class Downsampling(nn.Module):
    def __init__(self,patch,num_decoder,out_channel) -> None:
        super().__init__()
        self.patch=patch
        in_channel = 1
        self.out_channel = out_channel
        self.samplelayer = [0]*num_decoder
        self.device = "cuda:0"

        for i in range(num_decoder):
            self.samplelayer[i] = nn.ModuleList()
            self.samplelayer[i].append(nn.Conv2d(in_channel,out_channel[i],
                                                 kernel_size=(patch[i],1),stride=(patch[i],1),
                                                 device=self.device))
            self.samplelayer[i].append(nn.SiLU())
            self.samplelayer[i].append(nn.Conv2d(out_channel[i],out_channel[i],
                                                 kernel_size=(1,out_channel[i]),stride=(1,out_channel[i]),
                                                 device=self.device))
    
    def forward(self,states,i):
        B,N,T,D = states.shape
        states = states.reshape(B*N,1,T,D)
        for layer in self.samplelayer[i]:
            states=layer(states)
        states = states.transpose(1,2)
        states = states.reshape(B,N,T//self.patch[i],D)
        return states