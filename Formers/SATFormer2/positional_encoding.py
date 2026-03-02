import torch
from torch import nn
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, hidden_dim, patch ,dropout=0.1, max_len: int = 2000): # max len for patch
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dim = hidden_dim
        self.position_embedding = torch.empty(1 , max_len, hidden_dim, requires_grad=True,device='cuda:0')
        self.num=num=len(patch)
        self.pool=[]
        for i in range(num):
            self.pool.append(nn.Conv1d(in_channels=hidden_dim,out_channels=hidden_dim,kernel_size=patch[i],device='cuda:0'))

    def forward(self, input=None,layer=None,index=None):
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N, P, d].
            index (list or None): add positional embedding by index.

        Returns:
            torch.tensor: output sequence
        """
        batch_size,num_nodes,_,_ = input[0].shape
        embed = self.position_embedding
        # tp_enc_2d = PositionalEncoding1D(self.embed_dim).to(input.device)
        ret=[]
        for i,x in enumerate(input):
            x = x.reshape(batch_size*num_nodes,-1,self.embed_dim)
            patches = x.shape[1]
            embed = self.pool[i](embed.transpose(-1,-2)).transpose(-1,-2)

            if index != None:
                x = x + embed[:,index[i],:]
            else:
                x = x + embed[:,:patches,:]
                
            x = self.dropout(x)

            ret.append(x.reshape(batch_size,num_nodes,patches,self.embed_dim))
        return ret


class SpacePositionalEncoding(nn.Module):
    def __init__(self, hidden_dim ,dropout=0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dim = hidden_dim
        self.position_embedding = torch.empty(1 , max_len, hidden_dim, requires_grad=True,device='cuda:0')
    def forward(self, x):
        _,_,num_nodes,_ = x.shape
        embed = self.position_embedding
        x = x + embed[:,:num_nodes,:]
        x = self.dropout(x)
        return x
