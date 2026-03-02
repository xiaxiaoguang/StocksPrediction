import torch
from torch import nn
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, hidden_dim, patch1,patch2 ,dropout=0.1 , maxPatch=48, max_len: int = 1000): # max len for patch
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.maxPatch = maxPatch
        self.position_embedding = nn.Parameter(torch.empty(1 , max_len, hidden_dim), requires_grad=True)

        self.pool1 = nn.Conv2d(1 ,1, kernel_size=(patch1,1),stride=(patch1,1))
        self.pool2 = nn.Conv2d(1 ,1, kernel_size=(patch2,1),stride=(patch2,1))

    def forward(self, input=None,layer=None,index=None ):
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N, P, d].
            index (list or None): add positional embedding by index.

        Returns:
            torch.tensor: output sequence
        """
        # breakpoint()
        batch_size, num_nodes, num_patch, num_feat = input.shape
        if layer >= 0:
            embed = self.position_embedding
        if layer >= 1:
            embed = self.pool1(embed)
        if layer >= 2:
            embed = self.pool2(embed)
        input = input.reshape(batch_size*num_nodes,num_patch,num_feat)
        if index is not None :
            input = input + embed[:,index]
            input = input.reshape(batch_size,num_nodes,num_patch , num_feat)
            return input
        input = input + embed[:,:num_patch,:]
        # try sine positional encoding with learnable encoding
        tp_enc_2d = PositionalEncoding2D(num_feat).to(input.device)
        input = input.reshape(batch_size,num_nodes,num_patch , num_feat)
        input= input + tp_enc_2d(input)
        return input
