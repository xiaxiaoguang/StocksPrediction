import torch
from torch import nn
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, hidden_dim, windowsize ,dropout=0.1 , maxPatch=48, max_len: int = 4200): # max len for patch
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.maxPatch = maxPatch
        self.windowsize = windowsize
        self.position_embedding = nn.Parameter(torch.empty(max_len, hidden_dim), requires_grad=True)
        self.space_embedding = nn.Parameter(torch.empty(max_len,hidden_dim),requires_grad=True)
        self.avgpool = nn.AvgPool1d(kernel_size=windowsize,stride=windowsize)

    def forward(self, input1=None, input2=None, input3=None, index=None):
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N, P, d].
            index (list or None): add positional embedding by index.

        Returns:
            torch.tensor: output sequence
        """
        if input2 is None:
            assert index is not None, "Error Mode 1"
            batch_size, num_patch1 , num_patch3,  num_feat = input3 .shape        
            input3 = input3.reshape(batch_size*num_patch1,num_patch3 , num_feat)
            # embed3 = self.avgpool(self.space_embedding.transpose(0,1)).transpose(0,1)
            input3 = input3 + self.space_embedding[index].unsqueeze(0)
            input3 = input3.reshape(batch_size,num_patch1,num_patch3 , num_feat)
            return input3
        
        batch_size, num_nodes, num_patch2, num_feat = input2.shape        
        input2 = input2.reshape(batch_size*num_nodes,num_patch2 , num_feat)
        embed2 = self.avgpool(self.position_embedding.transpose(0,1)).transpose(0,1)
        
        if input1 is None :
            assert index is not None  , "Error Mode 2"
            input2 = input2 + embed2[index].unsqueeze(0)
            input2 = input2.reshape(batch_size,num_nodes,num_patch2 , num_feat)
            return input2
        
        input2 = input2 + embed2[:num_patch2,:].unsqueeze(0)
        
        num_patch1 = input1.shape[2]
        input1 = input1.reshape(batch_size*num_nodes,num_patch1 , num_feat)
        input1 = input1 + self.position_embedding[:num_patch1,:].unsqueeze(0)
        # try space embedding , a question, do i need add time embedding to the input3?
        input1 = input1.reshape(batch_size*num_patch1,num_nodes,num_feat)
        input1 = input1 + self.space_embedding[:num_nodes,:].unsqueeze(0)        
        
        input2 = input2.reshape(batch_size*num_patch2,num_nodes,num_feat)
        input2 = input2 + self.space_embedding[:num_nodes,:].unsqueeze(0)

        _, _, num_patch3 , _ = input3.shape
        input3 = input3.reshape(batch_size*num_patch2,num_patch3,num_feat)
        # embed3 = self.avgpool(self.space_embedding.transpose(0,1)).transpose(0,1)
        input3 = input3 + self.space_embedding[:num_patch3,:].unsqueeze(0)
        # try sine positional encoding with learnable encoding
        tp_enc_2d = PositionalEncoding2D(num_feat).to(input1.device)
        input1 = input1.reshape(batch_size,num_nodes,num_patch1 , num_feat)
        input1= input1 + tp_enc_2d(input1)
        
        input2 = input2.reshape(batch_size,num_nodes,num_patch2 , num_feat)
        input2= input2 + tp_enc_2d(input2)

        input3 = input3.reshape(batch_size,num_patch2, num_patch3 , num_feat)
        return input1,input2,input3
