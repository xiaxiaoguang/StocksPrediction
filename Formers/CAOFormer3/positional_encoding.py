import torch
from torch import nn
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer


class OldPositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, hidden_dim, dropout=0.1, max_len: int = 2016):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Parameter(torch.empty(max_len, hidden_dim), requires_grad=True)

    def forward(self, input_data, index=None, abs_idx=None):
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N, P, d].
            index (list or None): add positional embedding by index.

        Returns:
            torch.tensor: output sequence
        """
        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        input_data = input_data.view(batch_size*num_nodes, num_patches, num_feat)
        # positional encoding
        if index is None:
            pe = self.position_embedding[:input_data.size(1), :].unsqueeze(0)
        else:
            pe = self.position_embedding[index].unsqueeze(0)
            
        input_data = input_data + pe
        input_data = self.dropout(input_data)
        # reshape
        input_data = input_data.view(batch_size, num_nodes, num_patches, num_feat)
        return input_data


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, hidden_dim, dropout=0.1 , maxPatch=48, max_len: int = 24000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.maxPatch = maxPatch
        self.position_embedding = nn.Parameter(torch.empty(max_len, hidden_dim), requires_grad=True)
        # self.position_embedding = nn.Embedding(max_len , hidden_dim)

    def forward(self, input_data, index=None, abs_idx=None):
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N, P, d].
            index (list or None): add positional embedding by index.

        Returns:
            torch.tensor: output sequence
        """
        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        input_data = input_data.reshape(batch_size, num_nodes*num_patches, num_feat)
        # positional encoding
        if index is None:
            NotImplementedError
        #     pe = self.position_embedding[:input_data.size(1), :].unsqueeze(0)
        # else:
        apex = torch.arange(0,num_nodes*self.maxPatch,self.maxPatch,device=index.device,dtype=torch.long).unsqueeze(1)
        index = index.unsqueeze(0)
        index = (index + apex).view(-1)
        
        pe = self.position_embedding[index].unsqueeze(0)
        input_data = input_data + pe
        input_data = self.dropout(input_data)
        # reshape
        input_data = input_data.view(batch_size, num_nodes, num_patches, num_feat)
        
        # try sine positional encoding with learnable encoding
        tp_enc_2d = PositionalEncoding2D(num_feat).to(input_data.device)
        input_data= input_data + tp_enc_2d(input_data)

        return input_data
