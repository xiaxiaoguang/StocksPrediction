

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ML_Patch_Embedding(nn.Module):
    def __init__(self, d_model, d_out, patch_level,function):
        super().__init__()
        self.d_model = d_model
        self.patch_level = patch_level
        self.function = function

        self.patch_len_list =[d_model//(2**i) for i in range(self.patch_level)]
        self.t_linears = nn.ModuleList([])
    
        for patch_len in self.patch_len_list:
            self.t_linears.append(self.function(patch_len, d_out))
        
    def _seq_to_patch(self, x):
        patch_list=[]
        for patch_len in self.patch_len_list:
            split_seq = [x[:,:,i:i+patch_len,...] for i in range(0, x.shape[2], patch_len)]
            last_split_seq = split_seq[-1]
            if last_split_seq.shape[2] != patch_len:
                last_split_seq_shape=list(last_split_seq.shape)
                last_split_seq_shape[2] = patch_len-last_split_seq_shape[2]
                zero_tensor = torch.zeros(last_split_seq_shape).to(x.device)
                last_split_seq = torch.concat([last_split_seq,zero_tensor],dim=2)
                split_seq[-1]=last_split_seq
            patch_list.append(split_seq)
        return patch_list
    
    def _patch_to_seq(self, x):
        seq_list = [torch.mean(torch.stack(item, dim=-1), dim=-1) for item in x]
        if len(seq_list[0].shape) == 4:
            return torch.concat(seq_list, dim=-2)
        else:
            return torch.concat(seq_list, dim=-1)

    def forward(self, x):
        patches = self._seq_to_patch(x)
        res = [[self.t_linears[idx](item) for item in patches[idx]] for idx, patch in enumerate(patches)]
        result = self._patch_to_seq(res) #list([B F T D]*P)(多个patch)
        return result
