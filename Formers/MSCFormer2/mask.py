import random
import torch 
from torch import nn

class MaskGenerator(nn.Module):
    """Mask generator."""

    def __init__(self, num_tokens ,patch , mask_ratio , device = None):
        super().__init__()
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        self.device = device
        self.sort = True
        self.patch = patch
        self.num=num=len(patch)

    def uniform_rand(self):
        now_tokens = self.num_tokens
        mask = torch.randperm(now_tokens,device=self.device)
        mask_len = int(now_tokens * self.mask_ratio)

        masked_token = mask[:mask_len]
        unmasked_token = mask[mask_len:]
        masked_token, _ = torch.sort(masked_token)
        unmasked_token, _ = torch.sort(unmasked_token)
        self.masked = [masked_token]
        self.unmasked = [unmasked_token]

        for i in range(self.num-1):
            now_tokens = now_tokens * self.patch[self.num-i-1]
            mask = torch.randperm(now_tokens,device=self.device)
            mask_len = int(now_tokens * self.mask_ratio)
            masked_token = mask[:mask_len]
            unmasked_token = mask[mask_len:]
            masked_token, _ = torch.sort(masked_token)
            unmasked_token, _ = torch.sort(unmasked_token)
            self.masked.append(masked_token)
            self.unmasked.append(unmasked_token)

        return self.masked,self.unmasked
    
    def forward(self):
        self.out1,self.out2 = self.uniform_rand()
        return self.out1,self.out2
