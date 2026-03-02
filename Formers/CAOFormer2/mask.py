import random
import torch 
from torch import nn



class MaskGenerator(nn.Module):
    """Mask generator."""

    def __init__(self, num_tokens, mask_ratio , device = None):
        super().__init__()
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        self.device = device
        self.sort = True

    # def uniform_rand(self):
    #     mask = list(range(int(self.num_tokens)))
    #     random.shuffle(mask)
    #     mask_len = int(self.num_tokens * self.mask_ratio)
    #     self.masked_tokens = mask[:mask_len]
    #     self.unmasked_tokens = mask[mask_len:]
    #     if self.sort:
    #         self.masked_tokens = sorted(self.masked_tokens)
    #         self.unmasked_tokens = sorted(self.unmasked_tokens)
    #     return self.unmasked_tokens, self.masked_tokens

    def uniform_rand(self):
        mask = torch.randperm(self.num_tokens ,device = self.device)
        mask_len = int(self.num_tokens * self.mask_ratio)
        self.masked_tokens = mask[:mask_len]
        self.unmasked_tokens = mask[mask_len:]
        if self.sort:
            self.masked_tokens, _ = torch.sort(self.masked_tokens)
            self.unmasked_tokens, _ = torch.sort(self.unmasked_tokens)
        return self.unmasked_tokens, self.masked_tokens
    
    def forward(self):
        self.unmasked_tokens, self.masked_tokens = self.uniform_rand()
        return self.unmasked_tokens, self.masked_tokens
