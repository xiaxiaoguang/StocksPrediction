import random
import torch 
from torch import nn



class MaskGenerator(nn.Module):
    """Mask generator."""

    def __init__(self, num_tokens ,patch1 , patch2 , mask_ratio , device = None):
        super().__init__()
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        self.device = device
        self.sort = True
        self.patch1 = patch1
        self.patch2 = patch2
 
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

        self.masked2 = mask[:mask_len]
        self.unmasked2 = mask[mask_len:]

        if self.sort:
            self.masked2, _ = torch.sort(self.masked2)
            self.unmasked2, _ = torch.sort(self.unmasked2)
        
        self.masked1 = None
        self.unmasked1 = None
        for i in self.masked2:
            mask = torch.randperm(self.patch2 ,device = self.device)
            if self.masked1 is None:
                self.masked1 = mask[:-1] + i * self.patch2 # mention
                self.unmasked1 = (mask[-1:] + i * self.patch2)
            else :
                self.masked1 = torch.cat((self.masked1, mask[:-1]  + i * self.patch2))
                self.unmasked1 = torch.cat((self.unmasked1, mask[-1:] + i * self.patch2))

        mask = torch.arange(start=0,end=self.patch2,step=1,device=self.device)
        for i in self.unmasked2:
            self.unmasked1 = torch.cat((self.unmasked1,mask + i * self.patch2))
        
        self.masked0 = None
        self.unmasked0 = None
        for i in self.masked1:
            mask = torch.randperm(self.patch1 ,device = self.device)
            if self.masked0 is None:
                self.masked0 = mask[:-1] + i * self.patch1
                self.unmasked0 = (mask[-1:] + i * self.patch1)
            else :
                self.masked0 = torch.cat((self.masked0,mask[:-1]+i*self.patch1))
                self.unmasked0 = torch.cat((self.unmasked0,mask[-1:]+i*self.patch1))

        mask = torch.arange(start=0,end=self.patch1,step=1,device=self.device)
        for i in self.unmasked1:
            self.unmasked0 = torch.cat((self.unmasked0, mask + i * self.patch1))

        return self.masked0 ,self.unmasked0 , self.unmasked1, self.masked1, self.unmasked2, self.masked2
    
    def forward(self):
        self.masked0 ,self.unmasked0,self.unmasked1, self.masked1, self.unmasked2, self.masked2 = self.uniform_rand()
        return self.masked0 ,self.unmasked0,self.unmasked1, self.masked1, self.unmasked2, self.masked2
