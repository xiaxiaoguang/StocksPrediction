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
        self.patch=patch
        self.num=num=len(patch)

    def uniform_rand(self):
        mask = torch.randperm(self.num_tokens,device=self.device)
        mask_len = int(self.num_tokens * self.mask_ratio)

        self.masked2 = mask[:mask_len]
        self.unmasked2 = mask[mask_len:]
        self.masked2, _ = torch.sort(self.masked2)
        self.unmasked2, _ = torch.sort(self.unmasked2)
        
        self.masked = [self.masked2]
        self.unmasked = [self.unmasked2]
        for i in range(self.num-1):
            sufmasked = None
            sufunmasked = None            
            premasked = self.masked[i]
            preunmasked=self.unmasked[i]
            nowpatch = self.patch[self.num-i-1]
            
            for j in premasked:
                mask = torch.randperm(nowpatch ,device = self.device)
                # tmp = random.random()
                # if tmp < self.mask_ratio:
                mask_len = int(nowpatch * self.mask_ratio + 0.5)
                # else :
                #     mask_len = int(nowpatch * self.mask_ratio - 0.5)

                if sufmasked is None:
                    sufmasked = mask[:mask_len] + j * nowpatch
                    sufunmasked = mask[mask_len:] + j * nowpatch
                else :
                    sufmasked = torch.cat((sufmasked,mask[:mask_len] + j * nowpatch))
                    sufunmasked = torch.cat((sufunmasked,mask[mask_len:] + j * nowpatch))

            mask=torch.arange(start=0,end=nowpatch,step=1,device=self.device)
            for j in preunmasked:
                sufunmasked = torch.cat((sufunmasked,mask+j*nowpatch))
            self.masked.append(sufmasked)
            self.unmasked.append(sufunmasked)

        return self.masked,self.unmasked
    
    def forward(self):
        self.out1,self.out2 = self.uniform_rand()
        return self.out1,self.out2
