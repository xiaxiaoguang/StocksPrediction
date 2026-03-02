from torch import nn
import torch
class Polymeric(nn.Module):
    """Patchify time series."""
    def __init__(self, patch_size):
        super().__init__()
        self.average = nn.AvgPool2d(kernel_size=(patch_size,1),stride=(patch_size,1))
        self.maxpool = nn.AvgPool2d(kernel_size=(patch_size,1),stride=(patch_size,1))
    def forward(self, history):
        history1=self.average(history)
        history2=self.average(history)
        output=torch.cat((history1,history2),dim=-1)
        return output
