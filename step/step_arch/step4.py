import torch
import math
from torch import nn
from torch_geometric.nn import GCNConv
from Formers.CAOFormer import CAOFormer
from .GCNConv import GCNLayer,gcn


class STEP4(nn.Module):
    """Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting"""

    def __init__(self, dataset_name, pre_trained_tsformer_path, 
                 tsformer_args, in_channels, out_channels,num_nodes):
        super().__init__()
        self.dataset_name = dataset_name
        self.pre_trained_tsformer_path = pre_trained_tsformer_path
        self.tsformer = CAOFormer(**tsformer_args)
        
        self.li1  = nn.Linear(in_channels,in_channels//2)
        self.act = nn.GELU()
        self.li2  = nn.Linear(in_channels//2,out_channels)

        self.learnable_graph = nn.Parameter(torch.rand(num_nodes,num_nodes))
        self.gcnlayer=gcn(in_channels,in_channels,dropout=0.1)

        self.load_pre_trained_model()

    def load_pre_trained_model(self):
        checkpoint_dict = torch.load(self.pre_trained_tsformer_path)
        self.tsformer.load_state_dict(checkpoint_dict["model_state_dict"])
        for param in self.tsformer.parameters():
            param.requires_grad = False

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, 
                future_data: torch.Tensor, batch_seen: int, epoch: int, **kwargs) -> torch.Tensor:
        
        long_term_history = long_history_data # torch.Size([32, 2016, 207, 3])
        with torch.no_grad():
            hidden_states = self.tsformer(long_term_history)
        
        y_hat = self.gcnlayer(hidden_states,[self.learnable_graph])
        y_hat = self.li2(self.act(self.li1(y_hat)))

        if epoch is not None and batch_seen % 200 == 0 :
            with open("test0.out",'a') as f:
                print(self.learnable_graph[0,:100],file=f)
                print(y_hat[0,:100,0],file=f)
                
        return y_hat.unsqueeze(1)