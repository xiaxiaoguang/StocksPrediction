import torch
from torch import nn
import math

from .tsformer import tsformer

from .graphwavenet import GraphWaveNet
from .graphwavenet import GraphWaveNet2
from .graphwavenet import GraphWaveNet3
from .graphwavenet import GraphWaveNet4
from .stgmamba import KFGN_Mamba

from .discrete_graph_learning import DiscreteGraphLearning   # 一般的版本
from .discrete_graph_learning2 import DiscreteGraphLearning2 # 想要利用space decoder的信息同时处理
from .discrete_graph_learning3 import DiscreteGraphLearning3 # 想再后端增加一个decoder，处理空间信息
from .discrete_graph_learning4 import DiscreteGraphLearning4
from .discrete_graph_learning5 import DiscreteGraphLearning5

class STEP3(nn.Module):
    """Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting"""

    def __init__(self, dataset_name, pre_trained_tsformer_path, 
                 tsformer_args, backend_args , dgl_args):
        super().__init__()
        self.dataset_name = dataset_name
        self.pre_trained_tsformer_path = pre_trained_tsformer_path
        # self.backend = GraphWaveNet(**backend_args)
        self.backend = GraphWaveNet4(**backend_args)
        self.discrete_graph_learning = DiscreteGraphLearning5(**dgl_args)

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, **kwargs) -> torch.Tensor:

        short_term_history = history_data     # [B, L, N, 3] torch.Size([32, 12, 207, 3])
        long_term_history = long_history_data # torch.Size([32, 2016, 207, 3])
        batch_size, _, num_nodes, _ = short_term_history.shape
        bernoulli_unnorm, hidden_states, adj_knn, sampled_adj = self.discrete_graph_learning(long_term_history)
        
        # self.cnt+=1
        # if self.cnt % 300 == 0 :
        # with open("test1.out",'a') as f:
        #         print(hidden_states[0,0,0],file=f)
        # breakpoint()
        # hidden_states = hidden_states[:, :, -1, :]
        
        y_hat = self.backend(short_term_history, hidden_states=hidden_states, sampled_adj=sampled_adj).transpose(1, 2)

        gsl_coefficient = 0 
        
        return y_hat.unsqueeze(-1), bernoulli_unnorm.softmax(-1)[..., 0].clone().reshape(batch_size, num_nodes, num_nodes), adj_knn, gsl_coefficient
