from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, remove_self_loops
from torch_geometric.nn.inits import glorot, zeros
import torch
from torch import nn, Tensor

class GCNLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, improved: bool = False,
                 cached: bool = False, 
                 normalize: bool = True, bias: bool = True, **kwargs):
        super(GCNLayer, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = nn.Linear(in_channels, out_channels, bias=False)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor = None) -> Tensor:
        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple of node features as input while this layer does not support bipartite message passing. Please try other layers such as 'SAGEConv' or 'GraphConv' instead")
        breakpoint()

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = self.gcn_norm(edge_index, edge_weight, x.size(1), self.improved, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

        x = self.lin(x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def gcn_norm(self, edge_index: Tensor, edge_weight: Tensor, num_nodes: int, 
                 improved: bool, dtype: torch.dtype):

        deg_inv_sqrt = edge_index.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

        edge_weight = deg_inv_sqrt * edge_weight * deg_inv_sqrt

        return edge_index, edge_weight
    
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()
    def forward(self , x , A):
        A = A.to(x.device)
        if len(A.shape) == 3:
            x = torch.einsum('ncvl,nvw->ncwl',(x,A))
        else:
            if len(x.shape)==4:
                x = torch.einsum('ncvl,vw->ncwl',(x,A))
            else :
                x = torch.einsum('bnd,nm->bmd',(x,A))
            # 就是个矩阵乘法的封装,保证了第三维相乘
            # torch.Size([6, 32, 500, 143]) * torch.Size([6, 500, 500])
        return x.contiguous()

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=1,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = nn.Linear(c_in,c_out)
        self.dropout = nn.Dropout(dropout)
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1,a) 
                out.append(x2)
                x1 = x2
        h = torch.cat(out,dim=-1)
        h = self.mlp(h)
        h = self.dropout(h)
        return h
