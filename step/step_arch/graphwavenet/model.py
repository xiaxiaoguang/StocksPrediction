import torch
from torch import nn
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self , x , A):
        A = A.to(x.device)
        if len(A.shape) == 3:
            x = torch.einsum('ncvl,nvw->ncwl',(x,A))
        else:
            x = torch.einsum('ncvl,vw->ncwl',(x,A))
            # 就是个矩阵乘法的封装,保证了第三维相乘
            # torch.Size([6, 32, 500, 143]) * torch.Size([6, 500, 500])
        return x.contiguous()

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3 , num_nodes=500):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim,hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(num_nodes))
        self.layers.append(nn.Dropout())
        for i in range(num_layers-2):
            self.layers.append(nn.Linear(hidden_dim,hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm2d(num_nodes))
            self.layers.append(nn.Dropout())
        self.layers.append(nn.Linear(hidden_dim,output_dim))
    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x


class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            # support 提供了图的一个序列
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a) 
                # 每次多乘上一个a
                out.append(x2)
                x1 = x2
        # 和support求一个前缀积 
        h = torch.cat(out,dim=1) 
        # out : torch.Size([6, 224, 500, 143])
        # 把所有积连起来
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        # 最后mlp意思一下
        return h

class GraphWaveNet(nn.Module):
    """
        Paper: Graph WaveNet for Deep Spatial-Temporal Graph Modeling.
        Link: https://arxiv.org/abs/1906.00121
        Ref Official Code: https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py
    """

    def __init__(self, num_nodes, support_len, dropout=0.3, gcn_bool=True, t_len = 12,
                  addaptadj=True, aptinit=None, embed_dim=96, in_dim=2 , out_dim=12,
                   residual_channels=32, dilation_channels=32,skip_channels=256,
                   end_channels=512,kernel_size=2,blocks=4,layers=2, **kwargs):
        """
            kindly note that although there is a 'supports' parameter, 
            we will not use the prior graph if there is a learned dependency graph.
            Details can be found in the feed forward function.
        """
        
        super(GraphWaveNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.in_dim = in_dim
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.fc_his = nn.Linear(embed_dim, embed_dim)
        self.numcount = 0
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1,1))
        
        receptive_field = 1

        self.t_len = t_len
        self.supports_len = support_len

        if gcn_bool and addaptadj:
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 20), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(20, num_nodes), requires_grad=True)
                # 这里的20应该是类似高维特征的东西
                self.supports_len +=1
            else:
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, 
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, 
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=residual_channels, 
                                                     kernel_size=(1, 1)))
                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels, 
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                self.t_len -= (kernel_size-1)*new_dilation
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))
        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=skip_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=skip_channels, out_channels=out_dim, kernel_size=(1,1), bias=True)
        
        # self.end_conv   = nn.Conv2d(in_channels=skip_channels,out_channels=out_dim,kernel_size=(1,1),bias=True)
        # self.end_mlp = MLP(self.t_len,256,in_dim,num_layers=3,num_nodes=num_nodes)
        self.receptive_field = receptive_field

    def _calculate_random_walk_matrix(self, adj_mx):
        B, N, N = adj_mx.shape

        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[1])).unsqueeze(0).expand(B, N, N).to(adj_mx.device)
        # 后面哪些就是Batch个N*N对角矩阵！size是 B,N,N
        d = torch.sum(adj_mx, 2) # 度数
        d_inv = 1. / d # 度数倒数
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(adj_mx.device), d_inv)
        # 0替换掉inf
        d_mat_inv = torch.diag_embed(d_inv)
        # one hot 嵌入得到 B , N , N
        random_walk_mx = torch.bmm(d_mat_inv, adj_mx)
        # bmm 就是batch production，相当于 d_mat_inv的二三维和adj_mx的二三维乘起来
        # 函数作用就是除以度数！
       
        return random_walk_mx

    def forward(self, input, hidden_states, sampled_adj):
        """feed forward of Graph WaveNet.

        Args:
            input (torch.Tensor): input history MTS with shape [B, L, N, C].
            His (torch.Tensor): the output of TSFormer of the last patch (segment) with shape [B, N, d].
            adj (torch.Tensor): the learned discrete dependency graph with shape [B, N, N].

        Returns:
            torch.Tensor: prediction with shape [B, N, L]
        """
        input = input.transpose(1, 3) # reshape input: [B, L, N, C] -> [B, C, N, L]
        input = input[:, :self.in_dim, :, :]        
        in_len = input.size(3)
        
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        
        x = self.start_conv(x) #升维度
        skip = 0
        # ====== if use learned adjacency matrix, then reset the self.supports ===== #

        self.supports = [self._calculate_random_walk_matrix(sampled_adj),
                          self._calculate_random_walk_matrix(sampled_adj.transpose(-1, -2))]
        # self[1]是反向图
        
        # calculate the current adaptive adj matrix
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            # self.nodevec1 是 N,10 , self.nodevec2 是 10,N ,乘起来得到这个图 
            new_supports = self.supports + [adp]
            # 相当于在这里修订一下
            
        # WaveNet layers
        for i in range(self.blocks * self.layers):
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            #(dilation, init_dilation) = self.dilations[i]
            #residual = dilation_func(x, `dilation, init_dilation, i)
            residual = x
            # dilated convolution 这里就是用大小为2的方法逐层扩展信息
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            # torch.Size([8, 2, 500, 144]) -> torch.Size([8, 32, 500, 143]) 减一了
            gate = self.gate_convs[i](residual) # Conv1d 卷积 1*2 
            gate = torch.sigmoid(gate)
            # 这里是Gate TCN
            x = filter * gate
            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            # torch.Size([8, 32, 500, 143])升维了torch.Size([8, 256, 500, 143])
            # 1*1 卷积核 升到 skip 维度
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                    #torch.Size([8, 256, 500, 143])选择这条路，也就是图可以修改，而且所有图，最后都会mlp
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x) # 归一化得到x            
        # hidden_states = self.fc_his(hidden_states)        # B, N, D
        hidden_states = hidden_states.transpose(1, 2).unsqueeze(-1)
        skip = skip + hidden_states

        x = nn.SiLU()(skip)
        x = nn.SiLU()(self.end_conv_1(x)) 
        x = self.end_conv_2(x)
        # reshape output: [B, P, N, 1] -> [B, N, P]
        x = x.squeeze(-1).transpose(1, 2)
        x = x.unsqueeze(-1)

        self.numcount += 1 

        if self.numcount % 150 == 0:
            with open("test2.out",'a') as f:
                print(x[0,:,0,0].flatten(),"\n",skip[0,:,0,0],file=f)

        return x
