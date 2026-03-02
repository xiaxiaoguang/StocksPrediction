# Discrete Graph Learning
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from basicts.utils import load_pkl
from Formers.MSCFormer.transformer_layers import TransformerLayers3
from .similarity import batch_cosine_similarity, batch_dot_similarity


def sample_gumbel(shape, eps=1e-20, device=None):
    uniform = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(uniform + eps) + eps))


def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps, device=logits.device)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).to(logits.device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

class DiscreteGraphLearning3(nn.Module):
    """Dynamic graph learning module."""

    def __init__(self, dataset_name, k , input_seq_len,output_seq_len, 
                 pls=5 , move_window=5, decoder_depth=1,embed_dim=128,num_nodes=500,
                 mlp_ratio=4,num_heads=4,dropout=0.5,num_decoder=4):
        super().__init__()
        self.pls=pls
        self.k = k          # the "k" of knn graph
        self.num_nodes = {"METR-LA": 207, "PEMS04": 307, "PEMS03": 358, "PEMS-BAY": 325, "PEMS07": 883, "PEMS08": 170,"StockD":500,"StockDEBUG":500}[dataset_name]
        self.train_length = {"METR-LA": 23990, "PEMS04": 13599, "PEMS03": 15303, "PEMS07": 16513,\
                              "PEMS-BAY": 36482, "PEMS08": 14284,"StockD":2271,"StockDEBUG":2271}[dataset_name]
        self.dim_fc = {"METR-LA": 383552, "PEMS04": 217296, "PEMS03": 244560, "PEMS07": 263920,\
                        "PEMS-BAY": 583424, "PEMS08": 228256,"StockD":36112,"StockDEBUG":35984}[dataset_name]
        
        self.TARGET_FEATURES = [0]

        self.node_target = 0
        if dataset_name.startswith("StockD"):
            self.TARGET_FEATURES = list(range(45))
            self.node_target = 1

        self.node_feats = torch.from_numpy(load_pkl("datasets/" + dataset_name + "/data_in{0}_out{1}.pkl".format(input_seq_len, output_seq_len))["processed_data"]).float()[:self.train_length, :, self.node_target]
        self.embedding_dim = 64
        ## network structure

        tmp = move_window

        self.conv1 = torch.nn.Conv1d(1, 8, tmp, stride=1)  # .to(device)
        self.conv2 = torch.nn.Conv1d(8, 16, tmp, stride=1)  # .to(device)
        self.fc = torch.nn.Linear(self.dim_fc, self.embedding_dim)

        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(self.embedding_dim)

        self.fc_cat = nn.Linear(self.embedding_dim, 2)
        self.fc_out = nn.Linear((self.embedding_dim) * 2, self.embedding_dim)
        self.dropout = nn.Dropout(0.5)

        self.decoder_num = num_decoder
        self.decoder = TransformerLayers3(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)
        self.fc_his=[]
        self.bn_his=[]
        for i in range(self.decoder_num):
            self.fc_his.append(torch.nn.Linear(embed_dim,embed_dim,device='cuda:0'))
            self.bn_his.append(torch.nn.BatchNorm1d(num_nodes,device='cuda:0'))
        # reference code https://github.com/chaoshangcs/GTS/blob/8ed45ff1476639f78c382ff09ecca8e60523e7ce/model/pytorch/model.py#L149
        def encode_one_hot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
            labels_one_hot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
            return labels_one_hot

        self.rel_rec = torch.FloatTensor(np.array(encode_one_hot(np.where(np.ones((self.num_nodes, self.num_nodes)))[0]), dtype=np.float32))
        self.rel_send = torch.FloatTensor(np.array(encode_one_hot(np.where(np.ones((self.num_nodes, self.num_nodes)))[1]), dtype=np.float32))

    def get_k_nn_neighbor(self, data, pls=5 , k=10*500, metric="cosine"):
        """找到k近邻然后构图
        data: tensor B, N, D
        metric: cosine or dot
        """
        if metric == "cosine":
            batch_sim = batch_cosine_similarity(data, data)
        elif metric == "dot":
            batch_sim = batch_dot_similarity(data, data)    # B, N, N
        else:
            assert False, "unknown metric"
        batch_size, num_nodes, _ = batch_sim.shape
        adj = batch_sim.view(batch_size, num_nodes*num_nodes)
        res = torch.zeros_like(adj)
        top_k, indices = torch.topk(adj, k, dim=-1) # B,N*K 
        res.scatter_(-1, indices, top_k)
        adj = torch.where(res != 0, 1.0, 0.0).detach().clone()
        adj = adj.view(batch_size, num_nodes, num_nodes)

        top_pls,plsedge = torch.topk(batch_sim,pls,dim=2)
        res = torch.zeros_like(batch_sim)
        res.scatter_(-1,plsedge,top_pls)
        adj += torch.where(res!=0,1.0,0.0).detach().clone()
        # adj.requires_grad = True #注意这里
        return adj

    def forward(self, long_term_history, tsformer):
        device = long_term_history.device
        batch_size, _, num_nodes, _ = long_term_history.shape

        global_feat = self.node_feats.to(device).transpose(1, 0).view(num_nodes, 1, -1) #torch.Size([207, 1, 23990]) , 207 是图节点数
        global_feat = self.dropout(global_feat)

        global_feat = self.bn1(F.relu(self.conv1(global_feat)))  # torch.Size([207, 8, 23981])      
        global_feat = self.bn2(F.relu(self.conv2(global_feat))) # torch.Size([207, 16, 23972])
        global_feat = global_feat.view(num_nodes, -1) # global feat 
        global_feat = self.bn3(F.relu(self.fc(global_feat)))# fc 是 linear 而不是1d卷积
        global_feat = global_feat.unsqueeze(0).expand(batch_size, num_nodes, -1)    # Gi in Eq. (2) torch.Size([32, 207, 100])
        node_feat = global_feat


        receivers = torch.matmul(self.rel_rec.to(node_feat.device), node_feat)
        senders = torch.matmul(self.rel_send.to(node_feat.device), node_feat)
        edge_feat = torch.cat([senders, receivers], dim=-1)
        edge_feat = torch.relu(self.fc_out(edge_feat))
        bernoulli_unnorm = self.fc_cat(edge_feat)
        
        sampled_adj = gumbel_softmax(bernoulli_unnorm, temperature=0.5, hard=True)
        sampled_adj = sampled_adj[..., 0].clone().reshape(batch_size, num_nodes, -1)
        mask = torch.eye(num_nodes, num_nodes).unsqueeze(0).bool().to(sampled_adj.device)
        sampled_adj.masked_fill_(mask, 0)

        with torch.no_grad():
            hidden_states = tsformer(long_term_history[..., self.TARGET_FEATURES ])
        
        real_states = [hidden_states[self.decoder_num-1]]*self.decoder_num
        for i in range(self.decoder_num-2,-1,-1): # 最后一天的信息除外吗
            real_states[i] = self.decoder(tgt=hidden_states[i],memory=F.relu(real_states[i+1]))
            real_states[i] = self.bn_his[i](self.fc_his[i](real_states[i]))
            
        adj_knn = self.get_k_nn_neighbor(real_states[0].reshape(batch_size, num_nodes, -1), \
                                         pls=self.pls,
                                         k=self.k*self.num_nodes, metric="cosine")
        
        mask = torch.eye(num_nodes, num_nodes).unsqueeze(0).bool().to(adj_knn.device)
        adj_knn.masked_fill_(mask, 0)
        return bernoulli_unnorm, real_states[0], adj_knn, sampled_adj
