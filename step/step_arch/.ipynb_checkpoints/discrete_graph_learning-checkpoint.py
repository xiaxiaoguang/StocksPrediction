# Discrete Graph Learning
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from basicts.utils import load_pkl

from .similarity import batch_cosine_similarity, batch_dot_similarity


def sample_gumbel(shape, eps=1e-20, device=None):
    uniform = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(uniform + eps) + eps))


def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps, device=logits.device)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
    
    # 给logits施加一个随机扰动，然后返回比较大的那一侧为1，随机扰动的占比会比较大（两个负log
    # logits [32, 42849 , 2]
    """Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y

    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """
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


class DiscreteGraphLearning(nn.Module):
    """Dynamic graph learning module."""

    def __init__(self, dataset_name, k, input_seq_len, output_seq_len):
        super().__init__()

        self.k = k          # the "k" of knn graph
        self.num_nodes = {"METR-LA": 207, "PEMS04": 307, "PEMS03": 358, "PEMS-BAY": 325, "PEMS07": 883, "PEMS08": 170,"StockD":500}[dataset_name]
        # 这个超参数是正常的
        self.train_length = {"METR-LA": 23990, "PEMS04": 13599, "PEMS03": 15303, "PEMS07": 16513, "PEMS-BAY": 36482, "PEMS08": 14284,"StockD":2338}[dataset_name]
        self.node_feats = torch.from_numpy(load_pkl("datasets/" + dataset_name + "/data_in{0}_out{1}.pkl".format(input_seq_len, output_seq_len))["processed_data"]).float()[:self.train_length, :, 0]

        # CNN for global feature extraction
        ## for the dimension, see https://github.com/zezhishao/STEP/issues/1#issuecomment-1191640023
        self.dim_fc = {"METR-LA": 383552, "PEMS04": 217296, "PEMS03": 244560, "PEMS07": 263920, "PEMS-BAY": 583424, "PEMS08": 228256,"StockD":37120}[dataset_name]
        # 这里是他黔驴技穷了，只能使用超参数而不能提前算出来
        self.embedding_dim = 100
        ## network structure
        self.conv1 = torch.nn.Conv1d(1, 8, 10, stride=1)  # .to(device)
        self.conv2 = torch.nn.Conv1d(8, 16, 10, stride=1)  # .to(device)
        self.fc = torch.nn.Linear(self.dim_fc, self.embedding_dim) 
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(self.embedding_dim)

        # FC for transforming the features from TSFormer
        ## for the dimension, see https://github.com/zezhishao/STEP/issues/1#issuecomment-1191640023
        self.dim_fc_mean = {"METR-LA": 16128, "PEMS-BAY": 16128, "PEMS03": 16128 * 2, "PEMS04": 16128 * 2, "PEMS07": 16128, "PEMS08": 16128 * 2,"StockD":30}[dataset_name]
        self.fc_mean = nn.Linear(self.dim_fc_mean, 100)

        # discrete graph learning
        self.fc_cat = nn.Linear(self.embedding_dim, 2)
        self.fc_out = nn.Linear((self.embedding_dim) * 2, self.embedding_dim)
        self.dropout = nn.Dropout(0.5)
        
        # reference code https://github.com/chaoshangcs/GTS/blob/8ed45ff1476639f78c382ff09ecca8e60523e7ce/model/pytorch/model.py#L149
        def encode_one_hot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
            labels_one_hot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
            return labels_one_hot

        self.rel_rec = torch.FloatTensor(np.array(encode_one_hot(np.where(np.ones((self.num_nodes, self.num_nodes)))[0]), dtype=np.float32))
        self.rel_send = torch.FloatTensor(np.array(encode_one_hot(np.where(np.ones((self.num_nodes, self.num_nodes)))[1]), dtype=np.float32))
    # self.rel_rec 就是 num_node * num_node , num_node 大小的tensor，然后前num_node个为1 0 0 0...的one hot，之后num_node个为0 1 0 0 .... 这种规律
    # self.rel_send 就是大小和上面那个一样 但是每 num_node 行构成一个num_node 的对角矩阵
    def get_k_nn_neighbor(self, data, k=11*207, metric="cosine"):
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
        top_k, indices = torch.topk(adj, k, dim=-1)
        res.scatter_(-1, indices, top_k)
        adj = torch.where(res != 0, 1.0, 0.0).detach().clone()
        adj = adj.view(batch_size, num_nodes, num_nodes)
        adj.requires_grad = False
        return adj

    def forward(self, long_term_history, tsformer):
        """Learning discrete graph structure based on TSFormer.

        Args:
            long_term_history (torch.Tensor): very long-term historical MTS with shape [B, P * L, N, C], which is used in the TSFormer.
                                                P is the number of segments (patches), and L is the length of segments (patches).
            tsformer (nn.Module): the pre-trained TSFormer.

        Returns:
            torch.Tensor: Bernoulli parameter (unnormalized) of each edge of the learned dependency graph. Shape: [B, N * N, 2].
            torch.Tensor: the output of TSFormer with shape [B, N, P, d].
            torch.Tensor: the kNN graph with shape [B, N, N], which is used to guide the training of the dependency graph.
            torch.Tensor: the sampled graph with shape [B, N, N].
        """
        # long_term_history torch.Size([32, 2016, 207, 3])
        # 2016 is time length , 207 is variable numbers, what is 3 ?
        device = long_term_history.device
        batch_size, _, num_nodes, _ = long_term_history.shape
        # generate global feature
        # self.node_feats torch.Size([23990, 207])
        global_feat = self.node_feats.to(device).transpose(1, 0).view(num_nodes, 1, -1) #torch.Size([207, 1, 23990]) , 207 是图节点数
        # 这里做的是上卷积
        global_feat = self.bn1(F.relu(self.conv1(global_feat)))  # torch.Size([207, 8, 23981])      
        global_feat = self.bn2(F.relu(self.conv2(global_feat))) # torch.Size([207, 16, 23972])
        
        global_feat = global_feat.view(num_nodes, -1) # global feat 
        global_feat = F.relu(self.fc(global_feat))# fc 是 linear 而不是1d卷积
        global_feat = self.bn3(global_feat)
        global_feat = global_feat.unsqueeze(0).expand(batch_size, num_nodes, -1)                     # Gi in Eq. (2) torch.Size([32, 207, 100])

        # generate dynamic feature based on TSFormer
        hidden_states = tsformer(long_term_history[..., [0]])
        # hidden_states torch.Size([32, 207, 168, 96])
        # The dynamic feature has now been removed,
        # as we found that it could lead to instability in the learning of the underlying graph structure.
        # dynamic_feat = F.relu(self.fc_mean(hidden_states.reshape(batch_size, num_nodes, -1)))     # relu(FC(Hi)) in Eq. (2)
        # time series feature

        # global_feat size is torch.Size([32, 207, 100]) , 100 is embedding dimension
        node_feat = global_feat

        # learning discrete graph structure
        # node feat 理论上来讲就是W系数矩阵吧
        # 乘上向量的输入得到输出的维度。
        receivers = torch.matmul(self.rel_rec.to(node_feat.device), node_feat)
        senders = torch.matmul(self.rel_send.to(node_feat.device), node_feat)
        # 相当于说 ： 我们对于每条边都有receiver和sender两方贡献，receiver是入点的信息，sender代表终点的信息
        # 我们需要区分入点和终点具体的贡献是啥，所以我们才用了这两个
        
        edge_feat = torch.cat([senders, receivers], dim=-1)
        edge_feat = torch.relu(self.fc_out(edge_feat))
        # torch.Size([32, 42849, 100]) 把receiver和sender分开,之后再合起来
        # Bernoulli parameter (unnormalized) Theta_{ij} in Eq. (2)
        bernoulli_unnorm = self.fc_cat(edge_feat)
        # torch.Size([32, 42849, 2]) 
        
        # sampling
        ## differentiable sampling via Gumbel-Softmax in Eq. (4)
        sampled_adj = gumbel_softmax(bernoulli_unnorm, temperature=0.5, hard=True)
        # 就是bernoulli_unnorm里面2元组大的为1，小的为0
        sampled_adj = sampled_adj[..., 0].clone().reshape(batch_size, num_nodes, -1)
        # torch.Size([32, 207, 207]) , 相当于选择0代表的那些边
        ## remove self-loop
        mask = torch.eye(num_nodes, num_nodes).unsqueeze(0).bool().to(sampled_adj.device)
        # torch.Size([1, 207, 207])  对角矩阵罢了
        sampled_adj.masked_fill_(mask, 0)

        # prior graph based on TSFormer
        adj_knn = self.get_k_nn_neighbor(hidden_states.reshape(batch_size, num_nodes, -1), k=self.k*self.num_nodes, metric="cosine")
        # 余弦相似度建图！这里直接利用TSFormer强大能力的假设！
        mask = torch.eye(num_nodes, num_nodes).unsqueeze(0).bool().to(adj_knn.device)
        adj_knn.masked_fill_(mask, 0)

        # sample_adj 就是这个模型算出来的邻接矩阵，然后adj_knn就是根据余弦相似度直接选择的K近邻图！
        return bernoulli_unnorm, hidden_states, adj_knn, sampled_adj
