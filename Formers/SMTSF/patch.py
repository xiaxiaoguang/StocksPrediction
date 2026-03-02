from torch import nn

class PatchEmbedding(nn.Module):
    """Patchify time series."""

    def __init__(self, patch_sizeA , patch_sizeB,vision, in_channel,patch_length, embed_dim, norm_layer,num_feats = 45):
        super().__init__()
        self.output_channel = embed_dim
        self.patch1 = patch_sizeA
        self.patch2 = patch_sizeB
        self.input_channel = in_channel

        mid_dim = embed_dim*2
        # spc_dim = embed_dim//2
        
        self.week_embedding = nn.ModuleList()
        self.week_embedding.append(nn.Conv2d(in_channel,embed_dim,
                                            kernel_size=(1, num_feats),stride=(1, 1)))
        self.week_embedding.append(nn.ReLU())
        self.week_embedding.append(nn.BatchNorm2d(embed_dim)) # try InstanceNorm2d / LayerNorm2d
        self.week_embedding.append(nn.Conv2d(embed_dim,mid_dim,
                                            kernel_size=(patch_sizeA,1),stride=(patch_sizeA, 1)))
        self.week_embedding.append(nn.ReLU())
        self.week_embedding.append(nn.BatchNorm2d(mid_dim)) # try InstanceNorm2d / LayerNorm2d
        self.week_embedding.append(nn.Conv2d(mid_dim,embed_dim,
                                                kernel_size=(1,1),stride=(1, 1)))
       
        self.week_activate = nn.Tanh()

        self.month_embedding = nn.ModuleList()
        self.month_embedding.append(nn.Conv2d(embed_dim,mid_dim,
                                                kernel_size=(1,1),stride=(1,1)))
        self.month_embedding.append(nn.ReLU())
        self.month_embedding.append(nn.BatchNorm2d(mid_dim)) # try InstanceNorm2d / LayerNorm2d
        self.month_embedding.append(nn.Conv2d(mid_dim,mid_dim,
                                                kernel_size=(patch_sizeB,1),stride=(patch_sizeB,1)))
        self.month_embedding.append(nn.ReLU())
        self.month_embedding.append(nn.BatchNorm2d(mid_dim)) # try InstanceNorm2d / LayerNorm2d
        self.month_embedding.append(nn.Conv2d(mid_dim,embed_dim,
                                                kernel_size=(1,1),stride=(1,1)))
        
        self.space_embedding = nn.ModuleList()
        self.space_embedding.append(nn.Conv2d(embed_dim,mid_dim,
                                              kernel_size=(1,1),stride=(1,1)))
        self.space_embedding.append(nn.ReLU())
        self.space_embedding.append(nn.BatchNorm2d(mid_dim))
        self.space_embedding.append(nn.Conv2d(mid_dim ,mid_dim,
                                              kernel_size=(vision,1),padding=((vision-1)//2,0)))
        self.space_embedding.append(nn.ReLU())
        self.space_embedding.append(nn.BatchNorm2d(mid_dim))
        self.space_embedding.append(nn.Conv2d(mid_dim, embed_dim,
                                              kernel_size=(1,1),stride=(1,1)))
        
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

    def forward(self, long_term_history):
        """
        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, 1, P * L],
                                                which is used in the TSFormer.
                                                P is the number of segments (patches).

        Returns:
            torch.Tensor: patchified time series with shape [B, N, d, P]
        """
        batch_size, num_nodes, num_feat, len_time_series = long_term_history.shape
        long_term_history = long_term_history.unsqueeze(-1)
        long_term_history = long_term_history.reshape(batch_size*num_nodes, 1, len_time_series, num_feat)

        for layers in self.week_embedding:
                long_term_history = layers(long_term_history)
        week_data = self.norm_layer(long_term_history)

        month_term = self.week_activate(long_term_history)

        for layers in self.month_embedding:
             month_term = layers(month_term)
        month_data = self.norm_layer(month_term) # torch.Size([500, 256, 50, 1])
        space_term = month_term.clone().reshape(batch_size*month_data.shape[-2],self.output_channel,num_nodes,1)
        for layers in self.space_embedding:
             space_term = layers(space_term)
        space_data = self.norm_layer(space_term) # torch.Size([1,200, 500, 256])

        week_data = week_data.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)    # B, N, d, P
        assert week_data.shape[-1] == len_time_series // self.patch1
        month_data = month_data.squeeze(-1).view(batch_size,num_nodes,self.output_channel, -1)
        assert month_data.shape[-1] == week_data.shape[-1] // self.patch2# torch.Size([1, 500, 256, 50])
        space_data = space_data.squeeze(-1).view(batch_size,month_data.shape[-1],self.output_channel,-1)
        # assert space_data.shape[-1] == num_nodes                         # torch.Size([1, 200, 256, 500])
        return week_data,month_data,space_data
