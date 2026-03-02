from torch import nn
import torch

class PatchEmbedding(nn.Module):
    """Patchify time series."""

    def __init__(self, patch_size, embed_dim, norm_layer, num_feats = 45):
        super().__init__()
        self.output_channel = embed_dim
        self.patch = patch_size
        self.num = num = len(patch_size)
        self.device='cuda:0'
        # self.inlayer1 = nn.Sequential(
        #         (nn.Conv2d(in_channel,embed_dim,kernel_size=(1,num_feats),device=self.device)),
        #         (nn.BatchNorm2d(embed_dim,device=self.device)),
        #         (nn.SiLU())
        #     )
        # self.inlayer2 = nn.Sequential(
        #         (nn.Conv2d(in_channel,embed_dim,kernel_size=(1,num_feats),device=self.device)),
        #         (nn.BatchNorm2d(embed_dim,device=self.device)),
        #         (nn.SiLU())
        #     )
        self.outputlayer = nn.Linear(num_feats*2,embed_dim)
        self.embedding = []        
        for i in range(num):
            tmp2 = nn.AvgPool1d(kernel_size=patch_size[i],stride=patch_size[i])
            tmp3 = nn.MaxPool1d(kernel_size=patch_size[i],stride=patch_size[i])
            self.embedding.append((tmp2,tmp3))


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
        long_term_history = long_term_history.reshape(batch_size*num_nodes, num_feat, len_time_series)
        
        output = []
        avghistory = long_term_history
        maxhistory = long_term_history

        for layer in self.embedding:
            avgp,maxp = layer
            avghistory = avgp(avghistory)
            maxhistory = maxp(maxhistory)
            midvalue = torch.cat((avghistory,
                                maxhistory),
                                dim=1).transpose(-1,-2)
            midvalue = self.outputlayer(midvalue).view(batch_size, num_nodes, -1, self.output_channel)
            output.append(midvalue)

        return output
