from torch import nn
import torch

class PatchEmbedding(nn.Module):
    """Patchify time series."""

    def __init__(self, patch_size, in_channel, embed_dim, norm_layer):
        super().__init__()
        self.output_channel = embed_dim
        mid_dim = 16
        self.patch = patch_size
        self.num = num = len(patch_size)
        self.device='cuda:0'
        self.embedding = []
        self.outputlayer = nn.Conv2d(mid_dim,1,kernel_size=(1,1),device=self.device)
        self.inlayer = nn.Sequential(
                (nn.Conv2d(in_channel, mid_dim,kernel_size=(1,1),device=self.device)),
                (nn.BatchNorm2d(mid_dim,device=self.device)),
                (nn.ReLU())
            )
        for i in range(num):
            tmp2 = nn.Sequential(
                (nn.Conv2d(mid_dim ,mid_dim, kernel_size=(patch_size[i],1),stride=(patch_size[i],1),device=self.device)),
                (nn.BatchNorm2d(mid_dim,device=self.device)),
            )
            self.embedding.append(tmp2)

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
        history = long_term_history.reshape(batch_size*num_nodes, 1, len_time_series, num_feat)
        output = []
        history = self.inlayer(history)
        for i,layer_patch in enumerate(self.embedding):
            history = torch.relu(layer_patch(history))

            output.append(self.outputlayer(history).squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1))
        return output
