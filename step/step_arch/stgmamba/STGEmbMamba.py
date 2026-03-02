import torch.nn as nn
import torch
from mamba_ssm import Mamba

class OldPatchEmbedding(nn.Module):
    """Patchify time series."""

    def __init__(self, patch_size, in_channel, embed_dim, norm_layer, num_feats=45):
        super().__init__()
        self.output_channel = embed_dim
        self.len_patch = patch_size
        self.input_channel = in_channel
        self.output_channel = embed_dim
        self.input_embedding = nn.Conv2d(
            in_channel,
            embed_dim,
            kernel_size=(self.len_patch, num_feats),
            stride=(self.len_patch, num_feats))
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

    def forward(self, long_term_history):
        batch_size, num_nodes, num_feat, len_time_series = long_term_history.shape
        long_term_history = long_term_history.unsqueeze(-1)
        long_term_history = long_term_history.reshape(batch_size * num_nodes, 1, len_time_series, num_feat)
        output = self.input_embedding(long_term_history)
        output = self.norm_layer(output)
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)
        assert output.shape[-1] == len_time_series / self.len_patch
        return output



class STGEmbMamba(nn.Module):
    def __init__(self, patch_size, in_channel, embed_dim, norm_layer, num_feats, fea_size, d_model, d_state, d_conv, expand, n_layers):
        super(STGEmbMamba, self).__init__()
        self.patch_embedding = OldPatchEmbedding(patch_size, in_channel, embed_dim, norm_layer, num_feats)
        self.fea_size = fea_size
        self.embed_dim = embed_dim
        self.node_num = None

        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(n_layers)
        ])
        
        self.linear = nn.Linear(d_model, 1)

    def forward(self, inputs):
        batch_size, step_size, node_num = inputs.shape
        self.node_num = node_num
        
        inputs = inputs.permute(0, 2, 1).unsqueeze(2)  # [batch_size, node_num, 1, step_size]
        embedded = self.patch_embedding(inputs)  # [batch_size, node_num, embed_dim, step_size]
        
        embedded = embedded.view(batch_size, node_num * step_size, self.embed_dim)  # [batch_size, node_num * step_size, embed_dim]
        
        for layer in self.mamba_layers:
            embedded = layer(embedded)
        
        output = self.linear(embedded)  # [batch_size, node_num * step_size, 1]
        output = output.view(batch_size, node_num, step_size).permute(0, 2, 1).contiguous()  # [batch_size, step_size, node_num]
        
        return output

if __name__ == "__main__":
    # Example usage
    patch_size = 1
    in_channel = 1
    embed_dim = 96
    norm_layer = None
    num_feats = 1
    fea_size = 96
    d_model = 96
    d_state = 64
    d_conv = 4
    expand = 2
    n_layers = 2

    model = STGEmbMamba(patch_size, in_channel, embed_dim, norm_layer, num_feats, fea_size, d_model, d_state, d_conv, expand, n_layers).cuda()

    # Test
    input_tensor = torch.randn(48, 12, 312).cuda()
    output = model(input_tensor)
    print(output.shape)  # Should output [48, 12, 312]
