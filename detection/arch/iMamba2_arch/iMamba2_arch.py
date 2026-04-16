import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Embed import DataEmbedding_inverted

# Ensure you have installed mamba-ssm: pip install mamba-ssm
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    print("Please install mamba-ssm to use iMamba architecture.")

class iMamba2AnomalyDetector(nn.Module):
    """
    iMamba: Inverted Mamba for Global Time Series Anomaly Detection.
    Replaces Transformer blocks with Selective State Space Models (Mamba).
    """
    def __init__(self, seq_len, d_model, dropout, d_state=16, d_conv=4, expand=2, e_layers=2, use_norm=True):
        super(iMamba2AnomalyDetector, self).__init__()
        self.seq_len = seq_len
        self.use_norm = use_norm
        
        # Inverted Embedding: Each variate's history (L) becomes a single token of size d_model
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout=dropout)
        
        # Replace Transformer Encoder with a Stack of Mamba Layers
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,    # Model dimension
                d_state=d_state,    # SSM state dimension
                d_conv=d_conv,      # Local convolution width
                expand=expand,      # Block expansion factor
            ) for _ in range(e_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Anomaly Detection Head
        # Pooling Mean and Max to capture both trend and spikes across variates
        self.detector = nn.Sequential(
            nn.Linear(d_model, 1),
        )
        self.detector2 = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1) # Output raw logits [B, 1]
        )

    def detect(self, x_enc):
        # x_enc shape: [B, L, N] (Batch, Seq_Len, Num_Variates)
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Embedding: [B, L, N] -> [B, N, d_model]
        # In iMamba, we treat the N variates as the sequence length for the SSM
        enc_out = self.enc_embedding(x_enc) 
         
        # Forward through Mamba Layers
        for layer in self.layers:
            # Mamba block includes its own projection and selection logic
            # Residual connection and Norm added manually for stability
            res = enc_out
            enc_out = layer(enc_out)
            enc_out = self.dropout(enc_out) + res
            enc_out = self.norm(enc_out)

        # Final Classification
        logits_local = self.detector(enc_out).squeeze(-1) # [B, N]
        
        mean_pool = torch.mean(enc_out, dim=1) # [B, E]
        max_pool = torch.max(enc_out, dim=1)[0] # [B, E]
        global_representation = torch.cat([mean_pool, max_pool], dim=-1) # [B, 2 * E]
        logits_global = self.detector2(global_representation) # [B, 1]

        logits ={
            'local':logits_local,
            'global':logits_global,
        }

        return logits

    def forward(self, data, **kwargs):
        # history_data: [B, L, N, 1]
        if isinstance(data, (list, tuple)):
            history_data = data[0]
        else:
            history_data = data
            
        x_enc = history_data.squeeze(-1)
        logits = self.detect(x_enc)
        return logits