import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Embed import DataEmbedding_inverted

class iRNNAnomalyDetector(nn.Module):
    """
    iRNN: Inverted RNN for Global Time Series Anomaly Detection.
    Uses a Bidirectional GRU to process the inverted variate dimensions.
    """
    def __init__(self, seq_len, d_model, dropout, e_layers=2, use_norm=True, rnn_type='GRU'):
        super(iRNNAnomalyDetector, self).__init__()
        self.seq_len = seq_len
        self.use_norm = use_norm
        
        # Inverted Embedding: [B, L, N] -> [B, N, d_model]
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout=dropout)
        
        # RNN Encoder (using GRU for better convergence/speed)
        # hidden_size is d_model // 2 because it's bidirectional
        self.rnn = getattr(nn, rnn_type)(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=e_layers,
            dropout=dropout if e_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        self.norm = nn.LayerNorm(d_model)
        
        # Anomaly Detection Head
        self.detector = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def detect(self, x_enc):
        # x_enc shape: [B, L, N]
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Embedding: [B, N, d_model]
        enc_out = self.enc_embedding(x_enc) 
         
        # RNN Forward: enc_out is [Batch, Variates, d_model]
        # output: [Batch, Variates, d_model] (since it's bidirectional)
        enc_out, _ = self.rnn(enc_out)
        
        # Residual + Norm
        enc_out = self.norm(enc_out)

        # --- GLOBAL AGGREGATION ---
        mean_pool = torch.mean(enc_out, dim=1) # [B, d_model]
        max_pool = torch.max(enc_out, dim=1)[0] # [B, d_model]
        
        global_representation = torch.cat([mean_pool, max_pool], dim=-1) # [B, 2 * d_model]

        return self.detector(global_representation)

    def forward(self, data, **kwargs):
        history_data = data[0] if isinstance(data, (list, tuple)) else data
        x_enc = history_data.squeeze(-1)
        return self.detect(x_enc)