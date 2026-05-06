import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """A standard Feed-Forward Network used as a single Expert."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))

class MoELayer(nn.Module):
    """Mixture of Experts Layer with Top-K routing."""
    def __init__(self, d_model, num_experts, d_ff, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router calculates logits for each expert
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([Expert(d_model, d_ff, dropout) for _ in range(num_experts)])

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # 1. Routing: Calculate probabilities for each expert
        router_logits = self.router(x) # [B, N, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # 2. Top-K selection
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1) # [B, N, top_k]
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True) # Normalize weights
        
        # Flatten for easier processing
        flat_x = x.view(-1, d_model)
        flat_expert_indices = selected_experts.view(-1, self.top_k)
        flat_weights = routing_weights.view(-1, self.top_k)
        
        # Initialize an empty tensor for the output
        flat_output = torch.zeros_like(flat_x)

        # 3. Process tokens through their selected experts
        for i, expert in enumerate(self.experts):
            # Find tokens that selected the current expert
            expert_mask = (flat_expert_indices == i)
            token_indices, k_indices = expert_mask.nonzero(as_tuple=True)
            
            if token_indices.numel() > 0:
                expert_inputs = flat_x[token_indices]
                expert_outputs = expert(expert_inputs)
                
                # Apply the routing weight
                weights = flat_weights[token_indices, k_indices].unsqueeze(-1)
                
                # Add the expert's output to the final output
                flat_output.index_add_(0, token_indices, expert_outputs * weights)
                
        return flat_output.view(batch_size, seq_len, d_model)

class SpaceEncoderLayer(nn.Module):
    """Transformer Layer using Multi-Head Attention and MoE."""
    def __init__(self, d_model, n_heads, num_experts, d_ff, top_k=2, dropout=0.1):
        super().__init__()
        # Spatial Self-Attention (Stocks interacting with stocks)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.moe = MoELayer(d_model, num_experts, d_ff, top_k, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention Block
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # MoE Block
        moe_out = self.moe(x)
        x = self.norm2(x + self.dropout(moe_out))
        return x

class SpaceEncoder(nn.Module):
    """Main Space Encoder Module."""
    def __init__(self, spc_configs):
        super().__init__()
        d_model = spc_configs['d_model']
        n_heads = getattr(spc_configs, 'n_heads', 8)
        num_layers = getattr(spc_configs, 'num_layers', 2)
        num_experts = getattr(spc_configs, 'num_experts', 4)
        top_k = getattr(spc_configs, 'top_k', 2)
        d_ff = getattr(spc_configs, 'd_ff', d_model * 4)
        dropout = getattr(spc_configs, 'dropout', 0.5)
        num_stocks = getattr(spc_configs, 'num_stocks', 50)

        self.pos_embed = nn.Parameter(torch.randn(1, num_stocks, d_model) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList([
            SpaceEncoderLayer(d_model, n_heads, num_experts, d_ff, top_k, dropout)
            for _ in range(num_layers)
        ])
        # self.layers = nn.ModuleList([nn.LSTM(d_model,
        #                      hidden_size=d_model // 2,
        #                      num_layers=num_layers,
        #                      dropout=dropout,
        #                      batch_first=True,
        #                      bidirectional=True,)])


    def forward(self, x):
        # x expected shape: [Batch, Num_Stocks, d_model]
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)
        
        for layer in self.layers:
            x = layer(x)
        return x