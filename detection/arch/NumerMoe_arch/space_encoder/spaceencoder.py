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

class DenseMoELayer(nn.Module):
    """Ablation: Passes tokens through all experts and averages them (No routing)."""
    def __init__(self, d_model, num_experts, d_ff, top_k=None, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        
        # Notice: No self.router is defined here.
        self.experts = nn.ModuleList([Expert(d_model, d_ff, dropout) for _ in range(num_experts)])

    def forward(self, x):
        # Process x through all experts simultaneously
        # List of tensors of shape [Batch, Seq_Len, d_model]
        expert_outputs = [expert(x) for expert in self.experts]

        # Stack along a new dimension and average
        # Shape becomes [Batch, Seq_Len, d_model, num_experts]
        stacked_outputs = torch.stack(expert_outputs, dim=-1)
        
        # Average across the experts
        out = torch.mean(stacked_outputs, dim=-1)
        
        # Create dummy uniform weights to prevent the load balancing loss from crashing
        # Shape: [Batch, Seq_Len, num_experts]
        dummy_weights = torch.ones(x.size(0), x.size(1), self.num_experts, device=x.device) / self.num_experts
        
        return out

class GiantExpertLayer(nn.Module):
    """Ablation: A single expert scaled up to match the parameter count of the MoE."""
    def __init__(self, d_model, num_experts, d_ff, top_k=None, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        
        # Scale the hidden dimension to roughly equal the total parameters of N experts
        giant_d_ff = d_ff * num_experts
        
        # A single massive expert
        self.expert = Expert(d_model, giant_d_ff, dropout)

    def forward(self, x):
        # Process directly through the single giant expert
        out = self.expert(x)
        
        # Create dummy uniform weights to prevent the load balancing loss from crashing
        dummy_weights = torch.ones(x.size(0), x.size(1), self.num_experts, device=x.device) / self.num_experts
        
        return out

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

class SpatialMixerBlock(nn.Module):
    """A single Mixer block that processes spatial correlations and feature correlations."""
    def __init__(self, num_stocks, d_model, dropout=0.5):
        super().__init__()
        
        # 1. Stock/Spatial Mixing (Mixes information across different stocks)
        self.norm1 = nn.LayerNorm(d_model)
        self.stock_mixer = nn.Sequential(
            nn.Linear(num_stocks, num_stocks),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Feature Mixing (Mixes information within the d_model representation)
        self.norm2 = nn.LayerNorm(d_model)
        self.feature_mixer = nn.Sequential(
            nn.Linear(d_model, d_model * 4), # Standard 4x expansion
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: [Batch, Num_Stocks, d_model]
        
        # Step 1: Stock Mixing
        residual = x
        x = self.norm1(x)
        x = x.transpose(1, 2)          # Shape: [Batch, d_model, Num_Stocks]
        x = self.stock_mixer(x)        # Mix across stocks
        x = x.transpose(1, 2)          # Shape: [Batch, Num_Stocks, d_model]
        x = x + residual               # Skip connection
        
        # Step 2: Feature Mixing
        residual = x
        x = self.norm2(x)
        x = self.feature_mixer(x)      # Mix across features
        x = x + residual               # Skip connection
        
        return x



class SpaceEncoder(nn.Module):
    """Main Space Encoder Module."""
    def __init__(self, spc_configs):
        super().__init__()
        d_model = spc_configs.get('d_model', 512)
        n_heads = getattr(spc_configs, 'n_heads', 8)
        num_layers = getattr(spc_configs, 'num_layers', 2)
        num_experts = getattr(spc_configs, 'num_experts', 4)
        top_k = getattr(spc_configs, 'top_k', 2)
        d_ff = getattr(spc_configs, 'd_ff', d_model * 4)
        dropout = getattr(spc_configs, 'dropout', 0.5)
        num_stocks = getattr(spc_configs, 'num_stocks', 50)
        
        # NEW: Strategy toggle for ablation ('moe', 'vanilla', or 'mixer')
        self.encoder_type = getattr(spc_configs, 'encoder_type', 'moe')

        self.pos_embed = nn.Parameter(torch.randn(1, num_stocks, d_model) * 0.1)
        self.pos_drop = nn.Dropout(p=dropout)
        
        if self.encoder_type == 'moe':
            # Your original MoE layers
            self.layers = nn.ModuleList([
                SpaceEncoderLayer(d_model, n_heads, num_experts, d_ff, top_k, dropout)
                for _ in range(num_layers)
            ])
            
        elif self.encoder_type == 'vanilla':
            # Official PyTorch Vanilla Transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='gelu', # GeLU is standard for modern transformers
                batch_first=True   # CRITICAL: Ensures input shape is [Batch, Seq_Len, d_model]
            )
            self.vanilla_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
        elif self.encoder_type == 'mixer':
            # The NumerMixer we built previously
            self.layers = nn.ModuleList([
                SpatialMixerBlock(num_stocks, d_model, dropout)
                for _ in range(num_layers)
            ])

    def forward(self, x):
        # x expected shape: [Batch, Num_Stocks, d_model]
        
        # Positional embedding is usually kept for both MoE and Vanilla Transformers
        if self.encoder_type in ['moe', 'vanilla']:
            x = x + self.pos_embed[:, :x.size(1), :]
            x = self.pos_drop(x)
        
        # Route to the correct encoder
        if self.encoder_type == 'vanilla':
            # PyTorch's native transformer handles the layer looping internally
            x = self.vanilla_transformer(x)
        else:
            # For custom MoE or Mixer blocks
            for layer in self.layers:
                # If your MoE layer returns a tuple (x, routing_weights), handle it here:
                # For this snippet, assuming it just returns x like your original draft
                if isinstance(layer(x), tuple):
                    x, _ = layer(x) 
                else:
                    x = layer(x)
                    
        return x