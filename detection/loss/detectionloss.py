import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryDetectionLoss(nn.Module):
    """
    Standard Binary Cross Entropy Loss for Anomaly Detection.
    Expects raw logits (no sigmoid) from the model to ensure numerical stability.
    """
    def __init__(self, pos_weight=None):
        super(BinaryDetectionLoss, self).__init__()
        # pos_weight helps combat class imbalance. 
        # E.g., if you have 10x more normal days than anomalies, pos_weight=10.
        if pos_weight is not None:
            self.pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        # Ensure labels are float to match logits
        labels = labels.float()
        
        # Move pos_weight to the correct GPU dynamically
        weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
        
        # Calculate loss
        return loss_fn(logits, labels)
    
class MultiObjectiveDetectionLoss(nn.Module):
    """
    Computes loss for Global Trend, Individual Stocks, and MoE Load Balancing.
    """
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.1, 
                 pos_weight_global=None, pos_weight_local=None, use_focal=False):
        super().__init__()
        self.alpha = alpha      # Weight for global loss
        self.beta = beta        # Weight for individual stock loss
        self.gamma = gamma      # Weight for MoE load balancing loss
        self.use_focal = use_focal # Toggle for Focal Loss
        
        pw_g = torch.tensor([pos_weight_global], dtype=torch.float32) if pos_weight_global else None
        pw_l = torch.tensor([pos_weight_local], dtype=torch.float32) if pos_weight_local else None
        
        # If using focal loss, we need the raw unreduced BCE loss first
        reduction = 'none' if use_focal else 'mean'
        self.loss_global = nn.BCEWithLogitsLoss(pos_weight=pw_g, reduction=reduction)
        self.loss_local = nn.BCEWithLogitsLoss(pos_weight=pw_l, reduction=reduction)

    def _focal_loss_wrapper(self, bce_loss, targets, focal_alpha=0.25, focal_gamma=2.0):
        # bce_loss is already computed with pos_weights
        pt = torch.exp(-bce_loss) 
        focal_weight = (focal_alpha * targets + (1 - focal_alpha) * (1 - targets)) * (1 - pt) ** focal_gamma
        return (focal_weight * bce_loss).mean()

    def forward(self, preds: dict, targets: dict, **kwargs):
        # 1. Global Trend Loss
        logits_g = preds['global']
        labels_g = targets['global'].float()
        
        if self.loss_global.pos_weight is not None:
            self.loss_global.pos_weight = self.loss_global.pos_weight.to(logits_g.device)
        loss_g = self.loss_global(logits_g, labels_g)
        if self.use_focal:
            loss_g = self._focal_loss_wrapper(loss_g, labels_g)
            
        # 2. Individual Stocks Loss
        logits_l = preds['local']
        labels_l = targets['local'].float()

        if self.loss_local.pos_weight is not None:
            self.loss_local.pos_weight = self.loss_local.pos_weight.to(logits_l.device)
        loss_l = self.loss_local(logits_l, labels_l)
        if self.use_focal:
            loss_l = self._focal_loss_wrapper(loss_l, labels_l)
        
        # 3. MoE Load Balancing Loss
        loss_moe = 0.0
        if 'routing_weights' in preds and preds['routing_weights'] is not None:
            # Expected shape: [Batch, Seq_Len, Num_Experts] or [Batch, Num_Experts]
            gates = preds['routing_weights'] 
            
            # Flatten all dimensions except the expert dimension
            gates = gates.view(-1, gates.size(-1))
            num_experts = gates.size(-1)
            
            # Calculate the mean probability assigned to each expert across the batch
            mean_probs = gates.mean(dim=0)
            
            # Continuous load balancing proxy: E * sum(P_i^2)
            # Minimizing this forces the distribution toward 1/E (uniform)
            loss_moe = num_experts * torch.sum(mean_probs * mean_probs)
        
        # 4. Combine
        # total_loss = (self.alpha * loss_g) + (self.beta * loss_l) + (self.gamma * loss_moe)
        total_loss = loss_l
        return total_loss