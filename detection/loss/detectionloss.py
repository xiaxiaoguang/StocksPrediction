import torch
import torch.nn as nn

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
    Computes loss for both Global Trend [B, 1] and Individual Stocks [B, N].
    """
    def __init__(self, alpha=0.5, beta=0.5, pos_weight_global=None, pos_weight_local=None):
        super(MultiObjectiveDetectionLoss, self).__init__()
        self.alpha = alpha  # Weight for global loss
        self.beta = beta    # Weight for individual stock loss
        
        # Pos weights to handle class imbalance independently
        pw_global = torch.tensor([pos_weight_global], dtype=torch.float32) if pos_weight_global else None
        pw_local = torch.tensor([pos_weight_local], dtype=torch.float32) if pos_weight_local else None
        
        self.loss_global = nn.BCEWithLogitsLoss(pos_weight=pw_global)
        self.loss_local = nn.BCEWithLogitsLoss(pos_weight=pw_local)

    def forward(self, preds: dict, targets: dict, **kwargs):
        """
        Expects preds and targets to be dictionaries containing:
        - 'global': Tensor of shape [B, 1]
        - 'local': Tensor of shape [B, N]
        """
        # 1. Global Trend Loss
        logits_g = preds['global']
        labels_g = targets['global'].float()
        
        if self.loss_global.pos_weight is not None:
            self.loss_global.pos_weight = self.loss_global.pos_weight.to(logits_g.device)
            
        loss_g = self.loss_global(logits_g, labels_g)
        
        # 2. Individual Stocks Loss
        logits_l = preds['local']
        labels_l = targets['local'].float()
        
        if self.loss_local.pos_weight is not None:
            self.loss_local.pos_weight = self.loss_local.pos_weight.to(logits_l.device)
            
        loss_l = self.loss_local(logits_l, labels_l)
        
        # 3. Combine
        total_loss = (self.alpha * loss_g) + (self.beta * loss_l)
        
        return total_loss