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
    Computes loss for both Global Trend [B, 1] and Individual Stocks [B, N].
    """
    def __init__(self, alpha=0.5, beta=0.5, pos_weight_global=None, pos_weight_local=None):
        super(MultiObjectiveDetectionLoss, self).__init__()
        self.alpha = alpha  # Weight for global loss
        self.beta = beta    # Weight for individual stock loss
        
        # Pos weights to handle class imbalance independently
        pw_global = torch.tensor([pos_weight_global], dtype=torch.float32) if pos_weight_global else None
        pw_local = torch.tensor([pos_weight_local], dtype=torch.float32) if pos_weight_local else None
        print("pos_weight_global, pos_weight_local",pos_weight_global,pos_weight_local)
        self.loss_global = nn.BCEWithLogitsLoss(pos_weight=pw_global)
        self.loss_local = nn.BCEWithLogitsLoss(pos_weight=pw_local)

    def forward(self, preds: dict, targets: dict, **kwargs):
        """
        Expects preds and targets to be dictionaries containing:
        - 'global': Tensor of shape [B, 1]`
        - 'local': Tensor of shape [B, N]
        """
        # # 1. Global Trend Loss
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
        total_loss = (loss_l)
        
        return total_loss
    
class AdaptiveMultiObjectiveLoss(nn.Module):
    """
    Computes Adaptive Loss for Global Trend [B, 1] and Individual Stocks [B, N].
    Automatically balances tasks using Uncertainty Weighting and handles 
    imbalance using Focal Loss.
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super(AdaptiveMultiObjectiveLoss, self).__init__()
        
        # 1. Learnable task weights (initialized to 0)
        # We use log variance to avoid division by zero during training
        self.log_var_g = nn.Parameter(torch.zeros(1))
        self.log_var_l = nn.Parameter(torch.zeros(1))
        
        # 2. Focal Loss parameters
        self.gamma = gamma
        self.alpha_focal = alpha # Optional: standard alpha balancing for Focal Loss

    def focal_loss_with_logits(self, logits, targets):
        """
        Numerically stable Focal Loss implementation.
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # pt is the probability of the true class
        pt = torch.exp(-bce_loss) 
        
        # Calculate focal term
        focal_term = (1 - pt) ** self.gamma
        
        # Optional alpha weighting (different from your task alpha!)
        # alpha_t = self.alpha_focal * targets + (1 - self.alpha_focal) * (1 - targets)
        
        focal_loss = focal_term * bce_loss
        return focal_loss.mean()

    def forward(self, preds: dict, targets: dict, **kwargs):
        # Extract predictions and targets
        logits_g = preds['global']
        labels_g = targets['global'].float()
        
        logits_l = preds['local']
        labels_l = targets['local'].float()

        # 1. Compute Individual Focal Losses
        loss_g = self.focal_loss_with_logits(logits_g, labels_g)
        loss_l = self.focal_loss_with_logits(logits_l, labels_l)

        # 2. Apply Uncertainty Weighting
        # precision = exp(-log_var) = 1 / sigma^2
        precision_g = torch.exp(-self.log_var_g)
        precision_l = torch.exp(-self.log_var_l)

        # loss = precision * L + log_var
        weighted_loss_g = precision_g * loss_g + self.log_var_g
        weighted_loss_l = precision_l * loss_l + self.log_var_l

        # 3. Combine
        total_loss = weighted_loss_g + weighted_loss_l
        
        return total_loss