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