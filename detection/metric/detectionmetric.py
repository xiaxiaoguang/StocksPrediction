import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def _prepare_tensors(logits: torch.Tensor, labels: torch.Tensor):
    """Helper to safely move PyTorch tensors to CPU numpy arrays for sklearn."""
    # 1. Apply sigmoid to get probabilities (0.0 to 1.0)
    probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
    
    # 2. Threshold at 0.5 for hard predictions (0 or 1)
    preds = (probs > 0.5).astype(int)
    
    # 3. Format labels
    targets = labels.detach().cpu().numpy().astype(int).flatten()
    
    return probs, preds, targets

def detection_accuracy(logits: torch.Tensor, labels: torch.Tensor, null_val=None):
    _, preds, targets = _prepare_tensors(logits, labels)
    return accuracy_score(targets, preds)

def detection_precision(logits: torch.Tensor, labels: torch.Tensor, null_val=None):
    _, preds, targets = _prepare_tensors(logits, labels)
    return precision_score(targets, preds, zero_division=0)

def detection_recall(logits: torch.Tensor, labels: torch.Tensor, null_val=None):
    _, preds, targets = _prepare_tensors(logits, labels)
    return recall_score(targets, preds, zero_division=0)

def detection_f1(logits: torch.Tensor, labels: torch.Tensor, null_val=None):
    _, preds, targets = _prepare_tensors(logits, labels)
    return f1_score(targets, preds, zero_division=0)

def detection_auc(logits: torch.Tensor, labels: torch.Tensor, null_val=None):
    probs, _, targets = _prepare_tensors(logits, labels)
    
    # AUC fails and throws an error if a batch only contains one class (e.g., all 0s).
    # We catch that here and return 0.0 to prevent the training loop from crashing.
    if len(np.unique(targets)) <= 1:
        return 0.0
        
    return roc_auc_score(targets, probs)