import torch

def detection_accuracy(preds: dict, targets: dict, task: str, null_val=None):
    # Unpack dict, apply sigmoid, threshold, and flatten
    p = (torch.sigmoid(preds[task]) > 0.5).float().view(-1)
    t = targets[task].view(-1)
    return (p == t).float().mean()

def detection_precision(preds: dict, targets: dict, task: str, null_val=None):
    p = (torch.sigmoid(preds[task]) > 0.5).float().view(-1)
    t = targets[task].view(-1)
    
    tp = (p * t).sum()
    fp = (p * (1.0 - t)).sum()
    
    if tp + fp == 0: return torch.tensor(0.0, device=p.device)
    return tp / (tp + fp)

def detection_recall(preds: dict, targets: dict, task: str, null_val=None):
    p = (torch.sigmoid(preds[task]) > 0.5).float().view(-1)
    t = targets[task].view(-1)
    
    tp = (p * t).sum()
    fn = ((1.0 - p) * t).sum()
    
    if tp + fn == 0: return torch.tensor(0.0, device=p.device)
    return tp / (tp + fn)

def detection_f1(preds: dict, targets: dict, task: str, null_val=None):
    precision = detection_precision(preds, targets, task)
    recall = detection_recall(preds, targets, task)
    
    if precision + recall == 0: return torch.tensor(0.0, device=precision.device)
    return 2.0 * (precision * recall) / (precision + recall)

def detection_auc(preds: dict, targets: dict, task: str, null_val=None):
    # AUC needs raw probabilities, not binary 0/1 predictions
    probs = torch.sigmoid(preds[task]).view(-1)
    t = targets[task].view(-1)
    
    total_pos = t.sum()
    total_neg = t.numel() - total_pos
    
    if total_pos == 0 or total_neg == 0: return torch.tensor(0.0, device=probs.device)
        
    desc_score_indices = torch.argsort(probs, descending=True)
    targets_sorted = t[desc_score_indices]
    
    tps = torch.cumsum(targets_sorted, dim=0)
    fps = torch.cumsum(1.0 - targets_sorted, dim=0)
    
    tpr = tps / total_pos
    fpr = fps / total_neg
    
    return torch.trapz(tpr, fpr)