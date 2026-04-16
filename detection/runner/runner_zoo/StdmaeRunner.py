import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np

# Assuming these are available in your environment based on the iTransformer example
from ..base_adt_runner import AnomalyDetectionRunner 
from detection.metric import detection_accuracy, detection_precision, detection_recall, detection_f1, detection_auc

class STDMAEAnomalyRunner(AnomalyDetectionRunner):
    """
    Runner for Anomaly Detection using STDMAE.
    Uses long_history for graph learning and short_history for detection.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)

        # 1. Setup Classification Metrics
        self.metrics = {
            "Accuracy": detection_accuracy,
            "Precision": detection_precision,
            "Recall": detection_recall,
            "F1": detection_f1,
            "AUC": detection_auc
        }
        
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        # Note: target_features is usually not needed for binary classification [B, 1]

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        """
        Feed forward process for STDMAE Anomaly Detection.
        
        Args:
            data (tuple): (history_data, long_history_data, labels) 
                         provided by the AnomalyDetectionDataset
        Returns:
            tuple: (logits, labels, extra_info) 
                  where extra_info contains graph data for potential GSL losses
        """
        # 1. Unpack data (Note: Anomaly tasks usually don't have 'future_data')
        # Ensure your Dataset class provides long_history_data
        history_data, labels = data
        
        history_data = self.to_running_device(history_data)      # [B, L_short, N, C]
        long_history_data = self.to_running_device(history_data) # [B, L_long, N, C]
        labels = self.to_running_device(labels)                  # [B, 1]

        # 2. Feature Selection
        history_data = self.select_input_features(history_data)
        long_history_data = self.select_input_features(long_history_data)

        # 3. Model Inference
        # STDMAE returns: (prediction, pred_adj, prior_adj, gsl_coefficient)
        logits = self.model(
            history_data=history_data, 
            long_history_data=long_history_data, 
            future_data=None, 
            batch_seen=iter_num, 
            epoch=epoch
        )

        # 4. Shape Validation
        # Based on your previous requirement: output shape should be [B, 1]
        batch_size = history_data.shape[0]
        if logits.shape != (batch_size, 1):
            # Fallback/Safety: If backend returns [B, N], condense to [B, 1]
            if len(logits.shape) > 1 and logits.shape[1] > 1:
                logits = torch.mean(logits, dim=1, keepdim=True)

        # 5. Visualization (Optional)
        if train and (iter_num % 500 == 0) and epoch is not None:
            self._visualize_anomaly(history_data, logits, labels, epoch)

        # Return logits and labels for the Loss function (e.g., Binary Cross Entropy)
        # We also return GSL components in case your loss function includes Graph Sparsity constraints
        return logits, labels

    def _visualize_anomaly(self, history, logits, labels, epoch):
        """Visualizes the short-term history and anomaly probability."""
        save_path = os.path.join(self.ckpt_save_dir, 'plots')
        os.makedirs(save_path, exist_ok=True)

        # Average across nodes for visualization
        hist_plot = history[0, :, :, 0].mean(dim=-1).detach().cpu().numpy()
        prob = torch.sigmoid(logits[0]).item()
        true_label = labels[0].item()

        plt.figure(figsize=(8, 4))
        plt.plot(hist_plot, label='Input Series (Mean)')
        plt.title(f"Epoch {epoch} | GT: {int(true_label)} | Pred Prob: {prob:.4f}")
        plt.legend()
        plt.savefig(os.path.join(save_path, f"val_{epoch}_{np.random.randint(100)}.png"))
        plt.close()