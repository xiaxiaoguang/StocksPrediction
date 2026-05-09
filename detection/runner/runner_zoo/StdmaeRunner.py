import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
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
            "Accuracy": partial(detection_accuracy, task='global'),
            "F1": partial(detection_f1, task='global'),
            "AUC": partial(detection_auc, task='global'),
            # Local (Stock-Specific) Metrics
            "Local_Accuracy": partial(detection_accuracy, task='local'),
            "Local_Precision": partial(detection_precision, task='local'),
            "Local_Recall": partial(detection_recall, task='local'),
            "Local_F1": partial(detection_f1, task='local'),
            "Local_AUC": partial(detection_auc, task='local'),        
        }
        self.use_local=True

        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)

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
        labels['global'] = self.to_running_device(labels['global'])              # [B, N]
        labels['local'] = self.to_running_device(labels['local'])              # [B, N]

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

        if train and ((iter_num % 1000) == 0) and epoch is not None:
            try:
                self._visualize_results(history_data, logits['global'], labels['global'], epoch)
            except Exception as e:
                print(f"Visualization failed at epoch {epoch}: {e}")

        return logits, labels


    def _visualize_results(self, history, logits, labels, epoch):
        """
        Visualizes the input sequence and the model's binary prediction.
        history: [B, L, N, C]
        logits: [B, 1]
        labels: [B, 1]
        """
        save_path = os.path.join(self.ckpt_save_dir, 'plots')
        os.makedirs(save_path, exist_ok=True)

        # 1. Get the first sample in the batch
        # Calculate the market average across all N nodes for the plot
        hist_avg = history[0, :, :, 0].mean(dim=-1).detach().cpu().numpy()
        
        # 2. Extract Prediction and Target
        prob = torch.sigmoid(logits[0]).detach().cpu().item()
        pred_class = 1 if prob > 0.5 else 0
        true_class = int(labels[0].detach().cpu().item())

        # 3. Plot
        plt.figure(figsize=(10, 5))
        x_hist = np.arange(len(hist_avg))
        plt.plot(x_hist, hist_avg, label='Market Average (History)', color='blue', linewidth=2)

        # Color code the title: Green if correct, Red if wrong
        title_color = 'green' if pred_class == true_class else 'red'
        title_text = f"Epoch {epoch} | Target: {true_class} | Pred: {pred_class} (Prob: {prob:.2f})"
        
        plt.title(title_text, color=title_color, fontweight='bold')
        plt.xlabel("Time Steps")
        plt.ylabel("Normalized Average Price")
        plt.axvline(x=len(hist_avg)-1, color='gray', linestyle=':', label='Forecast Point')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        plt.savefig(os.path.join(save_path, f"epoch_{epoch}_anomaly_val.png"))
        plt.close()
