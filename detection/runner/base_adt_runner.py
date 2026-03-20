import math
import torch
import numpy as np
import functools
from typing import Union, Tuple, Optional

from metric import *
from basicts.runners import BaseRunner

class AnomalyDetectionRunner(BaseRunner):
    """
    Runner for binary anomaly detection in time series.
    Features:
        - Predicts 0 (Normal) or 1 (Anomaly) for a given future window.
        - Metrics: Accuracy, Precision, Recall, F1-Score, AUC.
        - Best model is selected based on the highest F1-Score at validation.
        - Handles severe class imbalance inherent in anomaly detection.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)

        self.dataset_name = cfg["DATASET_NAME"]
        self.dataset_type = cfg["DATASET_TYPE"]
        self.evaluate_on_gpu = cfg["TEST"].get("USE_GPU", True)

        # Loss Function (e.g., nn.BCEWithLogitsLoss)
        self.loss = cfg["TRAIN"]["LOSS"]
        
        # We no longer need MAE/RMSE. We register classification metrics.
        self.metrics = {
            "Accuracy": detection_accuracy,
            "Precision": detection_precision,
            "Recall": detection_recall,
            "F1": detection_f1,
            "AUC": detection_auc,
        }

    def init_training(self, cfg: dict):
        super().init_training(cfg)
        for key in self.metrics.keys():
            self.register_epoch_meter("train_" + key, "train", "{:.4f}")

    def init_validation(self, cfg: dict):
        super().init_validation(cfg)
        for key in self.metrics.keys():
            self.register_epoch_meter("val_" + key, "val", "{:.4f}")

    def init_test(self, cfg: dict):
        super().init_test(cfg)
        for key in self.metrics.keys():
            self.register_epoch_meter("test_" + key, "test", "{:.4f}")

    # --- Dataset Builders ---
    # Simplified from your original code. Just passing the correct kwargs.
    def build_train_dataset(self, cfg: dict):
        dataset_args = cfg.get("DATASET_ARGS", {})
        dataset_args["data_file_path"] = f"{cfg['TRAIN']['DATA']['DIR']}/data_anomaly.pkl"
        dataset_args["index_file_path"] = f"{cfg['TRAIN']['DATA']['DIR']}/index_anomaly.pkl"
        dataset_args["label_file_path"] = f"{cfg['TRAIN']['DATA']['DIR']}/label_anomaly.pkl"
        dataset_args["mode"] = "train"
        
        dataset = cfg["DATASET_CLS"](**dataset_args)
        self.iter_per_epoch = math.ceil(len(dataset) / cfg["TRAIN"]["DATA"]["BATCH_SIZE"])
        return dataset

    @staticmethod
    def build_val_dataset(cfg: dict):
        dataset_args = cfg.get("DATASET_ARGS", {})
        dataset_args["data_file_path"] = f"{cfg['VAL']['DATA']['DIR']}/data_anomaly.pkl"
        dataset_args["index_file_path"] = f"{cfg['VAL']['DATA']['DIR']}/index_anomaly.pkl"
        dataset_args["label_file_path"] = f"{cfg['VAL']['DATA']['DIR']}/label_anomaly.pkl"
        dataset_args["mode"] = "valid"
        return cfg["DATASET_CLS"](**dataset_args)

    @staticmethod
    def build_test_dataset(cfg: dict):
        dataset_args = cfg.get("DATASET_ARGS", {})
        dataset_args["data_file_path"] = f"{cfg['TEST']['DATA']['DIR']}/data_anomaly.pkl"
        dataset_args["index_file_path"] = f"{cfg['TEST']['DATA']['DIR']}/index_anomaly.pkl"
        dataset_args["label_file_path"] = f"{cfg['TEST']['DATA']['DIR']}/label_anomaly.pkl"
        dataset_args["mode"] = "test"
        return cfg["DATASET_CLS"](**dataset_args)

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        """
        Must return (logits, labels)
        Logits shape: [B, 1]
        Labels shape: [B, 1]
        """
        raise NotImplementedError()

    def calculate_metrics(self, logits: torch.Tensor, labels: torch.Tensor, prefix: str):
        """Helper to compute and update classification metrics."""
        # Convert logits to probabilities (if using BCEWithLogitsLoss)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        targets = labels.detach().cpu().numpy().astype(int)

        for metric_name, metric_func in self.metrics.items():
            try:
                if metric_name == "AUC":
                    # AUC requires probabilities, not hard predictions
                    # Also, AUC fails if a batch only contains one class (all 0s)
                    if len(np.unique(targets)) > 1:
                        val = metric_func(targets, probs)
                    else:
                        continue 
                else:
                    val = metric_func(targets, preds)
                
                self.update_epoch_meter(f"{prefix}_{metric_name}", float(val))
            except ValueError:
                pass # Catch errors for batches with only one class

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        logits, labels = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
        
        # Calculate Loss (NO INVERSE SCALING)
        loss = self.loss(logits, labels)
        
        # Calculate Metrics
        self.calculate_metrics(logits, labels, prefix="train")
        
        return loss

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        logits, labels = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)
        self.calculate_metrics(logits, labels, prefix="val")

    @torch.no_grad()
    # @master_only # Assuming this decorator exists in your environment
    def test(self):
        self.logger.info("Starting evaluation on test dataset...")
        all_logits = []
        all_labels = []

        for _, data in enumerate(self.test_data_loader):
            logits, labels = self.forward(data, epoch=None, iter_num=None, train=False)
            all_logits.append(logits)
            all_labels.append(labels)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Move to CPU for sklearn metrics
        probs = torch.sigmoid(all_logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        targets = all_labels.cpu().numpy().astype(int)

        # Calculate Overall Test Metrics
        for metric_name, metric_func in self.metrics.items():
            if metric_name == "AUC":
                if len(np.unique(targets)) > 1:
                    metric_val = metric_func(targets, probs)
                else:
                    metric_val = 0.0
            else:
                metric_val = metric_func(targets, preds)
                
            self.update_epoch_meter("test_" + metric_name, float(metric_val))
            self.logger.info(f"Test Overall {metric_name}: {metric_val:.4f}")

    # @master_only
    def on_validating_end(self, train_epoch: Optional[int]):
        if train_epoch is not None:
            # Crucial: We now want the GREATEST F1-Score, not the lowest Error!
            self.save_best_model(train_epoch, "val_F1", greater_best=True)