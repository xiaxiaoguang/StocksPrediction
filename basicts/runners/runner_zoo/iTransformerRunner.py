import torch
import torch.nn as nn
from ..base_tsf_runner import BaseTimeSeriesForecastingRunner
from basicts.metrics import *
import os
import matplotlib.pyplot as plt
import numpy as np


class iTransformerRunner(BaseTimeSeriesForecastingRunner):
    """Simple Runner: select forward features and target features."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)

        self.metrics = cfg.get("METRICS", {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape, "MRLoss":MRLoss
                                            , "Best_Return_10" : bstReturn(10), "Our_Return_10": predReturn(10) , 
                                            "Random_Return_10": RndReturn(10) , "Average_Return": AvgR,
                                            "Sharperatio_10":SR(10),"Success_rate":successK(10)})

        # self.metrics = cfg.get("METRICS", {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape})        
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)

        # if cfg.get("StartTest",False):
        #     self.ckpt_save_dir2 = self.ckpt_save_dir 
        #     self.ckpt_save_dir = cfg.StartTest.ckpt_save_dir

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select input features.

        Args:
            data (torch.Tensor): input history data, shape [B, L, N, C]

        Returns:
            torch.Tensor: reshaped data
        """

        # select feature using self.forward_features
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select target feature.

        Args:
            data (torch.Tensor): prediction of the model with arbitrary shape.

        Returns:
            torch.Tensor: reshaped data with shape [B, L, N, C]
        """

        # select feature using self.target_features
        data = data[:, :, :, self.target_features]
        return data

    def _visualize_results(self, history, real, prediction, epoch):
            """
            Internal helper to plot and save time series.
            history: [B, L_in, N, C]
            real: [B, L_out, N, C]
            prediction: [B, L_out, N, C]
            """
            # Ensure the directory exists
            # breakpoint()
            save_path = os.path.join(self.ckpt_save_dir, 'plots')
            os.makedirs(save_path, exist_ok=True)

            # Move to CPU and pick the first batch, first node, first feature
            # Shape: [Length]
            hist_sample = history[0, :, 0, 0].detach().cpu().numpy()
            real_sample = real[0, :, 0, 0].detach().cpu().numpy()
            pred_sample = prediction[0, :, 0, 0].detach().cpu().numpy()

            plt.figure(figsize=(10, 5))
            
            # Plot history
            x_hist = np.arange(len(hist_sample))
            plt.plot(x_hist, hist_sample, label='History', color='blue')

            # Plot Ground Truth and Prediction
            x_pred = np.arange(len(hist_sample), len(hist_sample) + len(real_sample))
            plt.plot(x_pred, real_sample, label='Ground Truth', color='green', alpha=0.7)
            plt.plot(x_pred, pred_sample, label='Prediction', color='red', linestyle='--')

            plt.axvline(x=len(hist_sample)-1, color='gray', linestyle=':', label='Forecast Start')
            plt.title(f"Epoch {epoch} - Time Series Forecasting")
            plt.legend()
            
            # Save and close
            plt.savefig(os.path.join(save_path, f"epoch_{epoch}_val.png"))
            plt.close()

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        """Feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history ata).
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value)
        """
        # preprocess
        future_data, history_data = data
        history_data = self.to_running_device(history_data)      # B, L, N, C
        future_data = self.to_running_device(future_data)       # B, L, N, C
        # label = self.to_running_device(label)
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        # curriculum learning
        if self.cl_param is None:
            prediction_data = self.model(history_data=history_data, future_data=future_data_4_dec, batch_seen=iter_num, epoch=epoch, train=train)
        else:
            task_level = self.curriculum_learning(epoch)
            prediction_data = self.model(history_data=history_data, future_data=future_data_4_dec, batch_seen=iter_num, epoch=epoch, train=train,\
                                                                                                                     task_level=task_level)
        # feed forward
        assert list(prediction_data.shape)[:3] == [batch_size, length, num_nodes], \
            "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"

        prediction = self.select_target_features(prediction_data)
        real_value = self.select_target_features(future_data_4_dec)

        if train and ((iter_num % 1000) == 0) and epoch is not None:
            # You can wrap this in a try-except to ensure plotting doesn't crash the training
            try:
                self._visualize_results(history_data, real_value, prediction, epoch)
            except Exception as e:
                print(f"Visualization failed at epoch {epoch}: {e}")

        return prediction, real_value
