import torch
import torch.nn as nn
from ..base_tsf_runner import BaseTimeSeriesForecastingRunner
from basicts.metrics import *

import os
import matplotlib.pyplot as plt
import numpy as np


class NumerionRunner(BaseTimeSeriesForecastingRunner):
    """Simple Runner: select forward features and target features."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.metrics = cfg.get("METRICS", {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape, "IC":IC,"Sharperatio_10":SR(10)}) #"MRLoss":MRLoss
                                            # "Best_Return_10" : bstReturn(10), "Our_Return_10": predReturn(10) , 
                                            # "Random_Return_10": RndReturn(10) , "Average_Return": AvgR,
                                            # "Sharperatio_10":SR(10),"Success_rate":successK(10), "MDD_10":MDD(10),
                                            # "Best_Return_30" : bstReturn(30), "Our_Return_30": predReturn(30) , 
                                            # "Random_Return_30": RndReturn(30) , "Average_Return": AvgR,
                                            # "Sharperatio_30":SR(30),"Success_rate":successK(30)})

        # self.metrics = cfg.get("METRICS", {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape})        
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)

        if cfg.get("StartTest",False):
            self.ckpt_save_dir2 = self.ckpt_save_dir 
            self.ckpt_save_dir = cfg.StartTest.ckpt_save_dir

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
            
    def _visualize_results(self, history, real, prediction, epoch, iter_num=0):
            """
            Internal helper to plot and save time series.
            history: [B, L_in, N, C]
            real: [B, L_out, N, C]
            prediction: [B, L_out, N, C]
            """
            import os
            import numpy as np
            import matplotlib.pyplot as plt
            
            # Ensure the directory exists
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

            # --- THE FIX ---
            # 1. Grab the last point of the history to anchor the line
            last_hist_val = hist_sample[-1]
            start_x = len(hist_sample) - 1

            # 2. Prepend the last history point to the arrays so a line can be drawn
            plot_real = np.concatenate(([last_hist_val], real_sample))
            plot_pred = np.concatenate(([last_hist_val], pred_sample))

            # 3. Create the x-axis for the prediction, starting from the last history index
            x_pred = np.arange(start_x, start_x + len(plot_real))

            # 4. Plot with explicit markers ('o' for real, 'X' for prediction)
            plt.plot(x_pred, plot_real, label='Ground Truth', color='green', alpha=0.7, marker='o')
            plt.plot(x_pred, plot_pred, label='Prediction', color='red', linestyle='--', marker='X', markersize=8)

            plt.axvline(x=start_x, color='gray', linestyle=':', label='Forecast Start')
            plt.title(f"Epoch {epoch} - Time Series Forecasting")
            plt.legend()
            
            # Save and close
            plt.savefig(os.path.join(save_path, f"epoch_{epoch}_{iter_num}_val.png"))
            plt.close()
    # def _visualize_results(self, history, real, prediction, epoch, iter_num=0):
    #         """
    #         Internal helper to plot and save time series.
    #         history: [B, L_in, N, C] - Contains raw prices
    #         real: [B, L_out, N, C] - Contains ground truth log returns
    #         prediction: [B, L_out, N, C] - Contains predicted log returns
    #         """
    #         save_path = os.path.join(self.ckpt_save_dir, 'plots')
    #         os.makedirs(save_path, exist_ok=True)

    #         # Move to CPU and pick the first batch, first node, first feature
    #         # Shape: [Length]
    #         hist_sample = history[0, :, 0, 0].detach().cpu().numpy()
    #         real_returns = real[0, :, 0, 0].detach().cpu().numpy()
    #         pred_returns = prediction[0, :, 0, 0].detach().cpu().numpy()

    #         # 1. Anchor the reconstruction using the last known price in history (P_t)
    #         p_t = hist_sample[-1]

    #         # 2. Reconstruct prices from log returns using cumulative sum
    #         # P_{t+k} = P_t * exp(cumsum(r))
    #         real_prices = p_t * np.exp(np.cumsum(real_returns))
    #         pred_prices = p_t * np.exp(np.cumsum(pred_returns))

    #         plt.figure(figsize=(10, 5))
            
    #         # Plot history (Raw Prices)
    #         x_hist = np.arange(len(hist_sample))
    #         plt.plot(x_hist, hist_sample, label='History (Price)', color='blue')

    #         # Plot Ground Truth and Prediction (Reconstructed Prices)
    #         x_pred = np.arange(len(hist_sample), len(hist_sample) + len(real_prices))
    #         plt.plot(x_pred, real_prices, label='Ground Truth (Reconstructed)', color='green', alpha=0.7)
    #         plt.plot(x_pred, pred_prices, label='Prediction (Reconstructed)', color='red', linestyle='--')

    #         plt.axvline(x=len(hist_sample)-1, color='gray', linestyle=':', label='Forecast Start')
    #         plt.title(f"Epoch {epoch} - Price Reconstruction from Log Returns")
    #         plt.legend()
            
    #         # Save and close
    #         plt.savefig(os.path.join(save_path, f"epoch_{epoch}_{iter_num}_val.png"))
    #         plt.close()

    def fourier_resample(self, data: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Resamples time series using Fourier Transform (Sinc Interpolation).
        Input: [B, Channels, L]
        Output: [B, Channels, target_len]
        """
        B, C, L = data.shape
        
        # 1. Move to frequency domain
        # Use rfft for real-valued time series
        freq_data = torch.fft.rfft(data, dim=-1)
        
        # 2. Adjust the number of frequency bins
        # We need to construct the new spectrum
        # target_len // 2 + 1 is the number of frequency bins for real signals
        target_freq_len = target_len // 2 + 1
        
        # Prepare a zero tensor for the new spectrum
        new_freq_data = torch.zeros(B, C, target_freq_len, device=data.device, dtype=freq_data.dtype)
        
        # Copy existing frequencies (we crop if shrinking, pad if expanding)
        copy_len = min(freq_data.shape[-1], target_freq_len)
        new_freq_data[..., :copy_len] = freq_data[..., :copy_len]
        
        # Scale amplitude to maintain signal energy
        new_freq_data *= (target_len / L)
        
        # 3. Move back to time domain
        resampled_data = torch.fft.irfft(new_freq_data, n=target_len, dim=-1)
        
        return resampled_data

    # Update your existing function
    def resample_time_series(self, data: torch.Tensor, target_len: int) -> torch.Tensor:
        B, L, N, C = data.shape
        data = data.permute(0, 2, 3, 1).reshape(B, N * C, L)
        # Call the new spectral resampler
        data_resampled = self.fourier_resample(data, target_len)
        return data_resampled.reshape(B, N, C, target_len).permute(0, 3, 1, 2)

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        # 1. Preprocess & Scale
        future_data, history_data, label = data
        future_data = self.to_running_device(future_data)       # [B, 12, N, C]
        if hasattr(self.model,'configs'):
            target_len = self.model.configs.pred_len  # 96
        else :
            target_len = self.model.pred_len
        future_data = future_data[:,-target_len:,:,:]
        future_extended_dec = self.select_input_features(future_data)
        future_extended = self.resample_time_series(future_extended_dec, target_len)

        history_data = self.to_running_device(history_data)      # [B, 48, N, C]
        history_extended = self.select_input_features(history_data)
        if hasattr(self.model,'configs'):
            target_len = self.model.configs.seq_len  # 96
        else :
            target_len = self.model.seq_len
        history_extended = self.resample_time_series(history_extended, target_len)

        batch_size, original_his, _, _ = history_data.shape
        _, original_fut, _, _ = future_data.shape

        # 3. Model Inference (Expects 96, Outputs 96)
        if self.cl_param is None:
            prediction_24 = self.model(
                history_data=history_extended, 
                future_data=future_extended_dec, 
                batch_seen=iter_num, epoch=epoch, train=train
            )
        else:
            task_level = self.curriculum_learning(epoch)
            prediction_24 = self.model(
                history_data=history_extended, 
                future_data=future_extended_dec, 
                batch_seen=iter_num, epoch=epoch, train=train,
                task_level=task_level
            )
        
        if not train or (epoch is not None and ((iter_num % 1000) == 0)):
            try:
                # We visualize the 24-day version to see the actual performance
                self._visualize_results(history_extended, future_extended, prediction_24, epoch, iter_num)
            except Exception as e:
                print(f"Visualization failed at epoch {epoch}: {e}")

        prediction_24 = self.resample_time_series(prediction_24, original_fut)

        # 6. Post-process & Scale back
        prediction = self.select_target_features(prediction_24)
        real_value = self.select_target_features(future_extended_dec)
        return prediction, real_value