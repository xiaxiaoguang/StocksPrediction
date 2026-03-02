import torch
import torch.nn as nn
from ..base_tsf_runner import BaseTimeSeriesForecastingRunner
from basicts.metrics import *
from einops import repeat,rearrange

class MambaRunner(BaseTimeSeriesForecastingRunner):
    """Simple Runner: select forward features and target features."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)

        self.metrics = cfg.get("METRICS", {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape})
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)

        if cfg.get("StartTest",False):
            self.ckpt_save_dir2 = self.ckpt_save_dir 
            self.ckpt_save_dir = cfg.StartTest.ckpt_save_dir
        
        self.weak_noise = True
        if cfg.get("StrongNoise",False):
            self.weak_noise = False

        self.ConL = False
        if cfg.get("ContrastiveLoss",False):
            self.ConL = True
            self.RepeatK = cfg.get("RepeatTimes",3)
            self.noise = cfg.get("RepeatNoise",0)

        self.Prob = False
        if cfg.get("AdjustProb",False):
            self.Prob = True
            self.n    = cfg.DATASET_INPUT_LEN - 1
        
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
        data = data[:, :, :, self.target_features]
        return data
    def repeat_contrastive(self, history_data: torch.Tensor, _repeat: int, noise_std: float):
        B,L,N,C = history_data.shape
        if self.model_name == 'imamba':
            history_data = history_data.transpose(1,2)
        
        data = repeat(history_data, 'B L N C -> B (L R) N C', R=_repeat)
        
        if False :
            L = history_data.size(1)
            for i in range(L):
                start_idx = i * _repeat
                end_idx = (i + 1) * _repeat - 1
                noise[:, start_idx, :, :] = 0  # Set the first element of each block to zero
                noise[:, end_idx, :, :] = 0    # Set the last element of each block to zero
        
        if self.weak_noise :
            data = rearrange(data,"B A N C -> (B N C) A")
            freq_domain = torch.fft.fft(data,dim=1)
            noise = torch.randn(data.shape,device=data.device) * noise_std
            freq_domain[:L]  += noise[:L]
            data_with_noise = torch.fft.ifft(freq_domain,dim=1).real
            data_with_noise = rearrange(data_with_noise,"(B N C) A -> B A N C",B=B,N=N,C=C)
        else :
            noise = torch.randn(data.shape,device=data.device) * noise_std
            data_with_noise = data + noise

        if self.model_name == 'imamba':
            data_with_noise = data_with_noise.transpose(1,2)
        return data_with_noise
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
        future_data, history_data = data
        history_data = self.to_running_device(history_data)      # B, L, N, C
        future_data = self.to_running_device(future_data)       # B, L, N, C
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)
        tmp =  self.model(history_data=history_data, future_data=future_data_4_dec, batch_seen=iter_num, epoch=epoch, train=train)
        
        if self.Prob:
            prediction_data,hid1,base = tmp
        else :
            prediction_data,hid1 = tmp
# history_data = self.repeat_contrastive(history_data ,_repeat = 3, noise_std = 0.05)
# hid2 = self.model(history_data=history_data, future_data=future_data_4_dec, ConL=True)
        if self.ConL:
            history_data = self.repeat_contrastive(history_data ,_repeat = self.RepeatK , noise_std = self.noise)
            hid2 = self.model(history_data=history_data, future_data=future_data_4_dec, 
                              batch_seen=iter_num, epoch=epoch, train=train, ConL=True)

        assert list(prediction_data.shape)[:3] == [batch_size, length, num_nodes], \
            "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        
        prediction = self.select_target_features(prediction_data)
        real_value = self.select_target_features(future_data_4_dec)
        
        if self.ConL == False:
            return prediction, real_value
        else :
            if self.Prob==False:
                return prediction, real_value, hid1 , hid2
            else :
                return prediction, real_value, hid1 , hid2, base
