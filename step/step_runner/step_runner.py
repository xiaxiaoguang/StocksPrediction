import torch

from basicts.runners import BaseTimeSeriesForecastingRunner
from basicts.metrics import *

class STEPRunner(BaseTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.metrics = cfg.get("METRICS", {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape , 
                                           "Average_Return":AvgR,
                                           "Best_Return_10":bstReturn(10),"Our_Return_10":predReturn(10),
                                           "Random_Return_10":RndReturn(10),"Success_rate_10":successK(10),"SR_10":SR(10),
                                           "Best_Return_30":bstReturn(30),"Our_Return_30":predReturn(30),
                                           "Random_Return_30":RndReturn(30),"Success_rate_30":successK(30),"SR_30":SR(30),
                                           "Best_Return_90":bstReturn(90),"Our_Return_90":predReturn(90),
                                           "Random_Return_90":RndReturn(90),"Success_rate_90":successK(90),"SR_90":SR(90)
                                           })
        
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)

        if cfg.get("StartTest",False):
            self.ckpt_save_dir2 = self.ckpt_save_dir 
            self.ckpt_save_dir = cfg.StartTest.ckpt_save_dir

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select input features and reshape data to fit the target model.

        Args:
            data (torch.Tensor): input history data, shape [B, L, N, C].

        Returns:
            torch.Tensor: reshaped data
        """

        # select feature using self.forward_features
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select target features and reshape data back to the BasicTS framework

        Args:
            data (torch.Tensor): prediction of the model with arbitrary shape.

        Returns:
            torch.Tensor: reshaped data with shape [B, L, N, C]
        """

        # select feature using self.target_features
        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: tuple, epoch:int = None, iter_num: int = None, train:bool = True, **kwargs) -> tuple:
        """feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history data). [B, L, N, C] for each of them
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value)
        """
        # preprocess
        future_data, history_data, long_history_data, label = data
        history_data        = self.to_running_device(history_data)      # B, L, N, C
        long_history_data   = self.to_running_device(long_history_data) # B, L, N, C
        future_data         = self.to_running_device(future_data)       # B, L, N, C
        label               = self.to_running_device(label)

        history_data = self.select_input_features(history_data)  # torch.Size([32, 12, 207, 3])
        long_history_data = self.select_input_features(long_history_data)# torch.Size([32, 2016, 207, 3])

        # feed forward
        prediction, pred_adj, prior_adj, gsl_coefficient = \
            self.model(history_data=history_data, long_history_data=long_history_data, future_data=None, batch_seen=iter_num, epoch=epoch)
        
        # if self.label_file_path is not None :
        # judge = label[:,:,:,3].unsqueeze(-1)
        # judge = torch.where(abs(judge)- 0.1 > 0 , 0.0 , 1.0)
        label = label[:,:,:,2].unsqueeze(-1)
        future_data = future_data * label
        label = label.unsqueeze(-1)
        prediction = prediction * label

        # mask the useless nodes
        # mask_tool = future_data.sum(dim=-1)
        # mask = torch.where(mask_tool == 0,0,1).unsqueeze(-1).unsqueeze(-1)
        # prediction = prediction * mask

        batch_size, length, num_nodes, _ = future_data.shape
        assert list(prediction.shape)[:3] == [batch_size, length, num_nodes], \
            "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        
        if iter_num is not None and iter_num % 100 == 0 and train is True:
            with open("test1.out",'a') as f:
                # print(prediction[0,0,:,1,0],future_data[0,0,:,1],file=f)
                a,b,c,d = PredR(prediction,future_data)
                b=b.view(batch_size,10)
                c=c.view(batch_size,10)
                d=d.view(batch_size,10)
                print(b,"\n",c,"\n",d,"\n",file=f)
        # post process
        if prediction.shape[-2] != 1:
            prediction = self.select_target_features(prediction) # torch.Size([32, 12, 207, 1])
        real_value = self.select_target_features(future_data)
        return prediction, real_value, pred_adj, prior_adj, gsl_coefficient,epoch
