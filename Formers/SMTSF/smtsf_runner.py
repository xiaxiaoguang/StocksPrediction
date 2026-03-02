import torch

from easytorch.utils.dist import master_only
from basicts.data.registry import SCALER_REGISTRY
from basicts.runners import BaseTimeSeriesForecastingRunner


class SMTSFRunner(BaseTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)
        self.Test=False
        if cfg.get("StartTest",False):
            self.Test=True
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
        future_data, history_data,label  = data
        history_data    = self.to_running_device(history_data)      # B, L, N, C
        # future_data     = self.to_running_device(future_data)       # B, L, N, C
        label           = self.to_running_device(label)             # B ,L, N, 5
        
        judge = label[:,:,:,3].unsqueeze(-1)
        judge = torch.where(abs(judge)- 0.1 > 0 , 0.0 , 1.0)
        label = label[:,:,:,2].unsqueeze(-1) * judge
        history_data = history_data * label

        month_masked_tokens, month_label_masked_tokens, space_masked_tokens, space_label_masked_tokens = self.model(history_data=history_data, future_data=None, batch_seen=iter_num, epoch=epoch)

        return month_masked_tokens, month_label_masked_tokens, space_masked_tokens, space_label_masked_tokens

    @torch.no_grad()
    @master_only
    def test(self):
        """Evaluate the model.

        Args:
            train_epoch (int, optional): current epoch if in training process.
        """
        # test loop
        prediction1 = []
        real_value1 = []
        prediction2 = []
        real_value2 = []
        for _, data in enumerate(self.test_data_loader):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)
            prediction1.append(forward_return[0])        # preds = forward_return[0]
            prediction2.append(forward_return[2])        # preds = forward_return[0]
            real_value1.append(forward_return[1])        # testy = forward_return[1]
            real_value2.append(forward_return[3])        # preds = forward_return[0]
        prediction1 = torch.cat(prediction1, dim=0)
        real_value1 = torch.cat(real_value1, dim=0)
        prediction2 = torch.cat(prediction2, dim=0)
        real_value2 = torch.cat(real_value2, dim=0)        
        
        for metric_name, metric_func in self.metrics.items():
            if self.evaluate_on_gpu:
                metric_item = self.metric_forward(metric_func, [prediction1, real_value1])
                metric_item2= self.metric_forward(metric_func, [prediction2, real_value2])
            else:
                metric_item = self.metric_forward(metric_func, [prediction1.detach().cpu(), real_value1.detach().cpu()])
                metric_item2= self.metric_forward(metric_func, [prediction2.detach().cpu(), real_value2.detach().cpu()])
            self.update_epoch_meter("test_"+metric_name, (metric_item+metric_item2).item())
            
    