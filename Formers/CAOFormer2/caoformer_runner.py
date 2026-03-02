import torch

from easytorch.utils.dist import master_only
from basicts.data.registry import SCALER_REGISTRY
from basicts.runners import BaseTimeSeriesForecastingRunner
from basicts.metrics import masked_mae, masked_rmse, masked_mape , PLKnce

class CAOFormerRunner(BaseTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)
        self.Test=False
        self.name=cfg["DATASET_NAME"]
        self.metrics = {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape}
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
        if "StockD" in self.name:
            future_data, history_data, _ = data
        else :
            future_data, history_data = data
        history_data    = self.to_running_device(history_data)      # B, L, N, C
        future_data     = self.to_running_device(future_data)       # B, L, N, C
        batch_size, length, num_nodes, _ = future_data.shape
        history_data    = self.select_input_features(history_data) # B,L,N,C
        alongTime = torch.argsort(torch.mean(history_data[:,:,:,self.target_features],dim=2),dim=1)
        alongSpace = torch.argsort(torch.mean(future_data[:,:,:,self.target_features],dim=1),dim=1)
        predict_future, real_future, hidden_space, hidden_time = self.model(history_data=history_data, future_data=future_data, batch_seen=iter_num, epoch=epoch)
        return predict_future, real_future, hidden_space, hidden_time, alongSpace, alongTime , iter_num

    @torch.no_grad()
    @master_only
    def test(self):
        """Evaluate the model.

        Args:
            train_epoch (int, optional): current epoch if in training process.
        """
        # test loop
        prediction = []
        real_value = []
        for _, data in enumerate(self.test_data_loader):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)
            prediction.append(forward_return[0])        # preds = forward_return[0]
            real_value.append(forward_return[1])        # testy = forward_return[1]``

        safe_memory = len(self.test_data_loader) // 2
        prediction = torch.cat(prediction[:safe_memory], dim=0)
        real_value = torch.cat(real_value[:safe_memory], dim=0) # torch.Size([1712, 1080, 207])
        # re-scale data
        prediction = SCALER_REGISTRY.get(self.scaler["func"])(
            prediction, **self.scaler["args"])
        real_value = SCALER_REGISTRY.get(self.scaler["func"])(
            real_value, **self.scaler["args"])
        
        for metric_name, metric_func in self.metrics.items():
            if self.evaluate_on_gpu:
                metric_item = self.metric_forward(metric_func, [prediction, real_value])
            else:
                metric_item = self.metric_forward(metric_func, [prediction.detach().cpu(), real_value.detach().cpu()])
            self.update_epoch_meter("test_"+metric_name, metric_item.item())
            
    