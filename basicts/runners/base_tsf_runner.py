import math
import os
import functools
from typing import Tuple, Union, Optional

import torch
import numpy as np
from easytorch.utils.dist import master_only

from .base_runner import BaseRunner
from ..data import SCALER_REGISTRY
from ..utils import load_pkl
from ..metrics import *
from torch.utils.tensorboard import SummaryWriter

import os
import matplotlib.pyplot as plt
import pandas as pd # Added for easy choice logging

class BaseTimeSeriesForecastingRunner(BaseRunner):
    """
    Runner for short term multivariate time series forecasting datasets.
    Typically, models predict the future 12 time steps based on historical time series.
    Features:
        - Evaluate at horizon 3, 6, 12, and overall.
        - Metrics: MAE, RMSE, MAPE. The best model is the one with the smallest mae at validation.
        - Loss: MAE (masked_mae). Allow customization.
        - Support curriculum learning.
        - Users only need to implement the `forward` function.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)

        self.dataset_name = cfg["DATASET_NAME"]
        # different datasets have different null_values, e.g., 0.0 or np.nan.
        self.null_val = cfg["TRAIN"].get("NULL_VAL", np.nan)    # consist with metric functions
        self.dataset_type = cfg["DATASET_TYPE"]
        self.evaluate_on_gpu = cfg["TEST"].get("USE_GPU", True)     # evaluate on gpu or cpu (gpu is faster but may cause OOM)

        # read scaler for re-normalization
        self.scaler = load_pkl("{0}/scaler_in{1}_out{2}.pkl".format(cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"]))
        # define loss
        self.loss = cfg["TRAIN"]["LOSS"]
        # define metric
        self.metrics = {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape}
        # curriculum learning for output. Note that this is different from the CL in Seq2Seq archs.
        self.cl_param = cfg.TRAIN.get("CL", None)
        if self.cl_param is not None:
            self.warm_up_epochs = cfg.TRAIN.CL.get("WARM_EPOCHS", 0)
            self.cl_epochs = cfg.TRAIN.CL.get("CL_EPOCHS")
            self.prediction_length = cfg.TRAIN.CL.get("PREDICTION_LENGTH")
            self.cl_step_size = cfg.TRAIN.CL.get("STEP_SIZE", 1)
        # evaluation horizon
        self.predlen = cfg["DATASET_OUTPUT_LEN"]
        self.evaluation_horizons = [_ - 1 for _ in cfg["TEST"].get("EVALUATION_HORIZONS", range(1, self.predlen+1))]
        assert min(self.evaluation_horizons) >= 0, "The horizon should start counting from 0."


    def init_training(self, cfg: dict):
        """Initialize training.

        Including loss, training meters, etc.

        Args:
            cfg (dict): config
        """

        super().init_training(cfg)
        for key, _ in self.metrics.items():
            self.register_epoch_meter("train_"+key, "train", "{:.4f}")
        if cfg.get("StartTest",False):

            if cfg.StartTest.get("test_line",False):
                self.test_line(cfg)
                self.num_epochs = 0
            elif cfg.StartTest.get("test_whole",False):
                self.test_whole(cfg)
                self.num_epochs = 0
            else :
                # self.validate(cfg)
                self.test_process(cfg)
                self.num_epochs = 0

    def init_validation(self, cfg: dict):
        """Initialize validation.

        Including validation meters, etc.

        Args:
            cfg (dict): config
        """

        super().init_validation(cfg)
        for key, _ in self.metrics.items():
            self.register_epoch_meter("val_"+key, "val", "{:.4f}")

    def init_test(self, cfg: dict):
        """Initialize test.

        Including test meters, etc.

        Args:
            cfg (dict): config
        """

        super().init_test(cfg)
        for key, _ in self.metrics.items():
            self.register_epoch_meter("test_"+key, "test", "{:.4f}")

    def build_train_dataset(self, cfg: dict):
        """Build MNIST train dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """

        data_file_path = "{0}/data_in{1}_out{2}.pkl".format(cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        index_file_path = "{0}/index_in{1}_out{2}.pkl".format(cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        label_file_path = "{0}/label_in{1}_out{2}.pkl".format(cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path

        if "csi" in cfg["DATASET_NAME"]:
            dataset_args["label_file_path"] = label_file_path
        else :
            dataset_args["label_file_path"] = None

        dataset_args["mode"] = "train"
        
        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("train len: {0}".format(len(dataset)))

        batch_size = cfg["TRAIN"]["DATA"]["BATCH_SIZE"]
        self.iter_per_epoch = math.ceil(len(dataset) / batch_size)

        return dataset

    @staticmethod
    def build_val_dataset(cfg: dict):
        """Build MNIST val dataset

        Args:
            cfg (dict): config

        Returns:
            validation dataset (Dataset)
        """
        data_file_path = "{0}/data_in{1}_out{2}.pkl".format(cfg["VAL"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        index_file_path = "{0}/index_in{1}_out{2}.pkl".format(cfg["VAL"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        label_file_path = "{0}/label_in{1}_out{2}.pkl".format(cfg["VAL"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])

        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path

        if "csi" in cfg["DATASET_NAME"]:
            dataset_args["label_file_path"] = label_file_path
        else :
            dataset_args["label_file_path"] = None        
        dataset_args["mode"] = "valid"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("val len: {0}".format(len(dataset)))

        return dataset

    @staticmethod
    def build_test_dataset(cfg: dict):
        """Build MNIST val dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """

        data_file_path = "{0}/data_in{1}_out{2}.pkl".format(cfg["TEST"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        index_file_path = "{0}/index_in{1}_out{2}.pkl".format(cfg["TEST"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        label_file_path = "{0}/label_in{1}_out{2}.pkl".format(cfg["TEST"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        if "csi" in cfg["DATASET_NAME"]:
            dataset_args["label_file_path"] = label_file_path
        else :
            dataset_args["label_file_path"] = None
        dataset_args["mode"] = "test"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("test len: {0}".format(len(dataset)))

        return dataset

    def curriculum_learning(self, epoch: int = None) -> int:
        """Calculate task level in curriculum learning.

        Args:
            epoch (int, optional): current epoch if in training process, else None. Defaults to None.

        Returns:
            int: task level
        """
        if epoch is None:
            return self.prediction_length
        epoch -= 1
        # generate curriculum length
        if epoch < self.warm_up_epochs:
            # still warm up
            cl_length = self.prediction_length
        else:
            _ = ((epoch - self.warm_up_epochs) // self.cl_epochs + 1) * self.cl_step_size
            cl_length = min(_, self.prediction_length)

        return cl_length

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        """Feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history data). [B, L, N, C] for each of them
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value). [B, L, N, C] for each of them.
        """

        raise NotImplementedError()

    def metric_forward(self, metric_func, args):
        """Computing metrics.

        Args:
            metric_func (function, functools.partial): metric function.
            args (list): arguments for metrics computation.
        """

        if isinstance(metric_func, functools.partial) and list(metric_func.keywords.keys()) == ["null_val"]:
            # support partial(metric_func, null_val = something)
            metric_item = metric_func(*args)
        elif callable(metric_func):
            # is a function
            metric_item = metric_func(*args, null_val=self.null_val)
        else:
            raise TypeError("Unknown metric type: {0}".format(type(metric_func)))
        return metric_item

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Training details.

        Args:
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader
            epoch (int): current epoch.
            iter_index (int): current iter.

        Returns:
            loss (torch.Tensor)
        """
        iter_num = (epoch-1) * self.iter_per_epoch + iter_index
        forward_return = list(self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True))
        # re-scale data
        prediction_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[0], **self.scaler["args"])
        real_value_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[1], **self.scaler["args"])
        # loss
        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return[0] = prediction_rescaled[:, :cl_length, :, :]
            forward_return[1] = real_value_rescaled[:, :cl_length, :, :]
        else:
            forward_return[0] = prediction_rescaled
            forward_return[1] = real_value_rescaled
        # breakpoint()
        loss = self.metric_forward(self.loss, forward_return)
        # metrics
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return[:2])
            self.update_epoch_meter("train_"+metric_name, metric_item.item())
        return loss

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        """Validation details.

        Args:
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader
            train_epoch (int): current epoch if in training process. Else None.
            iter_index (int): current iter.
        """

        forward_return = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)
        # re-scale data
        prediction_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[0], **self.scaler["args"])
        real_value_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[1], **self.scaler["args"])
        # metrics
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, [prediction_rescaled, real_value_rescaled])
            self.update_epoch_meter("val_"+metric_name, metric_item.item())

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
            real_value.append(forward_return[1])        # testy = forward_return[1]

        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)
        # re-scale data
        prediction = SCALER_REGISTRY.get(self.scaler["func"])(
            prediction, **self.scaler["args"])
        real_value = SCALER_REGISTRY.get(self.scaler["func"])(
            real_value, **self.scaler["args"])
        # summarize the results.
        # test performance of different horizon
        for i in self.evaluation_horizons:
            # For horizon i, only calculate the metrics **at that time** slice here.
            pred = prediction[:, i, :, :]
            real = real_value[:, i, :, :]
            # metrics
            metric_repr = ""
            for metric_name, metric_func in self.metrics.items(): 
                if metric_name == "MAE" or metric_name == "RMSE" or metric_name == "MAPE":
                    metric_item = self.metric_forward(metric_func, [pred, real])
                    metric_repr += ", Test {0}: {1:.4f}".format(metric_name, metric_item.item())
            log = "Evaluate best model on test data for horizon {:d}" + metric_repr
            log = log.format(i+1)
            self.logger.info(log)
        

        # test performance overall
        for metric_name, metric_func in self.metrics.items():
            if self.evaluate_on_gpu:
                metric_item = self.metric_forward(metric_func, [prediction, real_value])
            else:
                metric_item = self.metric_forward(metric_func, [prediction.detach().cpu(), real_value.detach().cpu()])
            self.update_epoch_meter("test_"+metric_name, metric_item.item())

    @master_only
    def on_validating_end(self, train_epoch: Optional[int]):
        """Callback at the end of validating.

        Args:
            train_epoch (Optional[int]): current epoch if in training process.
        """

        if train_epoch is not None:
            self.save_best_model(train_epoch, "val_MAE", greater_best=False)

    @torch.no_grad()
    @master_only
    def test_whole(self, cfg: dict = None):
        """Evaluate the ? days return rate."""
        self.init_test(cfg)
        self.model.eval()
        
        # Setup PDF save directory
        save_plot_dir = os.path.join(self.ckpt_save_dir, 'test')
        os.makedirs(save_plot_dir, exist_ok=True)

        handvalue = torch.ones((1,), dtype=torch.float)
        lst_inv = torch.zeros((self.predlen,), dtype=torch.float)
        lst_ret = torch.zeros((self.predlen,), dtype=torch.float)
        K = 4
        handvalue = self.to_running_device(handvalue)
        lst_ret = self.to_running_device(lst_ret)
        lst_inv = self.to_running_device(lst_inv)

        dataloader = self.test_data_loader
        if cfg.StartTest.get("UseTrain", False):
            print("Mention : Use training data to test")           
            dataloader = self.train_data_loader
        elif cfg.StartTest.get("UseValid", False):
            print("MENTION: Use validation data to test")
            self.init_validation(cfg)
            dataloader = self.val_data_loader

        maxL = max(1, dataloader.__len__() // 15)

        total_investment = []
        history_portfolio = []
        success_indices = []
        success_values = []
        investment_choices = [] # To store the "choices" for the external file

        for nowi, data in enumerate(dataloader):
            # Capture the indices chosen by the strategy (we need to modify predReturn to return them)
            # For now, let's assume we capture them here
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)
            last_price = data[1][:,-1:,:,:].to(forward_return[0].device)
            lst_pre = forward_return[0]
            lst_rel = forward_return[1]
            prev_total = (handvalue + torch.sum(lst_ret)).item()
            # Logic for return calculation
            if cfg.StartTest.get("RandomSelect", False):
                # Note: You'd need to modify RndReturn to return indices too if you want them logged
                ret = RndReturn(50)(lst_pre, lst_rel, last_price)
                our_ret = max(ret.unsqueeze(-1), 1) * (handvalue / K)
            else:
                # Assuming predReturn modified to return (reward, indices) or similar
                # If predReturn only returns reward, we just track the reward value
                func = predReturn_Short(20) if cfg.StartTest.get("AllowShort", True) else predReturn(20)
                ret = func(lst_pre, lst_rel, last_price)
                our_ret = ret.unsqueeze(-1) * (handvalue / K)

            lst_ret = torch.cat((lst_ret[1:], our_ret), dim=0)
            lst_inv = torch.cat((lst_inv[1:], (handvalue / K)), dim=0)
            handvalue -= (handvalue / K)
            current_portfolio_value = (handvalue + torch.sum(lst_ret)).item()
            history_portfolio.append(current_portfolio_value)

            # --- LOGIC FOR SUCCESS TRACKING ---
            # If current value is higher than previous, it's a "Success"
            profit = current_portfolio_value - prev_total
            if profit > 0:
                success_indices.append(nowi)
                success_values.append(current_portfolio_value)
            
            # Store choice data (Step, Value, Profit)
            investment_choices.append({
                "step": nowi,
                "portfolio_value": current_portfolio_value,
                "profit": profit,
                "is_success": profit > 0
            })

        # --- SAVE CHOICES TO FILE ---
        df_choices = pd.DataFrame(investment_choices)
        df_choices.to_csv(os.path.join(save_plot_dir, 'investment_choices.csv'), index=False)

        # --- PLOTTING ---
        plt.figure(figsize=(12, 6))
        # breakpoint()
        # 1. Main Portfolio Line
        plt.plot(history_portfolio, label='Portfolio Value', color='#1f77b4', linewidth=1.5)
        
        # 2. Success Markers (Green dots where we earned money)
        plt.scatter(success_indices, success_values, color='green', s=10, 
                    label='Profitable Step', alpha=0.5, marker='^')

        # 3. Log Scale Transformation
        plt.yscale('log') 
        
        plt.title(f"Growth Visualization (Log Scale) - Final: {current_portfolio_value:.2f}")
        plt.xlabel("Trading Steps")
        plt.ylabel("Value (Log Scale)")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_plot_dir, 'test_log_performance.pdf'))
        plt.close()
    
    @torch.no_grad()
    @master_only
    def test_line(self, cfg: dict = None):

        self.init_test(cfg)
        self.model.eval()
        
        # Setup PDF save directory
        save_plot_dir = os.path.join(self.ckpt_save_dir, 'test')
        os.makedirs(save_plot_dir, exist_ok=True)

        dataloader = self.test_data_loader
        
        if cfg.StartTest.get("UseTrain", False):
            dataloader = self.train_data_loader
            print("Mention : Use training data to test")  
        elif cfg.StartTest.get("UseValid", False):
            print("MENTION: Use validation data to test")
            self.init_validation(cfg)
            dataloader = self.val_data_loader

        maxL = max(1, dataloader.__len__() // 15) 
        
        tag = cfg.StartTest.get("select", [0]) 
        if isinstance(tag, int):
            tag = [tag]

        # --- NEW: Dictionary to store trajectories for each tag ---
        plot_data = {t: {'real': [], 'pred': []} for t in tag}
        interval = 3
        cnt = interval-1

        for nowi, data in enumerate(dataloader):

            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)
            
            prel = data[0].shape[1]
            numn = data[0].shape[2]

            lst_pre = forward_return[0].view(prel, numn)
            lst_rel = forward_return[1].view(prel, numn)

            cnt += 1
            if cnt == interval:
                # breakpoint()
                cnt = 0
                for i in tag:
                    # Accumulate data
                    for tmp in range(interval): 
                        pred_val = lst_pre[tmp][i].item()
                        real_val = lst_rel[tmp][i].item()
                        plot_data[i]['pred'].append(pred_val)
                        plot_data[i]['real'].append(real_val)
                    if nowi % maxL == 0:
                        print(f"Now {nowi}/{dataloader.__len__()} | Tag {i} | Pred: {pred_val:.4f} | Real: {real_val:.4f}")

        # --- NEW: Generate and save a PDF plot for each tag ---
        for i, data_dict in plot_data.items():
            plt.figure(figsize=(10, 5))
            
            # Use arange for x-axis to ensure proper alignment
            steps = range(len(data_dict['real']))
            plt.plot(steps, data_dict['real'], label='Real Price', color='green', alpha=0.7)
            plt.plot(steps, data_dict['pred'], label='Predicted Price', color='red', linestyle='--', alpha=0.9)
            
            plt.title(f"Time Series Forecasting - Tag {i}")
            plt.xlabel("Time Steps")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            
            # Save as PDF
            pdf_path = os.path.join(save_plot_dir, f'vis_tag_{i}_interval{interval}.pdf')
            plt.savefig(pdf_path, format='pdf')
            plt.close()
