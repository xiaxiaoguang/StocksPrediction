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

        if "StockD" in cfg["DATASET_NAME"]:
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

        if "StockD" in cfg["DATASET_NAME"]:
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
        if "StockD" in cfg["DATASET_NAME"]:
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

        forward_return = self.forward(data=data, epoch=None, iter_num=None, train=False)
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
            

    @torch.no_grad()
    @master_only
    def test_whole(self,cfg: dict = None):
        """Evaluate the ? days return rate.
        """
        self.init_test(cfg)
        # print(self.ckpt_save_dir,self.ckpt_save_dir2)
        tensorboard_writer = SummaryWriter(os.path.join(self.ckpt_save_dir2, 'tensorboard'))
        self.model.eval()
        # test loop
        # self.register_epoch_meter("test2_whole_return", "test2", "{:.3f} ($)", plt=False)
        # self.register_epoch_meter("test2_whole_hand", "test2", "{:.3f} ($)", plt=False)
        handvalue = torch.ones((1,),dtype=torch.float)
        lst_inv = torch.zeros((self.predlen,),dtype=torch.float)
        lst_ret = torch.zeros((self.predlen,),dtype=torch.float)
        K = 4
        handvalue = self.to_running_device(handvalue)
        lst_ret = self.to_running_device(lst_ret)
        lst_inv = self.to_running_device(lst_inv)

        dataloader = self.test_data_loader
        if cfg.StartTest.get("UseTrain",False):
            print("Mention : Use training data to test")           
            dataloader = self.train_data_loader
        elif cfg.StartTest.get("UseValid",False):
            print("MENTION:Use validation data to test")
            self.init_validation(cfg)
            dataloader = self.val_data_loader

        maxL = dataloader.__len__() // 15
        total_investment = []

        for nowi, data in enumerate(dataloader):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)
            lst_pre = forward_return[0]
            lst_rel = forward_return[1]

            total_investment.append(lst_ret[0] - lst_inv[0])
            handvalue += lst_ret[0]
            
            if cfg.StartTest.get("RandomSelect",False):
                ret = RndReturn(50)(lst_pre,lst_rel)
                our_ret = max(ret.unsqueeze(-1),1) * (handvalue / K)
            else :
                if cfg.StartTest.get("AllowShort",True):
                    ret = predReturn_Short(20)(lst_pre,lst_rel)
                else :
                    ret = predReturn(20)(lst_pre,lst_rel)
                    
                our_ret = ret.unsqueeze(-1) * (handvalue / K)

            lst_ret = torch.cat((lst_ret[1:],our_ret),dim=0)
            lst_inv = torch.cat((lst_inv[1:],(handvalue / K)),dim=0)
            handvalue -= (handvalue / K)

            tensorboard_writer.add_scalar("test2_whole_return", (handvalue + torch.sum(lst_ret)).item(), nowi)
            tensorboard_writer.add_scalar("test2_whole_hand", handvalue.item(), nowi)
            if nowi % maxL == 0:
                with open(self.ckpt_save_dir+"/rc.txt","a") as f : 
                    print(f"Now {nowi}/{maxL*20} we have {(handvalue + torch.sum(lst_ret)).item()}")
                    print(f"Now {nowi}/{maxL*20} we have {(handvalue + torch.sum(lst_ret)).item()}",file=f)
        
        total_investment = torch.tensor(total_investment)
        sr = total_investment.mean()/torch.sqrt(total_investment.var())
        with open(self.ckpt_save_dir+"/rc.txt","a") as f :  #这个会复制粘贴到奇怪的地方去
            print(f"Finally we have {(handvalue + torch.sum(lst_ret)).item()}, sr rate is {sr}")
            print(f"Finally we have {(handvalue + torch.sum(lst_ret)).item()}, sr rate is {sr}",file=f)

        self.tensorboard_writer.close()

    @torch.no_grad()
    @master_only
    def test_line(self,cfg: dict = None):
        self.init_test(cfg)
        # print(self.ckpt_save_dir,self.ckpt_save_dir2)
        tensorboard_writer = SummaryWriter(os.path.join(self.ckpt_save_dir2, 'tensorboard'))
        self.model.eval()
        dataloader = self.test_data_loader
        
        if cfg.StartTest.get("UseTrain",False):
            dataloader = self.train_data_loader
            print("Mention : Use training data to test")  
        elif cfg.StartTest.get("UseValid",False):
            print("MENTION:Use validation data to test")
            self.init_validation(cfg)
            dataloader = self.val_data_loader

        maxL = dataloader.__len__() // 15
        tag = cfg.StartTest.get("select",0)
        for nowi, data in enumerate(dataloader):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)
            
            prel = data[0].shape[1]
            numn = data[0].shape[2]

            lst_pre = forward_return[0].reshape(numn,prel)# 这里必须保证forward_return第三维是0
            lst_rel = forward_return[1].reshape(numn,prel)

            for i in tag:
                NV = data[0][0,0,i,-1]
                PV = data[1][0,-1,i,-1]
                tensorboard_writer.add_scalars(f"test3_value_tag{i}",  {'real': NV,'pred':PV * (1 + lst_pre[i][0])} , nowi)
                # tensorboard_writer.add_scalar(f"test3_value_tag{i}", , nowi)
                if nowi % maxL == 0:
                    print(f"Now {nowi}/{maxL*20} and tag {i} we have pred {PV * (1 + lst_pre[i][0])}, real {NV}, Predicted pct_chg {lst_pre[i][0]} , real {lst_rel[i][0]}")

            # tensorboard_writer.add_scalar("test3_pred_value", lst_rel[tag][0].item(), nowi)
        self.tensorboard_writer.close()

    @master_only
    def on_validating_end(self, train_epoch: Optional[int]):
        """Callback at the end of validating.

        Args:
            train_epoch (Optional[int]): current epoch if in training process.
        """

        if train_epoch is not None:
            self.save_best_model(train_epoch, "val_MAE", greater_best=False)
