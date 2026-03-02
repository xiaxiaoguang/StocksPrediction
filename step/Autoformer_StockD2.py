import os
import sys


# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
import torch.nn as nn
import numpy as np

from easydict import EasyDict
from basicts.utils.serialization import load_adj


from .step_loss import autoformer_loss
from .step_data import ForecastingDataset

from basicts.data import TimeSeriesForecastingDataset
from basicts.archs.arch_zoo.autoformer_arch import Autoformer
from basicts.runners.runner_zoo.autoformer_runner import AutoformerRunner

CFG = EasyDict()


CFG.StartTest = EasyDict()
CFG.StartTest.YES = True
CFG.StartTest.test_whole = True
CFG.StartTest.ckpt_save_dir = ""

# ================= general ================= #
CFG.DESCRIPTION = "Autoformer(StockD) configuration"
CFG.RUNNER = AutoformerRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "StockD"
CFG.DATASET_TYPE = "Stock data"
CFG.DATASET_INPUT_LEN = 144
CFG.DATASET_OUTPUT_LEN = 12
CFG.GPU_NUM = 1

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 0
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# 昨日、私はパンを食べませんでした 
# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "Autoformer"
CFG.MODEL.ARCH = Autoformer

CFG.MODEL.PARAM = {
    "seq_len":144,
    "label_len":144,
    "pred_len":12,
    "output_attention":0,
    "time_of_day_size":0,
    "day_of_week_size":0,
    "embed":"timeF",
    "moving_avg":15,
    "enc_in":500,
    "dec_in":500,
    "activation":nn.ReLU,
    "d_model":1024,
    "num_time_features":44,
    "dropout":0.05,
    "factor":1,
    "output_attention":False,
    "d_ff":2048,
    "e_layers":4,
    "d_layers":3,
    "n_heads":8,
    "c_out":500
}

# 前向传递的特征和后向传递的特征
CFG.MODEL.FORWARD_FEATURES = list(range(45))
CFG.MODEL.TARGET_FEATURES = [1]
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True

# ================= optim ================= #
CFG.TRAIN = EasyDict()

CFG.TRAIN.LOSS = autoformer_loss
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":0.001,
    "weight_decay":1.0e-5,
    "eps":1.0e-8,
}

CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "milestones":[1, 18, 36, 54, 72],
    "gamma":0.5
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 3.0
}
CFG.TRAIN.NUM_EPOCHS = 2
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 10
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 8
CFG.TRAIN.DATA.PIN_MEMORY = True
# curriculum learning
CFG.TRAIN.CL = EasyDict()
CFG.TRAIN.CL.WARM_EPOCHS = 0
CFG.TRAIN.CL.CL_EPOCHS = 6
CFG.TRAIN.CL.PREDICTION_LENGTH = 12

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 10
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 8
CFG.VAL.DATA.PIN_MEMORY = True

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# evluation
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 10
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 8
CFG.TEST.DATA.PIN_MEMORY = True
