import os
import sys


# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
import torch.nn as nn
from easydict import EasyDict
from basicts.utils.serialization import load_adj

from .step_loss import normal_loss
from .step_data import ForecastingDataset

from basicts.data import TimeSeriesForecastingDataset
from basicts.archs.arch_zoo.agcrn_arch import AGCRN
from basicts.runners.runner_zoo.agcrn_runner import AGCRNRunner

CFG = EasyDict()


CFG.StartTest = EasyDict()
CFG.StartTest.YES = True
CFG.StartTest.test_whole = True
CFG.StartTest.UseTrain = True
CFG.StartTest.ckpt_save_dir = "benchmark/AGCRN_100/1b0b55bad566934a2997ec170393cc03"
ALL_BATCH = 1

# ================= general ================= #
CFG.DESCRIPTION = "AGCRN(StockD) configuration"
CFG.RUNNER = AGCRNRunner
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
# 昨日私パン食べるませんでした
# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "AGCRN"
CFG.MODEL.ARCH = AGCRN

CFG.MODEL.PARAM = {
    "num_nodes":500,
    "input_dim":45,
    "rnn_units":10,
    "output_dim":2,
    "horizon":12,
    "num_layers":3,
    "embed_dim":128,
    "cheb_k":6,
    "default_graph":torch.randn((500,500))
}

# 前向传递的特征和后向传递的特征
CFG.MODEL.FORWARD_FEATURES = list(range(45))
CFG.MODEL.TARGET_FEATURES = [1]
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True

# ================= optim ================= #
CFG.TRAIN = EasyDict()

CFG.TRAIN.LOSS = normal_loss
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":0.005,
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
CFG.TRAIN.DATA.BATCH_SIZE = ALL_BATCH
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 4
CFG.TRAIN.DATA.PIN_MEMORY = True

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = ALL_BATCH
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 4
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
CFG.TEST.DATA.BATCH_SIZE = ALL_BATCH
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 4
CFG.TEST.DATA.PIN_MEMORY = True
