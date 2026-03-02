import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict
from basicts.utils.serialization import load_adj

from basicts.archs import mamba2
from basicts.runners import MambaRunner
from step.step_loss import mamba_loss
from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import *

CFG = EasyDict()

# 全数据 但是参数量倍增！
# 强噪声 噪声强度0.04
# 新型loss，contrastive中增加了negative的系数和对应的信息
# ================= general ================= #
CFG.DESCRIPTION = "mamba(METR-LA) configuration"
CFG.RUNNER = MambaRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "METR-LA"
CFG.DATASET_TYPE = "METR-LA data"


CFG.ContrastiveLoss = True
CFG.RepeatTimes = 3
CFG.StrongNoise= True
CFG.RepeatNoise = 0.04

CFG.METRICS = {
    "MAE":masked_mae,
    "RMSE":masked_rmse,
    "MAPE":masked_mape,
    "MSE":masked_mse,
}

ALL_BATCH_SIZE = 32
SEQ_LEN = 12
OUT_LEN  = 12
NUM_NODES = -1
EMBED_DIM = 96
INPUT_LEN = SEQ_LEN
NUM_EPOCHS = 200

CFG.DATASET_INPUT_LEN = SEQ_LEN
CFG.DATASET_OUTPUT_LEN = OUT_LEN
CFG.GPU_NUM = 1

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 0
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True


# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "mamba"
CFG.MODEL.ARCH = mamba2
CFG.MODEL.PARAM = {
    "input_len":INPUT_LEN,
    "pred_len":OUT_LEN,
    "input_dim":3,
    "embed_dim":EMBED_DIM,
    "d_state"  :EMBED_DIM-32,
    "d_conv"   :4,
    "expand"   :2,
    "n_layers" :3,
}

CFG.MODEL.FORWARD_FEATURES = None
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = mamba_loss
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":0.001,
    "amsgrad":True, # ?
    "weight_decay":2e-5,
    "eps":1.0e-8,
}
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "CosineAnnealingLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "T_max"  : 50,
    "eta_min":1e-4,
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 3.0
}
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = ALL_BATCH_SIZE
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
CFG.VAL.DATA.BATCH_SIZE = ALL_BATCH_SIZE
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
CFG.TEST.DATA.BATCH_SIZE = ALL_BATCH_SIZE
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 4
CFG.TEST.DATA.PIN_MEMORY = True
