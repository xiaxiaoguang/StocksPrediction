import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict
from basicts.utils.serialization import load_adj

from basicts.archs import imamba
from basicts.runners import SimpleTimeSeriesForecastingRunner as imambarunner
from .step_loss import normal_Stock

from basicts.data import TimeSeriesForecastingDataset


CFG = EasyDict()
CFG.NOTE = {"iMamba for spot data"}

CFG.StartTest = EasyDict()
CFG.StartTest.YES = True
CFG.StartTest.test_whole = False # 是否进行预测累计收益率的测试
CFG.StartTest.AllowShort = True
CFG.StartTest.UseTrain = False  # 是否使用训练集进行测试
CFG.StartTest.UseValid = False # 是否使用验证集测试
CFG.StartTest.test_line = True # 是否进行绘制某只股票的价格曲线的测试
CFG.StartTest.select = [0,1,2,3,4,5] # 选择哪只股票？
CFG.StartTest.ckpt_save_dir = "checkpoints/imamba_300/b54f56cb57bb22f5a65caa99683a6dea"


# ================= general ================= #
CFG.DESCRIPTION = "imamba(Crypto) configuration"
CFG.RUNNER = imambarunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "Crypto"
CFG.DATASET_TYPE = "Crypto Market"

ALL_BATCH_SIZE = 1
SEQ_LEN = 192
OUT_LEN  = 1
NUM_NODES = 181
EMBED_DIM = 96
INPUT_LEN = SEQ_LEN

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
CFG.MODEL.NAME = "imamba"
CFG.MODEL.ARCH = imamba
CFG.MODEL.PARAM = {
    "num_len":INPUT_LEN,
    "pred_len":OUT_LEN,
    "embed_dim":EMBED_DIM,
    "d_state":EMBED_DIM,
    "d_conv" :4,
    "expand" :2,
    "n_layers":3,
}

CFG.MODEL.FORWARD_FEATURES = [3]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = normal_Stock
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":2e-4,
    "amsgrad":True, # ?
    "weight_decay":2e-5,
    "eps":1.0e-8,
}

CFG.TRAIN.NUM_EPOCHS = 2
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "CosineAnnealingLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "T_max": CFG.TRAIN.NUM_EPOCHS,
    "eta_min":1e-5,
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 3.0
}
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
CFG.VAL.DATA.BATCH_SIZE = 1
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
CFG.TEST.DATA.BATCH_SIZE = 1
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 4
CFG.TEST.DATA.PIN_MEMORY = True
