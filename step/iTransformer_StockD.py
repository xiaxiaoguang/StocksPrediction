import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict
from basicts.utils.serialization import load_adj

from basicts.archs import iTransformer
from basicts.runners import iTransformerRunner
from .step_loss import normal_loss
from basicts.data import TimeSeriesForecastingDataset


CFG = EasyDict()
CFG.NOTE = {"新crypto数据,SPOT数据,新iTransformer"}

# ================= general ================= #
CFG.DESCRIPTION = "iTransformer(Crypto) configuration"
CFG.RUNNER = iTransformerRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "StockD"
CFG.DATASET_TYPE = "Finance data"

ALL_BATCH_SIZE = 8
SEQ_LEN = 256 #尝试800 - 1
OUT_LEN  = 1
EMBED_DIM = 32
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
CFG.MODEL.NAME = "iTransformer"
CFG.MODEL.ARCH = iTransformer
CFG.MODEL.PARAM = {
    "seq_len":INPUT_LEN,
    "pred_len":OUT_LEN,
    "d_model":EMBED_DIM,
    "dropout":0.1,
    "d_ff"   :2048,
    "n_heads":8,
    "e_layers":2,
    "use_norm":True,
}

CFG.MODEL.FORWARD_FEATURES = [1]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = normal_loss
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":1e-3,
    "amsgrad":True, # ?
    "weight_decay":1e-4,
    "eps":1.0e-8,
}

CFG.TRAIN.NUM_EPOCHS = 200
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
