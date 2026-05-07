import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict
from basicts.utils.serialization import load_adj

from .arch import iRNNAnomalyDetector
from .runner import iTransformer2AnomalyRunner
from .loss import MultiObjectiveDetectionLoss
from .data import AnomalyDetectionDataset


CFG = EasyDict()
CFG.TRAIN = EasyDict()
CFG.NOTE = {"Anomaly Detection 3"}

CFG.TEST_ONLY = False
# CFG.TEST_ONLY = True
# CFG.TRAIN.CKPT_SAVE_DIR = "/home/benyan2023/workspace/STEP/STEP/checkpoints/iRNNAnomalyDetector_100/"
# CFG.MD5 = "latest/"
# ================= general ================= #
CFG.DESCRIPTION = "iRNN (AD) configuration"
CFG.RUNNER = iTransformer2AnomalyRunner
CFG.DATASET_CLS = AnomalyDetectionDataset
CFG.DATASET_NAME = "Minute_Origin_dataA"
CFG.DATASET_NAME = "Minute_Origin_dataA_300"
CFG.DATASET_TYPE = "Finance data"

ALL_BATCH_SIZE = 128
SEQ_LEN = 24
OUT_LEN  = 1
EMBED_DIM = 128

from .NumerMoe_AD import DATAPARAM
CFG.DATAPARAM = DATAPARAM


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
CFG.MODEL.NAME = "iRNNAnomalyDetector"
CFG.MODEL.ARCH = iRNNAnomalyDetector
CFG.MODEL.PARAM = {
    "seq_len":SEQ_LEN ,
    "d_model":EMBED_DIM,
    "dropout":0.5,
    "e_layers":4,
    "use_norm":True,
}

CFG.MODEL.FORWARD_FEATURES = None
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True

# ================= optim ================= #
CFG.TRAIN.LOSS = MultiObjectiveDetectionLoss
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":1e-3,
    "amsgrad":True, # ?
    "weight_decay":1e-4,
    "eps":1.0e-8,
}

CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "CosineAnnealingLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "T_max": CFG.TRAIN.NUM_EPOCHS // 4,
    "eta_min":1e-5,
}
# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 3.0
}
if not CFG.TEST_ONLY:
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
CFG.TRAIN.DATA.SHUFFLE = False
CFG.TRAIN.DATA.NUM_WORKERS = 1
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
CFG.VAL.DATA.NUM_WORKERS = 1
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
CFG.TEST.DATA.NUM_WORKERS = 1
CFG.TEST.DATA.PIN_MEMORY = True
