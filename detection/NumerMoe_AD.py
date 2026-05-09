import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict
from basicts.utils.serialization import load_adj

from .arch import NumerMoe
from .runner import iTransformer2AnomalyRunner
from .loss import MultiObjectiveDetectionLoss
from .data import AnomalyDetectionDataset

CFG = EasyDict()
CFG.TRAIN = EasyDict()
CFG.NOTE = {"NumerMOE REAL COMPLEX"}

CFG.TEST_ONLY = False
# CFG.TEST_ONLY = True
# CFG.TRAIN.CKPT_SAVE_DIR = "/home/benyan2023/workspace/STEP/STEP/checkpoints/NumerMoe_100/"
# CFG.MD5 = "c2d690c7ed349a154a77cd6dd41e294a"

# ================= general ================= #
CFG.DESCRIPTION = "NumerMoe (AD) configuration"
CFG.RUNNER = iTransformer2AnomalyRunner
CFG.DATASET_CLS = AnomalyDetectionDataset
CFG.DATASET_NAME = "Minute_Origin_dataA"
CFG.DATASET_NAME = "Minute_Origin_dataA_300"
CFG.DATASET_TYPE = "Finance data"

ALL_BATCH_SIZE = 128
SEQ_LEN = 48
OUT_LEN  = 64
D_FF = 512
NUM_VARIABLE = 50
DROPOUT = 0.5

DATAPARAM = {
    "seq_len": SEQ_LEN,
    "num_anomalies": 0,
    "x_h": 0.3, "x_f": 0.3,
    "y_h": 1, "y_f": 4,
    "z_h": 1, "z_f": 1,
}


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
CFG.MODEL.NAME = "NumerMoe"
CFG.MODEL.ARCH = NumerMoe

CFG.MODEL.PARAM = {
    "tim_configs":{
        'seq_len':SEQ_LEN,
        'pre_len':OUT_LEN,
        'input_dim':1,
        'd_model':[64,16],
        "n_layer":2,
        "dropout":DROPOUT,
        "patch_level":-1,
    },

    "spc_configs":{
        'd_model':OUT_LEN*2,
        'n_heads':    4,
        'num_layers' :4,
        'num_experts':4,
        'top_k'      :2,
        'dropout':DROPOUT,
        'd_ff':D_FF,
        'num_stocks':NUM_VARIABLE,
    }
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
    "weight_decay":1e-5,
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
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
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
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = False

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
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False
