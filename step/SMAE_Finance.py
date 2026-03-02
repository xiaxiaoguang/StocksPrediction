import os
import sys
import random

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.losses import masked_rmse

from basicts.archs import Mask
from basicts.runners import MaskRunner
from .step_data import PretrainingDataset
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "SMAE(finance) configuration"
CFG.RUNNER = MaskRunner
CFG.DATASET_CLS = PretrainingDataset
CFG.DATASET_NAME = "finance"
CFG.DATASET_TYPE = "Pretrained finance data"
CFG.DATASET_INPUT_LEN = 144
CFG.DATASET_OUTPUT_LEN = 12
CFG.GPU_NUM = 1
BATCH_SIZE_ALL = 8
EMBED_DIM = 96

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 19260817
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "SMAE"
CFG.MODEL.ARCH = Mask
CFG.MODEL.PARAM = {
    "patch_size":CFG.DATASET_OUTPUT_LEN,
    "in_channel":1,
    "embed_dim":EMBED_DIM,
    "num_heads":4,
    "mlp_ratio":4,
    "dropout":0.1,
    "mask_ratio":0.5,
    "encoder_depth":4,
    "decoder_depth":1,
    "spatial":True,
    "mode":"pre-train"
}

CFG.MODEL.FORWARD_FEATURES = [0]
CFG.MODEL.TARGET_FEATURES = [0] # 这个不重要

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_rmse
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":0.001,
    "weight_decay":0,
    "eps":1.0e-8,
    "betas":(0.9, 0.95)
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "milestones":[40,70,100,150],
    "gamma":0.5
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
}
CFG.TRAIN.NUM_EPOCHS = 201
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
CFG.TRAIN.DATA.BATCH_SIZE =BATCH_SIZE_ALL
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
