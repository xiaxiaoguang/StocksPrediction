import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.losses import masked_mae
from CAOFormer3 import CAOFormer3
from CAOFormer3.caoformer_runner import CAOFormerRunner
from .step_data import PretrainingDataset


CFG = EasyDict()

ALLBATCH_SIZE = 4
INPUT_LEN = 2016
OUTPUT_LEN = 12
NUMNODES = 207
CFG.NOTE = {"Try small MAE"}
# ================= general ================= #
CFG.DESCRIPTION = "TSFormer(StockD) configuration"
CFG.RUNNER = CAOFormerRunner
CFG.DATASET_CLS = PretrainingDataset
CFG.DATASET_NAME = "METR-LA"
CFG.DATASET_TYPE = "Pretrained Stock Data"
CFG.DATASET_INPUT_LEN = INPUT_LEN
CFG.DATASET_OUTPUT_LEN = OUTPUT_LEN
CFG.GPU_NUM = 1

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 19260817
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "CAOFormer3"
CFG.MODEL.ARCH = CAOFormer3
CFG.MODEL.PARAM = {
    "patch_size":OUTPUT_LEN,
    "node_fusion":3,
    "embed_dim":96,
    "embed_dim2":384,
    "num_heads":4,
    "mlp_ratio":4,
    "dropout":0.1,
    "num_token":INPUT_LEN//OUTPUT_LEN, 
    "num_nodes":NUMNODES//3,
    "encoder1_depth":[1,1],
    "encoder2_depth":[1,1],
    "decoder_depth":2,
    "selected_feature":0, #这里需要尝试
    "num_feats":3,
    "pred_len":OUTPUT_LEN,
    "mode":"pre-train",
}
CFG.MODEL.FORWARD_FEATURES = [0,1,2]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = 150

CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":1e-3,
    "weight_decay":0,
    "eps":1.0e-8,
    "betas":(0.9, 0.95)
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "CosineAnnealingLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "T_max":CFG.TRAIN.NUM_EPOCHS,
    "eta_min":1e-5
}
# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
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
CFG.TRAIN.DATA.BATCH_SIZE = ALLBATCH_SIZE
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
CFG.VAL.DATA.BATCH_SIZE = ALLBATCH_SIZE
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
CFG.TEST.DATA.BATCH_SIZE = ALLBATCH_SIZE
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 4
CFG.TEST.DATA.PIN_MEMORY = True

