import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict
from basicts.utils.serialization import load_adj

from .step_arch import STEP2
from .step_runner import STEPRunner
from .step_loss import step_loss
from .step_loss import stock_loss

from .step_data import ForecastingDataset


CFG = EasyDict()
CFG.NOTE = {"人生来不是为了被打倒的"}


# ================= general ================= #
CFG.DATASET_NAME = "csi500"
CFG.DESCRIPTION = f"STEP({CFG.DATASET_NAME}) configuration"
CFG.RUNNER = STEPRunner
CFG.DATASET_CLS = ForecastingDataset
CFG.DATASET_TYPE = "Stock data"

ALL_BATCH_SIZE = 4
SEQ_LEN = 100
OUT_LEN  = 5
TSF_LEN = 100
NUM_NODES = 498
NODE_FUSION = 2
INPUT_LEN = TSF_LEN
D_MODEL = 128
D_MODEL2= 256
LR=1e-4

CFG.DATASET_INPUT_LEN = SEQ_LEN
CFG.DATASET_OUTPUT_LEN = OUT_LEN
CFG.DATASET_ARGS = {
    "seq_len": TSF_LEN
}
CFG.GPU_NUM = 1

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 0
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True


# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "STEP2"
CFG.MODEL.ARCH = STEP2
CFG.MODEL.PARAM = {
    "dataset_name": CFG.DATASET_NAME,
    "pre_trained_tsformer_path": "/home/benyan2023/workspace/STEP/STEP/checkpoints/SimpleFormer_96/f349672b60fecc21cd7a334b085b7e33/SimpleFormer_best_val_MAE.pt",
    "tsformer_args": {
        "patch_size":OUT_LEN,
        "node_fusion":NODE_FUSION,
        "embed_dim":D_MODEL,
        "embed_dim2":D_MODEL2,
        "num_heads":2,
        "mlp_ratio":4,
        "dropout":0.1,
        "num_token":INPUT_LEN//OUT_LEN,
        "num_nodes":NUM_NODES//NODE_FUSION,
        "encoder1_depth":2,
        "encoder2_depth":2,
        "decoder_depth":1,
        "selected_feature":1,
        "pred_len":OUT_LEN,
        "mode":"fine-tune",
    },
    "backend_args": {
                    "num_nodes" : NUM_NODES,
                    "support_len" : 2,
                    "dropout"   : 0.5,
                    "gcn_bool"  : True,
                    "addaptadj" : True,
                    "aptinit"   : None,
                    "t_len" : CFG.DATASET_INPUT_LEN,
                    "embed_dim" : D_MODEL,
                    "embed_dim2": D_MODEL2,
                    "in_dim"    : 45,
                    "out_dim"   : OUT_LEN,
                    "residual_channels" : 32,
                    "dilation_channels" : 32,
                    "skip_channels"     : D_MODEL,
                    "end_channels"      : 512,
                    "pred_len"          : OUT_LEN,
                    "kernel_size"       : 5,
                    "blocks"            : 5,
                    "layers"            : 3
    },
    "dgl_args": { # 需要根据数据集修改trainlength
                "dataset_name": CFG.DATASET_NAME,
                "k": 11,
                "pls":5,
                "move_window": 10,
                "num_nodes":NUM_NODES,
                "input_seq_len": CFG.DATASET_INPUT_LEN,
                "output_seq_len": CFG.DATASET_OUTPUT_LEN
    }
}
CFG.MODEL.FORWARD_FEATURES = list(range(45))
CFG.MODEL.TARGET_FEATURES = [1]
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = step_loss
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":LR,
    "amsgrad":True,
    "weight_decay":2e-5,
    "eps":1.0e-8,
}

CFG.TRAIN.NUM_EPOCHS = 100
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
