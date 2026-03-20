import os
import sys
import random

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict
from basicts.utils.serialization import load_adj

from .stdmae_arch import STDMAE
from .stdmae_runner import STDMAERunner

from .stdmae_data import ForecastingDataset
from basicts.data import TimeSeriesForecastingDataset
from .step_loss import normal_loss
from basicts.utils import load_adj

CFG = EasyDict()

# CFG.StartTest = EasyDict()
# CFG.StartTest.YES = False
# CFG.StartTest.CKPT_SAVE_DIR = "checkpoints/STDMAE_120/c0970cd6f337d6059269fcd03f84bae0"

# ================= general ================= #
CFG.DESCRIPTION = "STDMAE(StockD) configuration"
CFG.RUNNER = STDMAERunner
CFG.DATASET_CLS = ForecastingDataset
CFG.DATASET_NAME = "StockD"
CFG.DATASET_TYPE = "Stock Market"
CFG.DATASET_INPUT_LEN = 144
CFG.DATASET_OUTPUT_LEN = 12
CFG.DATASET_ARGS = {
    "seq_len": 144
    }
CFG.GPU_NUM = 1

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 19260817
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "STDMAE"
CFG.MODEL.ARCH = STDMAE
CFG.MODEL.PARAM = {
    "dataset_name": CFG.DATASET_NAME,
    "pre_trained_tmae_path": "checkpoints/SMAE_130/f85827afcd9c000263bd7d55ce09a2bc/SMAE_best_val_MAE.pt",
    "pre_trained_smae_path": "checkpoints/TMAE_120/0c502e790cfae1247fd54f88fc3f02dd/TMAE_best_val_MAE.pt",
    "mask_args": {
                    "patch_size":24,
                    "in_channel":1,
                    "embed_dim":512,
                    "num_heads":4,
                    "mlp_ratio":4,
                    "dropout":0.1,
                    "mask_ratio":0.6,
                    "encoder_depth":4,
                    "decoder_depth":1,
                    "selected_feature":1,
                    "mode":"forecasting"
    },
   "backend_args": {
                    "num_nodes" : 500,
                    "supports"  :[torch.rand((500,500))], 
                    "dropout"   : 0.3,
                    "gcn_bool"  : True,
                    "addaptadj" : True,
                    "embed"     : 512,
                    "aptinit"   : None,
                    "t_len" : CFG.DATASET_INPUT_LEN,
                    "embed_dim" : 512,
                    "in_dim"    : 45,
                    "out_dim"   : 12,
                    "residual_channels" : 32,
                    "dilation_channels" : 32,
                    "skip_channels"     : 256,
                    "end_channels"      : 512,
                    "kernel_size"       : 2,
                    "blocks"            : 4,
                    "layers"            : 2
    }
}
CFG.MODEL.FORWARD_FEATURES = list(range(45))
CFG.MODEL.TARGET_FEATURES = [1]
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS =  normal_loss
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":0.0006,
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
CFG.TRAIN.NUM_EPOCHS = 120

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
CFG.TRAIN.DATA.BATCH_SIZE = 8
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
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
CFG.VAL.DATA.BATCH_SIZE = 8
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
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
CFG.TEST.DATA.BATCH_SIZE = 8
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = True
