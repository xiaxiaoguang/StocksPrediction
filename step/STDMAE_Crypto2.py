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

from .step_data import ForecastingDataset
from .step_loss import normal_Stock

CFG = EasyDict()


CFG.StartTest = EasyDict()
CFG.StartTest.YES = True
CFG.StartTest.test_whole = False # 是否进行预测累计收益率的测试
CFG.StartTest.AllowShort = True
CFG.StartTest.UseTrain = False  # 是否使用训练集进行测试
CFG.StartTest.UseValid = False # 是否使用验证集测试
CFG.StartTest.test_line = True # 是否进行绘制某只股票的价格曲线的测试
CFG.StartTest.select = [0,1,2,3,4,5] # 选择哪只股票？
CFG.StartTest.ckpt_save_dir = "checkpoints/STDMAE_200/08e380840121c46cd93cb99dbd4e4ec3"

ALL_BATCH = 1

# ================= general ================= #
CFG.DESCRIPTION = "STDMAE(Crypto) configuration"
CFG.RUNNER = STDMAERunner
CFG.DATASET_CLS = ForecastingDataset
CFG.DATASET_NAME = "Crypto"
CFG.DATASET_TYPE = "Crypto Market"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 1
CFG.DATASET_ARGS = {
    "seq_len": 144
}
CFG.GPU_NUM = 1
NUMPATCHSIZE = 8
EMBED_DIM = 96
NUM_NODES = 181

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
    "pre_trained_tmae_path": "checkpoints/TMAE_200/dcdfdde75b4180b9b7c684e7167259a4/TMAE_best_val_MAE.pt",
    "pre_trained_smae_path": "checkpoints/SMAE_201/4231cfa7dddf502ea487e290ff6bcef7/SMAE_best_val_MAE.pt",
    "mask_args": {
                    "patch_size":12,
                    "in_channel":1,
                    "embed_dim":EMBED_DIM,
                    "num_heads":4,
                    "mlp_ratio":4,
                    "dropout":0.1,
                    "mask_ratio":0.5,
                    "encoder_depth":4,
                    "decoder_depth":1,
                    "num_feats":1,
                    "mode":"forecasting"
    },
   "backend_args": {
                    "num_nodes" : NUM_NODES,
                    "supports"  :[torch.rand((NUM_NODES,NUM_NODES))], 
                    "dropout"   : 0.3,
                    "gcn_bool"  : True,
                    "addaptadj" : True,
                    "embed"     : EMBED_DIM,
                    "aptinit"   : None,
                    "t_len" : CFG.DATASET_INPUT_LEN,
                    "in_dim"    : 1,
                    "out_dim"   : CFG.DATASET_OUTPUT_LEN,
                    "residual_channels" : 32,
                    "dilation_channels" : 32,
                    "skip_channels"     : 256,
                    "end_channels"      : 512,
                    "kernel_size"       : 2,
                    "blocks"            : 4,
                    "layers"            : 2
    }
}

CFG.MODEL.FORWARD_FEATURES = [3]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS =  normal_Stock
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":4e-4,
    "weight_decay":1.0e-5,
    "eps":1.0e-8,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "milestones":[1, 18, 36, 54, 72, 144],
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
CFG.TRAIN.DATA.BATCH_SIZE = NUMPATCHSIZE
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
CFG.VAL.DATA.BATCH_SIZE = 1
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
CFG.TEST.DATA.BATCH_SIZE = 1
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = True
