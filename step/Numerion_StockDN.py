import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict
from basicts.utils.serialization import load_adj

from basicts.archs import Numerion,NumerionArgs
from basicts.runners import NumerionRunner
from .step_loss import normal_loss
from basicts.data import TimeSeriesForecastingDataset

CFG = EasyDict()
CFG.NOTE = {"CSI500 New Data with Latest Numerion, retry with closed price"}

ALL_BATCH_SIZE = 8

# ALL_BATCH_SIZE = 1
# CFG.StartTest = EasyDict()
# CFG.StartTest.YES = True # 是否开始测试
# CFG.StartTest.ckpt_save_dir = "/home/benyan2023/workspace/STEP/STEP/checkpoints/Numerion_200/in48-pred3retry" # 选择哪个模型测试
# CFG.StartTest.test_whole = True # 是否进行预测累计收益率的测试
# CFG.StartTest.AllowShort = False # 是否允许做空
# CFG.StartTest.UseTrain = False  # 是否使用训练集进行测试
# CFG.StartTest.UseValid = False # 是否使用验证集测试
# CFG.StartTest.RandomSelect=False # 是否随机选股票
# CFG.StartTest.test_line = False # 是否进行绘制某只股票的价格曲线的测试
# CFG.StartTest.select = [0,1,2,3,4,5] # 选择哪只股票？


# ================= general ================= #
CFG.DESCRIPTION = "Numerion (CSI500) configuration"
CFG.RUNNER = NumerionRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "csi500"
CFG.DATASET_TYPE = "Finance data"


SEQ_LEN=12
OUT_LEN=3
NUM_NODES = 500

Args = NumerionArgs(
    seq_len=SEQ_LEN,
    pred_len=OUT_LEN,
    input_dim=NUM_NODES,
    d_model=[12,4],
    n_layer=2,
    patch_level=-1
)

# Args.seq_len = SEQ_LEN #尝试800 - 1
# Args.pred_len  = OUT_LEN
# Args.input_dim = 498
# Args.d_model = [64,16]
# Args.n_layer = 2
# Args.patch_level = -1

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
CFG.MODEL.NAME = "Numerion"
CFG.MODEL.ARCH = Numerion
CFG.MODEL.PARAM = {
    "configs":Args
}

CFG.MODEL.FORWARD_FEATURES = None
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
    "T_max": CFG.TRAIN.NUM_EPOCHS // 4,
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
