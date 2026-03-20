import os
import sys


# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict
from basicts.utils.serialization import load_adj

from .step_arch import STEP
from .step_runner import STEPRunner
from .step_loss import step_loss
from .step_data import ForecastingDataset


CFG = EasyDict()
# ================= Just for Testing ================= #

CFG.StartTest = EasyDict()
CFG.StartTest.YES = True # 是否开始测试
CFG.StartTest.ckpt_save_dir = "checkpoints/STEP_101/33c427ffd00bf49eb41c43fd4b930c5e" # 选择哪个模型测试
CFG.StartTest.test_whole = True # 是否进行预测累计收益率的测试
CFG.StartTest.UseTrain = True  # 是否使用训练集进行测试
CFG.StartTest.UseValid = False # 是否使用验证集测试
CFG.StartTest.test_line = True # 是否进行绘制某只股票的价格曲线的测试
CFG.StartTest.select = 0 # 选择哪只股票？


# ================= general ================= #
CFG.DESCRIPTION = "STEP(StockD) configuration"
CFG.RUNNER = STEPRunner
CFG.DATASET_CLS = ForecastingDataset
CFG.DATASET_NAME = "StockD"
CFG.DATASET_TYPE = "Stock data"

ALL_BATCH_SIZE = 1
SEQ_LEN = 15
TSF_LEN = 640
OUT_LEN  = 3
EMBED_DIM = 128

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

INPUT_LEN = 800
# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "STEP"
CFG.MODEL.ARCH = STEP
CFG.MODEL.PARAM = {
    "dataset_name": CFG.DATASET_NAME,
    "pre_trained_tsformer_path": "checkpoints/MSCFormer_101/5881534343b6f8e1e23b85e69e6f08e2/MSCFormer_101.pt",
    "tsformer_args": {
            "patch" : [2,2,2,2],
            "in_channel":1,
            "embed_dim":128,
            "num_heads":4,
            "mlp_ratio":4,
            "dropout":0.5,
            "num_token":INPUT_LEN//(16), #我吐了啊啊啊啊啊
            "mask_ratio":0.85, # high level masked!
            "num_decoder":4,
            "decoder_depth" : [1,1,1,1],
            "selected_feature":1, #我吐了啊啊啊啊啊
            "mode":"forecasting",
    },
    "backend_args": {
                    "num_nodes" : 500,
                    "support_len" : 2,
                    "dropout"   : 0.5,
                    "gcn_bool"  : True,
                    "addaptadj" : True,
                    "aptinit"   : None,
                    "t_len" : CFG.DATASET_INPUT_LEN,
                    "embed_dim" : EMBED_DIM,
                    "in_dim"    : 45,
                    "out_dim"   : OUT_LEN,
                    "residual_channels" : 32,
                    "dilation_channels" : 32,
                    "skip_channels"     : EMBED_DIM,
                    "end_channels"      : 512,
                    "kernel_size"       : 2,
                    "blocks"            : 5,
                    "layers"            : 2
    },
    "dgl_args": {
                "dataset_name": CFG.DATASET_NAME,
                "k": 10,
                "pls":3,
                "move_window": 8,
                "input_seq_len": CFG.DATASET_INPUT_LEN,
                "output_seq_len": CFG.DATASET_OUTPUT_LEN
    }
}
CFG.MODEL.FORWARD_FEATURES = list(range(45))
CFG.MODEL.TARGET_FEATURES = [1]
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True


# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 0
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = step_loss
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":0.002,
    "weight_decay":1.0e-5,
    "eps":1.0e-8,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "milestones":[1, 50, 100, 170],
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
