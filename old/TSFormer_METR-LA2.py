import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.losses import masked_mae

from CAOFormer import CAOFormer
from .step_runner import TSFormerRunner
from .step_data import PretrainingDataset


CFG = EasyDict()
INPUT_LEN = 2016
OUTPUT_LEN = 12
BATCH_SIZE = 8

# CFG.StartTest = EasyDict()
# CFG.StartTest.YES = True
# CFG.StartTest.ckpt_save_dir = "checkpoints/MSCFormer_100/1f274b5de30cdcc8e9826c0f122a4012"

CFG.NOTE = ["这里用的MTSFormer2的架构，选择了decoder接在后方，  测试一下泛化性是不是大胜而归，因为我发现训练还是需要更多的数据所以放开限制",
            "总的来说还行，我又去调整了一下PatchEmbedding,ReLU to Mish",
            "不应该两层decoder的，调回去+修改mask generator",
            "修改了一些参数 和MTSFormer的结构 使得询问也来自Encoder之后的信息"
            "再次修改架构 使得 tsformer encoding只使用encoder decoder利用月线还原周线"]

# ================= general ================= #
CFG.DESCRIPTION = "TSFormer(METR-LA) configuration"
CFG.RUNNER = TSFormerRunner
CFG.DATASET_CLS = PretrainingDataset
CFG.DATASET_NAME = "METR-LA"
CFG.DATASET_TYPE = "Traffic speed"
CFG.DATASET_INPUT_LEN = INPUT_LEN
CFG.DATASET_OUTPUT_LEN = OUTPUT_LEN
CFG.GPU_NUM = 1

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 0
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "CAOFormer"
CFG.MODEL.ARCH = CAOFormer
CFG.MODEL.PARAM = {
    "patch_size":12,
    "in_channel":1,
    "embed_dim":96,
    "embed_dim2":384,
    "num_heads":4,
    "mlp_ratio":4,
    "dropout":0.5,
    "num_token":INPUT_LEN//(12), #我吐了啊啊啊啊啊
    "encoder_depth":2,
    "encoder_depth2":2,
    "decoder_depth":1,
    "selected_feature":0, #我吐了啊啊啊啊啊
    "mode":"pre-train",
}
CFG.MODEL.FORWARD_FEATURES = [0]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = 151
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":0.0006,
    "weight_decay":0,
    "eps":1.0e-8,
    "betas":(0.9, 0.95)
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "CosineAnnealingLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "T_max": CFG.TRAIN.NUM_EPOCHS,
    "eta_min": 8e-5,
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
CFG.TRAIN.DATA.BATCH_SIZE = BATCH_SIZE
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
CFG.VAL.DATA.BATCH_SIZE = BATCH_SIZE
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
CFG.TEST.DATA.BATCH_SIZE = BATCH_SIZE
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = True
