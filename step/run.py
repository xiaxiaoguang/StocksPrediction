import os
import sys
from argparse import ArgumentParser

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../.."))
import torch
from basicts import launch_training,launch_test

torch.set_num_threads(2) # aviod high cpu avg usage


def parse_args():
    parser = ArgumentParser(description="Run time series forecasting model in BasicTS framework!")
    # parser.add_argument("-c", "--cfg", default="step/TSFormer_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="step/STEP_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="step/TSFormer_StockD4.py", help="training config")
    parser.add_argument("-c", "--cfg", default="step/STEP_StockD2.py", help="training config")

    # parser.add_argument("-c", "--cfg", default="step/TSFormer_PEMS04.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="step/STEP_PEMS04.py", help="training config")

    # parser.add_argument("-c", "--cfg", default="step/TSFormer_PEMS-BAY.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="step/STEP_PEMS-BAY.py", help="training config")

    # parser.add_argument("-c", "--cfg", default="step/TSFormer_PEMS08.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="step/STEP_PEMS08.py", help="training config")
    parser.add_argument("--test", action='store_true' , default=False, help="Is it testing models?")

    parser.add_argument("--gpus", default="1", help="visible gpus")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.test :
        launch_test(args.cfg,args.gpus)
    else: 
        launch_training(args.cfg, args.gpus)
