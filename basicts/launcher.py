from typing import Dict, Union
import traceback

import easytorch

def launch_training(cfg: Union[Dict, str], gpus: str = None, node_rank: int = 0):
    """Extended easytorch launch_training.

    Args:
        cfg (Union[Dict, str]): Easytorch config.
        gpus (str): set ``CUDA_VISIBLE_DEVICES`` environment variable.
        node_rank (int): Rank of the current node.
    """

    # pre-processing of some possible future features, such as:
    # registering model, runners.
    # config checking
    # pass
    # # launch training based on easytorch
    # try:
    #     easytorch.launch_training(cfg=cfg, devices=gpus, node_rank=node_rank)
    # except TypeError as e:
    #     if "launch_training() got an unexpected keyword argument" in repr(e):
    #         # NOTE: for earlier easytorch version
    easytorch.launch_training(cfg=cfg, gpus=gpus, node_rank=node_rank)
        # else:
        #     raise e
        
def test(cfg: Union[Dict, str], runner):
    runner.init_logger(logger_name='easytorch-testing', log_file_name='testing_log')
    try:
        runner.test(cfg)
    except BaseException as e:
        # log exception to file
        runner.logger.error(traceback.format_exc())
        raise e
    
def launch_test(cfg: Union[Dict, str], gpus: str = None, node_rank: int = 0):
    easytorch.launch_runner(cfg=cfg,fn=test,gpus=gpus)
