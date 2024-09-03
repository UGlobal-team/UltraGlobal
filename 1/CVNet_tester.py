r""" Test code of Correlation Verification Network """
# written by Seongwon Lee (won4113@yonsei.ac.kr)

import torch
import core.checkpoint as checkpoint
from config import cfg
from model.CVNet_Rerank_model import CVNet_Rerank
from test.test_model import test_model
from model.uGlobal import uGlobal
import logging
import os
import json

logger = logging.getLogger(__name__)

logger.setLevel(level = logging.INFO)

handler = logging.FileHandler("log.txt")   

handler.setLevel(logging.INFO)

logger.addHandler(handler)

logger.info("Start print log")  
def setup_model():
    """Sets up a model for training or testing and log the results."""
    # Build the model
    print("=> creating CVNet_Rerank model")
    model = CVNet_Rerank(cfg.MODEL.DEPTH, cfg.MODEL.HEADS.REDUCTION_DIM, cfg.SupG.relup)
    print(model)
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)

    return model

def load_model_state(model, path, config: dict):
    x = torch.load(path)
    model_type = config["model"]

    model.pan.load_state_dict(x['pan_state_dict'])
    if model_type == "resnet101":
        model.resnet101.load_state_dict(x['resnet101_state_dict'])
    else:
        model.resnet50.load_state_dict(x['resnet50_state_dict'])
    model.gemp.load_state_dict(x['gemp_state_dict'])
    model.rgem.load_state_dict(x['rgem_state_dict'])
    model.sgem.load_state_dict(x['sgem_state_dict'])
    model.head.load_state_dict(x['head_state_dict'])

def __main__():
    """Test the model."""
    if cfg.TEST.WEIGHTS == "":
        print("no test weights exist!!")
    else:
        # # Construct the model
        # model = setup_model()
        # # Load checkpoint
        # checkpoint.load_checkpoint(os.path.join(cfg.TEST.WEIGHTS, 'CVPR2022_CVNet_R50.pyth'), model)


        #our model
        path=r"..\model\model_state.pth"
        
        attributes = ['all_souls', 'oxford', 'ashmolean', 'balliol', 'bodleian', 'christ_church', 'cornmarket', 'hertford', 'keble', 'magdalen', 'pitt_rivers', 'radcliffe_camera', 'jesus', 'new', 'oriel', 'trinity', 'worcester']
        config = {
            'n_labels': len(attributes),
            'bs': 1,
            'lr': 1.5e-6,
            'warmup': 0.2,
            'w_decay': 0.001,
            'n_epochs': 1,
            'reduction_dim': 256 
        }
        model = uGlobal(config)
        load_model_state(model, path, config)

        test_model(model, cfg.TEST.DATA_DIR, cfg.TEST.DATASET_LIST, cfg.TEST.SCALE_LIST, cfg.SupG.rerank, cfg.SupG.gemp, cfg.SupG.rgem, cfg.SupG.sgem, cfg.SupG.onemeval, cfg.MODEL.DEPTH, logger)
