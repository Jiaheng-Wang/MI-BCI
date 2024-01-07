# --------------------------------------------------------
# IFNet
# Written by Jiaheng Wang
# --------------------------------------------------------
import os
import random
from yacs.config import CfgNode as CN


_C = CN()

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()

#Path to dataset
_C.DATA.DATA_PATH = '~/dataset dir/'
#_C.DATA.DATA_PATH = '~/Datasets/MI/BCIC/'
#_C.DATA.DATA_PATH = '~/Datasets/MI/openbmi62/'
_C.DATA.TRAIN_FILE = ['s1_calibration.mat']
_C.DATA.TEST_FILE = ['s1_feedback.mat']
_C.DATA.BATCH_SIZE = 32
_C.DATA.RTA = 5 #repeated trial augmentation
_C.DATA.K_FOLD = 5
_C.DATA.FOLD_STEP = 1
_C.DATA.BLOCK = True    #block-wise cv

_C.DATA.FILTER_BANK = [(4, 16), (16, 40)]
_C.DATA.FS = 256
_C.DATA.RESAMPLE = 1
_C.DATA.REF_DUR = 1 #time duration of baseline reference before the cue start

_C.DATA.REF_CHANS = []
_C.DATA.CHANS = [9, 10, 11, 12, 13, 14, 15,
         18, 19, 20, 21, 22, 23, 24,
         27, 28, 29, 30, 31, 32, 33,
         36, 37, 38, 39, 40, 41, 42,
         45, 46, 47, 48, 49, 50, 51,
                 ]
_C.DATA.CHANS = [c-1 for c in _C.DATA.CHANS]
_C.DATA.BAD_CHANS = [] # interpolated by spherical splines
_C.DATA.CHANS_NAMES = ["FP1","FPZ","FP2","AF7","AF3","AF4","AF8","F7","F5","F3","F1","FZ","F2","F4","F6",
    "F8","FT7","FC5","FC3","FC1","FCz","FC2","FC4","FC6","FT8","T7","C5","C3","C1","Cz","C2","C4",
    "C6","T8","TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8","P7","P5","P3","P1","PZ","P2",
    "P4","P6","P8","PO7","PO3","POz","PO4","PO8","O1","Oz","O2","F9","F10"]
_C.DATA.CHANS_LOC_FILE = '../../utils/getec62.sfp'

_C.DATA.MEAN = 0
_C.DATA.STD = 5.
_C.DATA.TIME_WIN = [0.5, 2.5]
_C.DATA.DUR = 2
_C.DATA.WIN_STEP = 4

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = 'IFNet'
_C.MODEL.NUM_CLASSES = 4
_C.MODEL.TIME_POINTS = int(_C.DATA.DUR * int(_C.DATA.FS / _C.DATA.RESAMPLE))
_C.MODEL.IN_CHANS = 35
_C.MODEL.PATCH_SIZE = 128   #temporal pooling size
_C.MODEL.EMBED_DIMS = 64
_C.MODEL.KERNEL_SIZE = 63
_C.MODEL.RADIX = 2

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.DEVICE = 0
_C.SEED = 2023
_C.SAVE = True
_C.EVAL = False
_C.EVAL_TAG = ''
_C.OUTPUT = 'output'
_C.TAG = 'IFNet_s1_calibration_2s'    #file name to log experiment results

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 1000
_C.TRAIN.BASE_LR = 2 ** -12
_C.TRAIN.WEIGHT_DECAY = 0.01
_C.TRAIN.LR_SCHEDULER = None

_C.TRAIN.RETRAIN = True
_C.TRAIN.RETRAIN_EPOCHS = 500

_C.TRAIN.OPTIMIZER = 'AdamW'
_C.TRAIN.CRITERION = 'CE'
_C.TRAIN.REPEAT = 1  #repeat training with differnt network initilization


def get_config():
    return _C.clone()