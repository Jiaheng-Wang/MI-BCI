# --------------------------------------------------------
# Feedback stage
# Written by Jiaheng Wang
# --------------------------------------------------------
from enum import Enum
import random


class MIType:
    TYPE_LEFT   = 0
    TYPE_RIGHT  = 1
    TYPE_UP     = 2
    TYPE_DOWN   = 3
    TYPE_REST   = 4

TARGET = [MIType.TYPE_LEFT, MIType.TYPE_RIGHT, MIType.TYPE_UP, MIType.TYPE_DOWN]    # note: keep MITypes in order for safe
DECODE_TYPE = [MIType.TYPE_LEFT, MIType.TYPE_RIGHT, MIType.TYPE_UP, MIType.TYPE_DOWN, ]

class mState:
    STATE_NULL          = -1
    STATE_PAUSE         = -2
    STATE_PROMPT        = 10
    STATE_ACTION        = 11
    STATE_FEEDBACK      = 12
    STATE_RESULT        = 13 # 14 correct, 15 false, 16 miss
    STATE_RELAX         = 17


class mStateDur:
    STATE_RELAX_DUR         = 2 # also noted as the preparation period
    STATE_PROMPT_DUR        = 0.5
    STATE_ACTION_DUR        = 2
    STATE_FEEDBACK_STEP     = 40 #msec
    STATE_FEEDBACK_DUR      = 10
    STATE_RESULT_DUR        = 1

class mModel:
    CSP     = 100
    FBCSP   = 101
    IFNET   = 102

speed = 25  # for a 1080 * 1080 2D plane
bias = 0
momentum = 0.8  # acceleration

RUN_NUM = 4
TRIAL_NUM = 40

IPADDRESS = '127.0.0.1'
PORT = 23618

TIME_LENGTH = 7 # data duration in storage
PULL_INTERVAL = 0.02
REF_DUR = 1.5
WIN_LENGTH = 2

REF_CHANS = [62, 63]
CHANS = [9, 10, 11, 12, 13, 14, 15,
         18, 19, 20, 21, 22, 23, 24,
         27, 28, 29, 30, 31, 32, 33,
         36, 37, 38, 39, 40, 41, 42,
         45, 46, 47, 48, 49, 50, 51,
         ]
CHANS = [c-1 for c in CHANS]

WORK_DIR = '../../'

BAD_CHANS = []
CHANS_NAMES = ["FP1","FPZ","FP2","AF7","AF3","AF4","AF8","F7","F5","F3","F1","FZ","F2","F4","F6",
    "F8","FT7","FC5","FC3","FC1","FCz","FC2","FC4","FC6","FT8","T7","C5","C3","C1","Cz","C2","C4",
    "C6","T8","TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8","P7","P5","P3","P1","PZ","P2",
    "P4","P6","P8","PO7","PO3","POz","PO4","PO8","O1","Oz","O2","F9","F10"]
CHANS_LOC_FILE = f'{WORK_DIR}/utils/getec62.sfp'

SUBJECT_NAME = 'Subject01'

MODEL_PATH = {
    mModel.FBCSP: f'{WORK_DIR}/MIFeedback/resources/{SUBJECT_NAME}/{SUBJECT_NAME}_FBCSP_s1_calibration_2s.pkl',
    mModel.IFNET: f'{WORK_DIR}/MIFeedback/resources/{SUBJECT_NAME}/{SUBJECT_NAME}_IFNet_s1_calibration_2s.pth',
}
MODEL_SET = [model for model in MODEL_PATH]
MODEL_QUEUE = []
for i in range(RUN_NUM // len(MODEL_SET)):
    random.shuffle(MODEL_SET)
    MODEL_QUEUE += MODEL_SET