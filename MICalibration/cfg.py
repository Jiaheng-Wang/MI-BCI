# --------------------------------------------------------
# Calibration stage
# Written by Jiaheng Wang
# --------------------------------------------------------
from enum import Enum


class MIType:
    TYPE_LEFT   = 0
    TYPE_RIGHT  = 1
    TYPE_UP     = 2
    TYPE_DOWN   = 3
    TYPE_REST   = 4

TARGET = [MIType.TYPE_LEFT, MIType.TYPE_RIGHT, MIType.TYPE_UP, MIType.TYPE_DOWN, ] # note: keep MITypes in order for safe

class mState:
    STATE_NULL          = -1
    STATE_PAUSE         = -2
    STATE_CROSS_BEEP    = 10
    STATE_CROSS         = 11
    STATE_INDICATOR     = 12
    STATE_RELAX         = 13

class mStateDur:
    STATE_CROSS_BEEP_DUR    = 1
    STATE_CROSS_DUR         = 1
    STATE_INDICATOR_DUR     = 4
    STATE_RELAX_DUR         = 1.5


RUN_NUM = 6
TRIAL_NUM = 40

IPADDRESS = '127.0.0.1'
PORT = 23618