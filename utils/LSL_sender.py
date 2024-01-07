# --------------------------------------------------------
# Simulated lsl stream with random signals
# Written by Jiaheng Wang
# --------------------------------------------------------
import time
from random import random as rand
from pylsl import StreamInfo, StreamOutlet
import pylsl
import numpy as np


cn = 64
fs = 256
info = StreamInfo('g.tec62', 'EEG', cn, fs, pylsl.cf_float32, 'BCILab')
outlet = StreamOutlet(info)
print("now sending data...")

while True:
    mysample = np.random.randn(cn).astype(np.float32)*25
    outlet.push_sample(mysample)
    time.sleep(1/fs)