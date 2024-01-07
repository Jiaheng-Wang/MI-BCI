# --------------------------------------------------------
# Feedback stage
# Written by Jiaheng Wang
# --------------------------------------------------------
import numpy as np
import pickle
from scipy import signal
import torch
from utils.pylsl_inlet import DataInlet
from ..cfg import *

from algorithms.IFNet.config import get_config

import sys, os
root = os.getcwd() + '/algorithms/'
[sys.path.append(root + dir) for dir in os.listdir(root)]


class Decoder():
    def __init__(self, dataInlet: DataInlet):
        self.dataInlet = dataInlet
        self.resample = 1
        self.ref = REF_DUR

    def setModel(self, path):
        self.model_path = path
        print(f'Using {path.split("/")[-1]} as decoders')

        if 'FBCSP' in self.model_path:
            with open(path, 'rb') as f:
                self.models = pickle.load(f)
            self.speed = speed
        elif 'IFNet' in self.model_path:
            self.models = torch.load(path)
            self.config = get_config()
            self.speed = speed
        else:
            self.models = None
            self.speed = speed
            print('warning, random decoder!')

    def process(self):
        self.dataInlet.mutex.lock()
        data = self.dataInlet.data
        self.dataInlet.mutex.unlock()

        if 'FBCSP' in self.model_path:
            '''FBCSP Decoder'''
            data = data[None, :, -int(self.dataInlet.fs * (WIN_LENGTH + self.ref)):]
            model = self.models[-1]
            filtered_data = model['fbank'].filter_data(data, model['window_details'])[..., ::self.resample]  # (F, N, C, T)

            x_features = []
            for j in range(len(TARGET)):
                x_features_fb = model['FBCSP'].transform(filtered_data, class_idx=j)
                x_features.append(model['feature_selection'][j].transform(x_features_fb))
            x_features = np.concatenate(x_features, axis=1)

            pred = model['clf'].predict_proba(x_features).reshape(-1)
            pred -= 1 / len(DECODE_TYPE)
            pred = pred.tolist()

        elif 'IFNet' in self.model_path:
            '''IFNet Decoder'''
            config = self.config

            data = data[None, :, -int(self.dataInlet.fs * (WIN_LENGTH + self.ref)):]
            # filter bank signal
            EEG_bank = []
            for bank in config.DATA.FILTER_BANK:
                parameter = signal.butter(N=5, Wn=bank, btype='bandpass', fs=config.DATA.FS)
                EEG_filtered = signal.lfilter(parameter[0], parameter[1], data)[..., -int(self.dataInlet.fs * WIN_LENGTH)::self.resample]
                EEG_bank.append(EEG_filtered)
            data = np.concatenate(EEG_bank, axis=1).astype(np.float32) / config.DATA.STD  # N,F*C,T

            model = self.models[0]
            model.eval()
            with torch.no_grad():
                pred = model(torch.from_numpy(data))
                pred = torch.softmax(pred, dim=1)
                pred = pred.numpy().reshape(-1)
                pred -= 1 / len(DECODE_TYPE)
            pred = pred.tolist()

        else:
            '''Random Decoder'''
            pred = [random.random() for _ in range(len(DECODE_TYPE))]

        idx = max(range(len(pred)), key=lambda i:pred[i])
        pred = {'value':pred[idx], 'TYPE':DECODE_TYPE[idx]}
        # hard threshold
        # pred['value'] = max(pred['value'] - 0.5, 0)
        pred['value'] *= self.speed
        return pred