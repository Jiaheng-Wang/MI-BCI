import torch
from torch.utils.data import Dataset
import math
import numpy as np
from scipy import io, signal
from sklearn.preprocessing import LabelEncoder
import random
import time
from ..utils.tools import make_interpolation_matrix


class EEG_Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        Dataset.__init__(self)
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, item):
        data = self.data[item]
        if self.transform:
            data = self.transform(data.clone())
        return data, self.labels[item]

    def __len__(self):
        return len(self.labels)


def seed_torch(seed=618):
    random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def window_split(x, y, config):
    fs = config.DATA.FS / config.DATA.RESAMPLE
    start = int(config.DATA.TIME_WIN[0] * fs)
    end = start + config.MODEL.TIME_POINTS
    step = int(config.DATA.WIN_STEP * fs)
    N, C, T = x.shape
    data = []
    while end <= T:
        data.append(x[..., start:end])
        start += step
        end += step
    labels = torch.stack([y] * len(data), dim=-1).reshape(-1)
    data = torch.stack(data, dim=1).reshape((-1, C, end - start))
    #print(f'window split {data.shape} {labels.shape}')
    return data, labels


def preprocess(data_file, config, stage='test'):
    '''preprocess EEG data and labels of shape (N,C,T) and (N)'''
    print(f'loading data from: {data_file}')
    data = io.loadmat(data_file)

    # label numbers must start at 0
    EEG = data['EEG_data'].astype(np.float64)
    EEG = EEG.transpose((2, 0, 1))
    labels = data['labels'].reshape(-1).astype(np.int32) # N
    labels = LabelEncoder().fit_transform(labels)

    if config.DATA.REF_CHANS:
        EEG_ref = EEG[:, config.DATA.REF_CHANS, :]
        EEG_ref = np.mean(EEG_ref, axis=1, keepdims=True)
        EEG = EEG - EEG_ref
        EEG = np.delete(EEG, config.DATA.REF_CHANS, axis=1)
    if config.DATA.BAD_CHANS:
        interpolation_matrix = make_interpolation_matrix(config.DATA.CHANS_NAMES, config.DATA.CHANS_LOC_FILE, config.DATA.FS, config.DATA.BAD_CHANS)
        EEG[:, config.DATA.BAD_CHANS, :] = np.matmul(interpolation_matrix, np.delete(EEG, config.DATA.BAD_CHANS, axis=1))
    if config.DATA.CHANS:
        EEG = EEG[:, config.DATA.CHANS, :]

    # # deal with drift channels
    # channels_names =  [config.DATA.CHANS_NAMES[idx] for idx in config.DATA.CHANS]
    #
    # parameter = signal.butter(N=5, Wn=4, btype='highpass', fs=config.DATA.FS)
    # EEG_inspect = signal.lfilter(parameter[0], parameter[1], EEG)
    # EEG_inspect = np.abs(EEG_inspect)[..., config.DATA.REF_DUR * config.DATA.FS:]
    # EEG_bads = np.any(EEG_inspect > 100, axis=-1)   # naive classification, bad drift channel?
    #
    # idxes = np.array(range(EEG.shape[1]))
    # for i in range(EEG.shape[0]):
    #     bads_idxes = idxes[EEG_bads[i]]
    #     if len(bads_idxes):
    #         #print(f'instance {i}    drift channels {bads_idxes}')
    #         interpolation_matrix = make_interpolation_matrix(channels_names, config.DATA.CHANS_LOC_FILE, config.DATA.FS, bads_idxes)
    #         EEG[i, bads_idxes, :] = np.matmul(interpolation_matrix, np.delete(EEG[i], bads_idxes, axis=0))
    # print(f'total drift instances: {np.sum(np.any(EEG_bads, axis=1))}')

    #filter bank signals
    EEG_bank=[]
    for bank in config.DATA.FILTER_BANK:
        parameter = signal.butter(N=5, Wn=bank, btype='bandpass', fs=config.DATA.FS)
        EEG_filtered = signal.lfilter(parameter[0], parameter[1], EEG)[..., config.DATA.REF_DUR * config.DATA.FS::config.DATA.RESAMPLE]
        EEG_bank.append(EEG_filtered)
    EEG = np.concatenate(EEG_bank, axis=1).astype(np.float32) #N,F*C,T

    EEG = (EEG - config.DATA.MEAN) / config.DATA.STD
    print(f'preprocessing data: {EEG.shape} {labels.shape}')
    print(np.max(EEG))
    return EEG[::-1], labels[::-1]


def merge_data_files(config, data_files): # for all given files
    subject_data, subject_labels = [], []
    for file in data_files:
        EEG_data, labels = preprocess(file, config, stage='train')  # (N, F*C, T), (N,)
        subject_data.append(EEG_data)
        subject_labels.append(labels)
    return subject_data, subject_labels


def k_fold_generator(config, data_files): #stratified block-wise CV
    subject = data_files[0].split('/')[-2]
    subject_data, subject_labels = merge_data_files(config, data_files)

    _, C, T = subject_data[0].shape
    cls_types = np.arange(config.MODEL.NUM_CLASSES)
    data_temp = np.zeros((0, C, T)).astype(np.float32)
    labels_temp = np.zeros((0,)).astype(np.int32)

    subject_cls_data, subject_cls_labels = [], []
    for data, labels in zip(subject_data, subject_labels):
        cls_data, cls_labels = [], []    # class-wise data and labels
        for cls in cls_types:
            idx = labels == cls
            cls_data.append(data[idx, ...])
            cls_labels.append(labels[idx])
            if not config.DATA.BLOCK:
                n = np.sum(idx)
                idx = np.random.choice(n, n, replace=False)
                cls_data[-1] = cls_data[-1][idx]
                cls_labels[-1] = cls_labels[-1][idx]
        subject_cls_data.append(cls_data)
        subject_cls_labels.append(cls_labels)

    k_fold = config.DATA.K_FOLD
    fold_step = config.DATA.FOLD_STEP
    for i in range(k_fold):
        for ii in range(int(1 // fold_step)):
            print(f'------{subject} data sampler------')
            k_train_x = []
            k_val_x = []
            k_train_y = []
            k_val_y = []

            for cls_data, cls_labels in zip(subject_cls_data, subject_cls_labels):
                for c_data, c_labels in zip(cls_data, cls_labels):
                    num_trials = c_labels.shape[0]
                    if not num_trials:
                        continue
                    fold_len = math.ceil(num_trials / k_fold)
                    fold_steps = int(fold_len * fold_step)
                    start = i * fold_len + ii * fold_steps
                    end = start + fold_len

                    if end > num_trials:
                        val_index = list(range(start, num_trials))
                        val_index.extend(range(0, end - num_trials))
                    else:
                        val_index = range(start, end)

                    k_val_x.append(c_data[val_index, ...])
                    k_val_y.append(c_labels[val_index])
                    k_train_x.append(np.delete(c_data, val_index, axis=0))
                    k_train_y.append(np.delete(c_labels, val_index, axis=0))

            yield np.concatenate(k_train_x, axis=0), np.concatenate(k_train_y, axis=0), \
                  np.concatenate(k_val_x, axis=0), np.concatenate(k_val_y, axis=0)