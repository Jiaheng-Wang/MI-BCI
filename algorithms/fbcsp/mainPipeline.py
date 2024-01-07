# --------------------------------------------------------
# Probabilistic FBCSP customized for online implementation
# Reference: https://github.com/fbcsptoolbox/fbcsp_code
# Written by Jiaheng Wang
# --------------------------------------------------------
from bin.MLEngine import MLEngine
import os
import csv


if __name__ == "__main__":
    configs = {
        'data_path' : '', #'~/data path for one subject/'
        'train_files': ['s1_calibration.mat',],
        'test_files': ['s1_feedback.mat'], # does nothing in online experiment
        'exp_tag' : 'FBCSP_s1_calibration_2s', # file name for saved models and results
        'kfold': 5,
        'ntimes': 1,
        'm_filters':2,
        'window_details': {'tmin':0.5, 'tmax':2.5},
        'ref': 1, # baseline duration before the cue onset
        'fs': 256,
        'channels':[c-1 for c in [
         9, 10, 11, 12, 13, 14, 15,
         18, 19, 20, 21, 22, 23, 24,
         27, 28, 29, 30, 31, 32, 33,
         36, 37, 38, 39, 40, 41, 42,
         45, 46, 47, 48, 49, 50, 51,
        ]], # for the 62-channel g.tec headset
        'bad_channels': [], # interpolated by spherical splines
        'channels_names': ["FP1", "FPZ", "FP2", "AF7", "AF3", "AF4", "AF8", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6",
                           "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4",
                           "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2",
                           "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "F9", "F10"],
        'channels_loc_file': '../../utils/getec62.sfp',
    }

    dataset_path = r'~\dataset dir/'
    val_acc_subjects, test_acc_subjects = [], []
    subjects = []
    for dir in os.listdir(dataset_path):
        print(f'Training on subject {dir}.\n')
        configs['data_path'] = dataset_path + dir + '/'

        ML_experiment = MLEngine(**configs)
        val_acc, test_acc = ML_experiment.experiment()

        val_acc_subjects.append(val_acc)
        test_acc_subjects.append(test_acc)
        subjects.append(dir)

    with open(f'{configs["exp_tag"]}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subject', 'val_acc', 'test_acc'])
        for subject, val_acc, test_acc in zip(subjects, val_acc_subjects, test_acc_subjects):
            writer.writerow([subject, f'{val_acc:.4f}', f'{test_acc:.4f}'])
        val_acc_avg  = sum(val_acc_subjects) / len(val_acc_subjects)
        test_acc_avg = sum(test_acc_subjects) / len(test_acc_subjects)
        writer.writerow(['Avg', f'{val_acc_avg:.4f}', f'{test_acc_avg:.4f}'])
    print(f"Subjects' val set average acc: {val_acc_avg:.4f}")
    print(f"Subjects' test set average acc: {test_acc_avg:.4f}")