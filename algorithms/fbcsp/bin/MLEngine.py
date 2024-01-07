# --------------------------------------------------------
# Probabilistic FBCSP customized for online implementation
# Modified from https://github.com/fbcsptoolbox/fbcsp_code
# --------------------------------------------------------
import numpy as np
import scipy.signal as signal
from scipy.signal import cheb2ord
from .FBCSP import FBCSP
from .Classifier import Classifier, FeatureSelect
from . import LoadData, Preprocess
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import LabelEncoder
from .tools import make_interpolation_matrix
from scipy import io
import pickle


class MLEngine:
    def __init__(self,data_path='', train_files=[], test_files = [], subject_id='',sessions=[1, 2],ntimes=1,kfold=5,m_filters=2,window_details={}, ref=1, fs=250, channels=(), **kwargs):
        self.data_path = data_path
        self.subject_id=subject_id
        self.train_files = train_files
        self.test_files = test_files
        self.sessions = sessions
        self.kfold = kfold
        self.ntimes=ntimes
        self.window_details = window_details
        self.ref = ref
        self.m_filters = m_filters
        self.fs = fs
        self.channels = channels
        self.kwargs = kwargs

    def experiment(self):
        fbank = FilterBank(self.fs, self.ref)
        fbank_coeff = fbank.get_filter_coeff()

        x_train, y_train = self.preprocess(self.train_files, fbank, self.kwargs)

        '''for N times x K fold CV'''
        # train_indices, test_indices = self.cross_validate_Ntimes_Kfold(y_labels,ifold=k)
        '''for K fold CV by sequential splitting'''
        # train_indices, test_indices = self.cross_validate_sequential_split(y_labels)
        '''for two fold CV in half half split'''
        #train_indices, test_indices = self.cross_validate_half_split(y_labels)

        best_val_acc = 0
        best_c = 1

        # CV process
        C = np.logspace(-5, 5, 11, base=2)
        for c in C:
            val_acc = []
            for i in range(self.kfold):
                train_indices, test_indices = self.cross_validate_sequential_split(y_train)
                train_idx = train_indices.get(i)
                test_idx = test_indices.get(i)
                print(f'Fold {str(i)}\n')
                y_train_k, y_test_k = self.split_ydata(y_train, train_idx, test_idx)
                x_train_k, x_test_k = self.split_xdata(x_train, train_idx, test_idx)

                y_classes_unique = np.unique(y_train_k)
                n_classes = len(np.unique(y_train_k))

                fbcsp = FBCSP(self.m_filters)
                fbcsp.fit(x_train_k, y_train_k)

                x_features_train = []
                x_features_test = []
                for j in range(n_classes):
                    cls_of_interest = y_classes_unique[j]
                    select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]

                    x_train_features_fb = fbcsp.transform(x_train_k, class_idx=cls_of_interest)
                    x_test_features_fb = fbcsp.transform(x_test_k, class_idx=cls_of_interest)
                    y_train_cls = np.asarray(select_class_labels(cls_of_interest, y_train_k))

                    feature_selection = FeatureSelect()
                    feature_selection.fit(x_train_features_fb, y_train_cls)

                    x_features_train.append(feature_selection.transform(x_train_features_fb))
                    x_features_test.append(feature_selection.transform(x_test_features_fb))
                x_features_train = np.concatenate(x_features_train, axis=1)
                x_features_test = np.concatenate(x_features_test, axis=1)

                classifier = SVC(C=c, kernel='rbf', probability=True, decision_function_shape='ovo')
                classifier.fit(x_features_train, y_train_k)

                y_train_predicted_multi = np.argmax(classifier.predict_proba(x_features_train), axis=1)
                y_test_predicted_multi = np.argmax(classifier.predict_proba(x_features_test), axis=1)

                tr_acc = np.sum(y_train_predicted_multi == y_train_k, dtype=np.float) / len(y_train_k)
                te_acc = np.sum(y_test_predicted_multi == y_test_k, dtype=np.float) / len(y_test_k)
                val_acc.append(te_acc)

            val_acc = np.mean(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_c = c

        # retrain the model using the whole training set with the best $c$
        self.models = []
        feature_selection = []

        fbcsp = FBCSP(self.m_filters)
        fbcsp.fit(x_train, y_train)

        y_classes_unique = np.unique(y_train)
        n_classes = len(np.unique(y_train))
        for j in range(n_classes):
            cls_of_interest = y_classes_unique[j]
            select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]
            x_train_features_fb = fbcsp.transform(x_train, class_idx=cls_of_interest)
            y_train_cls = np.asarray(select_class_labels(cls_of_interest, y_train))
            feature_selection.append(FeatureSelect())
            feature_selection[-1].fit(x_train_features_fb, y_train_cls)

        x_test, y_test = self.preprocess(self.test_files, fbank, self.kwargs)

        x_features_train = []
        x_features_test = []
        for j in range(n_classes):
            cls_of_interest = y_classes_unique[j]
            x_train_features_fb = fbcsp.transform(x_train, class_idx=cls_of_interest)
            x_test_features_fb = fbcsp.transform(x_test, class_idx=cls_of_interest)
            x_features_train.append(feature_selection[j].transform(x_train_features_fb))
            x_features_test.append(feature_selection[j].transform(x_test_features_fb))
        x_features_train = np.concatenate(x_features_train, axis=1)
        x_features_test = np.concatenate(x_features_test, axis=1)

        classifier = SVC(C=best_c, kernel='rbf', probability=True, decision_function_shape='ovo')
        classifier.fit(x_features_train, y_train)

        y_train_predicted_multi = np.argmax(classifier.predict_proba(x_features_train), axis=1)
        y_test_predicted_multi = np.argmax(classifier.predict_proba(x_features_test), axis=1)

        tr_acc = np.sum(y_train_predicted_multi == y_train, dtype=np.float) / len(y_train)
        te_acc = np.sum(y_test_predicted_multi == y_test, dtype=np.float) / len(y_test)

        print(f'C:{best_c}  val set acc: {best_val_acc:.4f}  test set acc: {te_acc:.4f}\n')

        model = {'window_details': self.window_details, 'fbank': fbank, 'FBCSP': fbcsp,
                 'feature_selection': feature_selection, 'clf': classifier}
        self.models.append(model)

        subject = self.data_path.split('/')[-2]
        pickle.dump(self.models, open(f'{self.data_path}/{subject}_{self.kwargs["exp_tag"]}.pkl', 'wb'))

        return best_val_acc, te_acc

    def preprocess(self, files, fbank, kwargs):
        data_total, labels_total = [], []
        for file in files:
            data = io.loadmat(self.data_path + file)
            EEG = data['EEG_data'].astype(np.float64)
            EEG = EEG.transpose((2, 0, 1))
            if kwargs['bad_channels']:
                interpolation_matrix = make_interpolation_matrix(kwargs['channels_names'], kwargs['channels_loc_file'],
                                                                self.fs, kwargs['bad_channels'])
                EEG[:, kwargs['bad_channels'], :] = np.matmul(interpolation_matrix,
                                                             np.delete(EEG,kwargs['bad_channels'], axis=1))
            if self.channels:
                EEG = EEG[:, self.channels, :]

            labels = data['labels'].reshape(-1).astype(np.int32)  # N
            labels = LabelEncoder().fit_transform(labels)
            data_total.append(EEG)
            labels_total.append(labels)

        eeg_data = {'x_data': np.concatenate(data_total, axis=0),
                    'y_labels': np.concatenate(labels_total, axis=0),
                    'fs': self.fs}

        print(eeg_data['x_data'].shape)
        print(eeg_data['y_labels'].shape)
        # idx = np.random.choice(range(eeg_data['y_labels'].shape[0]), eeg_data['y_labels'].shape[0], replace=False)
        # eeg_data['x_data'] = eeg_data['x_data'][idx]
        # eeg_data['y_labels'] = eeg_data['y_labels'][idx]

        filtered_data = fbank.filter_data(eeg_data.get('x_data'), self.window_details)  # (F, N, C, T)
        y_labels = eeg_data.get('y_labels')  # (N,)
        return filtered_data, y_labels

    def cross_validate_Ntimes_Kfold(self, y_labels, ifold=0):
        from sklearn.model_selection import StratifiedKFold
        train_indices = {}
        test_indices = {}
        random_seed = ifold
        skf_model = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=random_seed)
        i = 0
        for train_idx, test_idx in skf_model.split(np.zeros(len(y_labels)), y_labels):
            train_indices.update({i: train_idx})
            test_indices.update({i: test_idx})
            i += 1
        return train_indices, test_indices

    def cross_validate_sequential_split(self, y_labels):
        from sklearn.model_selection import StratifiedKFold
        train_indices = {}
        test_indices = {}
        skf_model = StratifiedKFold(n_splits=self.kfold, shuffle=False)
        i = 0
        for train_idx, test_idx in skf_model.split(np.zeros(len(y_labels)), y_labels):
            train_indices.update({i: train_idx})
            test_indices.update({i: test_idx})
            i += 1
        return train_indices, test_indices

    def cross_validate_half_split(self, y_labels):
        import math
        unique_classes = np.unique(y_labels)
        all_labels = np.arange(len(y_labels))
        train_idx =np.array([])
        test_idx = np.array([])
        for cls in unique_classes:
            cls_indx = all_labels[np.where(y_labels==cls)]
            if len(train_idx)==0:
                train_idx = cls_indx[:math.ceil(len(cls_indx)/2)]
                test_idx = cls_indx[math.ceil(len(cls_indx)/2):]
            else:
                train_idx=np.append(train_idx,cls_indx[:math.ceil(len(cls_indx)/2)])
                test_idx=np.append(test_idx,cls_indx[math.ceil(len(cls_indx)/2):])

        train_indices = {0:train_idx}
        test_indices = {0:test_idx}

        return train_indices, test_indices

    def split_xdata(self,eeg_data, train_idx, test_idx):
        x_train_fb=np.copy(eeg_data[:,train_idx,:,:])
        x_test_fb=np.copy(eeg_data[:,test_idx,:,:])
        return x_train_fb, x_test_fb

    def split_ydata(self,y_true, train_idx, test_idx):
        y_train = np.copy(y_true[train_idx])
        y_test = np.copy(y_true[test_idx])

        return y_train, y_test

    def get_multi_class_label(self,y_predicted, cls_interest=0):
        y_predict_multi = np.zeros((y_predicted.shape[0]))
        for i in range(y_predicted.shape[0]):
            y_lab = y_predicted[i, :]
            lab_pos = np.where(y_lab == cls_interest)[0]
            if len(lab_pos) == 1:
                y_predict_multi[i] = lab_pos
            elif len(lab_pos > 1):
                y_predict_multi[i] = lab_pos[0]
        return y_predict_multi

    def get_multi_class_regressed(self, y_predicted):
        y_predict_multi = np.asarray([np.argmax(y_predicted[i,:]) for i in range(y_predicted.shape[0])])
        return y_predict_multi


class FilterBank:
    def __init__(self, fs, ref):
        self.fs = fs
        self.ref = ref
        self.f_trans = 2
        self.f_pass = np.arange(4,40,4)
        self.f_width = 4
        self.gpass = 3
        self.gstop = 30
        self.filter_coeff={}

    def get_filter_coeff(self):
        Nyquist_freq = self.fs/2

        for i, f_low_pass in enumerate(self.f_pass):
            f_pass = np.asarray([f_low_pass, f_low_pass+self.f_width])
            f_stop = np.asarray([f_pass[0]-self.f_trans, f_pass[1]+self.f_trans])
            wp = f_pass/Nyquist_freq
            ws = f_stop/Nyquist_freq
            order, wn = cheb2ord(wp, ws, self.gpass, self.gstop)
            b, a = signal.cheby2(order, self.gstop, ws, btype='bandpass')
            self.filter_coeff.update({i:{'b':b,'a':a}})

        return self.filter_coeff

    def filter_data(self,eeg_data,window_details={}):
        n_trials, n_channels, n_samples = eeg_data.shape
        if window_details:
            n_samples = int(self.fs*(window_details.get('tmax')-window_details.get('tmin')))+0
        filtered_data=np.zeros((len(self.filter_coeff),n_trials,n_channels,n_samples))
        for i, fb in self.filter_coeff.items():
            b = fb.get('b')
            a = fb.get('a')
            eeg_data_filtered = np.asarray([signal.lfilter(b,a,eeg_data[j,:,:]) for j in range(n_trials)])
            if window_details:
                eeg_data_filtered = eeg_data_filtered[:,:,int((self.ref+window_details.get('tmin'))*self.fs):int((self.ref+window_details.get('tmax'))*self.fs)+0]
            filtered_data[i,:,:,:] = eeg_data_filtered

        return filtered_data

