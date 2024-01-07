import torch
import numpy as np


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, *args):
        for t in self.transforms:
            x = t(x, *args)
        return x


def make_interpolation_matrix(channels_names, fname, fs, bad_channels):
    import mne
    from mne.channels import montage
    from mne.bem import _check_origin
    from mne.channels.interpolation import _interpolate_bads_eeg, _make_interpolation_matrix
    from mne.utils import warn, logger

    channels_info = mne.create_info(ch_names=channels_names, sfreq=fs, ch_types='eeg')
    channels_info.set_montage(montage.read_custom_montage(fname=fname))

    origin = _check_origin('auto', channels_info)

    chs = channels_info['chs']
    pos = np.array([chs[k]['loc'][:3] for k in range(len(chs))])

    n_zero = np.sum(np.sum(np.abs(pos), axis=1) == 0)
    if n_zero > 1:  # XXX some systems have origin (0, 0, 0)
        raise ValueError('Could not extract channel positions for '
                         '{} channels'.format(n_zero))

    # test spherical fit
    distance = np.linalg.norm(pos - origin, axis=-1)
    distance = np.mean(distance / np.mean(distance))
    if np.abs(1. - distance) > 0.1:
        warn('Your spherical fit is poor, interpolation results are '
             'likely to be inaccurate.')

    good_channels = list(range(len(chs)))
    for bad in bad_channels:
        good_channels.remove(bad)
    pos_good = pos[good_channels] - origin
    pos_bad = pos[bad_channels] - origin
    logger.info('Computing interpolation matrix from {} sensor '
                'positions'.format(len(pos_good)))
    interpolation = _make_interpolation_matrix(pos_good, pos_bad)
    logger.info('Interpolating {} sensors'.format(len(pos_bad)))
    return interpolation