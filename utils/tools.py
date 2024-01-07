# --------------------------------------------------------
# Package protocol for network communication between the client and server
# Written by Jiaheng Wang
# --------------------------------------------------------
import struct
import json
import numpy as np


class Pack:
    '''
            4字节 特殊字符0x618
            4字节 包体长度
            4字节 CMD
            N字节 包体（json格式）
    '''
    magic_num = 0x618
    head_len = 12
    def pack(self, message: dict):
        message = json.dumps(message).encode()
        cmd = 1
        size = len(message)
        package = struct.pack(f'!3i{size}s', self.magic_num, size, cmd, message)
        return package


class Depack:
    '''
            4字节 特殊字符0x618
            4字节 包体长度
            4字节 CMD
            N字节 包体（json格式）
    '''
    magic_num = 0x618
    head_len = 12
    def __init__(self):
        self.buff = b''

    def depack(self, package: bytes):
        ls = []
        self.buff += package
        while True:
            if len(self.buff) >= self.head_len:
                magic, size, cmd = struct.unpack('!3i', self.buff[:self.head_len])
                assert magic == self.magic_num
                assert cmd == 1, 'only support json format'
                if len(package) >= self.head_len + size:
                    message = struct.unpack(f'!{size}s', self.buff[self.head_len: self.head_len + size])[0]
                    message = json.loads(message)
                    ls.append(message)
                    self.buff = self.buff[self.head_len + size:]
                else:
                    break
            else:
                break
        return ls


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