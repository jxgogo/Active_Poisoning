import numpy as np
import lib.utils as utils
from scipy.io import loadmat


def load(data_name, s_id, downsample=True):
    """ load ERN data """
    path = '../dataset/ActivePoisoningData/' + data_name + '/clean'

    data = loadmat(path + '/s{}.mat'.format(s_id))
    eeg = data['eeg']
    x = data['x']
    y = data['y']
    y = np.squeeze(y.flatten())
    s = s_id * np.ones(shape=[len(y)])
    # downsample
    if downsample:
        x1 = x[np.where(y == 0)]
        x2 = x[np.where(y == 1)]
        sample_num = min(len(x1), len(x2))
        idx1, idx2 = utils.shuffle_data(len(x1)), utils.shuffle_data(len(x2))
        x = np.concatenate([x1[idx1[:sample_num]], x2[idx2[:sample_num]]], axis=0)
        y = np.concatenate([np.zeros(shape=[sample_num]), np.ones(shape=[sample_num])], axis=0)
        s = s_id * np.ones(shape=[len(y)])

    return eeg, x, y, s

