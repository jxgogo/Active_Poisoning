import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import Callback


class TestCallback(Callback):
    def __init__(self, val_data, test_data, es):
        super().__init__()
        self.val_data = val_data
        self.test_data = test_data
        self.loss_hist = []
        self.acc_hist = []
        self.val_acc_hist = []
        self.es = es

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        val_loss, val_acc = self.model.evaluate(self.val_data[0], self.val_data[1], verbose=0)
        self.loss_hist.append(loss)
        self.acc_hist.append(acc)
        self.val_acc_hist.append(val_acc)
        print('\nTesting loss: {:.4f}, acc: {:.4f}, best val acc: {:.4f}, best acc: {:.4f}\n'.format(
            loss, acc, max(self.val_acc_hist), self.acc_hist[np.argmax(self.val_acc_hist).item()])
        )


class TestCallbackv1(Callback):
    def __init__(self, test_data):
        super().__init__()
        self.test_data = test_data
        self.loss_hist = []
        self.acc_hist = []

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        self.loss_hist.append(loss)
        self.acc_hist.append(acc)
        print('Epoch: {}, Testing loss: {:.4f}, acc: {:.4f}, best test acc: {:.4f}, on epoch: {}'.format(
            epoch, loss, acc, max(self.acc_hist), np.argmax(self.acc_hist).item())
        )


def standard_normalize(x, clip_range=None):
    x = (x - np.mean(x)) / np.std(x)
    if clip_range is not None:
        x = np.clip(x, a_min=clip_range[0], a_max=clip_range[1])
    return x


def split_data(data, split=0.8, shuffle=True):
    x = data[0]
    y = data[1]
    data_size = len(x)
    split_index = int(data_size * split)
    indices = np.arange(data_size)
    if shuffle:
        indices = np.random.permutation(indices)
    x_train = x[indices[:split_index]]
    y_train = y[indices[:split_index]]
    x_test = x[indices[split_index:]]
    y_test = y[indices[split_index:]]
    return x_train, y_train, x_test, y_test


def split_data_both(data, split=0.8, shuffle=True):
    x = data[0]
    x_poison = data[1]
    y = data[2]
    s = data[3]
    data_size = len(x)
    split_index = int(data_size * split)
    indices = np.arange(data_size)
    if shuffle:
        indices = np.random.permutation(indices)
    x_train = x[indices[:split_index]]
    x_train_poison = x_poison[indices[:split_index]]
    y_train = y[indices[:split_index]]
    s_train = s[indices[:split_index]]
    x_test = x[indices[split_index:]]
    y_test = y[indices[split_index:]]
    return x_train, x_train_poison, y_train, s_train, x_test, y_test


def get_attack_data(data, num=100, shuffle=True):
    x = data[0]
    y = data[1]
    data_size = len(x)
    indices = np.arange(data_size)
    if shuffle:
        indices = np.random.permutation(indices)
    x_test = x[indices[:num]]
    y_test = y[indices[:num]]
    x_train = x[indices[num:]]
    y_train = y[indices[num:]]
    return x_train, y_train, x_test, y_test


def shuffle_data(data_size, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    indices = np.arange(data_size)
    return np.random.permutation(indices)


def get_shuffle_indices(data_size):
    return np.random.permutation(np.arange(data_size))


def bca(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred)
    numb = m.shape[0]
    acc_each_label = 0
    for i in range(numb):
        acc = m[i, i] / np.sum(m[i, :], keepdims=False).astype(np.float32)
        acc_each_label += acc
    return acc_each_label / numb


def batch_iter(data, batchsize, shuffle=True):
    data = np.array(list(data))
    data_size = data.shape[0]
    num_batches = int((data_size - 1) / batchsize) + 1
    # Shuffle the data
    if shuffle:
        shuffle_indices = get_shuffle_indices(data_size)
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches):
        start_index = batch_num * batchsize
        end_index = min((batch_num + 1) * batchsize, data_size)
        yield shuffled_data[start_index:end_index]


def get_split_indices(data_size, split=[9, 1], shuffle=True):
    if len(split) < 2:
        raise TypeError(
            'The length of split should be larger than 2 while the length of your split is {}!'.format(len(split)))
    split = np.array(split)
    split = split / np.sum(split)
    if shuffle:
        indices = get_shuffle_indices(data_size)
    else:
        indices = np.arange(data_size)
    split_indices_list = []
    start = 0
    for i in range(len(split) - 1):
        end = start + int(np.floor(split[i] * data_size))
        split_indices_list.append(indices[start:end])
        start = end
    split_indices_list.append(indices[start:])
    return split_indices_list
