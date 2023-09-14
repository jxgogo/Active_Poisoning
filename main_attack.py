import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

import lib.utils as utils
from lib import ALstrategies
from lib.EA import align
from lib.load_data import load
from lib import models
from lib.methods import pulse_noise
import random

random_seed = 2106
os.environ['PYTHONHASHSEED'] = str(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

K.set_image_data_format('channels_first')

parser = argparse.ArgumentParser()
parser.add_argument('--p_rate', type=float, default=0.05, help='poison rate')
parser.add_argument('--gpu_n', type=int, default=0, help='name of GPU')
parser.add_argument('--data', type=str, default='ERN')
parser.add_argument('--model', type=str, default='ShallowCNN', choices=['EEGNet', 'DeepCNN', 'ShallowCNN'])
parser.add_argument('--TL', type=str, default='EA')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_n)
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.25
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

poisoning_rate = opt.p_rate
data_name = opt.data
model_used = opt.model
TL_method = opt.TL
subject_numbers = {'ERN': 16}
amplitudes = {'ERN': 1.0}
subject_number = subject_numbers[data_name]

npp_params = [amplitudes[data_name], 1, 0.2]
AL_methods = ['Random', 'MUS', 'MMCS', 'MDS', 'RDS', 'iRDM', 'MUS+MDS', 'MMCS+MDS']

save_dir = 'runs/attack_performance_{}/{}'.format(str(poisoning_rate), TL_method)
batch_size = 64
repeat = 5
epoches = 500
patience = 30

AL_raccs, AL_rbcas, AL_rasrs = [], [], []
# poisoning attack in cross-subject
for r in range(repeat):
    AL_accs, AL_bcas, AL_asrs = [], [], []
    s_id = np.arange(subject_number)
    results_dir = os.path.join(save_dir, data_name, model_used, 'run{}'.format(r))
    checkpoint_path = os.path.join(save_dir, data_name, model_used, 'run{}'.format(r), 'checkpoint')
    model_path = os.path.join(checkpoint_path, 'model.h5')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    for s in range(subject_number):
        # Load dataset
        train_idx = [x for x in range(subject_number)]
        train_idx.remove(s_id[s])
        _, x_train, y_train, s_train = load(data_name, train_idx[0], downsample=True)
        for i in train_idx[1:]:
            _, x_i, y_i, s_i = load(data_name, i, downsample=True)
            x_train = np.concatenate((x_train, x_i), axis=0)
            y_train = np.concatenate((y_train, y_i), axis=0)
            s_train = np.concatenate((s_train, s_i), axis=0)
        _, x_test, y_test, _ = load(data_name, s_id[s], downsample=False)

        # create NPP
        std = np.mean(np.std(x_train, axis=3))
        pulse = pulse_noise(x_train.shape[1:], freq=npp_params[1], sample_freq=128, proportion=npp_params[2])
        x_test_poison = npp_params[0]*std*pulse + x_test

        # build model
        nb_classes = len(np.unique(y_train))
        samples = x_train.shape[3]
        channels = x_train.shape[2]

        if model_used == 'EEGNet':
            model = models.EEGNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
            model_AL = models.EEGNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
        elif model_used == 'DeepCNN':
            model = models.DeepConvNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
            model_AL = models.DeepConvNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
        elif model_used == 'ShallowCNN':
            model = models.ShallowConvNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
            model_AL = models.ShallowConvNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
        else:
            raise Exception('No such model:{}'.format(model_used))

        # train unpoisoned model for AL
        data_size = y_train.shape[0]
        shuffle_index = utils.shuffle_data(data_size)
        x_train = x_train[shuffle_index]
        y_train = y_train[shuffle_index]
        s_train = s_train[shuffle_index]

        model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['acc', ])
        early_stop = EarlyStopping(monitor='val_acc', mode='max', patience=patience)
        model_checkpoint = ModelCheckpoint(filepath=checkpoint_path + '/{}_model.h5'.format(s), monitor='val_acc', mode='max',
                                           save_best_only=True)

        print('Training unpoisoned model')
        his = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            validation_split=0.2,
            shuffle=True,
            epochs=epoches,
            callbacks=[early_stop, model_checkpoint],
            verbose=0,
        )
        K.clear_session()

        poisoning_number = int(poisoning_rate * len(x_train))
        # select poison data using AL
        accs, bcas, asrs = [], [], []
        for AL_method in AL_methods:
            model = load_model(checkpoint_path + '/{}_model.h5'.format(s))
            if AL_method == 'Random':
                idx = ALstrategies.Random(x_train, y_train, poisoning_number)
            elif AL_method == 'MUS':
                idx = ALstrategies.Entropy(x_train, y_train, poisoning_number, model, active=False)
            elif AL_method == 'MMCS':
                idx = ALstrategies.MaxModelChange(x_train, y_train, poisoning_number, model, active=False)
            elif AL_method == 'MDS':
                idx = ALstrategies.MDS(x_train, y_train, poisoning_number, model)
            elif AL_method == 'RDS':
                idx = ALstrategies.RDS(x_train, y_train, poisoning_number, model)
            elif AL_method == 'MUS+MDS':
                idx = ALstrategies.MUS_MDS(x_train, y_train, poisoning_number, model)
            elif AL_method == 'MMCS+MDS':
                idx = ALstrategies.MMCS_MDS(x_train, y_train, poisoning_number, model)
            else:
                raise Exception('No such AL method:{}'.format(AL_method))

            x_poison, y_poison, s_poison = x_train[idx], y_train[idx], s_train[idx]
            x_poison = npp_params[0]*std*pulse + x_poison
            y_poison = np.ones(shape=y_poison.shape)  # target label = 1

            # add the poison data to train data
            x_train_poisoned = np.concatenate((x_train[np.delete(np.arange(len(x_train)), idx)], x_poison), axis=0)
            y_train_poisoned = np.concatenate((y_train[np.delete(np.arange(len(x_train)), idx)], y_poison), axis=0)
            s_train_poisoned = np.concatenate((s_train[np.delete(np.arange(len(x_train)), idx)], s_poison), axis=0)
            # TL
            if TL_method == 'EA':
                # x_train align
                x_train_aligned, _ = align(x_train_poisoned[np.where(s_train_poisoned == train_idx[0])].squeeze())
                y_train_aligned = y_train_poisoned[np.where(s_train_poisoned == train_idx[0])]
                for i in train_idx[1:]:
                    x_i_aligned, _ = align(x_train_poisoned[np.where(s_train_poisoned == i)].squeeze())
                    x_train_aligned = np.concatenate((x_train_aligned, x_i_aligned), axis=0)
                    y_train_aligned = np.concatenate((y_train_aligned, y_train_poisoned[np.where(s_train_poisoned == i)]), axis=0)
                # x_test align
                x_test_aligned, rf = align(x_test.squeeze())
                # x_test_poison align
                data_align = []
                for i in range(len(x_test_poison)):
                    data_align.append(np.dot(rf, x_test_poison.squeeze()[i]))
                x_test_poison_aligned = np.asarray(data_align).squeeze()

                x_train_aligned = x_train_aligned[:, np.newaxis, :, :]
                x_test_aligned = x_test_aligned[:, np.newaxis, :, :]
                x_test_poison_aligned = x_test_poison_aligned[:, np.newaxis, :, :]
            else:
                raise Exception('No such TL method:{}'.format(TL_method))

            # Train poisoned Model
            data_size = x_train_aligned.shape[0]
            shuffle_index = utils.shuffle_data(data_size)
            x_train_aligned = x_train_aligned[shuffle_index]
            y_train_aligned = y_train_aligned[shuffle_index]

            print('Training poisoned model')
            model_AL.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['acc', ])
            early_stop = EarlyStopping(monitor='val_acc', mode='max', patience=patience)
            model_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_acc', mode='max', save_best_only=True)

            his_AL = model_AL.fit(
                x_train_aligned, y_train_aligned,
                batch_size=batch_size,
                validation_split=0.2,
                shuffle=True,
                epochs=epoches,
                callbacks=[early_stop, model_checkpoint],
                verbose=0,
            )
            model_AL.load_weights(model_path)
            y_pred = np.argmax(model_AL.predict(x_test_aligned), axis=1)
            bca = utils.bca(y_test.squeeze(), y_pred)
            acc = np.sum(y_pred == y_test).astype(np.float32) / len(y_pred)
            accs.append(acc)
            bcas.append(bca)

            # poison performance
            idx = y_pred == y_test
            x_t_poison, y_t = x_test_poison_aligned[idx], y_test[idx]
            # idx = np.where(y_t == 0)
            idx = np.where(y_t != 1)
            x_t_poison, y_t = x_t_poison[idx], y_t[idx]
            if len(x_t_poison) != 0:
                p_pred = np.argmax(model_AL.predict(x_t_poison), axis=1)
                poison_s_rate = np.sum(p_pred == 1).astype(np.float32) / len(p_pred)
                print('{}_{}_{}: acc-{} bca-{} asr-{}'.format(data_name, model_used, AL_method, acc, bca, poison_s_rate))
                asrs.append(poison_s_rate)
            else:
                asrs.append(np.nan)
            K.clear_session()
        AL_accs.append(accs)
        AL_bcas.append(bcas)
        AL_asrs.append(asrs)

        np.savez(results_dir + '/s{}_new.npz'.format(s), accs=accs, bcas=bcas, asrs=asrs, AL_methods=AL_methods)

        print('target{}_allAL: acc-{}'.format(s, accs))
        print('target{}_allAL: bca-{}'.format(s, bcas))
        print('target{}_allAL: asr-{}'.format(s, asrs))
    AL_raccs.append(AL_accs)
    AL_rbcas.append(AL_bcas)
    AL_rasrs.append(AL_asrs)

    np.savez(results_dir + '/result_new.npz', AL_accs=AL_accs, AL_bcas=AL_bcas, AL_asrs=AL_asrs, AL_methods=AL_methods)

    AL_accs, AL_bcas, AL_asrs = np.asarray(AL_accs), np.asarray(AL_bcas), np.asarray(AL_asrs)
    for i in range(len(AL_methods)):
        print('{}_accs: {}'.format(AL_methods[i], AL_accs[:, i]))
        print('{}_bcas: {}'.format(AL_methods[i], AL_bcas[:, i]))
        print('{}_asrs: {}'.format(AL_methods[i], AL_asrs[:, i]))
        print('{}: mean acc={} mean bca={} mean asr={}'.format(AL_methods[i],
                                                               np.mean(AL_accs[:, i], axis=0),
                                                               np.mean(AL_bcas[:, i], axis=0),
                                                               np.nanmean(AL_asrs[:, i], axis=0)))
print('*'*100)
AL_raccs, AL_rbcas, AL_rasrs = np.asarray(AL_raccs), np.asarray(AL_rbcas), np.asarray(AL_rasrs)
for i in range(len(AL_methods)):
    print('{} performance'.format(AL_methods[i]))
    print('raccs:', np.mean(AL_raccs[:,:,i], 1))
    print('rbcas:', np.mean(AL_rbcas[:,:,i], 1))
    print('rpoison_rates:', np.nanmean(AL_rasrs[:,:,i], 1))
    print('Mean RCA={}, mean BCA={}, mean ASR={}'.format(np.mean(AL_raccs[:,:,i]), np.mean(AL_rbcas[:,:,i]), np.nanmean(AL_rasrs[:,:,i])))
