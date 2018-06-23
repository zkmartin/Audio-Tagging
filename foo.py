import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import librosa
from tframe.data.dataset import DataSet
from at.gpat_data import Gpat_set
from tframe.data.sequences.signals.signal_set import SignalSet
from at.data_utils import load_simulate_test_data

# csv_path = './data/original_data/train.csv'
# from sklearn.cross_validation import StratifiedKFold
# # from sklearn.model_selection import StratifiedKFold
#
# train = pd.read_csv(csv_path)
# LABELS = list(train.label.unique())
# label_idx = {label: i for i, label in enumerate(LABELS)}
# train.set_index("fname", inplace=True)
# train["label_idx"] = train.label.apply(lambda x: label_idx[x])
# # split the train_set and the val_set
# skf = StratifiedKFold(train.label_idx, n_folds=10)
# for (train_split, val_split) in skf:
# 	val_labels = train.label_idx[val_split]
# 	# for i in range(41):
# 	# 	print(len(val_labels[val_labels == i]))
# 	a = 1


# file_path = 'data/original_data/audio_test/0b0427e2.wav'
# data, _ = librosa.core.load(file_path, sr=16000)
# a = 1


# file_path = 'data/original_data/testdata_fs_16000/0b0427e2.tfds'
# file_path = 'data/original_data/testdata_fs_16000/ffeba2cd.tfds'
# data = SignalSet.load(file_path)
# a = 1

#
# data_fft = librosa.feature.spectral_centroid(data, sr=16000, n_fft=2048)
# a = 1

# with tf.device('/gpu:0'):
# 	a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# 	b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# with tf.device('/gpu:1'):
# 	c = tf.matmul(a, b)
# 	# Creates a session with log_device_placement set to True.
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
# # Runs the op.
# 	print(sess.run(c))

a = []
for i in a:
	assert isinstance(i, list)
	print(i)
