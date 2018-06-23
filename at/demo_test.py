import numpy as np

np.random.seed(1001)

import os
import shutil

import pandas as pd
from sklearn.cross_validation import StratifiedKFold

import librosa
import numpy as np
import scipy
from tensorflow.python.keras.utils import Sequence, to_categorical
from tframe.data.dataset import DataSet

COMPLETE_RUN = True
train = pd.read_csv("../data/original_data/train.csv")
test = pd.read_csv("../data/original_data/sample_submission.csv")


class Config(object):
	def __init__(self,
	             sampling_rate=16000, audio_duration=2, n_classes=41,
	             use_mfcc=False, n_folds=10, learning_rate=0.0001,
	             max_epochs=50, n_mfcc=20):
		self.sampling_rate = sampling_rate
		self.audio_duration = audio_duration
		self.n_classes = n_classes
		self.use_mfcc = use_mfcc
		self.n_mfcc = n_mfcc
		self.n_folds = n_folds
		self.learning_rate = learning_rate
		self.max_epochs = max_epochs
		
		self.audio_length = self.sampling_rate * self.audio_duration
		if self.use_mfcc:
			self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length / 512)), 1)
		else:
			self.dim = (self.audio_length, 1)


class DataGenerator(Sequence):
	def __init__(self, config, data_dir, list_IDs, labels=None,
	             batch_size=64, preprocessing_fn=lambda x: x):
		self.config = config
		self.data_dir = data_dir
		self.list_IDs = list_IDs
		self.labels = labels
		self.batch_size = batch_size
		self.preprocessing_fn = preprocessing_fn
		self.on_epoch_end()
		self.dim = self.config.dim
	
	def __len__(self):
		return int(np.ceil(len(self.list_IDs) / self.batch_size))
	
	def __getitem__(self, index):
		indexes = self.indexes[
		          index * self.batch_size:(index + 1) * self.batch_size]
		list_IDs_temp = [self.list_IDs[k] for k in indexes]
		return self.__data_generation(list_IDs_temp)
	
	def on_epoch_end(self):
		self.indexes = np.arange(len(self.list_IDs))
	
	def __data_generation(self, list_IDs_temp):
		cur_batch_size = len(list_IDs_temp)
		X = np.empty((cur_batch_size, *self.dim))
		
		input_length = self.config.audio_length
		for i, ID in enumerate(list_IDs_temp):
			file_path = self.data_dir + ID
			
			# Read and Resample the audio
			data, _ = librosa.core.load(file_path, sr=self.config.sampling_rate,
			                            res_type='kaiser_fast')
			
			# Random offset / Padding
			if len(data) > input_length:
				max_offset = len(data) - input_length
				offset = np.random.randint(max_offset)
				data = data[offset:(input_length + offset)]
			else:
				if input_length > len(data):
					max_offset = input_length - len(data)
					offset = np.random.randint(max_offset)
				else:
					offset = 0
				data = np.pad(data, (offset, input_length - len(data) - offset),
				              "constant")
			
			# Normalization + Other Preprocessing
			if self.config.use_mfcc:
				data = librosa.feature.mfcc(data, sr=self.config.sampling_rate,
				                            n_mfcc=self.config.n_mfcc)
				data = np.expand_dims(data, axis=-1)
			else:
				data = self.preprocessing_fn(data)[:, np.newaxis]
			X[i,] = data
		
		if self.labels is not None:
			y = np.empty(cur_batch_size, dtype=int)
			for i, ID in enumerate(list_IDs_temp):
				y[i] = self.labels[ID]
			return X, to_categorical(y, num_classes=self.config.n_classes)
		else:
			return X


def audio_norm(data):
	max_data = np.max(data)
	min_data = np.min(data)
	data = (data - min_data) / (max_data - min_data + 1e-6)
	return data - 0.5

LABELS = list(train.label.unique())
label_idx = {label: i for i, label in enumerate(LABELS)}
train.set_index("fname", inplace=True)
test.set_index("fname", inplace=True)
train["label_idx"] = train.label.apply(lambda x: label_idx[x])
if not COMPLETE_RUN:
	train = train[:2000]
	test = test[:2000]

config = Config(sampling_rate=16000, audio_duration=2, n_folds=10,
                learning_rate=0.001)
if not COMPLETE_RUN:
	config = Config(sampling_rate=100, audio_duration=1, n_folds=2, max_epochs=1)

data_generator = DataGenerator(config=config,
                               data_dir='../data/original_data/audio_train/',
                               list_IDs=train.index, labels=train["label_idx"])
batches = len(train.index) // 64
for i in range(batches):
	feature, target = data_generator[i]
	if i == 0:
		features = feature
		targets = target
	else:
		features = np.concatenate((features, feature), axis=0)
		targets = np.concatenate((targets, target), axis=0)
		
demo_data = DataSet(features=features, targets=targets)
demo_data.save('../data/processed_data/demo_data_0')
a = data_generator[2]
b = a[0]
c =1
for i in range(len(val_set.features)):
	if i == 0:
		features = GPAT.length_adapted(val_set.features[i],
		                               audio_length)
		features = np.reshape(features, (1, -1))
	# targets = batch_data[i].targets
	else:
		feature = GPAT.length_adapted(val_set.features[i],
		                              audio_length)
		feature = np.reshape(feature, (1, -1))
		features = np.concatenate((features, feature), axis=0)
# targets = np.concatenate((targets, batch_data[i].targets), axis=0)
targets = val_set.targets
features = np.expand_dims(features, axis=2)

