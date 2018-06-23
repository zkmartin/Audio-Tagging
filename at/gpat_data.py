import numpy as np
import librosa
from tframe.data.dataset import DataSet
from tframe.utils import misc
from tframe import checker
from tframe.data.sequences.signals.signal import Signal
from data.gpat import GPAT
from utils import plot_bar_diagram
from tframe.utils import console


class Gpat_set(DataSet):
	
	def __init__(self, features, targets, audio_length=32000, **kwargs):
		super(Gpat_set, self).__init__(features, targets, **kwargs)
		self.audio_length = audio_length
	
	def get_round_length(self, batch_size, num_steps=None):
		"""Get round length for training
		:param batch_size: Batch size. For irregular sequences, this value should
												be set to 1.
		:param num_steps: Step number. If provided, round length will be calculated
											 for RNN model
		:return: Round length for training
		"""
		# Make sure features exist
		self._check_feature()
		checker.check_positive_integer(batch_size, 'batch_size')
		if num_steps is None:
			# :: For feed-forward models
			return int(np.ceil(len(self.targets) / batch_size))
		else:
			# :: For recurrent models
			checker.check_type(num_steps, int)
			arrays = [self.features] if self.is_regular_array else self.features
			if num_steps < 0:
				return len(arrays)
			else:
				return int(sum([np.ceil(len(array) // batch_size / num_steps)
				                for array in arrays]))
	
	def gen_batches(self, batch_size, shuffle=False):
		round_len = self.get_round_length(batch_size)
		for i in range(round_len):
			if shuffle:
				indices = (self._rand_indices(size=batch_size))
			else:
				range(i * batch_size, min((i + 1) * batch_size, len(self.targets)))
			
			batch_features = []
			for indice in indices:
				batch_features.append(self.features[indice])
			for i in range(len(indices)):
				if i == 0:
					features = GPAT.length_adapted(batch_features[i], self.audio_length)
					mfccs = librosa.feature.mfcc(features, 16000, n_mfcc=50)
					mfccs = np.expand_dims(mfccs, axis=0)
					features = GPAT.audio_norm(features)
					features = np.reshape(features, (1, -1))
					# targets = batch_data[i].targets
				else:
					feature = GPAT.length_adapted(batch_features[i], self.audio_length)
					mfcc = librosa.feature.mfcc(feature, 16000, n_mfcc=50)
					mfcc = np.expand_dims(mfcc, axis=0)
					mfccs = np.concatenate((mfccs, mfcc), axis=0)
					feature = GPAT.audio_norm(feature)
					feature = np.reshape(feature, (1, -1))
					features = np.concatenate((features, feature), axis=0)
					# targets = np.concatenate((targets, batch_data[i].targets), axis=0)
			targets = self.targets[indices]
			features = np.expand_dims(features, axis=2)
			mfccs = np.expand_dims(mfccs, axis=-1)
			output_batch_data = DataSet(features, targets, data_dict={'mfcc':mfccs})
			yield output_batch_data
			
	def init_groups(self):
		if self.num_classes is None:return
		groups = []
		dense_labels = misc.convert_to_dense_labels(self.targets)
		for i in range(self.num_classes):
			samples = list(np.argwhere(dense_labels == i).ravel())
			groups.append(samples)
		self.properties[self.GROUPS] = groups
			
	def add_noise(self):
		features = []
		targets = []
		for i in range(len(self.features)):
			signal_energy = self.rms(self.features[i])
			noise = Signal.gaussian_white_noise()
			features.append(self.features[i] + noise)
			targets.append(self.targets[i])
			features.append(self.features[i])
			targets.append(self.targets[i])
			
		noise_set = Gpat_set(features=features, targets=targets)
		return noise_set
	
	@staticmethod
	def result_analyze(prods, dataset, audio_length=32000):
		"""the analysis of the classifiied results"""
		# the input dataset should be the raw data(list, unequal length)
		# the input dataset should have the properties NUM_CLASSES
		assert isinstance(dataset, DataSet)
		dense_prods = prods
		dense_labels = misc.convert_to_dense_labels(dataset.targets)
		assert len(dense_prods) == len(dense_labels)
		
		false_indices = list(np.argwhere(dense_prods != dense_labels).ravel())
		correct_indices = [i for i in range(len(dense_prods))
		                   if i not in false_indices]
		false_samples = [dataset.features[i] for i in false_indices]
		correct_samples = [dataset.features[i] for i in correct_indices]
		correct_labels = dense_labels[correct_indices]
		false_labels = dense_labels[false_indices]
		
		# analysis in aspects:
	  # the false samples distribution via classes, and the false class pointed to
	  # the distribution of the length of the false samples
		false_class_num = []
		for i in range(dataset.num_classes):
			false_class_num.append(len([false_labels[j]
			                            for j in range(len(false_labels))
			                            if false_labels[j] == i]))
			
		false_class_short_num = [len(false_samples[i])
		                     for i in range(len(false_samples))
		                     if len(false_samples[i]) < audio_length]
		console.supplement('.. Total Num: {}, Less than audio length num {}'. format(
			len(false_samples), len(false_class_short_num)))
		
		plot_bar_diagram(false_class_num, title='False class num of the categories')
		
		
	@staticmethod
	def split_data_set(split_indices, data_set):
		# TODO: only for features are list and targets are ndarrays
		assert isinstance(data_set, DataSet)
		split_features = []
		for id in split_indices:
			split_features.append(data_set.features[id])
		
		split_targets = data_set.targets[split_indices]
		split_data_set = DataSet(split_features, split_targets)
		return split_data_set
	
	@staticmethod
	def rms(data):
		return float(np.sqrt(np.mean(np.square(data))))
	
	@staticmethod
	def mfcc_preprocess(data):
		mean = np.mean(data, axis=0)
		std = np.std(data, axis=0)
		X_train = (data - mean) / std
		return X_train
		
if __name__ == '__main__':
	from tframe.data.sequences.signals.signal_set import SignalSet
	from tensorflow.python.keras.utils import to_categorical
	file_path = '../data/original_data/traindata_fs_44100/signal_data_0.tfds'
	data = SignalSet.load(file_path)
	features = data.signals
	for i in range(len(features)):
		if i == 0:
			targets = to_categorical(data['labels'][i], num_classes=41).reshape(1, -1)
		else:
			target = to_categorical(data['labels'][i], num_classes=41).reshape(1, -1)
			targets = np.concatenate((targets, target), axis=0)
	test_data = Gpat_set(features, targets)
	a = next(test_data.gen_batches(batch_size=32))
	b = 1
