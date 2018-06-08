import numpy as np
from tframe.data.dataset import DataSet
from tframe import checker
from data.gpat import GPAT

class Gpat_set(DataSet):
	
	def __init__(self, features, targets, audio_length=32000):
		super(Gpat_set, self).__init__(features, targets)
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
			indices = (np.random.randint(self.size, size=(batch_size,)) if shuffle
								 else range(i * batch_size,
														min((i + 1) * batch_size, len(self.targets))))
			batch_features = []
			for indice in indices:
				batch_features.append(self.features[indice])
			for i in range(len(indices)):
				if i == 0:
					features = GPAT.length_adapted(batch_features[i], self.audio_length)
					features = GPAT.audio_norm(features)
					features = np.reshape(features, (1, -1))
					# targets = batch_data[i].targets
				else:
					feature = GPAT.length_adapted(batch_features[i], self.audio_length)
					feature = GPAT.audio_norm(feature)
					feature = np.reshape(feature, (1, -1))
					features = np.concatenate((features, feature), axis=0)
					# targets = np.concatenate((targets, batch_data[i].targets), axis=0)
			targets = self.targets[indices]
			features = np.expand_dims(features, axis=2)
			output_batch_data = DataSet(features, targets)
			yield output_batch_data
			
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
