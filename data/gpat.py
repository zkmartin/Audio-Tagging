import os
import numpy as np
import librosa
import pandas as pd

from tframe import console
from tframe.data.sequences.signals.signal import Signal
from tframe.data.dataset import DataSet
from tframe.data.sequences.signals.signal_set import SignalSet

from data.utils import get_sizeof_series
from data.utils import un_zip
from data.utils import makdir

class GPAT(object):
	'''Free-Sound General-Purpose Audio Tagging Challenge Data'''
	
	DATA_NAME = "Free-Sound General-Purpose Audio Tagging Challenge Data"
	
	@classmethod
	def load_data_as_list(cls, root_data_dir, sample_rate=16000):
		train_data_dir = root_data_dir + '/' + 'traindata_fs_' + str(sample_rate)
		test_data_dir = root_data_dir + '/' + 'testdata_fs_' + str(sample_rate)
		# Check the existence of the data
		if not os.path.exists(train_data_dir):
			ori_train_data_dir = root_data_dir + '/' + 'audio_train'
			ori_test_data_dir = root_data_dir + '/' + 'audio_test'
			cls.load_as_signals_data(ori_train_data_dir,
			                         sample_rate=sample_rate)
			cls.load_as_signals_data(ori_test_data_dir,
			                         sample_rate=sample_rate)
		# Load the tfd data
		# TODO
		train_list_ver, train_list_unver = cls.load_singal_set_data(train_data_dir)
		test_list = cls.load_singal_set_data(test_data_dir, train=False)
		
		return train_list_ver, train_list_unver, test_list
		
	@classmethod
	def load_rnn_data(cls, root_dir, sample_rate=16000,
	                  validate_size=300,  assump_test=True):
		train_list_ver, train_list_unver, test_list = cls.load_data_as_list(root_dir,
		                                              sample_rate=sample_rate)
		val_indexes = np.random.randint(0, len(train_list_ver), size=validate_size)
		val_list = [train_list_ver[i] for i in val_indexes]
		non_val_list = [tr_list for tr_list in train_list_ver
		                if not tr_list in val_list]
		train_list = non_val_list + train_list_unver

		tra_sig_list, tra_sig_labels = cls.collecting_set(train_list)
		val_sig_list, val_sig_labels = cls.collecting_set(val_list)
		
		train_set = DataSet(features=tra_sig_list, targets=tra_sig_labels)
		val_set = DataSet(features=val_sig_list, targets=val_sig_labels)
		if assump_test is True:
			test_set = val_set
		else:
			test_sig_list = cls.collecting_set(test_list, labeled=False)
			test_set = DataSet(features=test_sig_list)
			
		return train_set, val_set, test_set
		
	@classmethod
	def load_zafar_data(cls, root_dir, sample_rate=16000, duration=2):
		train_list_ver, train_list_unver, test_list = cls.load_data_as_list(root_dir,
		                                              sample_rate=sample_rate)
		audio_length = sample_rate * duration
		train_list = train_list_ver + train_list_unver
		for train_set in train_list:
			for data in train_set.signals:
				data = cls.length_adapted(data, audio_length)
	
	@classmethod
	def load_as_signals_data(cls, data_dir,  sample_rate=None, labeled=True):
		# Checking the existence of the data
		# The download function is not supported yet
		ori_data_root_path = os.path.dirname(data_dir)
		if not os.path.exists(data_dir):
			zip_file = data_dir + '.zip'
			un_zip(zip_file, current_path=True)
		# Check the Sample_rate.
		sample_rate_ = 44100 if sample_rate is None else sample_rate
		sig = 'train' if 'train' in data_dir else 'test'
		output_path = ori_data_root_path + '/' + sig + 'data_fs_' + str(sample_rate_)
		# Check Dir
		makdir(output_path)
		# Get each file name and the labels from the .csv file
		if labeled:
			csv_file = 'train.csv'
			ver_file_indexes, ver_labels,  unver_file_indexes, unver_labels = \
				cls.load_labeled_data( ori_data_root_path + '/' + csv_file)
			cls.load_original_data(ver_file_indexes, output_path,
			                       sample_rate_=sample_rate_, labels=ver_labels)
			cls.load_original_data(unver_file_indexes, output_path, ver=False,
			                       sample_rate_=sample_rate_, labels=unver_labels)
		else:
			file_indexes = os.listdir(data_dir)
			cls.load_original_data(file_indexes, output_path,
			                       sample_rate_=sample_rate_)
				
	@staticmethod
	def load_singal_set_data(file_path, train=True):
		"""Load the whole files that the expanded-name
		are .tfds in the file_path """
		# !!!Temporary!!! This method should be added to the SignalSet object
		file_name_list = os.listdir(file_path)
		signal_set_list = []
		signal_set_list_s = []
		for file_name in file_name_list:
			if '.tfds' in file_name:
					file_path_abs = file_path + '/' + file_name
					sig_set = SignalSet.load(file_path_abs)
					if train is True:
						if 'un' in file_name:
							signal_set_list_s.append(sig_set)
						else:
							signal_set_list.append(sig_set)
						return signal_set_list, signal_set_list_s
					else:
						signal_set_list.append(sig_set)
						return signal_set_list
			
	@staticmethod
	def load_labeled_data(file_path):
		train = pd.read_csv(file_path)
		LABELS = list(train.label.unique())
		label_idx = {label: i for i, label in enumerate(LABELS)}
		train.set_index("fname", inplace=True)
		train["label_idx"] = train.label.apply(lambda x: label_idx[x])
		
		train_verified = train[train.manually_verified == 1]
		train_unverified = train[train.manually_verified == 0]
		
		return train_verified.index, train_verified["label_idx"], \
		       train_unverified.index, train_unverified["label_idx"]
	
	@staticmethod
	def load_original_data(file_indexes, output_path,
	                       sample_rate_=16000, labels=None, ver=True):
		signals_ = []
		labels_tmp = []
		file_id = 0
		memory_szie = 0
		for i, file_index in enumerate(file_indexes):
			file_path_wav = data_dir + '/' + file_index
			data, _ = librosa.core.load(file_path_wav, sample_rate_)
			memory_szie += get_sizeof_series(data)
			signal_data = Signal(data, fs=sample_rate_)
			signals_.append(signal_data)
			if labels is not None:
				labels_tmp.append(labels[file_index])
			if ((memory_szie/100) > 100) or (i == len(file_indexes) - 1):
				if labels is not None:
					signal_s = SignalSet(signals_, data_dict={'labels':labels_tmp})
					if ver is True:
						signal_file_name = output_path + '/' + 'ver_audio_data_' + str(file_id) + '.tfds'
					else:
						signal_file_name = output_path + '/' + 'unver_audio_data_' + str(file_id) + '.tfds'
				else:
					signal_s = SignalSet(signals_, data_dict=None)
					signal_file_name = output_path + '/' + 'audio_data_' + str(file_id) + '.tfds'
				signal_s.save(signal_file_name)
				signals_ = []
				file_id += 1
				memory_szie = 0
				labels_tmp = []
	@staticmethod
	def evaluate(f, dataset):
		if not callable(f): raise AssertionError('!! Input f mustbe callable')
		
	
	@staticmethod
	def audio_norm(data):
		max_data = np.max(data)
		min_data = np.min(data)
		data = (data - min_data) / (max_data - min_data + 1e-6)
		return data - 0.5

	@staticmethod
	def length_adapted(data, input_length):
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
		return data
	
	@staticmethod
	def collecting_set(signal_set_list, labeled=True):
		sig_list = []
		labels = []
		for sig in signal_set_list:
			assert isinstance(sig, SignalSet)
			sig_list += sig.signals
			if labeled is True:
				labels += sig.data_dict['labels']
				return sig_list, labels
			else:
				return sig_list
	
if __name__ == '__main__':
	data_dir = './original_data/audio_test'
	GPAT.load_as_signals_data(data_dir, sample_rate=16000, labeled=False)
	# data_dir = './original_data'
	# train_set, val_set, test_set = GPAT.load_rnn_data(data_dir, sample_rate=16000,
	#                                                   validate_size=300,
	#                                                   assump_test=True)

