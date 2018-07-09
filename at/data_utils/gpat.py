import os
import numpy as np
import librosa
import pandas as pd

from tframe.models.sl.classifier import Classifier
from tframe.utils import console
from tframe.utils.misc import convert_to_one_hot
from tframe.utils.misc import convert_to_dense_labels
from tframe.data.dataset import DataSet
from tframe.data.sequences.seq_set import SequenceSet
from at.data_utils.utils import check_dir
from at.data_utils.utils import pickle_data, read_pickle_data
from sklearn.cross_validation import StratifiedKFold


class GPAT(object):
	'''Free-Sound General-Purpose Audio Tagging Challenge Data'''
	
	DATA_NAME = "Free-Sound General-Purpose Audio Tagging Challenge Data"
	
	@classmethod
	def load_as_seq_set(cls, path, sample_rate=16000):
		train_path = os.path.join(path, 'audio_train')
		test_path = os.path.join(path, 'audio_test')
		train_csv_file = os.path.join(path, 'train.csv')
		submission_file = os.path.join(path, 'sample_submission.csv')
		
		train_csv = pd.read_csv(train_csv_file)
		sub_csv = pd.read_csv(submission_file)
		
		# get the verified or not indices
		verified_list = list(train_csv[train_csv.manually_verified == 1].index)
		verified_list_n = list(train_csv[train_csv.manually_verified == 0].index)
		assert(len(verified_list) + len(verified_list_n) == len(train_csv.index))
		
		# using fname as indexes
		cls.dataframe_set_index(train_csv, 'fname')
		cls.dataframe_set_index(sub_csv, 'fname')
		
		# load features and labels
		train_features = cls.read_wav_files(train_path, train_csv, sr=sample_rate)
		train_features_ver = [train_features[i] for i in verified_list]
		train_features_ver_n = [train_features[i] for i in verified_list_n]
		test_features = cls.read_wav_files(test_path, sub_csv, sr=sample_rate)
		train_csv = cls.add_label_idx(train_csv)
		targets = convert_to_one_hot(train_csv.label_idx, num_classes=41)
		
		train_set = SequenceSet(features=train_features,
		                       summ_dict={'targets': list(targets)})
		test_set = SequenceSet(features=test_features)
		train_set_ver = SequenceSet(features=train_features_ver,
		                            summ_dict={'targets':list(targets[verified_list])})
		train_set_ver_n = SequenceSet(features=train_features_ver_n,
		                              summ_dict={'targets':list(targets[verified_list_n])})
		# create save path
		dn = os.path.dirname
		data_root_path = dn(dn(dn(__file__)))
		paths = [data_root_path, 'data', 'processed_data',
		         'data_fs_{}'.format(sample_rate)]
		# Check path
		check_dir(paths)
		save_path =paths[0]
		paths.pop(0)
		for path in paths:save_path = os.path.join(save_path, path)
		# Save data set
		train_set.save(os.path.join(save_path, 'train_data'))
		test_set.save(os.path.join(save_path, 'test_data'))
		train_set_ver.save(os.path.join(save_path, 'train_data_ver'))
		train_set_ver_n.save(os.path.join(save_path, 'train_data_ver_n'))
	
	@classmethod
	def load_data_set(cls, path, th, random_pos=False, test_all=False,
	                  audio_length=32000, sr=16000, **kwargs):
		path = os.path.join(path, 'data_fs_{}'.format(sr))
		ver_only = kwargs.get('ver_only', False)
		id = kwargs.get('id', None)
		train_file_name = 'train_data.tfds' if not ver_only else 'train_data_ver.tfds'
		if id is not None: train_file_name = 'train_model_ver_{}.tfds'.format(id)
		train_file = os.path.join(path, train_file_name)
		test_file = os.path.join(path, 'test_data.tfds')
		dn = os.path.dirname
		# If there no files, load the raw data as sequenceset
		if not os.path.isfile(train_file):
			cls.load_as_seq_set(os.path.join(dn(dn(path)), 'raw_data'),
			                    sample_rate=sr)
		# Mean and std path
		root_dir = dn(dn(os.path.abspath(__file__)))
		mean_path = os.path.join(root_dir, 'means_stds')
		check_dir(mean_path)
		file_name_mean = th.job_dir + '_' + th.mark + '_mean.pkl'
		file_name_std = th.job_dir + '_' + th.mark + '_std.pkl'
		# If th.train is True
		if th.train:
			# Load train data
			train = SequenceSet.load(train_file)
			assert isinstance(train, SequenceSet)
			label_idx = (train.summ_dict['targets'])
			label_idx = [convert_to_dense_labels(label_idx[i])[0]
									 for i in range(len(label_idx))]
			label_idx = convert_to_dense_labels(np.array(label_idx))
			skf = StratifiedKFold(label_idx, n_folds=10)
			# TODO:
			fold = kwargs.get('fold', 0)
			for i, (tra, vali) in enumerate(skf):
				if i == fold:
					train_split = tra
					val_split = vali
			
			if not random_pos:
				data_set, train_mean, train_std = cls._batch_processor(audio_length,
				                                                       sr)(train)
				# not finished
				pickle_data(train_mean, os.path.join(mean_path, file_name_mean))
				pickle_data(train_std, os.path.join(mean_path, file_name_std))
				val_set = data_set[val_split]
				data_set = data_set[train_split]
			else:
				data_set = train[train_split]
				assert isinstance(data_set, SequenceSet)
				data_set.batch_preprocessor = cls._batch_processor(audio_length, sr)
				data_set.length_calculator = lambda x:1
				val_set = cls._batch_processor(audio_length, sr)(train[val_split])
			data_set.properties[data_set.NUM_CLASSES] = 41
			cls.init_groups(data_set)
			val_set.lengths = [len(train[val_split].features[i])
			                   for i in range(len(val_split))]
			test_set = None
		else:
			# Load test data
			console.show_status('Loading test data...')
			test = SequenceSet.load(test_file)
			assert isinstance(test, SequenceSet)
			if not test_all:
				if not random_pos:
					mean = read_pickle_data(os.path.join(mean_path, file_name_mean))
					std = read_pickle_data(os.path.join(mean_path, file_name_std))
					test_set = cls._batch_processor(audio_length, sr,
																					mean=mean, std=std)(test)
				else:
					test_set = cls._batch_processor(audio_length, sr)(test)
			else:
				# all_test_data_path = os.path.join(path, 'test_all_data.tfds')
				# test
				all_test_data_path = os.path.join(path, 'test_all_data.tfds.tfd')
				if not os.path.isfile(all_test_data_path):
					console.show_status('Splitting the test data ...')
					test_arrays, nums = cls.split_seqs(test.features,
																						block_length=audio_length)
					console.show_status('Splitting done ...')
					test_set = DataSet(features=test_arrays)
					test_set.nums = nums
					test_set.save(all_test_data_path)
					pickle_data(nums, os.path.join(path, 'all_test_data_nums.tfds'))
				else:
					# TODO
					test_set = DataSet.load(all_test_data_path)
					assert isinstance(test_set, DataSet)
					test_mfccs = arr_to_mfccs(test_set.features)
					test_features = np.empty(shape=test_set.features.shape)
					for i in range(len(test_features)):
						test_features[i, :] = GPAT.audio_norm(test_set.features[i, :])
					test_features = np.expand_dims(test_features, axis=-1)
					test_set = DataSet(features=test_features,
					                   data_dict={'mfcc':test_mfccs})
					test_set.nums = read_pickle_data(os.path.join(path,
					                                              'all_test_data_nums.tfds'))
			data_set, val_set = None, None
		if th.val_on_train_set:
			return data_set, val_set, test_set, train, train_split
		else:
			return data_set, val_set, test_set, None, None
	
	@classmethod
	def load_ver_n_data(cls, path, sr=16000, audio_length=32000, **kwargs):
		path = os.path.join(path, 'data_fs_{}'.format(sr))
		train_file_name = 'train_data_ver_n.tfds'
		id = kwargs.get('id', None)
		if id is not None: train_file_name = 'train_model_ver_n_{}.tfds'.format(id)
		train_file = os.path.join(path, train_file_name)
		train_test = SequenceSet.load(train_file)
		all_test_data_path = os.path.join(path, 'train_ver_n_test_all')
		if id is not None:
			all_test_data_path = os.path.join(path, 'train_ver_n_test_all_{}'.format(id))
		if not os.path.isfile(all_test_data_path + '.tfd'):
			console.show_status('Splitting the test data ...')
			test_arrays, nums = cls.split_seqs(train_test.features,
			                                   block_length=audio_length)
			console.show_status('Splitting done ...')
			test_set = DataSet(features=test_arrays)
			test_set.nums = nums
			test_set.save(all_test_data_path)
			pickle_data(nums, os.path.join(path, 'train_ver_n_all.pkl'))
		else:
			test_set = DataSet.load(all_test_data_path + '.tfd')
			assert isinstance(test_set, DataSet)
			test_mfccs = arr_to_mfccs(test_set.features)
			test_features = np.empty(shape=test_set.features.shape)
			for i in range(len(test_features)):
				test_features[i, :] = GPAT.audio_norm(test_set.features[i, :])
			test_features = np.expand_dims(test_features, axis=-1)
			test_targets = np.array(train_test.summ_dict['targets'])
			test_targets = np.reshape(test_targets, (test_targets.shape[0], -1))
			train_test_set = DataSet(features=test_features,
			                         data_dict={'mfcc':test_mfccs})
			train_test_set.labels = test_targets
			train_test_set.nums = read_pickle_data(os.path.join(path,
			                                              'train_ver_n_all.pkl'))
		return train_test_set
	
	@classmethod
	def modified_train_ver_n_data(cls, inds_path, data_path, id=0, sr=16000):
		# Load data
		path = os.path.join(data_path, 'data_fs_{}'.format(sr))
		train_ver_file_name = 'train_data_ver.tfds'
		train_file_name = 'train_data_ver_n.tfds' if id == 0 else 'train_model_ver_{}'.format(
			id)
		train_file = os.path.join(path, train_file_name)
		train_ver_file = os.path.join(path, train_ver_file_name)
		train_test = SequenceSet.load(train_file)
		train = SequenceSet.load(train_ver_file)
		
		# get the common indices
		indices = GPAT.read_indices(inds_path)
		com_inds = GPAT.inds_union(indices)
		train_model_ver = train_test[com_inds]
		ver_n_indices = [i for i in range(len(train_test.features))
		                                  if i not in com_inds]
		
		new_train_ver = SequenceSet(features=train.features + train_model_ver.features,
		                            summ_dict={'targets':train.summ_dict['targets']
		                                       + train_model_ver.summ_dict['targets']})
		# Save the new Sequenceset
		save_path = os.path.join(path, 'train_model_ver_{}'.format(id + 1))
		save_path_ver_n = os.path.join(path, 'train_model_ver_n_{}'.format(id + 1))
		new_train_ver.save(save_path)
		train_test[ver_n_indices].save(save_path_ver_n)
		console.show_status('new train set saved')
		
	@classmethod
	def _batch_processor(cls, e_length, sr, mean=None, std=None):
		def batch_preprocessor(seqset):
			assert isinstance(seqset, SequenceSet)
			features = []
			targets = []
			mfccs = []
			stfts = []
			for i in range(len(seqset.features)):
				feature = np.reshape(seqset.features[i], (-1, ))
				feature = cls.length_adapted(feature, e_length)
				mfcc = librosa.feature.mfcc(feature, sr=sr, n_mfcc=50)
				stf = np.abs(librosa.stft(feature))
				feature = GPAT.audio_norm(feature)
				mfcc = np.expand_dims(mfcc, -1)
				stf = np.expand_dims(stf, -1)
				features.append(feature)
				mfccs.append(mfcc)
				stfts.append(stf)
				if 'targets' in list(seqset.summ_dict.keys()):
					target = np.reshape(seqset.summ_dict['targets'][i], (1, -1))
					targets.append(target)
				else: targets=None
				# console.print_progress(i + 1, total)
			features = np.array(features)
			features = np.expand_dims(features, axis=-1)
			if targets is not None:
				targets = np.array(targets)
				targets = np.reshape(targets, (targets.shape[0], -1))
			mfccs = np.array(mfccs)
			stfts = np.array(stfts)
			# preprocess : train:fixed position, test fixed position fixed postion test
			# not preprocessing : train random position, test random position, val
			# random position
			if mean is not None:
				mfccs, train_mean, train_std = cls.preprocess(mfccs, mean=mean, std=std)
			else:
				if (len(seqset.features) > 3000) and targets is not None:
					mfccs, train_mean, train_std = cls.preprocess(mfccs, mean=mean, std=std)
			# TODO
			data_set = DataSet(features=features, targets=targets,
			                   data_dict={'mfcc':mfccs, 'stft':stfts})
			if mean is None and (len(seqset.features) > 3000) and targets is not None:
				return data_set, train_mean, train_std
			else:
				return data_set
		return batch_preprocessor
	
	@staticmethod
	def dataframe_set_index(frame, column_index):
		frame.set_index(column_index, inplace=True)
	
	@staticmethod
	def add_label_idx(frame):
		assert 'label' in frame.keys()
		LABELS = list(frame.label.unique())
		label_idx = {label: i for i, label in enumerate(LABELS)}
		# test.set_index("fname", inplace=True)
		frame["label_idx"] = frame.label.apply(lambda x: label_idx[x])
		return frame
	
	@staticmethod
	def read_wav_files(path, frame, sr=16000):
		data_list = []
		total = len(list(frame.index))
		console.show_status('Reading from {}'.format(path))
		for i, fname in enumerate(frame.index):
			file_path = os.path.join(path, fname)
			data, _ = librosa.core.load(file_path, sr=sr, res_type='kaiser_fast')
			data_list.append(data)
			console.print_progress(i + 1, total)
		return data_list
	
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
	def get_total_labels(file_path):
		train = pd.read_csv(file_path)
		return list(train.label.unique())
	
	@staticmethod
	def evaluate(model, data_set, th, save_prods=False, submit=True):
		assert isinstance(model, Classifier)
		if save_prods:
			raw_prods = model.classify(data_set, batch_size=th.val_batch_size,
			                           extractor=GPAT.raw_extractor)
			if hasattr(data_set, 'nums'):
				raw_prods = GPAT.test_all_prods_op(data_set, raw_prods)
			dn = os.path.dirname
			data_dir = dn(dn(os.path.abspath(__file__)))
			paths = [data_dir, 'prods', th.job_dir]
			check_dir(paths)
			save_path = paths[0]
			paths.pop(0)
			for path in paths: save_path = os.path.join(save_path, path)
			pickle_data(raw_prods, os.path.join(save_path, th.mark)+'.pkl')
		if submit:
			if hasattr(data_set, 'nums'):
				prods = model.classify(data_set, batch_size=th.val_batch_size,
				                       extractor=GPAT.raw_extractor)
				prods = GPAT.test_all_prods_op(data_set, prods)
				prods = GPAT.gpat_extractor(prods)
			else:
				prods = model.classify(data_set, batch_size=th.val_batch_size,
																	 extractor=GPAT.gpat_extractor)
			GPAT.make_submission_file(prods, th)
			
	@staticmethod
	def make_submission_file(prods, th=None, sub_name=None):
		# if th is None?
		dn = os.path.dirname
		csv_dir = dn(dn(dn(os.path.abspath(__file__))))
		tra_csv_path = os.path.join(csv_dir, 'data', 'raw_data', 'train.csv')
		sub_csv_path = os.path.join(csv_dir, 'data', 'raw_data', 'sample_submission.csv')
		console.show_status('Making submission file...')
		
		LABELS = GPAT.get_total_labels(tra_csv_path)
		top_3 = np.array(LABELS)[prods]
		predicted_labels = [' '.join(list(x)) for x in top_3]
		sub_csv = pd.read_csv(sub_csv_path)
		sub_csv['label'] = predicted_labels
		dn = os.path.dirname
		data_dir = dn(dn(os.path.abspath(__file__)))
		paths = [data_dir, 'submissions', th.job_dir]
		check_dir(paths)
		save_path = paths[0]
		paths.pop(0)
		for path in paths: save_path = os.path.join(save_path, path)
		if sub_name is None:
			save_path = os.path.join(save_path, th.mark + '.csv')
		else:
			if sub_name.split('.')[-1] is not 'csv':sub_name = sub_name+'.csv'
			save_path = os.path.join(save_path, sub_name)
		sub_csv[['fname', 'label']].to_csv(save_path, index=False)
		console.show_status('Submission file created')
		
	@staticmethod
	def gpat_extractor(prods):
		assert isinstance(prods, np.ndarray)
		pre_gores = np.argsort(-prods, axis=1)[:, :3]
		
		return pre_gores
	
	@staticmethod
	def raw_extractor(prods):
		return prods
	
	@staticmethod
	def audio_norm(data):
		max_data = np.max(data)
		min_data = np.min(data)
		data = (data - min_data) / (max_data - min_data + 1e-6)
		return data - 0.5
	
	@staticmethod
	def preprocess(data, mean=None, std=None):
		mean = np.mean(data, axis=0) if mean is None else mean
		std = np.std(data, axis=0) if std is None else std
		
		data = (data - mean)/std
		return data, mean, std
	
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
			data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
		return data
	
	@staticmethod
	def split_seqs(seqs, block_length):
		assert isinstance(seqs, list)
		num_record = []
		seqs_split, num= GPAT.split_seq(seqs[0], block_length)
		num_record.append(num)
		total = len(seqs) - 1
		for i in range(1, len(seqs)):
			seq, num = GPAT.split_seq(seqs[i], block_length)
			seqs_split = np.concatenate((seqs_split, seq), axis=0)
			num_record.append(num)
			console.print_progress(i + 1, total=total)
		return seqs_split, num_record
	
	@staticmethod
	def split_seq(seq, block_length):
		seq = np.reshape(seq, (-1, ))
		if len(seq) <= block_length:
			split_result = np.reshape(GPAT.length_adapted(seq, block_length),
			                          (1, -1))
			return (split_result, 1)
		else:
			if len(seq) > block_length:
				blocks = len(seq) // block_length
				remain_length = np.mod(len(seq), block_length)
				for i in range(blocks):
					if i == 0:
						split_result = np.reshape(seq[:block_length], (1, -1))
					else:
						tmp = np.reshape(seq[block_length * i:(i + 1) * (block_length)],
						                 (1, -1))
						split_result = np.concatenate((split_result, tmp), axis=0)
				if remain_length != 0:
					remain_block = np.reshape(seq[len(seq) - block_length:len(seq)],
					                          (1, -1))
					split_result = np.concatenate((split_result, remain_block))
					return (split_result, blocks + 1)
				else:
					return (split_result, blocks)
			else:
				return (np.reshape(seq, (1, -1)), 1)
	
	@staticmethod
	def targets_fill(targets, nums):
		assert isinstance(targets, np.ndarray)
		targets_filled = np.empty(shape=(sum(nums), targets.shape[1]))
		nums.insert(0, 0)
		for i in range(len(nums) - 1):
			for j in range(np.sum(nums[:(i + 1)]), np.sum(nums[:(i + 2)])):
				targets_filled[j, :] = targets[i, :]
				
		return targets_filled
		
	
	@staticmethod
	def init_groups(data_set):
		targets = data_set.targets if data_set.targets is not None else data_set.summ_dict['targets']
		targets = np.array(targets)
		targets = np.reshape(targets, (targets.shape[0], -1))
		if data_set.num_classes is None:return
		groups = []
		dense_labels = convert_to_dense_labels(targets)
		for i in range(data_set.num_classes):
			samples = list(np.argwhere(dense_labels == i).ravel())
			groups.append(samples)
		data_set.properties[data_set.GROUPS] = groups
		return data_set
	
	@staticmethod
	def test_all_prods_op(data_set, prods):
		assert isinstance(data_set, DataSet)
		assert hasattr(data_set, 'nums')
		nums = data_set.nums
		total_num = 0
		products = []
		prod = np.ones(prods[0].shape)
		for i, num in enumerate(nums):
			for j in range(total_num, total_num + num):
					prod *= prods[j]
			products.append(prod)
			total_num += num
			prod = np.ones(prods[0].shape)
		products = np.array(products)
		return products
	
	@staticmethod
	def inds_union(indices):
		assert isinstance(indices, list)
		for i in range(len(indices)):
			if i == 0:
				inds = indices[i]
			else:
				inds = GPAT.double_lists_union(inds, indices[i])
		return inds
	
	@staticmethod
	def double_lists_union(list_1, list_2):
		inds = [ind for ind in list_1 if ind in list_2]
		return inds
	
	@staticmethod
	def read_indices(inds_path):
		files = os.listdir(inds_path)
		indices = []
		for file in files:
			file_path = os.path.join(inds_path, file)
			indices.append(read_pickle_data(file_path))
		return indices
		
def arr_to_mfccs(data):
	mfccs = np.empty(shape=(data.shape[0], 50, 63, 1))
	for i in range(len(data)):
		mfcc = np.reshape(data[i, :], (-1, ))
		mfcc = librosa.feature.mfcc(mfcc, sr=16000, n_mfcc=50)
		mfcc = np.expand_dims(mfcc, axis=-1)
		mfccs[i, :] = mfcc
	return mfccs
	
if __name__ == '__main__':
	# path = '../../data/raw_data'
	# path = '../../data/processed_data/'
	# GPAT.load_as_seq_set(path, sample_rate=16000)
	# GPAT.load_data_set(path)
	a = np.ones(shape=(1, 5))
	b = np.ones(shape=(1, 5))*2
	c = np.ones(shape=(1, 5))*3
	d = [3, 2, 5]
	e = np.concatenate((a, b), axis=0)
	e = np.concatenate((e, c), axis=0)
	console.show_status('e = ')
	console.pprint(e)
	console.show_status('filled:')
	console.pprint(GPAT.targets_fill(e, d))
	


