import numpy as np
import librosa
import os
import pandas as pd

try:
  import cPickle as pickle
except:
  import pickle

from tframe import checker
from tframe import Classifier
from tframe.data.dataset import DataSet
from tframe.data.sequences.signals.signal_set import SignalSet
from tframe.data.sequences.signals.signal import Signal
from tframe import pedia
from data.gpat import GPAT
from at.gpat_data import Gpat_set
from tframe.utils.misc import convert_to_dense_labels
from tframe.utils import console
from tframe.utils.local import check_path
from tensorflow.python.keras.utils import to_categorical

from sklearn.cross_validation import StratifiedKFold


# path = '../data/original_data/traindata_fs_44100/signal_data_0.tfds'
# path = '../data/original_data/traindata_fs_16000'
path = '../data/original_data/traindata_fs_16000_all.tfd'
def load_data(path, csv_path):
  # TODO:
  train = pd.read_csv(csv_path)
  LABELS = list(train.label.unique())
  label_idx = {label: i for i, label in enumerate(LABELS)}
  train.set_index("fname", inplace=True)
  train["label_idx"] = train.label.apply(lambda x: label_idx[x])
  # split the train_set and the val_set
  skf = StratifiedKFold(train.label_idx, n_folds=10)
  
  for i, (train_split, val_split) in enumerate(skf):
    train_split_0 = train_split
    val_split_0 = val_split
    break
  audio_length = 32000
  data_set = DataSet.load(path)
  assert isinstance(data_set, DataSet)
  
  train_split_data = Gpat_set.split_data_set(train_split_0, data_set)
  val_set = Gpat_set.split_data_set(val_split_0, data_set)
  raw_val_set = val_set
  raw_val_set.properties[raw_val_set.NUM_CLASSES] = 41
  train_set = Gpat_set(features=train_split_data.features,
                       targets=train_split_data.targets,
                       NUM_CLASSES=41)
  
  train_set.init_groups()
  for i in range(len(val_set.features)):
    if i == 0:
      features = GPAT.length_adapted(val_set.features[i],
                                     audio_length)
      mfccs = librosa.feature.mfcc(features, 16000, n_mfcc=50)
      mfccs = np.expand_dims(mfccs, axis=0)
      features = np.reshape(features, (1, -1))
    # targets = batch_data[i].targets
    else:
      feature = GPAT.length_adapted(val_set.features[i],
                                    audio_length)
      mfcc = librosa.feature.mfcc(feature, 16000, n_mfcc=50)
      mfcc = np.expand_dims(mfcc, axis=0)
      mfccs = np.concatenate((mfccs, mfcc), axis=0)
      feature = np.reshape(feature, (1, -1))
      features = np.concatenate((features, feature), axis=0)
  targets = val_set.targets
  features = np.expand_dims(features, axis=2)
  mfccs = np.expand_dims(mfccs, axis=-1)
  val_set = DataSet(features, targets, data_dict={'mfcc':mfccs})
  test_set = val_set
  return train_set, val_set, test_set, raw_val_set

def evaluate(model, dataset, th, raw_dataset=None,
             scores=True, save_prods=False):
  assert isinstance(model, Classifier)
  assert isinstance(dataset, DataSet)
  if dataset.targets is not None:
    if not scores:
      model.evaluate_model(dataset)
    else:
      # prods = model.classify(data=dataset, extractor=gpat_extrator)
      prods_naif = model.classify(data=dataset, batch_size=64)
      Gpat_set.result_analyze(prods_naif, raw_dataset)
      if save_prods:
        prods_path = os.path.join('records', th.description)
        pickle_data(prods_naif, prods_path + 'prods.pkl')
      # comupte scores
      # gpat_scores = get_scores(prods, dataset.targets)
      # console.supplement('>> The Scores are {}'.format(gpat_scores))
  else:
    prods = model.classify(data=dataset, extractor=gpat_extrator,
                           batch_size=64)
    prods_path = 'prods/'
    check_path(prods_path)
    pickle_data(prods, os.path.join(prods_path, th.mark) + '.pkl')
    data_path = '../data/original_data'
    tra_csv_path = data_path + '/' + 'train.csv'
    sub_csv_path = data_path + '/' + 'sample_submission.csv'
    make_submission_file(prods, tra_csv_path, sub_csv_path, th)
    
def gpat_extrator(prods):
  assert isinstance(prods, np.ndarray)
  pre_gores = np.argsort(-prods, axis=1)[:, :3]
  
  return pre_gores

def make_submission_file(prods, tra_csv_path, sub_csv_path, th):
  console.supplement('>>Making submission file...')
  LABELS = GPAT.get_total_labels(tra_csv_path)
  top_3 = np.array(LABELS)[prods]
  predicted_labels = [' '.join(list(x)) for x in top_3]
  sub_csv = pd.read_csv(sub_csv_path)
  sub_csv['label'] = predicted_labels
  sub_csv[['fname', 'label']].to_csv(
    th.job_dir + "submission_{}.csv".format(th.mark), index=False)
  console.supplement('>> submission file created')
  
def get_scores(prediction, labels):
  label = convert_to_dense_labels(labels)
  scores = mapk(label, prediction)
  return scores
  
def apk(actual, predicted, k=3):
  if len(predicted) > k:
    predicted = predicted[:k]
	  
  score = 0.0
  num_hits = 0.0
  
  for i, p in enumerate(predicted):
    if p in actual and p not in predicted[:i]:
      num_hits += 1.0
      score += num_hits / (i + 1.0)
  
  if not actual:
    return 0.0
  
  return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
  return np.mean([apk([a], p, k) for a, p in zip(actual, predicted)])

def load_simulate_test_data():
  X = np.random.random(1000)
  X = np.reshape(X, (1, -1))
  X = np.expand_dims(X, axis=2)
  t = np.linspace(0, 10, 1000)
  Y = np.sin(t)
  Y = np.reshape(Y, (1, -1))
  return X, Y

class Config(object):
    def __init__(self,
                 sampling_rate_raw=16000, audio_duration=2, n_classes=41,
                 use_mfcc=False, n_folds=10, learning_rate=0.0001,
                 max_epochs=50, n_mfcc=20, sampling_rate=44100):
        self.sampling_rate_raw = sampling_rate_raw
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.audio_length_raw = self.sampling_rate_raw * self.audio_duration
        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)
        else:
            self.dim = (self.audio_length, 1)

def add_noise(coe, data, fs):
  
    signal_energy = Gpat_set.rms(data)
    intensity = coe * signal_energy
    noise = Signal.gaussian_white_noise(intensity, data.size, fs)
    noise = np.array(noise, dtype=np.float32)
    noise_data = noise + data
    return noise_data

def prepare_data(df, config, data_dir, noise=False):
    assert isinstance(config, Config)
    X = np.empty(shape=(df.shape[0], config.dim[0], config.dim[1], 1))
    X_t = np.empty(shape=(df.shape[0], config.sampling_rate_raw*config.audio_duration))
    input_length = config.audio_length
    for i, fname in enumerate(df.index):
        # print(fname)
        file_path = data_dir + fname
        data, _ = librosa.core.load(file_path, sr=config.sampling_rate,
                                    res_type="kaiser_fast")
        # raw_data, _ = librosa.core.load(file_path, sr=config.sampling_rate_raw,
        #                               res_type='kaiser_fast')
        
        if noise:
            data = add_noise(0.2, data, config.sampling_rate_raw)
            raw_data = add_noise(0.2, raw_data, config.sampling_rate)
        data = GPAT.length_adapted(data, config.audio_length)
        # raw_data = GPAT.length_adapted(raw_data, config.audio_length_raw)
        # X_t[i, :] = GPAT.audio_norm(raw_data)
        X_t[i, :] = GPAT.audio_norm(data)
        data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
        data = np.expand_dims(data, axis=-1)
        X[i,] = data
    return X, X_t

def preprocess(data, train_mean=None):
  if train_mean is None:
    mean = np.mean(data, axis=0)
  else:
    mean = train_mean
  std = np.std(data, axis=0)
  X_train = (data - mean) / std
  if train_mean is None:
    return X_train, mean
  else:
    return X_train
  

def load_demo_data(path):
  train = pd.read_csv("../data/original_data/train.csv")
  LABELS = list(train.label.unique())
  label_idx = {label: i for i, label in enumerate(LABELS)}
  train.set_index("fname", inplace=True)
  # test.set_index("fname", inplace=True)
  train["label_idx"] = train.label.apply(lambda x: label_idx[x])
  train_verified = train[train.manually_verified == 1]
  
  train_csv = train
  # test = pd.read_csv("../data/original_data/sample_submission.csv")
  config = Config(sampling_rate_raw=16000, audio_duration=2, n_folds=10,
                  learning_rate=0.001, use_mfcc=True, n_mfcc=50,
                  sampling_rate=16000)
  X_train, X_train_t = prepare_data(train_csv, config, path)
  # X_train_n = prepare_data(train, config, path, noise=True)
  y_train = to_categorical(train_csv.label_idx, num_classes=config.n_classes)
  X_train, train_mean = preprocess(X_train)
  pickle_data(train_mean, '../data/original_data/train_mean.pkl')
  # TODO:
  
  # split the train_set and the val_set
  skf = StratifiedKFold(train_csv.label_idx, n_folds=10)

  for i, (train_split, val_split) in enumerate(skf):
    if i == 1:
      train_split_0 = train_split
      val_split_0 = val_split
      break
		  
  X_train_t = np.expand_dims(X_train_t, axis=-1)
  
  features = X_train_t[train_split_0]
  targets = y_train[train_split_0]
  train_set = DataSet(features=features, targets=targets,
                      data_dict={'mfcc':X_train[train_split_0]})
  features = X_train_t[val_split_0]
  targets = y_train[val_split_0]
  val_set = DataSet(features=features, targets=targets,
                    data_dict={'mfcc': X_train[val_split_0]})
  
  return train_set, val_set

def load_test_data(path, train_mean):
  test = pd.read_csv("../data/original_data/sample_submission.csv")
  test.set_index("fname", inplace=True)
  
  train_csv = test
  # test = pd.read_csv("../data/original_data/sample_submission.csv")
  config = Config(sampling_rate_raw=16000, audio_duration=2, n_folds=10,
                  learning_rate=0.001, use_mfcc=True, n_mfcc=50,
                  sampling_rate=16000)
  X_train, X_train_t = prepare_data(train_csv, config, path)
  # X_train_n = prepare_data(train, config, path, noise=True)
  X_train = preprocess(X_train, train_mean=train_mean)
  # TODO:
  
  X_train_t = np.expand_dims(X_train_t, axis=-1)
  
  test_set = DataSet(features=X_train_t, data_dict={'mfcc':X_train})
  
  return test_set

def pickle_data(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

def read_pickle_data(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data
  
if __name__ == '__main__':
  # path = '../data/processed_data/traindata_fs_16000_all.tfd'
  # train_set, val_set, test_set = load_data(path)
  # a = next(train_set.gen_batches(batch_size=32))
  # b = next(val_set.gen_batches(batch_size=32))
  path = '../data/original_data/audio_train/'
  train_set, val_set = load_demo_data(path)
  # path = '../data/processed_data/traindata_fs_16000_all.tfd'
  # csv_path = '../data/original_data/train.csv'
  # train_set, val_set, test_set = load_data(path, csv_path)
  a = 1
  