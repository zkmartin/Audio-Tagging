import numpy as np

from tframe import checker
from tframe import Predictor
from tframe.data.dataset import DataSet
from tframe.data.sequences.signals.signal_set import SignalSet
from data.gpat import GPAT


def load_data(path, validate_size=300, sample_rate=16000, assump_test=True):
  data_sets = GPAT.load_rnn_data(path, sample_rate=sample_rate,
                                 validate_size=validate_size,
                                 assump_test=assump_test)
  return data_sets


def evaluate(model, data_set, plot=False):
  def f(u):
    assert isinstance(model, Predictor)
    return np.ravel(model.predict(DataSet(features=u)))
  return WHBM.evaluate(f, data_set, plot)


if __name__ == '__main__':
  train_set, test_set = load_data('./data', validate_size=0)
  assert isinstance(train_set, SignalSet)
  assert isinstance(test_set, SignalSet)
  train_set.plot(train_set)
  test_set.plot(train_set)