import tensorflow as tf
import numpy as np
from tframe import Predictor
from tframe import Classifier
from tframe.models.recurrent import Recurrent
from tframe.nets.resnet import ResidualNet
from tframe.nets.rnn_cells.lstms import BasicLSTMCell
from tframe.layers import Input, Linear, Activation
from tframe.layers import Conv1D, MaxPool2D, Conv2D
from tframe.layers import Dropout, BatchNorm
from tframe.layers import Flatten
from extensions import MaxPool1D
from data_utils.gpat_classsifer import Classifier_Gpat
from extensions import GlobalMaxPooling1D
from tframe.config import Config



def mlp(th):
  assert isinstance(th, Config)
  # Initiate a model
  model = Classifier(mark=th.mark)

  # Add input layer
  model.add(Input([32000]))
  # Add hidden layers
  for _ in range(th.num_blocks):
    model.add(Linear(output_dim=th.hidden_dim))
    model.add(BatchNorm())
    # model.add(BatchNormalization())
    model.add(Activation(th.actype1))
    # model.add(Dropout(0.9))
  # Add output layer
  model.add(Linear(output_dim=th.num_classes))
  model.add(Activation('softmax'))

  # Build model
  optimizer=tf.train.AdamOptimizer(learning_rate=th.learning_rate)
  model.build(optimizer=optimizer)

  return model


def lstm(th):
  assert isinstance(th, Config)
  # Initiate model
  th.mark = 'lstm_' + th.mark
  model = Predictor(mark=th.mark, net_type=Recurrent)

  # Add input layer
  model.add(Input(sample_shape=[th.memory_depth]))
  # Add hidden layers
  for _ in range(th.num_blocks):
    model.add(BasicLSTMCell(th.hidden_dim, with_peepholes=False))
  # Add output layer
  model.add(Linear(output_dim=1))

  # Build model
  optimizer = tf.train.AdamOptimizer(learning_rate=th.learning_rate)
  model.build_as_regressor(optimizer)

  return model

def lstm_test(th):
  assert isinstance(th, Config)
  # Initiate model
  th.mark = 'lstm_' + th.mark
  model = Classifier(mark=th.mark, net_type=Recurrent)

  # Add input layer
  model.add(Input(sample_shape=[th.memory_depth]))
  # Add hidden layers
  for _ in range(th.num_blocks):
    model.add(BasicLSTMCell(th.hidden_dim, with_peepholes=False))
  # Add output layer
  model.add(Linear(output_dim=41))
  model.add(Activation('softmax'))

  # Build model
  optimizer = tf.train.AdamOptimizer(learning_rate=th.learning_rate)
  model.build(optimizer=optimizer)

  return model

def conv_test(th):
  assert isinstance(th, Config)
  # Initiate model
  th.mark = 'cnn' + th.mark
  model = Classifier(mark=th.mark)
  
  # Add input layer
  model.add(Input(sample_shape=[32000, 1]))
  # Add hidden layers
  model.add(Conv1D(filters=16, kernel_size=9, padding='valid'))
  model.add(Activation('relu'))
  model.add(Conv1D(filters=16, kernel_size=9, padding='valid'))
  model.add(Activation('relu'))
  model.add(MaxPool1D(pool_size=16, strides=16))
  # model.add(Dropout(0.9))

  model.add(Conv1D(filters=32, kernel_size=3, padding='valid'))
  model.add(Activation('relu'))
  model.add(Conv1D(filters=32, kernel_size=3, padding='valid'))
  model.add(Activation('relu'))
  model.add(MaxPool1D(pool_size=4, strides=4))
  # model.add(Dropout(0.9))

  model.add(Conv1D(filters=32, kernel_size=3, padding='valid'))
  model.add(Activation('relu'))
  model.add(Conv1D(filters=32, kernel_size=3, padding='valid'))
  model.add(Activation('relu'))
  model.add(MaxPool1D(pool_size=4, strides=4))
  # model.add(Dropout(0.9))
  #
  model.add(Conv1D(filters=256, kernel_size=3, padding='valid'))
  model.add(Activation('relu'))
  model.add(Conv1D(filters=256, kernel_size=3, padding='valid'))
  model.add(Activation('relu'))
  model.add(GlobalMaxPooling1D())
  # model.add(Dropout(0.8))
  #
  model.add(Linear(output_dim=64))
  model.add(Activation('relu'))
  model.add(Linear(output_dim=1028))
  model.add(Activation('relu'))
  
  # Add output layer
  model.add(Linear(output_dim=41))
  model.add(Activation('softmax'))
  
  # Build model
  optimizer = tf.train.AdamOptimizer(learning_rate=th.learning_rate)
  model.build(optimizer=optimizer)
  
  return model

def conv_2d_test(th):
  assert isinstance(th, Config)
  # Initiate model
  th.mark = 'cnn_2d' + th.mark
  def data_dim(sample_rate=44100, duration=2, n_mfcc=40):
    audio_length = sample_rate * duration
    dim = (n_mfcc, 1 + int(np.floor(audio_length / 512)), 1)
    return dim
  dim = data_dim()
  
  model = Classifier(mark=th.mark)
  
  # Add input layer
  model.add(Input(sample_shape=[dim[0], dim[1], 1]))
  # Add hidden layers
  model.add(Conv2D(32, (4, 10), padding='same'))
  model.add(BatchNorm())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
  # model.add(Dropout(0.7))

  model.add(Conv2D(32, (4, 10), padding='same'))
  model.add(BatchNorm())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
  # model.add(Dropout(0.7))
  
  model.add(Conv2D(32, (4, 10), padding='same'))
  model.add(BatchNorm())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
  # model.add(Dropout(0.7))

  model.add(Conv2D(32, (4, 10), padding='same'))
  model.add(BatchNorm())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
  # model.add(Dropout(0.7))
  
  model.add(Flatten())
  model.add(Linear(output_dim=64))
  model.add(BatchNorm())
  model.add(Activation('relu'))
  
  # Add output layer
  model.add(Linear(output_dim=41))
  model.add(Activation('softmax'))
  
  # Build model
  optimizer = tf.train.AdamOptimizer(learning_rate=th.learning_rate)
  model.build(optimizer=optimizer)
  
  return model

def multiinput(th):
  assert isinstance(th, Config)
  # model = Classifier(mark=th.mark)
  model = Classifier_Gpat(mark=th.mark)
  
  def data_dim(sample_rate=16000, duration=2, n_mfcc=50):
    audio_length = sample_rate * duration
    dim = (n_mfcc, 1 + int(np.floor(audio_length / 512)), 1)
    return dim
  dim = data_dim()
  
  # Add hidden layers
  subnet = model.add(inter_type=model.CONCAT)
  # the net to process raw data
  subsubnet = subnet.add()
  # subsubnet.add(Input(sample_shape=[32000, 1], name='raw_data'))
  subsubnet.add(Input(sample_shape=[32000, 1]))
  subsubnet.add(Conv1D(filters=16, kernel_size=9, padding='valid'))
  subsubnet.add(BatchNorm())
  subsubnet.add(Activation('relu'))
  subsubnet.add(Conv1D(filters=16, kernel_size=9, padding='valid'))
  subsubnet.add(BatchNorm())
  subsubnet.add(Activation('relu'))
  subsubnet.add(MaxPool1D(pool_size=16, strides=16))
  subsubnet.add(Dropout(th.raw_keep_prob))

  subsubnet.add(Conv1D(filters=32, kernel_size=3, padding='valid'))
  subsubnet.add(Activation('relu'))
  subsubnet.add(Conv1D(filters=32, kernel_size=3, padding='valid'))
  subsubnet.add(Activation('relu'))
  subsubnet.add(MaxPool1D(pool_size=4, strides=4))
  subsubnet.add(Dropout(th.raw_keep_prob))

  subsubnet.add(Conv1D(filters=32, kernel_size=3, padding='valid'))
  subsubnet.add(Activation('relu'))
  subsubnet.add(Conv1D(filters=32, kernel_size=3, padding='valid'))
  subsubnet.add(Activation('relu'))
  subsubnet.add(MaxPool1D(pool_size=4, strides=4))
  # ! ! !

  subsubnet.add(Conv1D(filters=256, kernel_size=3, padding='valid'))
  subsubnet.add(BatchNorm())
  subsubnet.add(Activation('relu'))
  subsubnet.add(Conv1D(filters=256, kernel_size=3, padding='valid'))
  subsubnet.add(BatchNorm())
  subsubnet.add(Activation('relu'))
  subsubnet.add(GlobalMaxPooling1D())
  
  # the net to process mfcc features
  subsubnet = subnet.add()
  subsubnet.add(Input(sample_shape=[dim[0], dim[1], 1], name='mfcc'))
  subsubnet.add(Conv2D(32, (4, 10), padding='same'))
  subsubnet.add(BatchNorm())
  subsubnet.add(Activation('relu'))
  subsubnet.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
  subsubnet.add(Dropout(th.mfcc_keep_prob))

  subsubnet.add(Conv2D(32, (4, 10), padding='same'))
  subsubnet.add(BatchNorm())
  subsubnet.add(Activation('relu'))
  subsubnet.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
  subsubnet.add(Dropout(th.mfcc_keep_prob))
  #
  subsubnet.add(Conv2D(32, (4, 10), padding='same'))
  subsubnet.add(BatchNorm())
  subsubnet.add(Activation('relu'))
  subsubnet.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
  subsubnet.add(Dropout(th.mfcc_keep_prob))

  subsubnet.add(Conv2D(32, (4, 10), padding='same'))
  subsubnet.add(BatchNorm())
  subsubnet.add(Activation('relu'))
  subsubnet.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
  subsubnet.add(Dropout(th.mfcc_keep_prob))
  subsubnet.add(Flatten())


  subsubnet = subnet.add()
  subsubnet.add(Input(sample_shape=[1025, 63, 1], name='stft'))
  subsubnet.add(Conv2D(32, (4, 10), padding='same'))
  subsubnet.add(BatchNorm())
  subsubnet.add(Activation('relu'))
  subsubnet.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
  subsubnet.add(Dropout(th.mfcc_keep_prob))

  subsubnet.add(Conv2D(32, (4, 10), padding='same'))
  subsubnet.add(BatchNorm())
  subsubnet.add(Activation('relu'))
  subsubnet.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
  subsubnet.add(Dropout(th.mfcc_keep_prob))
  #
  subsubnet.add(Conv2D(32, (4, 10), padding='same'))
  subsubnet.add(BatchNorm())
  subsubnet.add(Activation('relu'))
  subsubnet.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
  subsubnet.add(Dropout(th.mfcc_keep_prob))

  subsubnet.add(Conv2D(32, (4, 10), padding='same'))
  subsubnet.add(BatchNorm())
  subsubnet.add(Activation('relu'))
  subsubnet.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
  subsubnet.add(Dropout(th.mfcc_keep_prob))
  subsubnet.add(Flatten())

  model.add(Dropout(th.concat_keep_prob))
  model.add(Linear(output_dim=128))
  model.add(BatchNorm())
  model.add(Activation('relu'))
  # model.add(Dropout(th.concat_keep_prob))
  #
  model.add(Linear(output_dim=64))
  model.add(BatchNorm())
  model.add(Activation('relu'))
  # model.add(Dropout(th.concat_part_prob))
  
  # Add output layer
  model.add(Linear(output_dim=41))
  model.add(Activation('softmax'))

  # Build model
  optimizer = tf.train.AdamOptimizer(learning_rate=th.learning_rate)
  model.build(optimizer=optimizer)

  return model

def multinput_mlp(th):
  assert isinstance(th, Config)
  model = Classifier(mark=th.mark)
  
  def data_dim(sample_rate=16000, duration=2, n_mfcc=50):
    audio_length = sample_rate * duration
    dim = (n_mfcc, 1 + int(np.floor(audio_length / 512)), 1)
    return dim
  dim = data_dim()
  
  # Add hidden layers
  subnet = model.add(inter_type=model.CONCAT)
  
  subsubnet = subnet.add()
  subsubnet.add(Input(sample_shape=[32000, 1]))
  subsubnet.add(Linear(output_dim=512))
  subsubnet.add(Activation('relu'))
  subsubnet.add(Linear(output_dim=256))
  subsubnet.add(Activation('relu'))
  
  subsubnet = subnet.add()
  subsubnet.add(Input(sample_shape=[dim[0], dim[1], 1], name='mfcc'))
  subsubnet.add(Conv2D(32, (4, 10), padding='same'))
  subsubnet.add(BatchNorm())
  subsubnet.add(Activation('relu'))
  subsubnet.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
  subsubnet.add(Dropout(0.8))

  # subsubnet.add(Conv2D(32, (4, 10), padding='same'))
  # subsubnet.add(BatchNorm())
  # subsubnet.add(Activation('relu'))
  # subsubnet.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
  # subsubnet.add(Dropout(0.8))

  subsubnet.add(Conv2D(32, (4, 10), padding='same'))
  subsubnet.add(BatchNorm())
  subsubnet.add(Activation('relu'))
  subsubnet.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
  subsubnet.add(Dropout(0.7))

  subsubnet.add(Flatten())

  model.add(Linear(output_dim=128))
  model.add(BatchNorm())
  model.add(Activation('relu'))

  model.add(Linear(output_dim=64))
  model.add(BatchNorm())
  model.add(Activation('relu'))
  
  model.add(Linear(output_dim=64))
  model.add(BatchNorm())
  model.add(Activation('relu'))

  # Add output layer
  model.add(Linear(output_dim=41))
  model.add(Activation('softmax'))

  # Build model
  optimizer = tf.train.AdamOptimizer(learning_rate=th.learning_rate)
  model.build(optimizer=optimizer)

  return model

def res_00(th):
  assert isinstance(th, Config)
  model = Classifier(mark=th.mark)
  
  def data_dim(sample_rate=16000, duration=2, n_mfcc=50):
    audio_length = sample_rate * duration
    dim = (n_mfcc, 1 + int(np.floor(audio_length / 512)), 1)
    return dim
  dim = data_dim()
  
  # Add hidden layers
  subnet = model.add(inter_type=model.CONCAT)
  # the net to process raw data
  subsubnet = subnet.add()
  # subsubnet.add(Input(sample_shape=[32000, 1], name='raw_data'))
  subsubnet.add(Input(sample_shape=[32000, 1]))
  subsubnet.add(Conv1D(filters=16, kernel_size=9, padding='valid'))
  subsubnet.add(BatchNorm())
  subsubnet.add(Activation('relu'))
  subsubnet.add(Conv1D(filters=16, kernel_size=9, padding='valid'))
  subsubnet.add(BatchNorm())
  subsubnet.add(Activation('relu'))
  subsubnet.add(MaxPool1D(pool_size=16, strides=16))
  subsubnet.add(Dropout(th.raw_keep_prob))
  
  subsubnet.add(Conv1D(filters=32, kernel_size=3, padding='valid'))
  subsubnet.add(Activation('relu'))
  subsubnet.add(Conv1D(filters=32, kernel_size=3, padding='valid'))
  subsubnet.add(Activation('relu'))
  subsubnet.add(MaxPool1D(pool_size=4, strides=4))
  subsubnet.add(Dropout(th.raw_keep_prob))
  
  subsubnet.add(Conv1D(filters=32, kernel_size=3, padding='valid'))
  subsubnet.add(Activation('relu'))
  subsubnet.add(Conv1D(filters=32, kernel_size=3, padding='valid'))
  subsubnet.add(Activation('relu'))
  subsubnet.add(MaxPool1D(pool_size=4, strides=4))
  
  subsubnet.add(Conv1D(filters=256, kernel_size=3, padding='valid'))
  subsubnet.add(BatchNorm())
  subsubnet.add(Activation('relu'))
  subsubnet.add(Conv1D(filters=256, kernel_size=3, padding='valid'))
  subsubnet.add(BatchNorm())
  subsubnet.add(Activation('relu'))
  subsubnet.add(GlobalMaxPooling1D())
  
  # the net to process mfcc features
  subsubnet = subnet.add()
  subsubnet.add(Input(sample_shape=[dim[0], dim[1], 1], name='mfcc'))
  subsubnet.add(Conv2D(32, (4, 10), padding='same'))
  subsubnet.add(BatchNorm())
  subsubnet.add(Activation('relu'))
  subsubnet.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
  subsubnet.add(Dropout(th.mfcc_keep_prob))
  
  net = subsubnet.add(ResidualNet())
  net.add(Conv2D(32, (4, 10), padding='same'))
  net.add(BatchNorm())
  net.add_shortcut()
  subsubnet.add(Activation('relu'))
  subsubnet.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
  subsubnet.add(Dropout(th.mfcc_keep_prob))
  #
  net = subsubnet.add(ResidualNet())
  net.add(Conv2D(32, (4, 10), padding='same'))
  net.add(BatchNorm())
  net.add_shortcut()
  subsubnet.add(Activation('relu'))
  subsubnet.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
  subsubnet.add(Dropout(th.mfcc_keep_prob))
  
  net = subsubnet.add(ResidualNet())
  net.add(Conv2D(32, (4, 10), padding='same'))
  net.add(BatchNorm())
  net.add_shortcut()
  subsubnet.add(Activation('relu'))
  subsubnet.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
  subsubnet.add(Dropout(th.mfcc_keep_prob))

  net = subsubnet.add(ResidualNet())
  net.add(Conv2D(32, (4, 10), padding='same'))
  net.add(BatchNorm())
  net.add_shortcut()
  subsubnet.add(Activation('relu'))
  subsubnet.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
  subsubnet.add(Dropout(th.mfcc_keep_prob))
  subsubnet.add(Flatten())

  subsubnet.add(Dropout(th.concat_keep_prob))
  model.add(Linear(output_dim=128))
  model.add(BatchNorm())
  model.add(Activation('relu'))
  #
  model.add(Linear(output_dim=64))
  model.add(BatchNorm())
  model.add(Activation('relu'))
  
  # Add output layer
  model.add(Linear(output_dim=41))
  model.add(Activation('softmax'))
  
  # Build model
  optimizer = tf.train.AdamOptimizer(learning_rate=th.learning_rate)
  model.build(optimizer=optimizer)
  
  return model

def multinput_ver_only(th):
  assert isinstance(th, Config)
  # model = Classifier(mark=th.mark)
  model = Classifier_Gpat(mark=th.mark)
  
  def data_dim(sample_rate=16000, duration=2, n_mfcc=50):
    audio_length = sample_rate * duration
    dim = (n_mfcc, 1 + int(np.floor(audio_length / 512)), 1)
    return dim
  dim = data_dim()
  
  # Add hidden layers
  subnet = model.add(inter_type=model.CONCAT)
  subsubnet = subnet.add()
  # the net to process raw data
  subsubnet.add(Input(sample_shape=[32000, 1]))
  def conv_bn_relu(filters, twod=True, bn=True):
    if twod:
      subsubnet.add(Conv2D(filters=filters,
                           kernel_size=(4, 10), padding='same'))
    else:
      subsubnet.add(Conv1D(filters=filters,
                           kernel_size=9, padding='valid'))
    if bn: subsubnet.add(BatchNorm())
    subsubnet.add(Activation('relu'))
  def maxpool_drop(pool_size, strides, twod=True, drop=True):
    if twod:
      subsubnet.add(MaxPool2D(pool_size=pool_size,
                              strides=strides))
    else:
      subsubnet.add(MaxPool1D(pool_size=pool_size,
                              strides=strides))
    if drop: subsubnet.add(Dropout(th.raw_keep_prob))
  for _ in range(th.raw_std_blocks):
    conv_bn_relu(32, twod=False, bn=True)
    maxpool_drop(pool_size=16, strides=16, drop=True, twod=False)
  conv_bn_relu(filters=32, twod=False, bn=True)
  subsubnet.add(Dropout(th.raw_keep_prob))
  subsubnet.add(GlobalMaxPooling1D())
  
  # the net to process mfcc features
  subsubnet = subnet.add()
  subsubnet.add(Input(sample_shape=[dim[0], dim[1], 1], name='mfcc'))
  for _ in range(th.mfcc_std_blocks):
    conv_bn_relu(filters=th.mfcc_cnn_filters, bn=True)
    maxpool_drop(pool_size=(2, 2), strides=(2, 2))
  subsubnet.add(Flatten())
  
  model.add(Dropout(th.concat_keep_prob))
  def linear_bn_relu(units, bn=True):
    model.add(Linear(output_dim=units))
    if bn:model.add(BatchNorm())
    model.add(Activation('relu'))
  
  for _ in range(th.concat_std_blocks):
    linear_bn_relu(th.concat_part_units)
  
  # Add output layer
  model.add(Linear(output_dim=41))
  model.add(Activation('softmax'))

  # Build model
  optimizer = tf.train.AdamOptimizer(learning_rate=th.learning_rate)
  model.build(optimizer=optimizer)

  return model


