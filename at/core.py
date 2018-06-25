import sys, os
import numpy as np
ROOT = os.path.abspath(__file__)
# Specify the directory depth with respect to the root of your project here
# (The project root usually holds your data folder and has a depth of 0)
DIR_DEPTH = 1
for _ in range(DIR_DEPTH + 1):
  ROOT = os.path.dirname(ROOT)
  sys.path.insert(0, ROOT)
from tframe import console, SaveMode
from tframe.trainers import SmartTrainerHub
from data_utils import load_data, evaluate, load_demo_data, load_test_data
from tframe.data.dataset import DataSet
from data_utils import load_simulate_test_data


from_root = lambda path: os.path.join(ROOT, path)

th = SmartTrainerHub(as_global=True)
th.data_dir = from_root('data/original_data')
th.job_dir = from_root('at/')

th.allow_growth = False
th.gpu_memory_fraction = 0.4

th.save_mode = SaveMode.ON_RECORD
th.warm_up_thres = 1
th.at_most_save_once_per_round = True
th.num_classes = 41

th.early_stop = True
th.idle_tol = 10


def activate():
  assert callable(th.model)
  model = th.model(th)
  # assert isinstance(model, )

  # Load data

  # path = '../data/original_data/traindata_fs_16000'fd
  # path = '../data/processed_data/traindata_fs_16000_all.tfd'
  # csv_path = '../data/original_data/train.csv'
  # train_set, val_set, test_set, raw_val_set = load_data(path, csv_path)
  if th.train:
    path = '../data/original_data/audio_train/'
    train_set, val_set = load_demo_data(path)
  else:
    path = '../data/original_data/audio_test/'
    test_set = load_test_data(path)
    
  # train_set2, val_set2, test_set2 = load_data(path2)
  # assert isinstance(train_set, DataSet)
  
  # Train or evaluate
  if th.train:
    model.train(train_set, validation_set=val_set, trainer_hub=th)
  else:
    model.launch_model(overwrite=False)
    evaluate(model, test_set, th, scores=True)
    # evaluate(model, val_set)
    # evaluate(model, test_set, plot=True)

  # End
  console.end()