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
from data_utils_temp import load_data, evaluate, load_demo_data, load_test_data
from tframe.data.dataset import DataSet
from tframe.config import Flag
from at.data_utils.gpat import GPAT


from_root = lambda path: os.path.join(ROOT, path)

class GpatHub(SmartTrainerHub):
  fold = Flag.integer(0, 'cross-validation fold id', name='fold')
  raw_keep_prob = Flag.float(0.7, 'raw part dropout keep prob',
                             name='raw_keep_prob')
  mfcc_keep_prob = Flag.float(0.7, 'mfcc part dropout keep prob')
  concat_keep_prob = Flag.float(0.9, 'concat part dropout keep prob')
  rand_pos = Flag.boolean(False, 'if rand position or not')
  test_all = Flag.boolean(False, 'Whether test all sequences or not')

GpatHub.register()

# th = SmartTrainerHub(as_global=True)
th = GpatHub(as_global=True)

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

  if th.rand_pos:th.mark += '_rand_pos'
  # Load data
  path = '../data/processed_data/'
  train_set, val_set, test_set = GPAT.load_data_set(path, th, random_pos=th.rand_pos,
                                                   test_all=th.test_all)
  
  # Train or evaluate
  if th.train:
    model.train(train_set, validation_set=val_set, trainer_hub=th)
  else:
    model.launch_model(overwrite=False)
    # evaluate(model, test_set, th, scores=True)
    GPAT.evaluate(model, test_set, th)
	  
  # End
  console.end()