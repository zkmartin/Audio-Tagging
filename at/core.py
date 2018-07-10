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
from at.data_utils.utils import pickle_data


from_root = lambda path: os.path.join(ROOT, path)

class GpatHub(SmartTrainerHub):
  fold = Flag.integer(0, 'cross-validation fold id', name='fold')
  raw_keep_prob = Flag.float(0.7, 'raw part dropout keep prob',
                             name='raw_keep_prob')
  mfcc_keep_prob = Flag.float(0.7, 'mfcc part dropout keep prob')
  concat_keep_prob = Flag.float(0.9, 'concat part dropout keep prob')
  rand_pos = Flag.boolean(False, 'if rand position or not')
  test_all = Flag.boolean(False, 'Whether test all sequences or not')
  val_on_train_set = Flag.boolean(False, 'Whether validate on tht training set')
  modify_train_ver_n = Flag.boolean(False, 'Whether to modify the unverified set')
  raw_std_blocks = Flag.integer(1, 'the standard blocks in the raw net')
  raw_cnn_filters = Flag.integer(32, 'the filters of the raw net')
  mfcc_std_blocks = Flag.integer(1, 'the standard blocks in the raw net')
  mfcc_cnn_filters = Flag.integer(32, 'the filters of the mfcc net')
  concat_std_blocks = Flag.integer(1, 'the standard blocks of the concat part')
  concat_part_units = Flag.integer(64, 'the concat part units')

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
  if th.rand_pos:th.mark += '_rand_pos'
  
  assert callable(th.model)
  model = th.model(th)
  # assert isinstance(model, )

  # Load data
  path = '../data/processed_data/'
  if not th.modify_train_ver_n:
    train_set, val_set, test_set, all_train_set, train_split = GPAT.load_data_set(
																											path, th,
																											random_pos=th.rand_pos,
																											test_all=th.test_all)
  else:
    test_set = GPAT.load_ver_n_data(path, id=1)
  # Train or evaluate
  if th.train and not th.val_on_train_set:
    model.train(train_set, validation_set=val_set, trainer_hub=th)
  if not th.train and not th.val_on_train_set:
    model.launch_model(overwrite=False)
    # evaluate(model, test_set, th, scores=True)
    if not th.modify_train_ver_n:
      GPAT.evaluate(model, test_set, th, save_prods=True)
    else:
      cor_inds, false_inds = model.evaluate_model(test_set,
                                                  batch_size=th.val_batch_size)
      
      pickle_data(cor_inds, os.path.join('./indices', th.mark + '.pkl'))
  if th.val_on_train_set:
    model.evaluate_model(val_set, batch_size=th.val_batch_size)
    
    
	  
  # End
  console.end()