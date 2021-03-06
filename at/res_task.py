import tensorflow as tf
import core
from tframe import console
from tframe.utils.misc import mark_str as ms
import model_lib as models
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def main(_):
	console.start('Multinput task')
	
	# Configurations
	th = core.th
	th.job_dir = 'res_task'
	th.model = models.res_00
	th.actype1 = 'relu'
	
	th.epoch = 5000
	th.learning_rate = 1e-3
	th.batch_size = 32
	th.validation_per_round = 1
	th.val_batch_size = th.batch_size
	th.print_cycle = 20
	th.patience = 100
	th.shuffle = True
	
	# th.train = False
	th.smart_train = False
	th.max_bad_apples = 4
	th.lr_decay = 0.6
	th.rand_over_classes = False
	
	th.save_model = True
	th.overwrite = True
	th.export_note = True
	th.summary = True
	th.monitor = False
	
	th.allow_growth = False
	th.gpu_memory_fraction = 0.4

	th.raw_keep_prob = 0.9
	th.mfcc_keep_prob = 0.7
	th.concat_keep_prob = 0.9
	th.fold = 0
	# th.shuffle = False

	th.rand_pos = True
	
	th.visible_gpu_id = '0'
	
	# description = 'cnn_raw_data_mfcc_random_rand'
	description = 'raw_data_mfcc_dropout_{}_{}'.format(
		th.mfcc_keep_prob, th.concat_keep_prob)
	th.mark = 'cnn_{}'.format(description)
	
	export_false = True
	core.activate()


if __name__ == '__main__':
	tf.app.run()
