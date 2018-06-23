import tensorflow as tf
import core
from tframe import console
import model_lib as models


def main(_):
  console.start('CNN task ')

  # Configurations
  th = core.th
  # th.model = models.conv_test
  th.model = models.conv_2d_test
  th.actype1 = 'relu'
  th.patience = 100

  th.epoch = 5000
  th.learning_rate = 1e-3
  th.batch_size = 64
  th.validation_per_round = 1
  th.print_cycle = 10
  th.shuffle = False

  # th.train = False
  th.smart_train = True
  th.max_bad_apples = 4
  th.lr_decay = 0.6

  th.save_model = True
  th.overwrite = True
  th.export_note = True
  th.summary = True
  th.monitor = False
  
  th.allow_growth = True
  # th.gpu_memory_fraction =
  description = 'conv_2d_add_noise'
  th.mark = 'cnn_{}x{}{}'.format(th.hidden_dim, th.num_blocks, description)

  core.activate()


if __name__ == '__main__':
  tf.app.run()