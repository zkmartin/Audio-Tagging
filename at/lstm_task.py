import tensorflow as tf
import core
from tframe import console
import model_lib as models


def main(_):
  console.start('LSTM task')

  # Configurations
  th = core.th
  th.model = models.lstm_test
  # th.model = models.lstm
  th.num_blocks = 1
  th.memory_depth = 3
  th.hidden_dim = 100

  th.epoch = 50000
  th.learning_rate = 1e-4
  th.batch_size = 512
  th.num_steps = 100
  th.val_preheat = 500
  th.validation_per_round = 0
  th.validate_cycle = 0
  th.print_cycle = 2

  th.train = True
  th.smart_train = False
  th.max_bad_apples = 4
  th.lr_decay = 0.5

  th.save_model = True
  th.overwrite = True
  th.export_note = True
  th.summary = False
  th.monitor = False

  description = ''
  th.mark = '{}x{}{}'.format(th.num_blocks, th.memory_depth, description)

  core.activate()


if __name__ == '__main__':
  tf.app.run()