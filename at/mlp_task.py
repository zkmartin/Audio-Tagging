import tensorflow as tf
import core
from tframe import console
import model_lib as models


def main(_):
  console.start('GPAT Classification task (MLP)')

  # Configurations
  th = core.th
  th.model = models.mlp
  th.num_blocks = 2
  th.hidden_dim = 500
  th.actype1 = 'relu'
  th.idle_tol = 30

  th.epoch = 500
  th.learning_rate = 1e-3
  th.batch_size = 64
  th.validation_per_round = 1
  th.print_cycle = 1
  th.shuffle = True

  # th.train = False
  th.smart_train = False
  th.max_bad_apples = 4
  th.lr_decay = 0.6

  th.save_model = True
  th.overwrite = True
  th.export_note = True
  th.summary = True
  th.monitor = False

  description = 'demo'
  th.mark = 'mlp_{}x{}{}'.format(th.hidden_dim, th.num_blocks, description)

  core.activate()


if __name__ == '__main__':
  tf.app.run()