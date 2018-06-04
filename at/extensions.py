from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.layers.layer import Layer
from tframe.layers.layer import single_input

from tframe.utils import get_scale
from tframe.core.decorators import init_with_graph
from tframe.core.function import Function

from tensorflow.python.layers.pooling import MaxPooling1D as MaxPool1D_

class MaxPool1D(Layer, MaxPool1D_):
	full_name = 'maxpool1d'
	abbreviation = 'maxpool'
	
	@init_with_graph
	def __init__(self, *args, **kwargs):
		super(Function, self).__init__(*args, **kwargs)
		
	@single_input
	def __call__(self, input_=None, **kwargs):
		assert isinstance(input_, tf.Tensor)
		output = MaxPool1D_.__call__(self, input_, scope=self.full_name)
		self.neuron_scale = get_scale(output)
		return output
