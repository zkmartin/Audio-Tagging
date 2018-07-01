from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe.models.sl.classifier import Classifier
from tframe.models.feedforward import Feedforward

from tframe.core.decorators import with_graph

class Classifier_Gpat(Classifier):
	
	def __init__(self, mark=None, net_type=Feedforward):
		super().__init__(mark=mark, net_type=net_type)
		
	@with_graph
	def classify(self, data, batch_size=None, extractor=None):
		probs = self._batch_evaluation(
			self._probabilities.tensor, data, batch_size, extractor)
		
		return probs
