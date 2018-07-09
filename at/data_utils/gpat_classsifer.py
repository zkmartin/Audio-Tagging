from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tframe.models.sl.classifier import Classifier
from tframe.models.feedforward import Feedforward
from tframe import DataSet

from tframe.core.decorators import with_graph
from data_utils.gpat import GPAT

from tframe.utils import console
from tframe.utils.misc import convert_to_dense_labels
from collections import Counter

class Classifier_Gpat(Classifier):
	
	def __init__(self, mark=None, net_type=Feedforward):
		super().__init__(mark=mark, net_type=net_type)
		
	
	@with_graph
	def evaluate_model(self, data, batch_size=None, extractor=None,
                     export_false=False, **kwargs):
		console.show_status('Evaluating classifier ...')
		assert isinstance(data, DataSet)
		
		preds = self.classify(data, batch_size=batch_size,
		                      extractor=GPAT.raw_extractor)
		# preds = GPAT.test_all_prods_op(data, preds)
		preds = np.argmax(preds, axis=-1)
		# targets = data.labels
		targets = data.targets
		# targets = np.reshape(targets, (targets.shape[0], -1))
		labels = convert_to_dense_labels(targets)
		false_indices = [ind for ind in range(len(preds))
		                 if preds[ind] != labels[ind]]
		correct_indices = [ind for ind in range(len(preds))
		                   if ind not in false_indices]
		assert len(false_indices) + len(correct_indices) == len(preds)
		false_labels = labels[false_indices]
		counter = Counter(false_labels)
		cou = counter.most_common(len(list(counter.keys())))
		
		false_samples = data[false_indices]
		short_samples = [arr for arr in false_samples.features if arr.size < 32000]
		
		false_samples_lengths = [data.lengths[i] for i in false_indices]
		less_audio_length = [false_samples_lengths[i]
		                     for i in range(len(false_samples_lengths))
		                     if false_samples_lengths[i] < 32000]
		
		console.show_status('total_num :')
		console.pprint(len(labels))
		console.show_status('False_labels_num:')
		console.pprint(len(false_labels))
		console.show_status('The false num of each label:')
		console.pprint(cou)
		console.show_status('Short samples num:')
		console.pprint(len(less_audio_length))
		
		return correct_indices, false_indices
		
	@with_graph
	def classify(self, data, batch_size=None, extractor=None):
		probs = self._batch_evaluation(
			self._probabilities.tensor, data, batch_size, extractor)
		
		return probs
	
if __name__ == '__main__':
	pass
