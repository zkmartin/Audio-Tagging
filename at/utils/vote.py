import numpy as np
from tframe import console
from collections import Counter


class VoteEngine(object):
	def __init__(self, P):
		assert isinstance(P, np.ndarray) and len(P.shape) == 2
		self._P = np.fliplr(np.sort(P, axis=1))
		self._C = np.fliplr(np.argsort(P, axis=1))
	
	# region : Public Methods
	
	def top_k(self, k):
		assert isinstance(k, int) and np.isscalar(k)
		result = []
		for i in range(k):
			index, indices_to_shift = self._top_1()
			result.append(index)
			self._shift(indices_to_shift)
		
		return result

	# endregion : Public Methods

	# region : Private Methods
	
	def _top_1(self):
		C, P = list(self._C[:, 0]), list(self._P[:, 0])
		counter = Counter(C)
		probs = [sum([p for p, c in zip(P, C) if c == cls])
		         for cls in counter.keys()]
		# Sort using counts
		best_labels = [c for c in counter.keys()
		               if counter[c] == counter.most_common(1)[0][1]]
		# Sort using probs
		if len(best_labels) > 1:
			scores = {c: p for c, p in zip(counter.keys(), probs) if c in best_labels}
			best_labels = sorted(best_labels, key=lambda c: scores[c], reverse=True)
			
		best_label = best_labels[0]
		indices = [i for i, c in enumerate(C) if c == best_label]
		
		return best_label, indices
			
		
		# labels_temp = list(self._P[:, 0])
		# count = Counter(labels_temp)
		# label_keys, labels_num = self.get_keys_values(count)
		# label_candidates = []
		# if labels_num[0] > labels_num[1]:
		# 	index = label_keys[0]
		# 	indices_to_shift = self.condition_index(self._P[:, 0],
		# 	                                        self._P[:, 0] == index)
		# else:
		# 	for i, num in enumerate(labels_num):
		# 		if num == labels_num[0]:
		# 			label_candidates.append(label_keys[i])
		# 	candidates_indices = []
		# 	for cand in label_candidates:
		# 		candidates_indices.append(self.condition_index(self._P[:, 0],
		# 		                                               self._P[:, 0] == cand))
		# 	candidates_indices_ravel = []
		# 	for indice in candidates_indices:
		# 		for ind in indice:
		# 			candidates_indices_ravel.append(ind)
		#
		# 	max_index = np.argsort(self._Q[:, 0][candidates_indices_ravel], axis=0)[-1]
		# 	dealer = max_index // len(label_candidates)
		# 	remainder = np.mod(max_index, len(label_candidates))
		# 	if remainder == 0:
		# 		index = label_candidates[dealer]
		# 		indices_to_shift = candidates_indices[dealer]
		# 	else:
		# 		index = label_candidates[dealer + 1]
		# 		indices_to_shift = candidates_indices[dealer + 1]
		#
		# return (index, indices_to_shift)

	def _shift(self, i):
		self._P[i] = np.roll(self._P[i], -1, axis=1)
		self._C[i] = np.roll(self._C[i], -1, axis=1)

	# endregion : Private Methods

	# region: Static Methods
	
	@staticmethod
	def get_keys_values(dic):
		keys = []
		values = []
		for (key, value) in dic.items():
			keys.append(key)
			values.append(value)
		return keys, values
	
	@staticmethod
	def condition_index(data, mask):
		assist_list = np.arange(data.size)
		assist_list = np.reshape(assist_list, data.shape)
		condition_index = assist_list[mask]
		
		return condition_index.tolist()
	# endregion: Static Methods
	

if __name__ == '__main__':
	a = np.random.randint(0, 10, size=(6, 6))
	console.show_status('a = ')
	console.pprint(a)
	
	P = np.fliplr(np.sort(a, axis=1))
	console.show_status('P = ')
	console.pprint(P)
	
	Q = np.fliplr(np.argsort(a, axis=1))
	console.show_status('Q = ')
	console.pprint(Q)
	
	indices = [2, 3]
	max_index = np.argsort(Q[:, 0][indices], axis=0)[-1]
	
	ve = VoteEngine(a)
	result = ve.top_k(3)
	

