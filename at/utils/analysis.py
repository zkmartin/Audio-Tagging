import os
import pandas as pd
from tframe import console

def compare_submmits(csv_path_1, csv_path_2):
	
	submission_1 = pd.read_csv(csv_path_1)
	submission_2 = pd.read_csv(csv_path_2)
	
	def get_labels_as_list(sub):
		labels = sub['label']
		labels_fir = [labels[i].split(' ')[0] for i in range(len(labels))]
		# labels_fir.append(labels[i])
		return labels_fir
	
	sub_1 = get_labels_as_list(submission_1)
	sub_2 = get_labels_as_list(submission_2)
	
	diff_indices = [ind for ind in range(len(sub_1))
	                if sub_1[ind] != sub_2[ind]]
	
	return diff_indices

def show_data_info(inds):


if __name__ == '__main__':
	csv_path_1 = '../submission_1.csv'
	csv_path_2 = '../strictly_vote_submissions.csv'
	indices = compare_submmits(csv_path_1, csv_path_2)
	console.pprint(len(indices))
	
