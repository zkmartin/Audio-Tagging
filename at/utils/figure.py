import numpy as np
import matplotlib.pyplot as plt

def plot_bar_diagram(data, *args, title=None, num_class=None):
	"""plot the bar_diagram of the data"""
	
	n_groups = len(data)
	if len(args) is not 0:
			pass
	fig, ax = plt.subplots()
	
	index = np.arange(n_groups)
	bar_width = 0.2
	
	opacity = 0.4
	error_config = {'ecolor': '0.3'}
	
	rects = plt.bar(index, data, bar_width,
	                 alpha=opacity,
	                 error_kw=error_config)
	plt.xlabel('Categories')
	plt.ylabel('False_num')
	plt.ylim(0, max(data) + 5)
	plt.title(title)
	# plt.xticks(index + 0.5*bar_width, ('A', 'B', 'C', 'D', 'E'))
	plt.xticks(index + 0.5*bar_width, (np.arange(n_groups)))
	plt.legend()
	
	plt.tight_layout()
	plt.show()
	
	
if __name__ == '__main__':
	
	means_men = (20, 35, 30, 35, 27)
	plot_bar_diagram(means_men)
	
