import multiprocessing
import os
from data_utils.gpat import GPAT
from multiinput_task import main
import subprocess

task_name = 'basis_task.py'
folds = [0]
raw_std_blocks = [2, 3, 5]
raw_cnn_filters = [8, 16, 32]
mfcc_std_blocks = []
mfcc_cnn_filters = []
concat_std_blocks = []
concat_part_units = []
visible_gpu_id = "0"
commands = []

# Create commands
for i in range(len(raw_std_blocks)):
	for j in range(len(raw_cnn_filters)):
		command = 'python ' + task_name + ' --raw_std_blocks {} ' \
	                                  '--raw_cnn_filters {}' \
	                                  ' --mfcc_std_blocks {}' \
	                                  '--mfcc_cnn_filters {}' \
	                                  '--gpu_id {}'.format(
			raw_std_blocks[i], raw_cnn_filters[j],
			raw_std_blocks[i], raw_cnn_filters[j], visible_gpu_id)
		commands.append(command)
		
# Execute all the commands
p1 = subprocess.Popen(commands[0])
p2 = subprocess.Popen(commands[1])
p1.wait()
p2.wait()

# inds_path = './indices'
# data_path = '../data/processed_data'
# GPAT.modified_train_ver_n_data(inds_path, data_path, id=0)