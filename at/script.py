import multiprocessing
import os
from subprocess import call
from data_utils.gpat import GPAT

task_name = 'multiinput_task.py'
folds = [0, 1]
visible_gpu_id = "0"
commands = []

for var in folds:
	command = 'python ' + task_name + ' --fold {} --gpu_id {}'.format(
		var, visible_gpu_id)
	commands.append(command)

p1 = multiprocessing.Process(target=os.system(commands[0]))
p2 = multiprocessing.Process(target=os.system(commands[1]))

p1.start()
p2.start()

p1.join()
p2.join()

print('All subprocesses done.')

# inds_path = './indices'
# data_path = '../data/processed_data'
# GPAT.modified_train_ver_n_data(inds_path, data_path, id=0)