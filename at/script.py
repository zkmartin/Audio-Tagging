from subprocess import call

task_name = 'multiinput_task.py'
folds = [1, 2]
visible_gpu_id = "0"

for var in folds:
	command = 'python ' + task_name + ' --fold {} --visible_gpu_id {}'.format(
		var, visible_gpu_id)
	call(command)