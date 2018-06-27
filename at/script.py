from subprocess import call

task_name = 'multiinput_task.py'
folds = [1, 2]

for var in folds:
	command = 'python ' + task_name + ' --fold {}'.format(var)
	call(command)