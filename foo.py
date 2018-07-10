from tframe import console
import time


# total = 1000
# start_time = time.time()
# for i in range(total):
# 	time.sleep(0.01)
# 	console.print_progress(i + 1, total, start_time)
	
# console.show_status('Done!!!!')

import multiprocessing
import os, time, random


def long_time_task(name, num):
	print('Run task %s (%s)...' % (name, os.getpid()))
	start = time.time()
	time.sleep(random.random() * 3)
	end = time.time()
	print('Task %s runs %0.2f seconds.' % (name, (end - start)))


if __name__ == '__main__':
	print('Parent process %s.' % os.getpid())
	p1 = multiprocessing.Process(target=long_time_task, args=('p1', 1))
	p2 = multiprocessing.Process(target=long_time_task, args=('p2', 1))

	p1.start()
	p2.start()
	print('Waiting for all subprocesses done...')

	p1.join()
	p2.join()
	print('All subprocesses done.')



