from tframe import console
import time


# total = 1000
# start_time = time.time()
# for i in range(total):
# 	time.sleep(0.01)
# 	console.print_progress(i + 1, total, start_time)
	
# console.show_status('Done!!!!')


import os
import numpy as np
from at.data_utils.gpat import GPAT

class a(object):
	test1 = 1

class b(a):
	test2 = 1
	
c = b()
print(isinstance(c, a))



