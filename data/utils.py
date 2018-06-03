import sys
import zipfile
import os
import shutil
import numpy as np

def get_sizeof_series(data):
	"""return a memory size of a sequence"""
	memo = 0
	for i in range(len(data)):
		memo += sys.getsizeof(data[i])
	memo = memo / (1024**2)
	return memo

def un_zip(file_name, current_path=False, extract_path=None):
	"""unzip zip file"""
	zip_file = zipfile.ZipFile(file_name)
	if current_path is True:
		_extract_path = os.path.dirname(os.path.abspath(file_name))
	else:
		_extract_path = extract_path
		if os.path.isdir(extract_path):
			pass
		else:
			os.mkdir(extract_path)
		
	for names in zip_file.namelist():
		zip_file.extract(names, _extract_path + '/')
	zip_file.close()

def makdir(path):
	if os.path.isdir(path):
		shutil.rmtree(path)
		os.mkdir(path)
	else:
		os.mkdir(path)

if __name__ == '__main__':
	a = np.array([1, 2, 3], dtype=np.float64)
	print(sys.getsizeof(a[0]))
