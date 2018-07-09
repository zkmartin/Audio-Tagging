import sys
import zipfile
import os
import shutil
import pickle
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
		# shutil.rmtree(path)
		# os.mkdir(path)
		pass
	else:
		os.mkdir(path)
		
def check_dir(paths):
	if isinstance(paths, str):paths = [paths]
	if len(paths) == 1:makdir(paths[0])
	path = paths[0]
	for i in range(len(paths)):
		makdir(path)
		if i == len(paths) - 1: break
		path = os.path.join(path, paths[i + 1])

def pickle_data(data, file_name):
	with open(file_name, 'wb') as f:
		pickle.dump(data, f)
	
def read_pickle_data(file_name):
	with open(file_name, 'rb') as f:
		data = pickle.load(f)
	return data

	
if __name__ == '__main__':
	root_path = os.path.dirname(os.path.abspath(__file__))
	paths = [root_path, 'test1', 'test2']
	check_dir(paths)
