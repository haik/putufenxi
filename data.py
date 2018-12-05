import scipy.io as sio
import h5py
import numpy as np

def loaddata(num):
	file = 'GaussianData.mat'
	# file = 'ringData.mat'
	data = sio.loadmat(file)
	matrix = np.array(data['Dataset'])
	print(matrix.shape)
	return matrix[:num]

def load_dot_mat():		
	# data_sets = ['be3', 'happy', 'hm', 'sp', 'tar']
	# num_classes = {'be3': 3,	'happy': 3,		'hm': 2,	'sp': 3,   	'tar': 6,}
	path = 'DB.mat'
	db_name = 'DB/' + 'be3'
	try:
		mat = sio.loadmat(path)
	except NotImplementedError:
		mat = h5py.File(path)
	return np.array(mat[db_name]).transpose()