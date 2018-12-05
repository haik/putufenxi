import numpy as np
import time
import pickle
from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt
from all_ss_module import *
from test_eigen_ss import *

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

if __name__ == '__main__':
	# data = loaddata(300)
	# n_samples = 200
	# noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
	# data, other = noisy_circles
	# print(data.shape)
	with open('data_400/l0_moons.data', 'rb') as f0:
		data = pickle.load(f0)
	with open('data_400/l1_moons.data', 'rb') as f1:
		l1 = pickle.load(f1)
	with open('data_400/l2_moons.data', 'rb') as f2:
		l2 = pickle.load(f2)

	size=len(l1)
	cluster_num = 2
	print(l1.shape, l2.shape)

	t7 = time.time()
	# print(np.max(l1), np.min(l1), np.max(l2), np.min(l2))
	V1, V2, T1, T2, toff, ton = LanczosTri(l1, l2, 80)
	t8 = time.time()
	print("Lanczos", t8-t7, toff, ton)
	# print("V", np.max(V1), np.min(V1), np.max(V1), np.min(V2))
	# print("T", np.max(T1), np.min(T1), np.max(T2), np.min(T2))

	# with open('data.v1', 'wb') as f1:
	# 	pickle.dump(V1, f1)
	# with open('data.v2', 'wb') as f2:
	# 	pickle.dump(V2, f2)

	# with open('data.t1', 'wb') as f1:
	# 	pickle.dump(T1, f1)
	# with open('data.t2', 'wb') as f2:
	# 	pickle.dump(T2, f2)

	# V = V1+V2
	# T = T1+T2
	# # print(T)
	# eigval, eigvec = np.linalg.eig(T)
	# # eigval, eigvec = computeEigVec(T)
	# # print(eigval)
	# # print(eigvec)
	# eigvec = np.dot(V, eigvec)

	# row = len(T1)
	# for i in range(row):
	# 	for j in range(row):
	# 		if j==(i-1) or j==i or j==(i+1):
	# 			T1[i,j] = 0
	# 			T2[i,j] = 0

	t9 = time.time()
	eigval1, eigval2, eigvec1, eigvec2, toff, ton = EigVecSs(T1, T2)
	t10 = time.time()
	print("Shift QR", t10-t9, toff, ton)
	eigvec1, eigvec2, toff, ton = MatMulss(V1, V2, eigvec1, eigvec2)
	eigval = eigval1 + eigval2
	eigval = np.diagonal(eigval)
	eigvec = eigvec1 + eigvec2
	# # print(eigvec.shape)
	
	eigval, eigvec = NewgetEigVec(eigval, eigvec, cluster_num)
	print(eigval)
	# print(eigvec)

	# refval, refvec = np.linalg.eig(l1+l2)
	# print(refval)

	clf = KMeans(n_clusters=cluster_num)
	s = clf.fit(eigvec)
	C = s.labels_
	centers = getCenters(data,C)
	plot(data,s.labels_,centers,cluster_num)

	
