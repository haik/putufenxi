# coding=utf-8
import scipy.io as sio
import h5py
import numpy as np
import time
from pytictoc import TicToc
from scipy.linalg import norm
import math
import random

from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt
import pickle

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

def binaryconvert(s):
	n = 32
	num1, num2 = s.shape
	u = np.empty((num1, num2, n), 'uint8')
	for i in range(n):
		u[:,:,i] = (s >> i) & 1
	return u

'''Bitwise operations'''
def TripleGenerate(numofrow, numofcol, numoftrip):
	triplet = np.random.randint(0, 2, (6, numofrow, numofcol, numoftrip))
	triplet[5] = ((triplet[0]+triplet[3]) * (triplet[1]+triplet[4]) - triplet[2]) % 2
	return triplet[0], triplet[1], triplet[2], triplet[3], triplet[4], triplet[5]

def BitMultiplyMatrix(u1, u2, v1, v2, ta1, tb1, tc1, ta2, tb2, tc2):
	u_a = (u1 - ta1 + u2 - ta2) % 2							# S1 <-> S2
	v_b = (v1 - tb1 + v2 - tb2) % 2
	z1 = (tc1 + u_a * tb1 + v_b * ta1)	% 2					# S1
	z2 = (tc2 + u_a * tb2 + v_b * ta2 + u_a * v_b) % 2		# S2
	return z1, z2

def BitAddition(u1, u2, v1, v2):			# input 10 based numbers
	n = 32
	num1 = u1.shape[0]
	num2 = u1.shape[1]
	krange = int(math.log(n, 2))

	t_off_21 = time.time()
	ta1, tb1, tc1, ta2, tb2, tc2 = TripleGenerate(num1, num2, n)
	t_off_22 = time.time()

	t_on_21 = time.time()
	p1 = u1 ^ v1
	p2 = u2 ^ v2
	s1, s2 = BitMultiplyMatrix(u1, u2, v1, v2, ta1, tb1, tc1, ta2, tb2, tc2)
	t_on_22 = time.time()

	t_off_31 = time.time()
	lm = 0
	for k in range(krange):
		lrange = 2**k
		mrange = n // (lrange+1)
		lm += lrange * mrange
	ta11, tb11, tc11, ta12, tb12, tc12 = TripleGenerate(num1, num2, lm)
	ta21, tb21, tc21, ta22, tb22, tc22 = TripleGenerate(num1, num2, lm)
	t_off_32 = time.time()

	t_on_31 = time.time()
	for k in range(krange):
		lrange = 2**k
		mrange = n // (lrange+1)
		for l in range(lrange):
			for m in range(mrange):
				klm = l * mrange + m
				pos1 = lrange + l + lrange*2*m
				pos2 = lrange - 1 + lrange*2*m
				if pos1 < n and pos2 < n:
					temp1, temp2 = BitMultiplyMatrix(p1[:,:,pos1], p2[:,:,pos1], s1[:,:,pos2], s2[:,:,pos2], ta11[:,:,1], tb11[:,:,1], tc11[:,:,1], ta12[:,:,1], tb12[:,:,1], tc12[:,:,1])
					s1[:,:,pos1] = s1[:,:,pos1] ^ temp1
					s2[:,:,pos1] = s2[:,:,pos1] ^ temp2
					p1[:,:,pos1], p2[:,:,pos1] = BitMultiplyMatrix(p1[:,:,pos1], p2[:,:,pos1], p1[:,:,pos2], p2[:,:,pos2], ta21[:,:,1], tb21[:,:,1], tc21[:,:,1], ta22[:,:,1], tb22[:,:,1], tc22[:,:,1])
	t_on_32 = time.time()

	t_on_41 = time.time()
	# w1 = np.empty((num1, num2, n), 'uint8')
	# w2 = np.empty((num1, num2, n), 'uint8')
	# w1[:,:,0] = u1[:,:,0] ^ v1[:,:,0]
	# w2[:,:,0] = u2[:,:,0] ^ v2[:,:,0]
	# w1[:,:,1:n] = u1[:,:,1:n] ^ v1[:,:,1:n] ^ s1[:,:,:(n-1)]
	# w2[:,:,1:n] = u2[:,:,1:n] ^ v2[:,:,1:n] ^ s2[:,:,:(n-1)]
	w1 = u1[:,:,n-1] ^ v1[:,:,n-1] ^ s1[:,:,n-2]
	w2 = u2[:,:,n-1] ^ v2[:,:,n-1] ^ s2[:,:,n-2]
	t_on_42 = time.time()

	t_off = t_off_22 - t_off_21 + t_off_32 - t_off_31
	t_on = t_on_22 - t_on_21 + t_on_32 - t_on_31 + t_on_42 - t_on_41
	return w1, w2, t_off, t_on

def BitExtractionMatrix(u1, u2):
	n = 32
	num1, num2 = u1.shape
	t_off_11 = time.time()
	r1 = np.random.randint(-2**(n-2),2**(n-2),(num1,num2))
	r2 = np.random.randint(-2**(n-2),2**(n-2),(num1,num2))
	rsum = r1 + r2
	q01 = np.random.randint(-2**(n-2),2**(n-2),(num1,num2))
	q02 = rsum ^ q01
	q1 = binaryconvert(q01)
	q2 = binaryconvert(q02)
	t_off_12 = time.time()
	
	''' T -> S1: r1, q1
		T -> S2: r2, q2 '''
	t_on_11 = time.time()
	u1 = u1 - r1
	u2 = u2 - r2
	v0b = u1 + u2
	v01 = np.random.randint(-2**(n-2),2**(n-2),(num1,num2))
	v02 = v0b ^ v01
	v1 = binaryconvert(v01)
	v2 = binaryconvert(v02)
	t_on_12 = time.time()

	w1, w2, t_off_0, t_on_0 = BitAddition(v1, v2, q1, q2)
	t_off = t_off_12 - t_off_11 + t_off_0
	t_on = t_on_12 - t_on_11 + t_on_0
	
	return w1, w2, t_off, t_on


################################Tensor operations###################################################

def BinaryConvertTensor(s):
	u = np.empty((num0, num1, num2, n), 'uint8')
	for i in range(n):
		u[:,:,:,i] = (s >> i) & 1
	return u

def TripleGenerateTensor(numoftensor, numofrow, numofcol, numoftrip):
	triplet = np.random.randint(0, 2, (6, numoftensor, numofrow, numofcol, numoftrip))
	triplet[5] = ((triplet[0]+triplet[3]) * (triplet[1]+triplet[4]) - triplet[2]) % 2
	return triplet[0], triplet[1], triplet[2], triplet[3], triplet[4], triplet[5]

def BitMultiplyTensor(u1, u2, v1, v2, ta1, tb1, tc1, ta2, tb2, tc2):
	u_a = (u1 - ta1 + u2 - ta2) % 2							# S1 <-> S2
	v_b = (v1 - tb1 + v2 - tb2) % 2
	z1 = (tc1 + u_a * tb1 + v_b * ta1)	% 2					# S1
	z2 = (tc2 + u_a * tb2 + v_b * ta2 + u_a * v_b) % 2		# S2
	return z1, z2

def BitAdditionTensor(u1, u2, v1, v2):			# input 10 based numbers
	krange = int(math.log(n, 2))

	t_off_21 = time.time()
	ta1, tb1, tc1, ta2, tb2, tc2 = TripleGenerateTensor(num0, num1, num2, n)
	t_off_22 = time.time()

	t_on_21 = time.time()
	p1 = u1 ^ v1
	p2 = u2 ^ v2
	s1, s2 = BitMultiplyTensor(u1, u2, v1, v2, ta1, tb1, tc1, ta2, tb2, tc2)
	t_on_22 = time.time()

	t_off_31 = time.time()
	lm = 0
	for k in range(krange):
		lrange = 2**k
		mrange = n // (lrange+1)
		lm += lrange * mrange
	ta11, tb11, tc11, ta12, tb12, tc12 = TripleGenerateTensor(num0, num1, num2, lm)
	ta21, tb21, tc21, ta22, tb22, tc22 = TripleGenerateTensor(num0, num1, num2, lm)
	t_off_32 = time.time()

	t_on_31 = time.time()
	for k in range(krange):
		lrange = 2**k
		mrange = n // (lrange+1)
		for l in range(lrange):
			for m in range(mrange):
				klm = l * mrange + m
				pos1 = lrange + l + lrange*2*m
				pos2 = lrange - 1 + lrange*2*m
				if pos1 < n and pos2 < n:
					temp1, temp2 = BitMultiplyTensor(p1[:,:,:,pos1], p2[:,:,:,pos1], s1[:,:,:,pos2], s2[:,:,:,pos2], ta11[:,:,:,1], tb11[:,:,:,1], tc11[:,:,:,1], ta12[:,:,:,1], tb12[:,:,:,1], tc12[:,:,:,1])
					s1[:,:,:,pos1] = s1[:,:,:,pos1] ^ temp1
					s2[:,:,:,pos1] = s2[:,:,:,pos1] ^ temp2
					p1[:,:,:,pos1], p2[:,:,:,pos1] = BitMultiplyTensor(p1[:,:,:,pos1], p2[:,:,:,pos1], p1[:,:,:,pos2], p2[:,:,:,pos2], ta21[:,:,:,1], tb21[:,:,:,1], tc21[:,:,:,1], ta22[:,:,:,1], tb22[:,:,:,1], tc22[:,:,:,1])
	t_on_32 = time.time()

	w1 = np.empty((num0, num1, num2, n), 'uint8')
	w2 = np.empty((num0, num1, num2, n), 'uint8')
	t_on_41 = time.time()
	w1[:,:,:,0] = u1[:,:,:,0] ^ v1[:,:,:,0]
	w2[:,:,:,0] = u2[:,:,:,0] ^ v2[:,:,:,0]
	w1[:,:,:,1:n] = u1[:,:,:,1:n] ^ v1[:,:,:,1:n] ^ s1[:,:,:,:(n-1)]
	w2[:,:,:,1:n] = u2[:,:,:,1:n] ^ v2[:,:,:,1:n] ^ s2[:,:,:,:(n-1)]
	t_on_42 = time.time()

	t_off = t_off_22 - t_off_21 + t_off_32 - t_off_31
	t_on = t_on_22 - t_on_21 + t_on_32 - t_on_31 + t_on_42 - t_on_41
	return w1, w2, t_off, t_on

def BitExtractionTensor(u1, u2):
	t_off_11 = time.time()
	r1 = np.random.randint(-2**(n-2),2**(n-2),(num0,num1,num2))
	r2 = np.random.randint(-2**(n-2),2**(n-2),(num0,num1,num2))
	rsum = r1 + r2
	q01 = np.random.randint(-2**(n-2),2**(n-2),(num0,num1,num2))
	q02 = rsum ^ q01
	q1 = BinaryConvertTensor(q01)
	q2 = BinaryConvertTensor(q02)
	t_off_12 = time.time()
	
	''' T -> S1: r1, q1
		T -> S2: r2, q2 '''
	t_on_11 = time.time()
	u1 = u1 - r1
	u2 = u2 - r2
	v0b = u1 + u2
	v01 = np.random.randint(-2**(n-2),2**(n-2),(num0,num1,num2))
	v02 = v0b ^ v01
	v1 = BinaryConvertTensor(v01)
	v2 = BinaryConvertTensor(v02)
	t_on_12 = time.time()

	w1, w2, t_off_0, t_on_0 = BitAdditionTensor(v1, v2, q1, q2)

	t_off = t_off_12 - t_off_11 + t_off_0
	t_on = t_on_12 - t_on_11 + t_on_0
	
	return w1[:,:,:,n-1], w2[:,:,:,n-1], t_off, t_on


#################################'''Vector operations'''#############################################

def TripleGenerate2n(numofrow):
	triplet = np.random.randint(0, 2**(l-2), (6, numofrow))
	triplet[5] = (triplet[0]+triplet[3]) * (triplet[1]+triplet[4]) - triplet[2]
	return triplet[0], triplet[1], triplet[2], triplet[3], triplet[4], triplet[5]

def MultiplyMatrix2n(u1, u2, v1, v2, ta1, tb1, tc1, ta2, tb2, tc2):
	u_a = u1 - ta1 + u2 - ta2							# S1 <-> S2
	v_b = v1 - tb1 + v2 - tb2
	z1 = tc1 + u_a * tb1 + v_b * ta1					# S1
	z2 = tc2 + u_a * tb2 + v_b * ta2 + u_a * v_b		# S2
	return z1, z2	

################################'''Matrix operations'''##########################################
def TripleMatrix(numofrow, numofcol):
	triplet = np.random.randint(0, 2, (6, numofrow, numofcol))
	triplet[5] = (triplet[0]+triplet[3]) * (triplet[1]+triplet[4]) - triplet[2]
	return triplet[0], triplet[1], triplet[2], triplet[3], triplet[4], triplet[5]

def MultiplyMatrix(u1, u2, v1, v2):
	dim1 = u1.shape[0]
	dim2 = u1.shape[1]
	trip = TripleMatrix(dim1, dim2)
	z1, z2 = MultiplyMatrix2n(u1, u2, v1, v2, trip[0], trip[1], trip[2], trip[3], trip[4], trip[5])
	return z1, z2	

################################'''Tensor operations'''##########################################
def TripleTensor(numoftensor, numofrow, numofcol):
	triplet = np.random.randint(0, 10, (6, numoftensor, numofrow, numofcol))
	triplet[5] = (triplet[0]+triplet[3]) * (triplet[1]+triplet[4]) - triplet[2]
	return triplet[0], triplet[1], triplet[2], triplet[3], triplet[4], triplet[5]

def MultiplyTensor(u1, u2, v1, v2):
	dim0 = u1.shape[0]
	dim1 = u1.shape[1]
	dim2 = u1.shape[2]
	trip = TripleTensor(dim0, dim1, dim2)
	z1, z2 = MultiplyMatrix2n(u1, u2, v1, v2, trip[0], trip[1], trip[2], trip[3], trip[4], trip[5])
	return z1, z2

####################################################################################################
def NumProductss(u1, u2, v1, v2):
	trip = TripleGenerate2n(1)
	z1, z2 = MultiplyMatrix2n(u1, u2, v1, v2, trip[0], trip[1], trip[2], trip[3], trip[4], trip[5])
	return z1, z2

def InnerProductss(a1, a2, b1, b2):  		# 求向量a的各元素的平方和，a=a1+a2
	dim = a1.shape
	trip = TripleGenerate2n(dim[0])
	product1, product2 = MultiplyMatrix2n(a1, a2, b1, b2, trip[0], trip[1], trip[2], trip[3], trip[4], trip[5])
	return np.sum(product1), np.sum(product2)

def DotProductss(a1, a2, b1, b2):
	dim = b1.shape
	c1 = np.repeat(a1, dim)
	c2 = np.repeat(a2, dim)
	trip = TripleGenerate2n(dim[0])
	product1, product2 = MultiplyMatrix2n(c1, c2, b1, b2, trip[0], trip[1], trip[2], trip[3], trip[4], trip[5])
	return product1, product2

def divss(b1, b2):
	if abs(b1+b2) < 0.0001:
		n1 = 0
		n2 = 0
	else:
		n1 = 0.5 / 100000
		n2 = 0.5 / 100000
		d1 = b1 / 100000
		d2 = b2 / 100000
		count = 0
		while True:
			count += 1
			f1 = 1 - d1
			f2 = 1 - d2
			n1, n2 = NumProductss(f1, f2, n1, n2)
			d1, d2 = NumProductss(f1, f2, d1, d2)
			if abs(d1+d2-1) < 0.00001:
				break
	return n1, n2

def sqrtss(a1, a2):
	x11 = a1
	x12 = a2
	x21 = a1 / 2
	x22 = a2 / 2
	count = 0
	while abs(x11+x12-x21-x22)> 0.0001:
		count += 1
		x11 = x21
		x12 = x22
		div_x11, div_x12 = divss(x11, x12)
		# print("div-count", countdiv)
		mul_x11, mul_x12 = NumProductss(a1, a2, div_x11, div_x12)
		x21 = (x11 + mul_x11) / 2
		x22 = (x12 + mul_x12) / 2
	# print("sqrt-count", count)
	return x11, x12

def isqrtss(a1, a2):
	b1, b2 = sqrtss(a1, a2)
	c1, c2 = divss(b1, b2)
	return c1, c2

def qrGSss(matrix1, matrix2):  		# Gram-Schimidt orthogonalization
	size = len(matrix1)
	Q1 = np.zeros((size, size))
	Q2 = np.zeros((size, size))
	R1 = np.zeros((size, size))
	R2 = np.zeros((size, size))
	for i in range(size):
		u1 = matrix1[:,i]
		u2 = matrix2[:,i]
		for j in range(i):
			R1[j,i], R2[j,i] = InnerProductss(Q1[:,j], Q2[:,j], u1, u2)
			t1, t2 = DotProductss(R1[j,i], R2[j,i], Q1[:,j], Q2[:,j])
			u1 = u1 - t1
			u2 = u2 - t2
		t3, t4 = InnerProductss(u1, u2, u1, u2)
		if (t3+t4) < 0.0001:
			Q1[:,i] = np.zeros(size)
			Q2[:,i] = np.zeros(size)
		else:
			R1[i,i], R2[i,i] = sqrtss(t3, t4)
			Rtemp1, Rtemp2 = isqrtss(t3, t4)
			Q1[:,i], Q2[:,i] = DotProductss(Rtemp1, Rtemp2, u1, u2)
	return Q1, Q2, R1, R2

def qrss(matrix01, matrix02):		# Householder transformation
	matrix1 = matrix01
	matrix2 = matrix02
	size = len(matrix1)
	Q1 = np.eye(size)
	Q2 = np.zeros((size, size))
	for i in range(size):
		temp1 = np.zeros((size-i))
		temp2 = np.zeros((size-i))
		x1 = matrix1[i:,i]
		x2 = matrix2[i:,i]
		s11, s12 = InnerProductss(x1, x2, x1, x2)
		alpha1, alpha2 = sqrtss(s11, s12)
		temp1[0] = alpha1
		temp2[0] = alpha2
		u1 = x1 - temp1
		u2 = x2 - temp2
		s21, s22 = InnerProductss(u1, u2, u1, u2)
		if (s21+s22)<0.0001:
			v1 = np.zeros((size-i, 1))
			v2 = np.zeros((size-i, 1))
		else:
			norm1, norm2 = isqrtss(s21, s22)
			v1, v2 = DotProductss(norm1, norm2, u1, u2)
			v1 = v1.reshape(size-i, 1)
			v2 = v2.reshape(size-i, 1)

		vvt1, vvt2 = MatMulss(v1, v2, v1.T, v2.T)
		Qt1 = np.eye(size)
		Qt2 = np.zeros((size, size))
		Qt1[i:, i:] = np.eye(size-i) - 2 * vvt1
		Qt2[i:, i:] = -2 * vvt2
		Q1, Q2 = MatMulss(Qt1, Qt2, Q1, Q2)
		matrix1, matrix2 = MatMulss(Q1, Q2, matrix01, matrix02)

	Q1 = Q1.T
	Q2 = Q2.T
	return Q1, Q2, matrix1, matrix2

def MatMulss(A1, A2, B1, B2):
	dim1 = A1.shape[0]
	dim2 = B1.shape[1]
	C1 = np.repeat(A1, dim2, axis=0)
	C2 = np.repeat(A2, dim2, axis=0)
	D1 = np.tile(B1.T, (dim1, 1))
	D2 = np.tile(B2.T, (dim1, 1))
	E1, E2 = MultiplyMatrix(C1, C2, D1, D2)
	F1 = np.sum(E1, axis=1)
	F2 = np.sum(E2, axis=1)
	F1 = F1.reshape((dim1,dim2), order='C')
	F2 = F2.reshape((dim1,dim2), order='C')
	return F1, F2

def NSIss(A1, A2, tol=1E-3, maxiter=20):                 # 1E-14
	'''Get Eigenvalues, Eigenvectors via QR Algorithm.
	Normalized Simultaneous QR Iteration in k steps.
	'''
	m = A1.shape[0]
	Q1 = np.identity(m)
	Q2 = np.zeros((m, m))

	residual = 10
	lprev = np.ones(m)
	ctr = 0
	while residual > tol:
		temp01, temp02 = MatMulss(A1, A2, Q1, Q2)        	# Q, R = qr(A @ Q)
		Q1, Q2, R1, R2 = qrss(temp01, temp02)
		
		temp1, temp2 = MatMulss(Q1.T, Q2.T, A1, A2)			# lam = np.diagonal(Q.T @ A @ Q)              # Rayleigh Quotient Matrix
		temp3, temp4 = MatMulss(temp1, temp2, Q1, Q2)

		lam1 = np.diagonal(temp3)
		lam2 = np.diagonal(temp4)
		lam = lam1+lam2
		residual = norm(lprev - np.sort(lam))
		lprev = np.sort(lam)

		ctr += 1
		if ctr == maxiter:
			break
	print("times of qr", ctr)  
	return lam1, lam2, Q1, Q2

def EigVecSs(L1, L2): 
    Q_eig1 = np.eye(len(L1))
    Q_eig2 = np.zeros((len(L1), len(L1)))
    for k in range(2):
        Q1, Q2, R1, R2 = qrss(L1, L2)
        Q_eig1, Q_eig2 = MatMulss(Q_eig1, Q_eig2, Q1, Q2)
        L1, L2 = MatMulss(R1, R2, Q1, Q2)
    return R1, R2, Q_eig1, Q_eig2

####################################################################################

def MatVecMulss(A1, A2, B1, B2):
	dim = A1.shape[0]
	C1 = np.zeros(dim)
	C2 = np.zeros(dim)
	for i in range(dim):
		C1[i], C2[i] = InnerProductss(A1[i], A2[i], B1, B2)
	return C1, C2

def qrformnss(matrix1, matrix2):
	sm = matrix1.shape[0]
	sn = matrix1.shape[1]
	Q1 = np.zeros((sm, 1))
	Q2 = np.zeros((sm, 1))
	R1 = np.zeros((sm, sn))
	R2 = np.zeros((sm, sn))
	for i in range(sn):
		u1 = matrix1[:,i]
		u2 = matrix2[:,i]
		for j in range(i):
			R1[j,i], R2[j,i] = InnerProductss(Q1[:,j], Q2[:,j], u1, u2)
			t1, t2 = DotProductss(R1[j,i], R2[j,i], Q1[:,j], Q2[:,j])
			u1 = u1 - t1
			u2 = u2 - t2
		t3, t4 = InnerProductss(u1, u2, u1, u2)
		if (t3+t4) < 0.00001:
			if i==0:
				Q1[:,i] = np.zeros(sm)
				Q2[:,i] = np.zeros(sm)
			else:
				qt1 = np.zeros(sm)
				qt2 = np.zeros(sm)
				Q1 = np.hstack((Q1, np.reshape(qt1, (sm, 1))))
				Q2 = np.hstack((Q2, np.reshape(qt2, (sm, 1))))
		else:
			R1[i,i], R2[i,i] = sqrtss(t3, t4)
			Rtemp1, Rtemp2 = divss(R1[i,i], R2[i,i])
			if i==0:
				Q1[:,i], Q2[:,i] = DotProductss(Rtemp1, Rtemp2, u1, u2)
			else:
				qt1, qt2 = DotProductss(Rtemp1, Rtemp2, u1, u2)
				Q1 = np.hstack((Q1, np.reshape(qt1, (sm, 1))))
				Q2 = np.hstack((Q2, np.reshape(qt2, (sm, 1))))
	return Q1, Q2, R1, R2


def LanczosTri(A1, A2):
	n = A1.shape[0]
	# x = np.ones(n)                      #Random Initial Vector
	np.random.seed(1)
	x1 = np.random.random(n)
	x2 = np.random.random(n)
	V1 = np.zeros((n,1))                 #Tridiagonalizing Matrix
	V2 = np.zeros((n,1))                 #Tridiagonalizing Matrix

	#Begin Lanczos Iteration
	norm_x1, norm_x2 = InnerProductss(x1, x2, x1, x2)  # q = x/np.linalg.norm(x)
	norm_x1, norm_x2 = isqrtss(norm_x1, norm_x2)
	q1, q2 = DotProductss(norm_x1, norm_x2, x1, x2)
	V1[:,0] = q1 								# V[:,0] = q
	V2[:,0] = q2
	
	r1, r2 = MatVecMulss(A1, A2, q1, q2)  		# r = A @ q
	a1, a2 = InnerProductss(q1, q2, r1, r2) 	# a1 = q.T @ r
	aq1, aq2 = DotProductss(a1, a2, q1, q2)  	# r = r - a1*q

	r1 = r1 - aq1
	r2 = r2 - aq2
	b1, b2 = InnerProductss(r1,r2, r1, r2)  	# b1 = np.linalg.norm(r)
	b1, b2 = sqrtss(b1, b2)
	ctr = 0
	for j in range(2, 170):
		v1 = q1 									# v = q
		v2 = q2
		divb1, divb2 = divss(b1, b2)				# q = r/b1
		q1, q2 = DotProductss(divb1, divb2, r1, r2)
		Aq1, Aq2 = MatVecMulss(A1, A2, q1, q2)		# r = A @ q - b1*v
		bv1, bv2 = DotProductss(b1, b2, v1, v2)
		r1 = Aq1 - bv1
		r2 = Aq2 - bv2 
		
		a1, a2 = InnerProductss(q1, q2, r1, r2)		# a1 = q.T @ r
		aq1, aq2 = DotProductss(a1, a2, q1, q2)		# r = r - a1*q
		r1 = r1 - aq1
		r2 = r2 - aq2
		b1, b2 = InnerProductss(r1, r2, r1, r2) 	# b1 = np.linalg.norm(r)
		b1, b2 = sqrtss(b1, b2)
		
		#Append new column vector at the end of V
		V1 = np.hstack((V1, np.reshape(q1,(n,1))))
		V2 = np.hstack((V2, np.reshape(q2,(n,1))))
		
		#Reorthogonalize all previous v's
		# V = np.linalg.qr(V)[0]
		# V = qrhh(V)[0]
		# V = gram_schmidt(V)[0]
		V1, V2, R1, R2 = qrformnss(V1, V2)

		ctr+=1
		
		if (b1+b2) == 0: 
			print("WARNING: Lanczos ended due to b1 = 0")
			return V1, V2 #Need to reorthonormalize
		
	#Tridiagonal matrix similar to A
	# T = V.T @ A @ V
	T1, T2 = MatMulss(V1.T, V2.T, A1, A2)
	T1, T2 = MatMulss(T1, T2, V1, V2)
	
	return V1, V2, T1, T2

#####################################################################################
def NewgetEigVec(eigval, eigvec, cluster_num):  #从拉普拉斯矩阵获得特征矩阵
	# eigval,eigvec = np.linalg.eig(L)
	dim = len(eigval)
	dictEigval = dict(zip(eigval,range(0,dim)))
	kEig = np.sort(eigval)[0:cluster_num]
	ix = [dictEigval[k] for k in kEig]
	return eigval[ix],eigvec[:,ix]

def getCenters(data,C):  # 获得中心位置
	centers = []
	for i in range(max(C)+1):
		points_list = np.where(C==i)[0].tolist()
		centers.append(np.average(data[points_list],axis=0))
	return centers

def randRGB():
	return (random.randint(0, 255)/255.0,
			random.randint(0, 255)/255.0,
			random.randint(0, 255)/255.0)

def plot(matrix,C,centers,k):
	colors = []
	for i in range(k):
		colors.append(randRGB())
	for idx,value in enumerate(C):
		plt.plot(matrix[idx][0],matrix[idx][1],'o',color=colors[int(C[idx])])
	for i in range(len(centers)):
		plt.plot(centers[i][0],centers[i][1],'rx')
	plt.show()


if __name__ == '__main__':
	t01 = time.time()
	# data = loaddata(100)
	# data = load_dot_mat()
	n_samples = 500
	noisy_circles = datasets.make_blobs(n_samples=n_samples, random_state=8)
	# noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
	# noisy_circles = datasets.make_moons(n_samples=n_samples, noise=.05)
	# X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
	# transformation = [[0.6, -0.6], [-0.4, 0.8]]
	# data = np.dot(X, transformation)
	data, other = noisy_circles
	t02 = time.time()
	print("load data", t02 - t01)


	with open('l0_blobs.data', 'wb') as f0:
		pickle.dump(data, f0)

	l = 4
	kn = 6
	n = 32	 		# bit-width
	
	row = data.shape[0]
	col = data.shape[1]
	num0 = row
	num1 = row
	num2 = row
	# print(np.min(data))
	# print(np.max(data))
	# data1 = np.random.randint(0.5*np.min(data), 0.5*np.max(data), (row, col))
	# data1 = np.random.randint(0, 1, (row, col))
	data1 = np.random.random((row,col))
	data2 = data - data1
	print("data", np.max(data1), np.min(data1), np.max(data2), np.min(data2))

	diff1 = np.zeros((row, row, col))
	diff2 = np.zeros((row, row, col))

	t11 = time.time()
	for i in range(row):
		for j in range(i+1, row):
			diff1[i][j] = data1[i] - data1[j]
			diff2[i][j] = data2[i] - data2[j]
	t12 = time.time()
	print("compute difference", t12 - t11)

	'''Compute the squared Euclidean distance'''
	t21 = time.time()
	dist1 = np.zeros((row, row))
	dist2 = np.zeros((row, row))
	for i in range(row):
		for j in range(i+1, row):
			dist1[i][j], dist2[i][j] = InnerProductss(diff1[i][j], diff2[i][j], diff1[i][j], diff2[i][j])
	dist1 += dist1.T
	dist2 += dist2.T
	t22 = time.time()
	print("dist", np.max(dist1), np.min(dist1), np.max(dist2), np.min(dist2))
	print("compute distance", t22-t21)
	# print("dist\n", dist1+dist2)

	wt1 = np.zeros((row, row), dtype=np.int32)
	wt2 = np.zeros((row, row), dtype=np.int32)
	t31 = time.time()
	weight1 = np.zeros((row, row, row), dtype=np.int32)
	weight2 = np.zeros((row, row, row), dtype=np.int32)
	
	for i in range(row):										# in a Matrix fashion
		weight1 = np.zeros((row, row), dtype=np.int32)
		weight2 = np.zeros((row, row), dtype=np.int32)
		for j in range(row):
			for k in range(row):
				weight1[j][k] = int((dist1[i][j] - dist1[i][k])*1000000)
				weight2[j][k] = int((dist2[i][j] - dist2[i][k])*1000000)
		w1, w2, t_off, t_on = BitExtractionMatrix(weight1, weight2)
		wtemp1 = w1.astype(np.int32)
		wtemp2 = w2.astype(np.int32)
		wn1, wn2 = MultiplyMatrix(wtemp1, -wtemp2, wtemp1, -wtemp2)
		wt1[i] = np.sum(wn1, axis=1)
		wt2[i] = np.sum(wn2, axis=1)
	# print("weight\n", wt1+wt2)  								# sort the distances
	print("weight", np.max(wt1), np.min(wt1), np.max(wt2), np.min(wt2))
	t32 = time.time()

	print("off", t_off)
	print("on", t_on)
	print("total", t32-t31)

	t61 = time.time()
	wt1 = (row-1-kn) - wt1 										# choose kn largest distance
	wt2 = 0 - wt2
	wxor1, wxor2, t_off, t_on = BitExtractionMatrix(wt1, wt2)
	t62 = time.time()
	print("weight matrix", t62-t61)

	t5 = time.time()
	wtemp3 = wxor1.astype(np.int32)
	wtemp4 = wxor2.astype(np.int32)
	wsum1, wsum2 = MultiplyMatrix(wtemp3, -wtemp4, wtemp3, -wtemp4)
	wsum1 -= np.diag(np.diag(wsum1))							# Set the elements in the diagnol to be 0
	wsum2 -= np.diag(np.diag(wsum2))

	wsum1 = (wsum1 + wsum1.T) / 2
	wsum2 = (wsum2 + wsum2.T) / 2
	t6 = time.time()
	print("wsum", np.max(wsum1), np.min(wsum1), np.max(wsum2), np.min(wsum2))
	print("xor to sum", t6-t5)
	d1 = np.diag(np.sum(wsum1, axis=1))							# D
	d2 = np.diag(np.sum(wsum2, axis=1))
	l1 = d1 - wsum1												# L = D - W
	l2 = d2 - wsum2

	with open('l1_blobs.data', 'wb') as f1:
		pickle.dump(l1, f1)
	with open('l2_blobs.data', 'wb') as f2:
		pickle.dump(l2, f2)

	############################################################
	# t7 = time.time()
	# size=len(l1)
	# cluster_num = 3
	# V1, V2, T1, T2 = LanczosTri(l1, l2)
	# # V = V1+V2
	# # T = T1+T2
	# # eigval, eigvec = np.linalg.eig(T)
	# # eigvec = np.dot(V, eigvec)
	# eigval1, eigval2, eigvec1, eigvec2 = EigVecSs(T1, T2)
	# eigvec1, eigvec2 = MatMulss(V1, V2, eigvec1, eigvec2)
	# t8 = time.time()
	# print("eigenvalues and eigenvectors", t8-t7)

	# eigval = np.diagonal(eigval1 + eigval2)
	# eigvec = eigvec1 + eigvec2
	# eigval, eigvec = NewgetEigVec(eigval, eigvec, cluster_num)
	# # print(eigval)
	# # print(eigvec)

	# clf = KMeans(n_clusters=cluster_num)
	# s = clf.fit(eigvec)
	# C = s.labels_
	# centers = getCenters(data,C)
	# plot(data,s.labels_,centers,cluster_num)

	
	
