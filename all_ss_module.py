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
import matplotlib.pyplot as plt
import pickle
from itertools import cycle, islice

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
	db_name = 'DB/' + 'happy'
	try:
		mat = sio.loadmat(path)
	except NotImplementedError:
		mat = h5py.File(path)
	return np.array(mat[db_name]).transpose()

def binaryconvert(s):
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
	l = 4
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
	triplet = np.random.randint(0, 10, (6, numofrow, numofcol))
	triplet[5] = (triplet[0]+triplet[3]) * (triplet[1]+triplet[4]) - triplet[2]
	return triplet[0], triplet[1], triplet[2], triplet[3], triplet[4], triplet[5]

def MultiplyMatrix(u1, u2, v1, v2):
	dim1 = u1.shape[0]
	dim2 = u1.shape[1]
	toff1 = time.time()
	trip = TripleMatrix(dim1, dim2)
	toff2 = time.time()
	ton1 = time.time()
	z1, z2 = MultiplyMatrix2n(u1, u2, v1, v2, trip[0], trip[1], trip[2], trip[3], trip[4], trip[5])
	ton2 = time.time()
	toff = toff2-toff1
	ton = ton2-ton1
	return z1, z2, toff, ton	

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
	toff1 = time.time()
	trip = TripleGenerate2n(1)
	toff2 = time.time()
	ton1 = time.time()
	z1, z2 = MultiplyMatrix2n(u1, u2, v1, v2, trip[0], trip[1], trip[2], trip[3], trip[4], trip[5])
	ton2 = time.time()
	toff = toff2-toff1
	ton = ton2-ton1
	return z1, z2, toff, ton

def InnerProductss(a1, a2, b1, b2):  		# 求向量a的各元素的平方和，a=a1+a2
	dim = a1.shape
	toff1 = time.time()
	trip = TripleGenerate2n(dim[0])
	toff2 = time.time()
	ton1 = time.time()
	product1, product2 = MultiplyMatrix2n(a1, a2, b1, b2, trip[0], trip[1], trip[2], trip[3], trip[4], trip[5])
	ton2 = time.time()
	toff = toff2-toff1
	ton = ton2-ton1
	return np.sum(product1), np.sum(product2), toff, ton

def DotProductss(a1, a2, b1, b2):
	dim = b1.shape
	c1 = np.repeat(a1, dim)
	c2 = np.repeat(a2, dim)
	toff1 = time.time()
	trip = TripleGenerate2n(dim[0])
	toff2 = time.time()
	ton1 = time.time()
	product1, product2 = MultiplyMatrix2n(c1, c2, b1, b2, trip[0], trip[1], trip[2], trip[3], trip[4], trip[5])
	ton2 = time.time()
	toff = toff2-toff1
	ton = ton2-ton1
	return product1, product2, toff, ton

def divss(b1, b2):
	toff = 0
	ton = 0
	if abs(b1+b2) < 1e-4:
		n1 = 0
		n2 = 0
		count = 0
	else:
		eps = 1e-6
		if abs(b1+b2)>1e5:
			eps = 1e-10
		n1 = 0.5 * eps
		n2 = 0.5 * eps
		d1 = b1 * eps
		d2 = b2 * eps
		count = 0
		while True:
			count += 1
			f1 = 1 - d1
			f2 = 1 - d2
			n1, n2, toff1, ton1 = NumProductss(f1, f2, n1, n2)
			d1, d2, toff2, ton2 = NumProductss(f1, f2, d1, d2)
			toff = toff + toff1 + toff2
			ton = ton + ton1 +ton2
			if abs(d1+d2-1) < 1e-5:
				break
	return n1, n2, toff, ton

def isqrtss(a1, a2):     # new version: use sqrt to estimate the initial value
	x11 = a1 * 0.5
	x12 = a2 * 0.5
	div_x11, div_x12, toff4, ton4 = divss(x11, x12)
	mul_x11, mul_x12, toff5, ton5 = NumProductss(a1, a2, div_x11, div_x12)
	x21 = (x11 + mul_x11) * 0.5
	x22 = (x12 + mul_x12) * 0.5
	x21, x22, toff6, ton6 = divss(x21, x22)
	x11 = a1
	x12 = a2
	count = 0
	toff = toff4+toff5+toff6
	ton = ton4+ton5+ton6
	while abs(x11+x12-x21-x22)> 0.0000001:
		count += 1
		x11 = x21
		x12 = x22
		temp_x1, temp_x2, toff1, ton1 = NumProductss(x11, x12, x11, x12)
		temp_x3, temp_x4, toff2, ton2 = NumProductss(a1, a2, temp_x1, temp_x2)
		temp_x3 = 3 - temp_x3
		temp_x4 = 0 - temp_x4
		x21, x22, toff3, ton3 = NumProductss(temp_x3, temp_x4, x11*0.5, x12*0.5)
		# print("div-count", countdiv)
		toff += toff1 + toff2 + toff3
		ton += ton1 + ton2 + ton3
		# print(count, x21+x22)
	return x11, x12, toff, ton

def MatMulss(A1, A2, B1, B2):
	dim1 = A1.shape[0]
	dim2 = B1.shape[1]
	C1 = np.repeat(A1, dim2, axis=0)
	C2 = np.repeat(A2, dim2, axis=0)
	D1 = np.tile(B1.T, (dim1, 1))
	D2 = np.tile(B2.T, (dim1, 1))
	E1, E2, toff, ton = MultiplyMatrix(C1, C2, D1, D2)
	F1 = np.sum(E1, axis=1)
	F2 = np.sum(E2, axis=1)
	F1 = F1.reshape((dim1,dim2), order='C')
	F2 = F2.reshape((dim1,dim2), order='C')
	return F1, F2, toff, ton

####################################################################################
def MatVecMulss(A1, A2, B1, B2):
	dim = A1.shape[0]
	# C1 = np.zeros(dim)
	# C2 = np.zeros(dim)
	# toff = 0
	# ton = 0
	# for i in range(dim):
	# 	C1[i], C2[i], toff1, ton1 = InnerProductss(A1[i], A2[i], B1, B2)
	# 	toff += toff1
	# 	ton += ton1
	D1 = np.tile(B1.T, (dim, 1))
	D2 = np.tile(B2.T, (dim, 1))
	E1, E2, toff, ton = MultiplyMatrix(A1, A2, D1, D2)
	C1 = np.sum(E1, axis=1)
	C2 = np.sum(E2, axis=1)
	return C1, C2, toff, ton

def qrformnss(matrix1, matrix2):
	sm = matrix1.shape[0]
	sn = matrix1.shape[1]
	Q1 = np.zeros((sm, 1))
	Q2 = np.zeros((sm, 1))
	R1 = np.zeros((sm, sn))
	R2 = np.zeros((sm, sn))
	toff = 0
	ton = 0
	for i in range(sn):
		u1 = matrix1[:,i]
		u2 = matrix2[:,i]
		toff1 = 0
		ton1 = 0
		for j in range(i):
			R1[j,i], R2[j,i], toff11, ton11 = InnerProductss(Q1[:,j], Q2[:,j], u1, u2)
			t1, t2, toff12, ton12 = DotProductss(R1[j,i], R2[j,i], Q1[:,j], Q2[:,j])
			u1 = u1 - t1
			u2 = u2 - t2
			toff1 += toff11 + toff12
			ton1 += ton11 + ton12
		t3, t4, toff2, ton2 = InnerProductss(u1, u2, u1, u2)
		toff3 = ton3 = toff4 = ton4 = toff5 = ton5 = toff6 = ton6 = 0
		if (t3+t4) < 0.000001:
			if i==0:
				Q1[:,i] = np.zeros(sm)
				Q2[:,i] = np.zeros(sm)
			else:
				qt1 = np.zeros(sm)
				qt2 = np.zeros(sm)
				Q1 = np.hstack((Q1, np.reshape(qt1, (sm, 1))))
				Q2 = np.hstack((Q2, np.reshape(qt2, (sm, 1))))
		else:
			# R1[i,i], R2[i,i], toff3, ton3 = sqrtss(t3, t4)
			# Rtemp1, Rtemp2, toff4, ton4 = divss(R1[i,i], R2[i,i])
			Rtemp1, Rtemp2, toff4, ton4 = isqrtss(t3, t4)
			R1[i,i], R2[i,i], toff3, ton3 = NumProductss(t3, t4, Rtemp1, Rtemp2)
			if i==0:
				Q1[:,i], Q2[:,i], toff5, ton5 = DotProductss(Rtemp1, Rtemp2, u1, u2)
			else:
				qt1, qt2, toff6, ton6 = DotProductss(Rtemp1, Rtemp2, u1, u2)
				Q1 = np.hstack((Q1, np.reshape(qt1, (sm, 1))))
				Q2 = np.hstack((Q2, np.reshape(qt2, (sm, 1))))
		toff += toff1 + toff2 + toff3 + toff4 + toff5 + toff6
		ton += ton1 + ton2 + ton3 + ton4 + ton5 + ton6
	return Q1, Q2, R1, R2, toff, ton

def LanczosTri(A1, A2, numiter):
	n = A1.shape[0]
	# x = np.ones(n)                      #Random Initial Vector
	# np.random.seed(1)
	x1 = np.random.randn(n)
	x2 = np.random.randn(n)
	V1 = np.zeros((n,1))                 #Tridiagonalizing Matrix
	V2 = np.zeros((n,1))                 #Tridiagonalizing Matrix

	#Begin Lanczos Iteration
	# q = x/np.linalg.norm(x)
	norm_x1, norm_x2, toff1, ton1 = InnerProductss(x1, x2, x1, x2)
	norm_x1, norm_x2, toff2, ton2 = isqrtss(norm_x1, norm_x2)
	q1, q2, toff3, ton3 = DotProductss(norm_x1, norm_x2, x1, x2)
	# V[:,0] = q
	V1[:,0] = q1
	V2[:,0] = q2
	# r = A @ q
	r1, r2, toff4, ton4 = MatVecMulss(A1, A2, q1, q2)
	# a1 = q.T @ r
	a1, a2, toff5, ton5 = InnerProductss(q1, q2, r1, r2)
	# r = r - a1*q
	aq1, aq2, toff6, ton6 = DotProductss(a1, a2, q1, q2)
	r1 = r1 - aq1
	r2 = r2 - aq2
	# b1 = np.linalg.norm(r)
	b1, b2, toff7, ton7 = InnerProductss(r1,r2, r1, r2)
	# b1, b2, toff8, ton8 = sqrtss(b1, b2)
	btemp1, btemp2, toff8, ton8 = isqrtss(b1, b2)
	b1, b2, toff9, ton9 = NumProductss(b1, b2, btemp1, btemp2)
	toff = toff1 + toff2 + toff3 + toff4 + toff5 + toff6 + toff7 + toff8 + toff9
	ton = ton1 + ton2 + ton3 + ton4 + ton5 + ton6 + ton7 + ton8 + ton9

	for j in range(2, numiter):
		# v = q
		v1 = q1
		v2 = q2
		# q = r/b1
		divb1, divb2, toff01, ton01 = divss(b1, b2)
		q1, q2, toff02, ton02 = DotProductss(divb1, divb2, r1, r2)
		# r = A @ q - b1*v
		Aq1, Aq2, toff03, ton03 = MatVecMulss(A1, A2, q1, q2)
		bv1, bv2, toff04, ton04 = DotProductss(b1, b2, v1, v2)
		r1 = Aq1 - bv1
		r2 = Aq2 - bv2 
		# a1 = q.T @ r
		a1, a2, toff05, ton05 = InnerProductss(q1, q2, r1, r2)
		# r = r - a1*q
		aq1, aq2, toff06, ton06 = DotProductss(a1, a2, q1, q2)
		r1 = r1 - aq1
		r2 = r2 - aq2
		# b1 = np.linalg.norm(r)
		b1, b2, toff07, ton07 = InnerProductss(r1, r2, r1, r2)
		# b1, b2, toff08, ton08 = sqrtss(b1, b2)
		btemp3, btemp4, toff08, ton08 = isqrtss(b1, b2)
		b1, b2, toff09, ton09 = NumProductss(b1, b2, btemp3, btemp4)
		#Append new column vector at the end of V
		# V = np.hstack((V, np.reshape(q,(n,1))))
		V1 = np.hstack((V1, np.reshape(q1,(n,1))))
		V2 = np.hstack((V2, np.reshape(q2,(n,1))))
		
		#Reorthogonalize all previous v's
		# V = np.linalg.qr(V)[0]
		# V = qrhh(V)[0]
		# V = gram_schmidt(V)[0]
		# V1, V2, R1, R2, toff09, ton09 = qrformnss(V1, V2)

		toff += toff01 + toff02 + toff03 + toff04 + toff05 + toff06 + toff07 + toff08
		ton += ton01 + ton02 + ton03 + ton04 + ton05 + ton06 + ton07 + ton08

		if (b1+b2) == 0: 
			print("WARNING: Lanczos ended due to b1 = 0")
			return V1, V2 #Need to reorthonormalize
		
	#Tridiagonal matrix similar to A
	# T = V.T @ A @ V
	T1, T2, toff10, ton10 = MatMulss(V1.T, V2.T, A1, A2)
	T1, T2, toff11, ton11 = MatMulss(T1, T2, V1, V2)
	toff += toff10 + toff11
	ton += ton10 + ton11
	
	return V1, V2, T1, T2, toff, ton

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

def plot_old(matrix,C,centers,k):
	colors = []
	for i in range(k):
		colors.append(randRGB())
	for idx,value in enumerate(C):
		plt.plot(matrix[idx][0],matrix[idx][1],'o',color=colors[int(C[idx])])
	for i in range(len(centers)):
		plt.plot(centers[i][0],centers[i][1],'rx')
	plt.show()

def plot(matrix,C,centers,k):
	colors = ['#ff7f00', '#4daf4a', '#a65628', 
		'#f781bf', '#984ea3', '#999999', '#e41a1c', '#377eb8', '#dede00']
	colors = colors[:k]
	for idx,value in enumerate(C):
		plt.scatter(matrix[idx][0],matrix[idx][1],s=30,color=colors[int(C[idx])])
	for i in range(len(centers)):
		plt.plot(centers[i][0],centers[i][1],'rx')
	plt.show()