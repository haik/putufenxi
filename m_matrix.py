import numpy as np
import time
from m_mult import *

def NumProductss(u1, u2, v1, v2):			# Number-Number
	t1 = time.time()
	trip = TripleGenerate2n(1)
	t2 = time.time()
	z1, z2 = MultiplyMatrix2n(u1, u2, v1, v2, trip[0], trip[1], trip[2], trip[3], trip[4], trip[5])
	t3 = time.time()
	return z1, z2, t2-t1, t3-t2

def DotProductss(a1, a2, b1, b2):			# Number-Vector
	dim = b1.shape
	c1 = np.repeat(a1, dim)
	c2 = np.repeat(a2, dim)
	t1 = time.time()
	trip = TripleGenerate2n(dim[0])
	t2 = time.time()
	product1, product2 = MultiplyMatrix2n(c1, c2, b1, b2, trip[0], trip[1], trip[2], trip[3], trip[4], trip[5])
	t3 = time.time()
	return product1, product2, t2-t1, t3-t2, t3-t1

def InnerProductss(a1, a2, b1, b2):  		# Vector-Vector
	dim = a1.shape
	t1 = time.time()
	trip = TripleGenerate2n(dim[0])
	t2 = time.time()
	product1, product2 = MultiplyMatrix2n(a1, a2, b1, b2, trip[0], trip[1], trip[2], trip[3], trip[4], trip[5])
	t3 = time.time()
	return np.sum(product1), np.sum(product2), t2-t1, t3-t2

def MatVecMulss(A1, A2, B1, B2):			# Matrix-Vector
	dim = A1.shape[0]
	D1 = np.tile(B1.T, (dim, 1))
	D2 = np.tile(B2.T, (dim, 1))
	E1, E2, toff, ton = MultiplyMatrix(A1, A2, D1, D2)
	C1 = np.sum(E1, axis=1)
	C2 = np.sum(E2, axis=1)
	return C1, C2, toff, ton

def MatVecMulssOld(A1, A2, B1, B2):
	dim = A1.shape[0]
	C1 = np.zeros(dim)
	C2 = np.zeros(dim)
	toff = 0
	ton = 0
	for i in range(dim):
		print(A1[i].shape, B1.shape)
		C1[i], C2[i], toff1, ton1 = InnerProductss(A1[i], A2[i], B1, B2)
		toff += toff1
		ton += ton1
	return C1, C2, toff, ton	

def MatVecMulssNew(A1, A2, B1, B2):			# Matrix-Vector
	dim = A1.shape
	D1 = np.tile(B1.T, (dim[0], 1))
	D2 = np.tile(B2.T, (dim[0], 1))
	A1 = A1.flatten()
	A2 = A2.flatten()
	D1 = D1.flatten()
	D2 = D2.flatten()

	t1 = time.time()
	trip = TripleGenerate2n(dim[0]*dim[1])
	t2 = time.time()
	E1, E2 = MultiplyMatrix2n(A1, A2, D1, D2, trip[0], trip[1], trip[2], trip[3], trip[4], trip[5])
	t3 = time.time()
	E1 = E1.reshape(dim[0], dim[1])
	E2 = E2.reshape(dim[0], dim[1])
	C1 = np.sum(E1, axis=1)
	C2 = np.sum(E2, axis=1)
	return C1, C2, t2-t1, t3-t2

def MatMulss(A1, A2, B1, B2):				# Matrix-Matrix
	dim1 = A1.shape[0]
	dim2 = B1.shape[1]
	t1 = time.time()
	C1 = np.repeat(A1, dim2, axis=0)
	C2 = np.repeat(A2, dim2, axis=0)
	D1 = np.tile(B1.T, (dim1, 1))
	D2 = np.tile(B2.T, (dim1, 1))
	t2 = time.time()
	E1, E2, toff, ton = MultiplyMatrix(C1, C2, D1, D2)
	t3 = time.time()
	F1 = np.sum(E1, axis=1)
	F2 = np.sum(E2, axis=1)
	t4 = time.time()
	F1 = F1.reshape((dim1,dim2), order='C')
	F2 = F2.reshape((dim1,dim2), order='C')
	return F1, F2, toff, ton+t2-t1+t4-t3


