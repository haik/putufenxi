# Test the performance of matrix multiplication

import numpy as np
from m_mult import *
from m_matrix import *

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

def MatMulss_all(A1, A2, B1, B2):				# Matrix-Matrix
	dim1 = A1.shape[0]
	dim2 = B1.shape[1]
	t1 = time.time()
	C1 = np.repeat(A1, dim2, axis=0)
	C2 = np.repeat(A2, dim2, axis=0)
	t2 = time.time()
	D1 = np.tile(B1.T, (dim1, 1))
	D2 = np.tile(B2.T, (dim1, 1))
	t3 = time.time()
	E1, E2, toff, ton = MultiplyMatrix(C1, C2, D1, D2)
	t4 = time.time()
	F1 = np.sum(E1, axis=1)
	F2 = np.sum(E2, axis=1)
	t5 = time.time()
	F1 = F1.reshape((dim1,dim2), order='C')
	F2 = F2.reshape((dim1,dim2), order='C')
	print(t2-t1, t3-t2, t4-t3, t5-t4)
	return F1, F2, toff, ton+t3-t1+t4-t3

dim = 1
a1 = np.random.randint(0, 100, size=(10,10))
a2 = np.random.randint(0, 100, size=(10,10))
b1 = np.random.randint(0, 100, size=(10,))
b2 = np.random.randint(0, 100, size=(10,))
t2 = time.time()
# trip = TripleMatrix(10, 10)
t3 = time.time()
# d = MultiplyMatrix(a1, a2, b1, b2, trip[0], trip[1], trip[2], trip[3], trip[4], trip[5])
d = MatVecMulssOld(a1, a2, b1, b2)
t4 = time.time()
print(a1+a2)
print(b1+b2)
print(d[0]+d[1])
print(t3-t2)
print(t4-t3)
# a1 = np.random.randint(0, 10, size=(dim, dim))
# a2 = np.random.randint(0, 10, size=(dim, dim))
# b1 = np.random.randint(0, 10, size=(dim, dim))
# b2 = np.random.randint(0, 10, size=(dim, dim))

# c = MatMulss_all(a1, a2, b1, b2)
# print(c[2], c[3])
# print(a1)
# c = np.repeat(a1, dim, axis=0)
# print(c)
# d = a1.ravel()
# print(d)
# e = np.tile(d,dim)
# print(e)
# t1 = time.time()
# c = MultiplyMatrix(a1, a2, b1, b2)
# t2 = time.time()
# a11 = a1.ravel()
# a21 = a2.ravel()
# b11 = b1.ravel()
# b21 = b2.ravel()
# trip = TripleGenerate2n(dim*dim)
# t3 = time.time()
# d = MultiplyMatrix2n(a11, a21, b11, b21, trip[0], trip[1], trip[2], trip[3], trip[4], trip[5])
# t4 = time.time()
# print(c[0]+c[1], c[2], c[3])
# print(d[0]+d[1])
# print(t3-t2, t4-t3)
