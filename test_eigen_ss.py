import numpy as np
from all_ss_module import *

def HouseholderSs(matrix1, matrix2):		# Householder transformation
	R1 = matrix1
	R2 = matrix2
	size = len(matrix1)
	Q1 = np.eye(size)
	Q2 = np.zeros((size, size))
	toff = 0
	ton = 0
	for i in range(size):
		x1 = R1[i:,i]
		x2 = R2[i:,i]
		temp1 = np.zeros((size-i))
		temp2 = np.zeros((size-i))
		temp1[0], temp2[0], toff1, ton1 = InnerProductss(x1, x2, x1, x2)
		# temp1[0], temp2[0], toff2, ton2 = sqrtss(temp1[0], temp2[0])
		temp11, temp12, toff2, ton2 = isqrtss(temp1[0], temp2[0])
		temp1[0], temp2[0], toff4, ton4 = NumProductss(temp1[0], temp2[0], temp11, temp12)
		
		u1 = x1 - temp1
		u2 = x2 - temp2
		beta1, beta2, toff3, ton3 = InnerProductss(u1, u2, u1, u2)
		# beta1, beta2, toff4, ton4 = sqrtss(beta1, beta2)
		if beta1+beta2<0.0001:
			v1 = np.zeros((size-i, 1))
			v2 = np.zeros((size-i, 1))
			toff5 = ton5 = 0
			toff6 = ton6 = 0
		else:
			# norm1, norm2, toff5, ton5 = divss(beta1, beta2)
			norm1, norm2, toff5, ton5 = isqrtss(beta1, beta2)
			v1, v2, toff6, ton6 = DotProductss(norm1, norm2, u1, u2)
			v1 = v1.reshape(size-i, 1)
			v2 = v2.reshape(size-i, 1)

		vvt1, vvt2, toff7, ton7 = MatMulss(v1, v2, v1.T, v2.T)
		Qt1 = np.eye(size)
		Qt2 = np.zeros((size, size))
		Qt1[i:, i:] = np.eye(size-i) - 2 * vvt1
		Qt2[i:, i:] = np.zeros((size-i, size-i)) - 2 * vvt2
		if i==0:
			Q1 = Qt1
			Q2 = Qt2
			toff8 = 0
			ton8 = 0
		else:
			Q1, Q2, toff8, ton8 = MatMulss(Qt1, Qt2, Q1, Q2)

		# Av1, Av2, toff7, ton7 = MatVecMulss(Q1[i:, i:].T, Q2[i:, i:].T, v1, v2)
		# Av1 = Av1.reshape(size-i, 1)
		# Av2 = Av2.reshape(size-i, 1)
		# vvA1, vvA2, toff8, ton8 = MatMulss(v1, v2, Av1.T, Av2.T)
		# Q1[i:, i:] -= 2 * vvA1
		# Q2[i:, i:] -= 2 * vvA2
		
		R1, R2, toff9, ton9 = MatMulss(Q1, Q2, matrix1, matrix2)


		toff = toff + toff1 + toff2 + toff3 + toff4 + toff5 + toff6 + toff7 + toff8 + toff9
		ton = ton + ton1 + ton2 + ton3 + ton4 + ton5 + ton6 + ton7 + ton8 + ton9
		
	Q1 = Q1.T
	Q2 = Q2.T
	return Q1, Q2, R1, R2, toff, ton

def EigVecSs(L1, L2): 
	Q_eig1 = np.eye(len(L1))
	Q_eig2 = np.zeros((len(L1), len(L1)))
	toff = 0
	ton = 0
	for k in range(2):
		Q1, Q2, R1, R2, toff1, ton1 = HouseholderSs(L1, L2)
		Q_eig1, Q_eig2, toff2, ton2 = MatMulss(Q_eig1, Q_eig2, Q1, Q2)
		L1, L2, toff3, ton3 = MatMulss(R1, R2, Q1, Q2)
		toff = toff + toff1 + toff2 + toff3
		ton = ton + ton1 + ton2 + ton3
	return R1, R2, Q_eig1, Q_eig2, toff, ton