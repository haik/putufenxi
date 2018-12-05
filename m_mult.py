import numpy as np
import time

#################################'''Vector operations'''#############################################

def TripleGenerate2n(numofrow):
	triplet = np.random.randint(0, 2**(8-2), (6, numofrow))
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
	triplet = np.random.randint(0, 2**(8-2), (6, numofrow, numofcol))
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