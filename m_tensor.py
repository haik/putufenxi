import numpy as np
import time

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