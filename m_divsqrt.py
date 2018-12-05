from m_matrix import *
import time
import struct

def divss(b1, b2):
	toff = 0
	ton = 0
	if abs(b1+b2) < 0.0001:
		n1 = 0
		n2 = 0
		count = 0
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
			n1, n2, toff1, ton1 = NumProductss(f1, f2, n1, n2)
			d1, d2, toff2, ton2 = NumProductss(f1, f2, d1, d2)
			toff = toff + toff1 + toff2
			ton = ton + ton1 +ton2
			if abs(d1+d2-1) < 0.00001:
				break
	return n1, n2, toff, ton

def sqrtss(a1, a2):
	x11 = a1
	x12 = a2
	x21 = a1 / 2
	x22 = a2 / 2
	count = 0
	toff = 0
	ton = 0
	while abs(x11+x12-x21-x22)> 0.0001:
		count += 1
		x11 = x21
		x12 = x22
		div_x11, div_x12, toff1, ton1 = divss(x11, x12)
		# print("div-count", countdiv)
		mul_x11, mul_x12, toff2, ton2 = NumProductss(a1, a2, div_x11, div_x12)
		x21 = (x11 + mul_x11) / 2
		x22 = (x12 + mul_x12) / 2
		toff = toff + toff1 + toff2
		ton = ton + ton1 + ton2
	# print("sqrt-count", count)
	return x11, x12, toff, ton

def isqrtss(a1, a2):
	b1, b2, toff1, ton1 = sqrtss(a1, a2)
	c1, c2, toff2, ton2 = divss(b1, b2)
	toff = toff1 + toff2
	ton = ton1 + ton2
	return c1, c2, toff, ton

def isqrtnew(a1, a2):
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

def isqrt(number):
	threehalfs = 1.5
	x2 = number * 0.5
	y = number

	packed_y = struct.pack('f', y)
	i = struct.unpack('i', packed_y)[0]
	i = 0x5f3759df - (i >> 1)
	packed_i = struct.pack('i', i)
	y = struct.unpack('f', packed_i)[0]

	y = y * (threehalfs - (x2 * y * y))
	return y

def isqrt_fast_ss(number1, number2):
	threehalfs = 1.5
	x21 = number1 * 0.5
	x22 = number2 * 0.5
	y1 = number1
	y2 = number2

	packed_y1 = struct.pack('f', y1)
	packed_y2 = struct.pack('f', y2)
	i1 = struct.unpack('i', packed_y1)[0]
	i2 = struct.unpack('i', packed_y2)[0]
	i1 = 0x5f3759df - (i1 >> 1)
	i2 = 0 - (i2 >> 1)
	packed_i1 = struct.pack('i', i1)
	packed_i2 = struct.pack('i', i2)
	y1 = struct.unpack('f', packed_i1)[0]
	y2 = struct.unpack('f', packed_i2)[0]
	y = y1+y2
	x2 = x21+x22
	y = y * (threehalfs - (x2 * y * y))
	
	return y

# a1 = 0.8
# a2 = 0.07
# t1 = time.time()
# r1, r2, off, on = isqrtss(a1, a2)
# t2 = time.time()
# r3 = isqrt(a1+a2)
# print(r3)
# print(1/np.sqrt(a1+a2))
# print(r1+r2)
# print(t2-t1, off, on)
