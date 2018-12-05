import numpy as np
import time
from m_divsqrt import *

def directisqrtss(a1, a2):
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
		print(count, x21+x22)
	return x11, x12, toff, ton

num = 100
a1 = np.random.randn(num)*10+1000
a2 = np.random.randn(num)*10
print(a1+a2)
r1 = np.random.randn(num)
r2 = np.random.randn(num)
s1 = np.random.randn(num)
s2 = np.random.randn(num)
off1 = np.random.randn(num)
on1 = np.random.randn(num)
off2 = np.random.randn(num)
on2 = np.random.randn(num)
for i in range(num):
	r1[i], r2[i], off1[i], on1[i] = directisqrtss(a1[i], a2[i])
	s1[i], s2[i], off2[i], on2[i] = isqrtss(a1[i], a2[i])

print(r1+r2, np.mean(off1), np.mean(on1))
print(s1+s2, np.mean(off2), np.mean(on2))
print(1/np.sqrt(a1+a2))