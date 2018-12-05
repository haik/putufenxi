from all_ss_module import *
import numpy as np

a1 = 100 * np.random.randn(100) + 300
a2 = 100 * np.random.randn(100) + 300
print(np.max(a1))
print(np.min(a1))
off = np.random.randn(100)
on = np.random.randn(100)
c1 = np.random.randn(100)
c2 = np.random.randn(100)
for i in range(100):
	c1[i], c2[i], off[i], on[i] = isqrtss(a1[i], a2[i])
print(c1+c2)
print(np.sum(off)/100, np.sum(on)/100)

# a1 = -1621.9672229674602
# a2 = 12430.4045286158
# c1, c2, off, on = isqrtss(a1, a2)
# print(c1+c2, off, on)