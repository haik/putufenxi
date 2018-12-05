from m_comp import *
import numpy as np

dim = 500
a1 = np.random.randint(-20000, 20000, size=(dim, dim))
a2 = np.random.randint(-20000, 20000, size=(dim, dim))
print(a1.shape)
print(a1+a2)

r1, r2, off, on = BitExtractionMatrix(a1, a2)
r0 = r1^r2
print(r0)
print(off, on) 

a0 = (a1+a2)<0
print(a0==r0)