import matplotlib.pyplot as plt
import pickle
import numpy as np
import time

with open('data_300/l0_circles.data', 'rb') as f0:
	circle = pickle.load(f0)
with open('data_400/l0_moons.data', 'rb') as f1:
	moon = pickle.load(f1)
with open('data_500/l0_blobs.data', 'rb') as f2:
	blob = pickle.load(f2)
# with open('data/l0_aniso.data', 'rb') as f3:
# 	aniso = pickle.load(f3)

t1 = time.time()
circle1 = np.random.randint(100*np.min(circle), 100*np.max(circle), size=(circle.shape[0], circle.shape[1]))
circle2 = circle - circle1
t2 = time.time()
moon1 = np.random.randint(100*np.min(moon), 100*np.max(moon), size=(moon.shape[0], moon.shape[1]))
moon2 = moon - moon1
t3 = time.time()
blob1 = np.random.randint(100*np.min(blob), 100*np.max(blob), size=(blob.shape[0], blob.shape[1]))
blob2 = blob - blob1
t4 = time.time()
# aniso1 = np.random.randint(100*np.min(aniso), 100*np.max(aniso), size=(row,col))
# aniso2 = aniso - aniso1
# t5 = time.time()
print(t2-t1, t3-t2, t4-t3)

size = 30
# plt.subplot(431)
plt.figure(7)
plt.scatter(circle[:,0], circle[:,1], s=size)
# plt.subplot(432)
plt.figure(8)
plt.scatter(circle1[:,0], circle1[:,1], s=size)
# plt.subplot(433)
plt.figure(9)
plt.scatter(circle2[:,0], circle2[:,1], s=size)
# plt.subplot(434)
plt.figure(1)
plt.scatter(moon[:,0], moon[:,1], s=size)
# plt.subplot(435)
plt.figure(2)
plt.scatter(moon1[:,0], moon1[:,1], s=size)
# plt.subplot(436)
plt.figure(3)
plt.scatter(moon2[:,0], moon2[:,1], s=size)
# plt.subplot(437)
plt.figure(4)
plt.scatter(blob[:,0], blob[:,1], s=size)
# plt.subplot(438)
plt.figure(5)
plt.scatter(blob1[:,0], blob1[:,1], s=size)
# plt.subplot(439)
plt.figure(6)
plt.scatter(blob2[:,0], blob2[:,1], s=size)
# plt.subplot(437)
# plt.figure(7)
# plt.scatter(aniso[:,0], blobs[:,1], s=15)
# # plt.subplot(438)
# plt.figure(8)
# plt.scatter(aniso1[:,0], blobs1[:,1], s=15)
# # plt.subplot(439)
# plt.figure(9)
# plt.scatter(aniso2[:,0], blobs2[:,1], s=15)

plt.show()
