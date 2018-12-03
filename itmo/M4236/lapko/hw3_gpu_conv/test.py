# N=1024. M=3. Обе матрицы состоят из единиц.
# N=1024. M=9. Обе матрицы состоят из единиц.
# N=1. M=9. Обе матрицы состоят из единиц.
# N=31. M=9. Обе матрицы состоят из единиц.
# N=1023. M=9. Обе матрицы состоят из единиц.
import subprocess

import numpy as np
from scipy.ndimage import convolve

fin = open("input.txt", 'w')

N = 5
M = 3

a = np.ones((N, N), dtype=int)
b = np.ones((M, M), dtype=int)
c = convolve(a, b, mode='constant', cval=0)

fin.write(f"{N} {M}\n")
for line in a:
    fin.write(" ".join(map(str, line)) + "\n")
for line in b:
    fin.write(" ".join(map(str, line)) + "\n")

fin.close()
cmd = './conv'
subprocess.call(cmd)

fout = open("output.txt", 'r')
c_ = list()
for line in fout:
    c_.append(list(map(int, line.split())))

c_ = np.array(c_, dtype=int)
assert np.array_equal(c, c_)
print(c_)
