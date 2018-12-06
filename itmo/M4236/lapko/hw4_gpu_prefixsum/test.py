# just preset N and `expected`

import subprocess
import numpy as np

N = 100000

a = np.ones((N,), dtype=int)
expected = np.arange(1, N + 1)

fin = open("input.txt", 'w')
fin.write(f"{N}\n")
fin.write(" ".join(map(str, a)))
fin.close()

cmd = './scan'
subprocess.call(cmd)

fout = open("output.txt", 'r')
line = next(fout)
c_ = list(map(int, line.split()))
c_ = np.array(c_, dtype=int)

assert np.array_equal(expected, c_), "Test FAIL!"
print("OK")
