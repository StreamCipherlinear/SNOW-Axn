"""Report the extreme entries of the correlation table built by profile.py.

Inputs: the author's scratch copy of profile.py under the system temp directory.
Outputs: the count and minimum of the non-zero entries and the argmin byte pair.
Paper object: none; development aid for the profile of Section 4.1.
"""

import sys, io, contextlib
import numpy as np
sys.path.insert(0, r'C:\Users\Lenovo\AppData\Local\Temp\snowa_study')
src = open(r'C:\Users\Lenovo\AppData\Local\Temp\snowa_study\profile.py').read()
head = src.split("nz = ~zero")[0]
exec(head)
nz = ~zero
print('zero points:', int(zero.sum()), ' nonzero:', int(nz.sum()))
print('acc min over nonzero:', acc[nz].min(), ' count acc<-108 among nonzero:', int(((acc < -108) & nz).sum()))
print('acc min over all:', acc.min())
idx = np.unravel_index(np.argmin(np.where(nz, acc, 1e9)), acc.shape)
print('argmin nonzero at', idx, 'acc=', acc[idx])
bi, cj = idx
print('B bytes:', B[bi], ' C bytes:', C[cj])
print('P bytes:', P[bi], ' Q bytes:', Q[cj])
