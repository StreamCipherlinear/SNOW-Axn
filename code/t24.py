# Sum tau^2 and the cross-term bound over the 24 strongest, reusing verify_exact.
import math
import numpy as np

ns = {}
exec(compile(open('verify_exact.py', encoding='utf-8-sig').read(),
             'verify_exact.py', 'exec'), ns)

tau, Cmat = ns['tau'], ns['Cmat']
off = np.abs(Cmat - np.eye(24))
w = np.outer(tau, tau)
s2 = float(np.sum(tau ** 2))
cross = float(np.sum(off * w))
s4 = float(np.sum(tau ** 4))

print('\n=== top-24 summary ===')
print('log2 sum tau^2      : %.3f' % math.log2(s2))
print('log2 sum tau^4      : %.3f' % math.log2(s4))
print('log2 sum_{i!=j}|C||t_i t_j| : %.3f' % math.log2(cross))
print('log2 max |C|        : %.3f' % math.log2(off.max()))
print('log2 max |t_i t_j|  : %.3f' % math.log2(w.max()))
print('log2 per-term bound : %.3f' % math.log2(off.max() * w.max()))
print('#offdiag terms      : %d' % np.count_nonzero(off))
print('log2 relative bound : %.3f' % math.log2(cross / s2))
print('log2 relative, per-term x terms : %.3f'
      % math.log2(552.0 * off.max() * w.max() / s2))
