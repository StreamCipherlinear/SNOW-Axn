# How many active bytes of the 24 strongest have a *tied* bytewise A* ?  If any,
# the spanning masks (hence W, hence the counts) depend on the tie-breaking.
import sys, io, contextlib
import numpy as np
sys.path.insert(0, r'C:\Users\Lenovo\AppData\Local\Temp\snowa_study')
with contextlib.redirect_stdout(io.StringIO()):
    from verify import LAT, ILAT, L, Linv, sigma, pullback
MU = (4, 5, 6, 12, 13, 14)
ILa = np.abs(ILAT.astype(np.int32))
T12v = np.zeros((256, 256), dtype=np.int32); T12n = np.zeros((256, 256), dtype=np.int32)
for m1 in range(256):
    pr = ILa[m1][None, :] * ILa
    pr[:, 0] = 0
    T12v[m1] = pr.max(axis=1)
    T12n[m1] = (pr == T12v[m1][:, None]).sum(axis=1) - (T12v[m1] == 0)
print('pairs (m1,m2) with more than one maximiser:', int((T12n > 1).sum()), 'out of 65536')
print('max number of maximisers:', int(T12n.max()))
# distribution
import collections
print('multiplicity histogram:', dict(collections.Counter(T12n.ravel().tolist())))

# which of the 24 triples' active bytes are affected?  rebuild them quickly
sys.argv = ['x']
exec(open(r'C:\Users\Lenovo\AppData\Local\Temp\snowa_study\verify_W2.py').read().split('CH = 1 << 20')[0].replace("PAIR = tuple(int(x) for x in sys.argv[1].split(','))", "PAIR = (22, 24)"))
hit = 0
for k in list(range(22)) + list(PAIR):
    Bb = [int(x) for x in B[bi[k]]]; Cb = [int(x) for x in C[cj[k]]]
    for i in MU:
        if T12n[Cb[i] ^ Bb[i], Cb[i]] > 1:
            hit += 1
            print('  triple %d byte %d: (%02x,%02x) has %d maximisers' %
                  (k, i, Cb[i] ^ Bb[i], Cb[i], int(T12n[Cb[i] ^ Bb[i], Cb[i]])))
print('tied active bytes among the 24 masks:', hit)
