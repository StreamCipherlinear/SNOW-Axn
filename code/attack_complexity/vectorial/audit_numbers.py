"""Numeric audit of every headline figure of Section 4 against the formulas."""
import math, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
L, PS = 512, 0.999992
DES = {'tau': -55.741, 'Bg': 243, 'N': 250.975, 'T': 252.201, 'M': 231.949}


def zt(Bg):
    t = math.sqrt(2.0 * (Bg + 1) * math.log(2.0))
    return t - (2.30753 + 0.27061 * t) / (1.0 + 0.99229 * t + 0.04481 * t * t)


def z_of(p):
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    return t - (2.30753 + 0.27061 * t) / (1.0 + 0.99229 * t + 0.04481 * t * t)


z1 = z_of(PS)


def attack(log2sei, v, Bg):
    log2M = math.log2((z1 + zt(Bg)) ** 2) - log2sei
    log2N = 0.5 * (log2M + L + 1 - Bg)
    T1 = 2 * 2 ** (0.5 * (log2M + L - Bg))
    T2 = Bg * 2 ** Bg + 2 ** log2M
    T = T1 + T2
    mem = max(2 ** log2N, v * 2 ** Bg)
    return log2M, log2N, math.log2(T), math.log2(mem)


sei1 = json.load(open(os.path.join(HERE, 'sei_nested.json')))[0]
sei24 = json.load(open(os.path.join(HERE, 'sei_nested_24.json')))[-1]
sei_des = 4 * DES['tau']

rows = []
for name, s, v in (('designers, re-evaluated', sei_des, 1), ('single mask v=1', sei1, 1), ('full space v=24', sei24, 24)):
    m, n, t, mem = attack(s, v, DES['Bg'] if name.startswith('designers') else 241)
    rows.append((name, s, m, n, t, mem))

print('%-24s %10s %10s %10s %10s %10s' % ('row', 'log2 tau', 'log2 M', 'log2 N', 'log2 T', 'log2 mem'))
for r in rows:
    print('%-24s %10.4f %10.4f %10.4f %10.4f %10.4f' % (r[0], r[1] / 4, r[2], r[3], r[4], r[5]))

print()
print('paper: designers tau 2^-55.741, Bg 243 ; single 250.028/250.919/228.056 ; full 248.909/250.182')
print('gain single vs designers: T %.3f  N %.3f  M %.3f' % (DES['T'] - rows[1][4], DES['N'] - rows[1][3], DES['M'] - rows[1][2]))
print('gain full   vs designers: T %.3f  N %.3f  M %.3f' % (DES['T'] - rows[2][4], DES['N'] - rows[2][3], DES['M'] - rows[2][2]))
print('gain full   vs single   : T %.3f  N %.3f' % (rows[1][4] - rows[2][4], rows[1][3] - rows[2][3]))
print()
print('tau* check: 2^-54.770580 vs designer 2^-55.741 gap %.4f' % (DES['tau'] + 54.770580))
print('0.771 below 2^-54 bound:', 54.770580 - 54.0)
