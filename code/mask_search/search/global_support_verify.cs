// Independent support census of Section 3.3, by row-space membership.
//
// Inputs : none; L, L^-1 and sigma are rebuilt instead of being read from Q.
// Outputs: one line per feasible pair and the same totals as the census.
// Call   : IndependentCensus.Run(); throws if the invariants fail.

using System;
using System.Collections.Generic;

// Independently construct the field matrices and use row-space membership
// instead of explicit kernel projections to repeat the full support census.
public static class IndependentCensus
{
    private static readonly int[] SR = {0,5,10,15,4,9,14,3,8,13,2,7,12,1,6,11};
    private static readonly int[] Sigma = {0,7,14,11,4,13,10,1,8,15,6,3,12,5,2,9};
    private static readonly int[,] MC = {{2,3,1,1},{1,2,3,1},{1,1,2,3},{3,1,1,2}};
    private static readonly int[,] InvMC = {{14,11,13,9},{9,14,11,13},{13,9,14,11},{11,13,9,14}};
    private static readonly byte[,] Mul = new byte[256,256], Q = new byte[16,16];
    private static readonly byte[] Inv = new byte[256];

    private static byte[,] MatrixMultiply(byte[,] a, byte[,] b)
    {
        var result = new byte[16,16];
        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 16; j++)
                for (int k = 0; k < 16; k++) result[i,j] ^= Mul[a[i,k], b[k,j]];
        return result;
    }

    private static void Init()
    {
        for (int a = 0; a < 256; a++) for (int b = 0; b < 256; b++) {
            int x = a, y = b, z = 0;
            while (y > 0) {
                if ((y & 1) != 0) z ^= x;
                x <<= 1;
                if (x >= 256) x ^= 0x11b;
                y >>= 1;
            }
            Mul[a,b] = (byte)z;
            if (z == 1) Inv[a] = (byte)b;
        }
        var l = new byte[16,16];
        var inverse = new byte[16,16];
        var sigma = new byte[16,16];
        for (int c = 0; c < 4; c++) for (int r = 0; r < 4; r++) for (int s = 0; s < 4; s++) {
            l[4*c+r, SR[4*c+s]] = (byte)MC[r,s];
            inverse[SR[4*c+r], 4*c+s] = (byte)InvMC[r,s];
        }
        for (int i = 0; i < 16; i++) sigma[i,Sigma[i]] = 1;
        var lr = MatrixMultiply(l, inverse);
        var rl = MatrixMultiply(inverse, l);
        var ss = MatrixMultiply(sigma, sigma);
        for (int i = 0; i < 16; i++) for (int j = 0; j < 16; j++) {
            int expected = i == j ? 1 : 0;
            if (lr[i,j] != expected || rl[i,j] != expected || ss[i,j] != expected)
                throw new InvalidOperationException("Linear-layer identity failed.");
        }
        var forward = MatrixMultiply(MatrixMultiply(inverse, sigma), l);
        for (int i = 0; i < 16; i++) for (int j = 0; j < 16; j++) Q[i,j] = forward[j,i];
    }

    private static bool Outside(byte[] v, byte[][] basis, int n)
    {
        for (int i = 0; i < n; i++) if (v[i] != 0) {
            if (basis[i] == null) return true;
            byte factor = v[i];
            for (int j = i; j < n; j++) v[j] ^= Mul[factor,basis[i][j]];
        }
        return false;
    }

    // Check that every required input and output coordinate is nonzero on the kernel.
    private static int Analyze(int inputSupport, int outputSupport, int a)
    {
        int[] indices = new int[a];
        int k = 0;
        for (int i = 0; i < 16; i++) if (((inputSupport >> i) & 1) != 0) indices[k++] = i;
        var basis = new byte[a][];
        int rank = 0;
        for (int o = 0; o < 16; o++) if (((outputSupport >> o) & 1) == 0) {
            var v = new byte[a];
            for (int j = 0; j < a; j++) v[j] = Q[o,indices[j]];
            for (int c = 0; c < a; c++) if (v[c] != 0) {
                if (basis[c] != null) {
                    byte factor = v[c];
                    for (int j = c; j < a; j++) v[j] ^= Mul[factor,basis[c][j]];
                } else {
                    byte factor = Inv[v[c]];
                    for (int j = c; j < a; j++) v[j] = Mul[factor,v[j]];
                    basis[c] = v;
                    rank++;
                    break;
                }
            }
        }
        if (rank == a) return -1;
        for (int c = 0; c < a; c++) {
            var v = new byte[a]; v[c] = 1;
            if (!Outside(v, basis, a)) return -1;
        }
        for (int o = 0; o < 16; o++) if (((outputSupport >> o) & 1) != 0) {
            var v = new byte[a];
            for (int j = 0; j < a; j++) v[j] = Q[o,indices[j]];
            if (!Outside(v, basis, a)) return -1;
        }
        return a-rank;
    }

    public static void Run()
    {
        Init();
        int[] popcount = new int[65536];
        var patterns = new List<int>[17];
        for (int i = 0; i < 17; i++) patterns[i] = new List<int>();
        for (int p = 1; p < 65536; p++) {
            popcount[p] = popcount[p >> 1] + (p & 1);
            patterns[popcount[p]].Add(p);
        }
        int[] groups = {0x8421, 0x4218, 0x2184, 0x1842};
        long all = 0, tested = 0, feasible = 0, equal = 0;
        long[] activity = new long[19];
        for (int b = 1; b <= 16; b++) foreach (int p in patterns[b])
            for (int a = 1; a <= 8 && 2*a+b <= 18; a++) foreach (int m in patterns[a]) {
                all++;
                bool valid = true;
                int db = 0;
                for (int j = 0; j < 4; j++) {
                    int outputWeight = popcount[(p >> (4*j)) & 15];
                    int inputWeight = popcount[m & groups[j]];
                    if (outputWeight > 0) {
                        if (outputWeight + inputWeight < 5) { valid = false; break; }
                        db += outputWeight + inputWeight - 4;
                    }
                }
                if (!valid) continue;
                tested++;
                int dc = Analyze(m, p, a);
                if (dc < 0) continue;
                feasible++;
                if (m == p) equal++;
                activity[2*a+b]++;
                Console.WriteLine("hit mu1={0:x4} mu2={1:x4} a={2} b={3} dB={4} dC={5}", m, p, a, b, db, dc);
            }
        Console.WriteLine("all={0} screened={1} feasible={2} same={3}", all, tested, feasible, equal);
        for (int i = 1; i <= 18; i++)
            if (activity[i] > 0) Console.WriteLine("activity={0} count={1}", i, activity[i]);
        if (all != 472985832 || tested != 925588 || feasible != 4 || equal != 4 || activity[18] != 4)
            throw new InvalidOperationException("Independent support verification did not match its expected invariants.");
    }
}
