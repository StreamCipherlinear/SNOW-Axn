// Complete support census of Section 3.3 (Table 2).
//
// Inputs : none; the trace-dual matrix Q is embedded below.
// Outputs: one line per feasible pair and the totals all/screened/feasible/same.
// Call   : SnowSupports.Run(); throws if the four-pair invariants fail.

using System;
using System.Collections.Generic;

// Complete support census.  Byte 0 is the least significant byte of a mask.
// Q is the transpose of L^{-1} sigma L in the GF(256) trace-dual byte basis.
public static class SnowSupports
{
    private static readonly byte[,] Q = {
        {28,11,29,14,9,18,14,18,27,13,26,9,13,23,11,22},
        {13,27,9,26,22,11,23,13,11,28,14,29,18,14,18,9},
        {9,23,13,18,26,13,29,11,14,22,11,18,27,9,28,14},
        {22,14,18,11,14,28,9,27,23,9,18,13,11,29,13,26},
        {29,14,14,22,28,18,9,9,13,18,27,13,11,11,26,23},
        {18,11,11,28,18,27,14,14,9,26,23,9,13,13,22,29},
        {26,9,9,23,29,22,13,13,11,18,28,11,14,14,27,18},
        {18,13,13,27,23,26,11,11,14,29,22,14,9,9,18,28},
        {27,13,26,9,13,23,11,22,28,11,29,14,9,18,14,18},
        {11,28,14,29,18,14,18,9,13,27,9,26,22,11,23,13},
        {14,22,11,18,27,9,28,14,9,23,13,18,26,13,29,11},
        {23,9,18,13,11,29,13,26,22,14,18,11,14,28,9,27},
        {13,18,27,13,11,11,26,23,29,14,14,22,28,18,9,9},
        {9,26,23,9,13,13,22,29,18,11,11,28,18,27,14,14},
        {11,18,28,11,14,14,27,18,26,9,9,23,29,22,13,13},
        {14,29,22,14,9,9,18,28,18,13,13,27,23,26,11,11}
    };
    private static readonly byte[,] Mul = new byte[256, 256];
    private static readonly byte[] Inv = new byte[256];

    private static void Init()
    {
        for (int a = 0; a < 256; a++)
            for (int b = 0; b < 256; b++) {
                int x = a, y = b, z = 0;
                while (y != 0) {
                    if ((y & 1) != 0) z ^= x;
                    x <<= 1;
                    if ((x & 256) != 0) x ^= 0x11b;
                    y >>= 1;
                }
                Mul[a, b] = (byte)z;
                if (z == 1) Inv[a] = (byte)b;
            }
    }

    // Return the kernel dimension, or -1 if a required input or output byte
    // is identically zero on the kernel.
    private static int CDim(int inputSupport, int outputSupport, int a)
    {
        int[] indices = new int[8], pivot = new int[8], free = new int[8], pivotRow = new int[8];
        int count = 0;
        for (int j = 0; j < 16; j++)
            if (((inputSupport >> j) & 1) != 0) indices[count++] = j;

        byte[,] rows = new byte[16, 8];
        int n = 0;
        for (int i = 0; i < 16; i++)
            if (((outputSupport >> i) & 1) == 0) {
                for (int j = 0; j < a; j++) rows[n, j] = Q[i, indices[j]];
                n++;
            }

        int rank = 0;
        for (int c = 0; c < a; c++) {
            int z = rank;
            while (z < n && rows[z, c] == 0) z++;
            if (z == n) continue;
            for (int j = 0; j < a; j++) {
                byte t = rows[z, j]; rows[z, j] = rows[rank, j]; rows[rank, j] = t;
            }
            byte iv = Inv[rows[rank, c]];
            for (int j = 0; j < a; j++) rows[rank, j] = Mul[iv, rows[rank, j]];
            for (int i = 0; i < n; i++)
                if (i != rank && rows[i, c] != 0) {
                    byte factor = rows[i, c];
                    for (int j = 0; j < a; j++) rows[i, j] ^= Mul[factor, rows[rank, j]];
                }
            pivot[rank] = c;
            pivotRow[c] = rank + 1;
            rank++;
        }
        if (rank == a) return -1;
        int nf = 0;
        for (int c = 0; c < a; c++) if (pivotRow[c] == 0) free[nf++] = c;
        for (int z = 0; z < rank; z++) {
            bool nonzero = false;
            for (int k = 0; k < nf; k++) if (rows[z, free[k]] != 0) { nonzero = true; break; }
            if (!nonzero) return -1;
        }
        for (int i = 0; i < 16; i++)
            if (((outputSupport >> i) & 1) != 0) {
                bool nonzero = false;
                for (int k = 0; k < nf; k++) {
                    int f = free[k];
                    byte v = Q[i, indices[f]];
                    for (int z = 0; z < rank; z++) v ^= Mul[Q[i, indices[pivot[z]]], rows[z, f]];
                    if (v != 0) { nonzero = true; break; }
                }
                if (!nonzero) return -1;
            }
        return nf;
    }

    public static void Run()
    {
        Init();
        int[] popcount = new int[65536];
        var patterns = new List<int>[17];
        for (int i = 0; i <= 16; i++) patterns[i] = new List<int>();
        for (int x = 1; x < 65536; x++) {
            popcount[x] = popcount[x >> 1] + (x & 1);
            patterns[popcount[x]].Add(x);
        }
        int[][] groups = {
            new[] {0,5,10,15}, new[] {3,4,9,14},
            new[] {2,7,8,13}, new[] {1,6,11,12}
        };
        long all = 0, tested = 0, feasible = 0, equal = 0;
        long[] activity = new long[19];
        for (int a = 1; a <= 8; a++) foreach (int m in patterns[a]) {
            int[] inputWeights = new int[4];
            for (int j = 0; j < 4; j++) foreach (int k in groups[j]) inputWeights[j] += (m >> k) & 1;
            for (int b = 1; b <= 18 - 2*a; b++) foreach (int p in patterns[b]) {
                all++;
                bool valid = true;
                int db = 0;
                for (int j = 0; j < 4; j++) {
                    int outputWeight = popcount[(p >> (4*j)) & 15];
                    if (outputWeight != 0) {
                        if (outputWeight + inputWeights[j] < 5) { valid = false; break; }
                        db += outputWeight + inputWeights[j] - 4;
                    }
                }
                if (!valid) continue;
                tested++;
                int dc = CDim(m, p, a);
                if (dc < 0) continue;
                feasible++;
                if (m == p) equal++;
                activity[2*a + b]++;
                Console.WriteLine("hit mu1={0:x4} mu2={1:x4} a={2} b={3} dB={4} dC={5}", m, p, a, b, db, dc);
            }
        }
        Console.WriteLine("all={0} screened={1} feasible={2} same={3}", all, tested, feasible, equal);
        for (int i = 1; i <= 18; i++)
            if (activity[i] > 0) Console.WriteLine("activity={0} count={1}", i, activity[i]);
        if (all != 472985832 || tested != 925588 || feasible != 4 || equal != 4 || activity[18] != 4)
            throw new InvalidOperationException("The complete support census did not match its expected invariants.");
    }
}
