# Code for "Improved Correlation and Distinguishing Attacks on SNOW-Axn"

The code reproduces the computational results of the paper "Improved Correlation and
Distinguishing Attacks on SNOW-Axn". It is organised in two categories: the mask search,
which establishes the correlation results on the three-word approximation, and the attack
complexity, which prices the single-approximation, vectorial and distinguishing attacks.

    code/
      README.md
      mask_search/
        correlation/      LAT and ILAT tables, the per-byte factorisation, and the
                          exact correlation of the optimal and designers' triples
        search/           support census, exact optimum, feasibility and dimension
                          checks, and the search for the 24-mask space W
      attack_complexity/
        single/           single-approximation fast correlation attack
        vectorial/        vectorial fast correlation attack
        distinguishing/   distinguishing attack
      dev/                development history, not needed for reproduction

## Mask search

### correlation

`verify.py` rebuilds GF(2^8), the AES S-box, the linear layers L and sigma, and the LAT
and ILAT tables, factorises the three-block correlation per byte, and evaluates the
optimal triple of Eq. (19) and the designers' triple of Appendix A of the design
document. The designers' trail is reproduced box by box: 18 active S-boxes with table
magnitudes 32, 28 and 20 and total correlation 2^-55.7414.

| Script | Paper object | Expected output |
| --- | --- | --- |
| `verify.py` | Table 1 (LAT and ILAT magnitude distribution), Eq. (19), Table 3 | `log2 tau* = -54.7706`, designers `-55.7414`, gap `0.9709` bits |
| `check_triple.py` | Eq. (19), recomputed from the mask bytes printed in the paper | `log2 tau=-54.770580` |
| `unit_tau.py` | Eq. (19), byte-by-byte unit check | `log2 tau = -54.77058031176958` |
| `verify_exact.py` | Section 4.1 correlation profile, Theorem 1 | 11,093,111 non-zero, max `log2\|tau\| = -54.7706` with multiplicity 1, 3,295 levels |
| `profile.py` | Theorem 1 by exhaustive enumeration of Theta_mu | 11,093,111 non-zero, max `log2\|tau\| = -54.7706`, 3,295 distinct levels |
| `levels.py` | Table 5, the level list near the head | `all levels 0..7 (22 masks) + 2 of the 4 masks at level 8`; `uniquely determined: False` |
| `ties.py` | multiplicity of the bytewise maximum among the 24 strongest | 11,476 of 65,536 pairs have more than one maximiser; 72 tied active bytes among the 24 masks |
| `fourth2.py` | profile sums and the two cost rules behind Section 4.2 | `sum tau^2 = 2^-97.938`, `sum tau^4 = 2^-213.382`, `sum tau^6 = 2^-326.187` |
| `t24.py` | tau^2, tau^4 and the cross-term bound over the 24 strongest | `log2 sum tau^2 = -106.272`, `log2 sum tau^4 = -216.881`, relative bound `-69.003` |
| `verify_W2.py` | span of one 22 + 2 selection of the 24 strongest (development aid) | `python verify_W2.py 22,24`: rank 24, 6,254,429 non-zero; superseded by `verify_list.py` |

### search

`global_support_census.cs` and `global_support_verify.cs` enumerate every support with at
most eighteen active S-boxes and find the same four, each feasible with activity 18.
`global_optimum_scan.py` evaluates the exact optimum of those four supports.
`feas.py`, `feas2.py` and `structure_check.py` check feasibility and the dimensions 16
and 8, and `screen24.py` screens the choice of the last two masks. The remaining scripts
search the nested spans of the profile and return the 24 triples that span W;
`verify_list.py` in `attack_complexity/distinguishing` enumerates the resulting span.

| Script | Paper object | Expected output |
| --- | --- | --- |
| `global_support_census.cs` | Section 3.3 support census, Table 2 | `all=472985832 screened=925588 feasible=4 same=4`, four hits with `activity=18` |
| `global_support_verify.cs` | the same census by row-space membership | identical output |
| `global_support_matrix.py` | byte matrices and the four MDS blocks behind the proof | `256` basis pullbacks agree, all `276` minors nonzero |
| `global_optimum_scan.py` | Eq. (18) and Table 2, exact optimum of the four supports | `mu=7070 ... log2tau=-54.770580312`, three at `-55.155870468` |
| `feas.py` | random sparse feasibility of the 0/1 model | `feasible witness found: False` |
| `feas2.py` | structured feasibility with gamma = sigma(L(alpha)) | `structured hits: 6` |
| `structure_check.py` | Section 3.2 dimensions of B_mu and C_mu | `dim B = 16, dim C = 8, \|Theta\| = 2^24` for all four patterns |
| `screen24.py` | rank screening of the six 22 + 2 selections | `rank384 = 24` for all six, `rank(B,C) = 20` or `21` |
| `verify_space.py` | the two basis-selection strategies; writes `sei_nested.json` next to itself | `nested best: k=1 log2 T=221.5536`, `greedy best: k=1 log2 T=221.5536` |
| `search_space.py` | nested spans of the k strongest masks | `best k=1 log2 T=221.5536 log2 SEI=-219.0823` |
| `space_search2.py` | nested spans, basis-linear variant | `best: k=16, log2 T=108.0907, log2 SEI=-101.6195` |
| `greedy_space.py` | greedy basis selection from the profile | `best k=1 log2 T=221.5536 log2 SEI=-219.0823` |
| `pair_search.py` | pairs and triples from a 400-mask pool | `best pair: SEI -218.6562`; `best triple: SEI -218.3277` |
| `push_k24.py` | extends the nested search to k = 24; writes `sei_nested_24.json` next to itself | `k=21 ... k=24 log2 SEI = -216.9048 ... -216.8437` |
| `rank_check.py` | GF(2) rank of the 24 strongest complete triples | `rank of the 24 strongest masks = 24` |
| `crosscheck_sei.py` | independent SEI for k = 1, 2, 3 and sampled k = 24 | `k=1 direct SEI = -219.082321`, `k=2 = -218.656208`, `k=3 = -218.327675` |

## Attack complexity

Each subsection names the attack it prices, its references and the paper section. All
costs are base-2 logarithms of data N, time T and memory.

### single

Single-approximation fast correlation attack, Meier and Staffelbach (1989) and Selcuk (2008).
Prices Section 3.5.

| Script | Computes | Expected output |
| --- | --- | --- |
| `audit_attack_costs.py` | Table 4 and the statistical constants of the first stage | `bg=241 ... time=250.919215250`; designers `bg=243 ... time=252.201402061` |

### vectorial

Vectorial fast correlation attack, Zhang, Liu, Meier, Sun, Feng and Wu (ASIACRYPT
2025). Prices Section 4.2. The statistic is a vector over the space, one sample carries
the whole vector, and the rule assigns N = 4 kappa ln2 / SEI(V_v) keystream words,
T = N + kappa 2^kappa in time and 2^kappa words of memory, where kappa = 240 is the
number of state bits decoded in one step.

| Script | Computes | Expected output |
| --- | --- | --- |
| `price.py` | Table 6, Eq. (22), Eq. (23) and the Result of Section 4.2 | `v = 24: log2 N = 226.222 log2 T = 247.907 log2 mem = 240.000`, gain `3.012` bits |
| `sei_nested.json`, `sei_nested_24.json` | cached capacities for v = 1..20 and v = 21..24, read by `price.py` | the same files that `verify_space.py` and `push_k24.py` regenerate |

### distinguishing

Distinguishing attack, Nyberg and Hermelin (2007) and Yang, Johansson and Maximov (2020).
Prices Section 5.

| Script | Computes | Expected output |
| --- | --- | --- |
| `verify_list.py` | enumerates the span W of `../data/w24.json` | rank 24, 6,138,951 non-zero, `log2_sum4=-216.8436669675`, `log2_sum6=-327.2032268545` |
| `verify_mask_relations.py` | separated-time linear masking kernel ranks | `T=(0, 2, 3) rank=383 nullity=1`, `T=(0, 1, 2, 4) rank=480 nullity=32` |

## Data

`../data/w24.json` holds the 24 triples that span W, with byte 0 at the left end and the
selection rule in the file. Its SHA-256 is
`68ef228943f9f4a2fafa571e28ce43bc06a9437b68662c12b22b8fe927a256e1`, and `../data/w24.log`
is the output of the last run of `verify_list.py`.

The cached capacities are `attack_complexity/vectorial/sei_nested.json` for k = 1..20 and
`sei_nested_24.json` for k = 21..24. `verify_space.py` and `push_k24.py` regenerate the
same files, byte for byte, next to themselves in `mask_search/search/`.

## Runtime

All Python scripts are Python 3 with `numpy`. Every script outside `dev/` locates `verify.py` and the data
from its own path, so it can be run from any directory. `verify_list.py` enumerates 2^24
sums and takes about two minutes. `profile.py`, `verify_exact.py`, `t24.py`,
`screen24.py`, `push_k24.py` and the nested searches take tens of seconds; the two C#
censuses take about three seconds each, and the other scripts finish in seconds.

## Licence

The code is released under the MIT licence; see `../LICENSE`. The paper text and figures
are released under CC-BY-4.0.

`dev/` holds development history, one-off probes and superseded drafts; it is not needed
for reproduction. It also keeps `vectorial_snow.py` and `verify_synchronous_collision.py`,
which price and check the earlier paired-clock model that the paper no longer uses.
