# Code for "Improved Correlation and Distinguishing Attacks on SNOW-Axn"

All Python scripts are Python 3 with `numpy`. Run them from this directory.
`dev/` holds development history and is not needed for reproduction.

## Scripts that produce a number quoted in the paper

| Script | Reproduces | Expected output |
| --- | --- | --- |
| `verify.py` | LAT and ILAT tables, the optimal triple of Eq. (optmasks), the designers' triple | `log2 tau* = -54.7706`, designers `-55.7414`, gap `0.9709` bits |
| `global_support_census.cs` | the support census of Section 3.3 | `all=472985832 screened=925588 feasible=4 same=4`, four hits with `activity=18` |
| `global_support_verify.cs` | the same census by row-space membership | identical output |
| `global_support_matrix.py` | the two byte matrices and the MDS blocks | both matrices agree on all 256 basis pullbacks, all 276 minors nonzero |
| `global_optimum_scan.py` | the exact optimum of the four pairs and the moment sums | `mu=7070 ... log2tau=-54.770580312`, three at `-55.155870468` |
| `audit_attack_costs.py` | Table 4 and the statistical constants | `bg=241 ... time=250.919215250`, designers `252.201402061` |
| `verify_list.py` | the span of `../data/w24.json` | rank 24, 6,138,951 non-zero, `log2_sum4=-216.8436669675`, `log2_sum6=-327.2032268545` |
| `verify_mask_relations.py` | the separated-time kernels of Section 5 | `T=(0, 2, 3) rank=383 nullity=1`, `T=(0, 1, 2, 4) rank=480 nullity=32` |
| `verify_synchronous_collision.py` | the vanished simultaneous collision of Section 4.2 | `kernel_dimension=0` for `Bg` 240, 241 and 243 |

## Scripts that support the argument rather than a number

`verify_exact.py` and `profile.py` enumerate the profile of Section 4.1, `levels.py` the level
list of Table 5, `ties.py` the multiplicity of the optimum, and `fourth2.py` the
fourth-power sums. `feas.py`, `feas2.py`, `screen24.py` and `structure_check.py` check the
feasibility and the dimensions of the four patterns. The remaining top-level files are
independent checks of the same numbers.

## Data

`../data/w24.json` holds the 24 triples that span W, with byte 0 at the left end and the
selection rule in the file. Its SHA-256 is
`68ef228943f9f4a2fafa571e28ce43bc06a9437b68662c12b22b8fe927a256e1`, and `../data/w24.log` is
the output of the last run of `verify_list.py`.

## Runtime

`verify_list.py` enumerates 2^24 sums and takes about two minutes. The two C# censuses take
about three seconds each, and the other scripts finish in seconds.

## Licence

The code is released under the MIT licence; see `../LICENSE`. The paper text and figures
are released under CC-BY-4.0.
