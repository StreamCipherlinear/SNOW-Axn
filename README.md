# Code and data for "Improved Correlation and Distinguishing Attacks on SNOW-Axn"

This repository holds the code and the data that reproduce the computational results of
the paper. The paper itself is not part of the repository.

SNOW-Axn is a 256-bit AEAD stream cipher of the SNOW family. Its designers assessed
resistance to linear cryptanalysis with a three-word approximation of correlation
2^-55.741, obtained from a heuristic search, which left open whether a better
approximation exists. The paper settles that model exactly: enumerating every support
with at most eighteen active S-boxes leaves four supports, and their exact evaluation
gives the optimum 2^-54.771, which is 0.971 bits above the designers' trail and 0.771
bits short of the 2^-54 bound implied by their count of active S-boxes. With a known
LFSR state the optimal approximation recovers the 512-bit state at 2^250.919 in time
and 2^250.028 in data and memory, while without a known state the strongest
approximations form a 24-dimensional distinguisher that needs 2^327.203 keystream
words.

## Repository layout

    code/          Python 3 and C# scripts; code/README.md maps each script to the
                   paper object it reproduces
    code/dev/      development history, not needed for reproduction
    data/          w24.json (the 24 triples spanning W) and w24.log

## Links

The repository is <https://github.com/StreamCipherlinear/SNOW-Axn>.

## Requirements

* Python 3.10 or newer with numpy. The scripts were last run with Python 3.12 and
  numpy 2.3.5.
* The .NET SDK or Mono for the two C# censuses. Each of `global_support_census.cs`
  and `global_support_verify.cs` defines a class with a static `Run()` method and no
  entry point, so compile it together with a one-line `Main` that calls
  `SnowSupports.Run()` or `IndependentCensus.Run()`.

All Python scripts are run from the `code` directory, for example:

    cd code
    python verify.py

## Reproducing the results

| Paper object | Script | Expected output |
| --- | --- | --- |
| Table 1 (LAT and ILAT magnitude distribution), Eq. (19) and Table 3 (optimal and designers' triples) | `verify.py` | `log2 tau* = -54.7706`, designers `-55.7414`, gap `0.9709` bits |
| Section 3.3 support census, Table 2 (the four supports) | `global_support_census.cs` | `all=472985832 screened=925588 feasible=4 same=4`, four hits with `activity=18` |
| Section 3.3, the same census by row-space membership | `global_support_verify.cs` | identical output |
| Section 3.3 byte matrices and MDS blocks (proof of the four supports) | `global_support_matrix.py` | both matrices agree on all 256 basis pullbacks, all 276 minors nonzero |
| Eq. (18) and Table 2: exact optimum of the four supports | `global_optimum_scan.py` | `mu=7070 ... log2tau=-54.770580312`, three at `-55.155870468` |
| Table 4 and Eq. (20): state-recovery cost | `audit_attack_costs.py` | `bg=241 ... time=250.919215250`, designers `252.201402061` |
| Table 6 and Section 5: span of W, fourth and sixth power sums | `verify_list.py` | rank 24, 6,138,951 non-zero, `log2_sum4=-216.8436669675`, `log2_sum6=-327.2032268545` |
| Section 5: separated-time kernels | `verify_mask_relations.py` | `T=(0, 2, 3) rank=383 nullity=1`, `T=(0, 1, 2, 4) rank=480 nullity=32` |
| Section 4.2: synchronous collision kernel | `verify_synchronous_collision.py` | `kernel_dimension=0` for `Bg` 240, 241 and 243 |

The remaining top-level scripts support the argument rather than a single number.
`verify_exact.py` and `profile.py` enumerate the correlation profile of Section 4.1,
`levels.py` its level list, `ties.py` the multiplicity of the optimum, `fourth2.py` the
power sums and the check-level costing, `feas.py`, `feas2.py`, `screen24.py` and
`structure_check.py` the feasibility and dimensions of the four supports, and
`verify_space.py`, `pair_search.py`, `greedy_space.py`, `search_space.py`,
`space_search2.py` and `push_k24.py` the nested-space search behind Table 6.
`crosscheck_sei.py`, `audit_exact.py`, `audit_numbers.py`, `t24.py`,
`vectorial_snow.py`, `price_space.py`, `price_final.py`, `check_triple.py`,
`unit_tau.py`, `rank_check.py` and `verify_W2.py` are independent recomputations of
numbers that the table already covers.

## Data

`data/w24.json` holds the 24 triples that span W, with byte 0 leftmost; the selection
rule is recorded in the file. Its SHA-256 is
`68ef228943f9f4a2fafa571e28ce43bc06a9437b68662c12b22b8fe927a256e1`, and
`data/w24.log` is the output of the last run of `code/verify_list.py`.

## Runtime

`verify_list.py` enumerates 2^24 sums and takes about two minutes. The two C# censuses
take about three seconds each, and the other scripts finish in seconds.

## Licence

The code and data in this repository are released under the MIT licence; see
`LICENSE`.

## Citation

    @misc{snowa2026,
      title  = {Improved Correlation and Distinguishing Attacks on {SNOW-Axn}},
      author = {{The SNOW-Axn authors}},
      year   = {2026},
      note   = {Manuscript; venue pending}
    }
