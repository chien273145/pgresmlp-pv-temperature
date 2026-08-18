# PG-ResMLP — Physics-guided TinyML virtual sensing of PV module temperature

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21990310.svg)](https://doi.org/10.5281/zenodo.21990310)

Code, data, and results for the paper:

> **Physics-guided TinyML virtual sensing of photovoltaic module temperature across climates, sampling resolutions, and data regimes**
> La Ngoc Chien, Nguyen Khanh Duy, Do Cao Trung — Hanoi University of Science and Technology
> *(under review; citation will be updated upon acceptance)*

A 705-parameter residual MLP anchored on the Faiman thermal model, evaluated across 5 sampling resolutions × 2 tasks (n = 20 seeds, Holm–Bonferroni), replicated on 3 additional climates, with pre-registered causal ablations, Mondrian split-conformal uncertainty, and bit-validated ESP32 deployment (2.8 kB, 31 μs).

## One-command verification

```bash
python reproduce_all.py --skip-preprocess
```

Re-verifies **every number in the paper** against the result files in `results/` (11/11 tables). Each experiment section prints its regeneration command.

## Repository layout

| Path | Contents |
|---|---|
| `experiments/` | Training/experiment scripts (E13 resolution sweep, E13b cross-site, E13c wind ablation, E13d baselines, E14 conformal, transfer & ablation, Hanoi validation) |
| `analysis/` | Statistics + figure generation (paired tests, Holm–Bonferroni, journal-style figures) |
| `results/` | All result CSVs — the source of truth for every table |
| `figures/` | Fig. 1–7, vector PDF + 600-dpi PNG |
| `data/` | Hanoi 94-h field dataset (ours) + processed PVDAQ features |
| `esp32/` | Pure-C float32 firmware (weights, scaler, inference) |
| `paper_q1_supplementary.md` | Auto-generated per-seed tables S1–S9 |

## Data sources

| Dataset | Access |
|---|---|
| NREL PVDAQ system 7333 (California, 2022–23) | DOI [10.25984/1846021](https://doi.org/10.25984/1846021) — raw files not redistributed here; `data/pvdaq_7333_v2_2022_2023.csv` is the processed feature table (regenerable via `experiments/preprocess_pvdaq.py`) |
| NREL PERT (Cocoa FL / Eugene OR / Golden CO) | [datahub.duramat.org](https://datahub.duramat.org/) — modules mSi0166 / mSi0247 |
| Hanoi rooftop 94-h field record | `data/hanoi_real_4day_with_wind.csv` (this work, CC BY 4.0) |

## Reproducibility notes

- Fixed seeds `range(20)`; `keras.utils.set_random_seed` applied **before** model construction — resume ≡ fresh run (unit-tested).
- Batch size `min(256, max(32, n/60))` guarantees a matched optimisation budget at every resolution.
- Gap-aware windowing: windows/lags valid only over timestamps contiguous at exactly Δt.
- Environment: Python 3.11, TensorFlow 2.20, scikit-learn, pandas, scipy (CPU-only; results tolerance ±0.01 °C across hardware).

## License

Code: MIT (see `LICENSE`). Hanoi field dataset: CC BY 4.0. NREL datasets remain under their original terms — obtain from the sources above.
