# DAS practical: Intelligent Earth CDT, Natural Hazards (Nov 2026)

A hands-on practical on Distributed Acoustic Sensing (DAS) data and machine-learning denoising with [DAS-N2N](https://github.com/sachalapins/DAS-N2N-torch): icequakes on an Antarctic ice stream, a micro-earthquake in a geothermal borehole, and whale calls on a seafloor telecom cable.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sachalapins/ie-cdt-das-practical-2026/blob/main/DAS_practical.ipynb)

**Students:** click the badge above, then *Runtime → Change runtime type → T4 GPU*, and run the first cell. Everything else (packages, helper code, data) downloads automatically. To run locally instead, download `DAS_practical.ipynb` and open it in Jupyter (Python ≥ 3.10); the first cell installs what's missing.

| File | What it is |
|---|---|
| `DAS_practical.ipynb` | Student notebook |
| `das_utils.py` | Helper functions (plotting, filters, scoring, DAS-N2N wrapper), fetched by the notebook |
| Release `v1.0` assets | The data files listed below (downloaded automatically by the notebook) |

A solutions notebook with demonstrator notes is distributed separately.

## Data files

| File | Size | Contents |
|---|---|---|
| `rutford_event1.npz` | 8 MB | Rutford Ice Stream DAS, 2020-01-17 01:30:18.73 UTC, 3.5 s, 986 channels, **both fibres** (`data`, `data_fibre2`) |
| `rutford_event2.npz` | 8 MB | As above, 04:42:06.90 UTC (after the DAS-N2N training period) |
| `rutford_fibre_pairs_10s.npz` | 23 MB | 10 s of both fibres (01:30:24.73 UTC) for training a Noise2Noise model (Part 7) |
| `rutford_geophones_ev1/ev2.mseed`, `rutford_geophones_stations.xml` | <1 MB | 3-component geophones (EarthScope network 6L), raw counts plus responses |
| `forge_event.npz` | 7 MB | Utah FORGE well 78-32 DAS, 2019-04-27 17:30:30 UTC, 2.5 s, 2000 Hz |
| `ooi_whales.npz` | 57 MB | OOI RCA south cable DAS, 2021-11-02 10:36:09.84 UTC, 15 s, 28–52 km |

The `.npz` files hold `data` (time × channel; int16 counts for Rutford/OOI, float32 for FORGE) plus scalar metadata: `fs`, `dx`, `x0`, `starttime`, `name`, `units`, `notes`.
Fibre 2 at Rutford is re-ordered so that `data_fibre2[:, c]` is co-located with `data[:, c]` (in the original TDMS file, fibre-1 channel *i* pairs with channel 2220 − *i*).

## Data sources and credits
* **Rutford Ice Stream DAS**: BEAMISH project; see Lapins et al. (2024), *GJI*, doi:10.1093/gji/ggad460. Geophones: EarthScope FDSN network 6L.
* **Utah FORGE DAS**: Utah FORGE / US DOE Geothermal Data Repository (April 2019 Silixa iDAS stimulation monitoring data).
* **OOI DAS**: Ocean Observatories Initiative Regional Cabled Array community DAS experiment (Nov 2021); see Wilcock, Abadi & Lipovsky (2023), doi:10.1121/10.0017104.

The code in this repository is MIT-licensed. The data files are **not** covered by that licence: they remain subject to each provider's terms and should be cited as above.
