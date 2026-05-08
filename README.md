# Earthquake Synchronization — Code and Data Repository

This repository contains the analysis code accompanying the GRL paper on synchronization between faults. It is organized into three self-contained modules, each with its own environment and data.

## Repository Structure

```
.
├── spring_block_sliders/   # Coupled spring-block slider model and simulations
├── japan_analysis/         # Numerical fault models supporting Nankai synchronization
└── repeaters/              # Repeating earthquake analysis (San Andreas & subduction zones)
```

---

## Modules

### `spring_block_sliders/`

Implements and analyzes a coupled spring-block slider model of two interacting faults using rate-and-state friction. The core class `CoupledSpringBlock` (in [two_spring_block.py](spring_block_sliders/two_spring_block.py)) integrates the equations of motion and tracks slip deficit, stress, and event timing.

The notebook [clock_advance.ipynb](spring_block_sliders/clock_advance.ipynb) explores:
- Synchronization behavior as a function of fault asymmetry (normal stress ratio)
- Phase of one fault in the earthquake cycle of the other
- Iterated phase maps and steady-state phase distributions
- Parallel parameter sweeps over coupling strength and stress ratio

No separate environment file is provided for this module; dependencies are `numpy`, `scipy`, `matplotlib`, `joblib`, and `tqdm`.

---

### `japan_analysis/`

Contains model output and analysis for Nankai Trough fault synchronization. Results provide numerical support for the spring-block findings.

**Key files:**
- [analysis_Nankai.ipynb](japan_analysis/analysis_Nankai.ipynb) — main analysis notebook
- [utils_Nankai.py](japan_analysis/utils_Nankai.py) — utility functions
- [sync_Nankai.sh](japan_analysis/sync_Nankai.sh) — script for syncing model output from Box
- `Nankai_output/` — model output directories (stored on Box; use `sync_Nankai.sh` to download)

**Setup:**
```bash
conda env create -f japan_analysis/enviroment.yml
```

---

### `repeaters/`

Analyzes repeating earthquakes along the creeping section of the San Andreas Fault and neighboring subduction zones. Uses a catalog from Li et al. (2023) and historical subduction zone records from Philibosian & Meltzner (2020).

**Key files:**
- [SAFOD_repeaters.ipynb](repeaters/SAFOD_repeaters.ipynb) — microseismicity near SAFOD (Fig. 3)
- [repeater_analysis.ipynb](repeaters/repeater_analysis.ipynb) — full creeping section catalog (Figs. 2 and S1)
- [repeater.py](repeaters/repeater.py), [earthquake.py](repeaters/earthquake.py), [catalog.py](repeaters/catalog.py) — supporting classes
- [phase_util.py](repeaters/phase_util.py), [figure_util.py](repeaters/figure_util.py) — phase and plotting utilities
- `data/` — earthquake catalogs and CRE catalog (Takaaki 2022)

**Setup:**
```bash
conda env create -f repeaters/requirements.yml
conda activate repeaters
```

---

## References

- Y. Li, R. Bürgmann, T. Taira, *Spatiotemporal Variations of Surface Deformation, Shallow Creep Rate, and Slip Partitioning Between the San Andreas and Southern Calaveras Fault.* J. Geophys. Res. Solid Earth 128, e2022JB025363 (2023).
- B. Philibosian, A. J. Meltzner, *Segmentation and supercycles: A catalog of earthquake rupture patterns from the Sumatran Sunda Megathrust and other well-studied faults worldwide.* Quat. Sci. Rev. 241, 106390 (2020).
