# PETSim — Interactive Brain Tumor Imaging Simulator

Educational desktop workstation for **computational simulation of brain-tumor PET imaging**.  
Built with **Python 3.11+**, **PySide6**, **NumPy**, and embedded **Matplotlib**.

> **Not a clinical diagnostic tool.**

---

## Current stage: Stage 2 — Acquisition

| Stage | Status | Contents |
| --- | --- | --- |
| **1** | Done | Dark three-panel UI, adjustable brain/tumor phantom, 48-crystal ring |
| **2** | **This release** | Positron annihilation, 511 keV back-to-back photons, detector efficiency, coincidences, animated acquisition |
| **3** | Not started | Sinograms / FBP / OSEM / ML-EM reconstruction (panel reserved, disabled) |

### Stage 2 model (intentional simplifications)

Positron annihilation:

\[
e^{+} + e^{-} \rightarrow 2\gamma \quad (511\,\mathrm{keV\ each})
\]

Coincidence detection probability with independent single-photon efficiency \(\varepsilon\):

\[
P(\mathrm{coincidence}) = \varepsilon^{2}
\]

**Assumptions for this stage:** ideal back-to-back emission, **no** tissue attenuation, **no** scattering, **no** positron range. Photon paths are computed from sampled annihilation sites to the circular detector ring.

---

## Features

| Panel | Role |
| --- | --- |
| **Simulation Controls** | Brain/tumor parameters; Start / Pause / Reset acquisition; event count; detector efficiency; animation speed; optional seed; live stats |
| **Acquisition Geometry** | 48-crystal ring + phantom; animated LORs (green = coincidence, red = miss); crystal illumination |
| **Reconstruction** | **Unavailable (Stage 3)** — shows coincidence count only; no fake images |

Changing brain or tumor parameters **resets** acquisition data so events from different uptake maps are never mixed.

---

## Project layout

```
petsim/
├── app.py / main_window.py / theme.py
├── acquisition_controller.py   # QTimer loop (UI ↔ physics)
├── models/                     # Phantom + detector geometry
├── physics/                    # Stage 2 engine (Qt-free)
│   ├── geometry.py             # Ray–ring intersection, crystal index
│   ├── pet_engine.py           # Sampling, photons, coincidences
│   └── acquisition.py          # Buffer + stats + config
└── widgets/                    # Controls, detector view, recon placeholder
tests/                          # Physics + offscreen UI tests
```

---

## Setup

```bash
git clone <repo-url> PETSim-Interactive-Brain-Tumor-Imaging-Simulator
cd PETSim-Interactive-Brain-Tumor-Imaging-Simulator

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e ".[dev]"            # optional: pytest
```

---

## Run

```bash
python -m petsim
```

Or after an editable install:

```bash
petsim
```

### Tests

```bash
pip install -e ".[dev]"
QT_QPA_PLATFORM=offscreen pytest
```

---

## Using Stage 2 acquisition

1. Adjust **brain / tumor** uptake as desired.
2. Set **Events** (1,000–20,000), **Det. efficiency** (20–100%), and **Anim. speed**.
3. Optionally enable a **Random seed** for reproducible runs.
4. Click **Start Acquisition**. Pause / Resume / Reset as needed.
5. Watch LORs update in the center panel; compare the live coincidence rate to \(\varepsilon^{2}\).

---

## License

MIT (see `pyproject.toml`).
