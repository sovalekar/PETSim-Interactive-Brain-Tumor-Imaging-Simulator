# PETSim — Interactive Brain Tumor Imaging Simulator

Educational desktop workstation for **computational simulation of brain-tumor PET imaging**.  
Built with **Python 3.11+**, **PySide6**, **NumPy**, and embedded **Matplotlib**.

> **Not a clinical diagnostic tool.**

---

## Stages

| Stage | Status | Contents |
| --- | --- | --- |
| **1** | Done | Dark three-panel UI, adjustable brain/tumor phantom, 48-crystal ring |
| **2** | Done (v0.2.0) | Annihilation sampling, 511 keV back-to-back photons, efficiency, coincidences, LOR animation |
| **3** | **This release (v0.3.0)** | Measured sinogram from coincidences + filtered backprojection (FBP) |

---

## Stage 3 overview

Workflow:

1. Adjust brain / tumor parameters.  
2. Run a Stage 2 acquisition (Start Acquisition).  
3. Click **Run Reconstruction**.  
4. PETSim bins measured LORs into a sinogram \(p(\theta, s)\).  
5. PETSim applies **filtered backprojection** to form a \(128\times 128\) activity image.  
6. View **Reconstructed Image**, **Sinogram**, or **Comparison** (acquired phantom vs FBP).

Reconstruction uses **only coincidence endpoints** (detector-ring hits). True annihilation positions are never reconstruction inputs.

### Sinogram

Each accepted coincidence defines a line of response (LOR). In normal form:

\[
x\cos\theta + y\sin\theta = s
\]

- \(\theta \in [0,\pi)\): angle of the LOR normal  
- \(s\): signed perpendicular distance from the scanner center  

Events are binned on a default grid of **180 angular × 128 radial** bins (ring radius \(s_{\max}=0.92\)).  
The sinogram is accumulated from **simulated detector measurements**, not from a Radon transform of the ground-truth phantom.

### Filtered backprojection (FBP)

1. For each angle, FFT the radial profile (with zero-padding).  
2. Multiply by the **ramp** filter \(|\omega|\), optionally with a **Hann** window.  
3. Inverse FFT → filtered projections \(q(s,\theta)\).  
4. Backproject with linear interpolation:

\[
f(x,y) \approx \frac{\pi}{N_\theta}\sum_{\theta} q(x\cos\theta + y\sin\theta,\,\theta)
\]

Negatives from filter ringing are kept in the raw array; the display view clips at zero without mutating the reconstruction.

### Model assumptions (still idealized)

- Ideal back-to-back emission, 2-D geometry  
- No attenuation, scatter, randoms, positron range, or TOF  
- 48 crystals → limited distinct LORs → angular sampling artifacts are expected  

---

## Project layout

```
petsim/
├── acquisition_controller.py
├── reconstruction_worker.py      # QThread FBP worker
├── models/                       # Phantom + detector geometry
├── physics/                      # Stage 2 acquisition engine
├── reconstruction/               # Stage 3 (Qt-free)
│   ├── lor.py                    # Endpoints → (θ, s)
│   ├── sinogram.py               # Binning + analytic fixtures
│   ├── fbp.py                    # Ramp / Hann FBP
│   └── pipeline.py               # End-to-end reconstruct_from_endpoints
└── widgets/                      # Controls, detector view, recon panel
tests/
```

---

## Setup

```bash
git clone <repo-url> PETSim-Interactive-Brain-Tumor-Imaging-Simulator
cd PETSim-Interactive-Brain-Tumor-Imaging-Simulator

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e ".[dev]"            # pytest
```

---

## Run the application

```bash
python -m petsim
```

### Acquisition + reconstruction

1. Set tumor / brain parameters.  
2. Choose event count, detector efficiency, optional seed; click **Start Acquisition**.  
3. Wait until acquisition completes (or pause mid-run — reconstruction uses recorded coincidences so far).  
4. In the right panel: choose Filter (**Ramp** / **Hann**), Display mode, colormap.  
5. Click **Run Reconstruction**.  
6. Inspect the image, sinogram, or side-by-side comparison.  
7. Changing the phantom **clears** acquisition and reconstruction so datasets are never mixed.

---

## Tests

```bash
pip install -e ".[dev]"
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

**macOS note:** PySide6 frequently **native-aborts** with ``QT_QPA_PLATFORM=offscreen``.  
``tests/conftest.py`` automatically remaps Darwin ``offscreen`` → ``minimal`` before Qt loads.  
``MainWindow`` / Matplotlib QtAgg checks run in a **subprocess** so a plugin abort cannot kill the parent pytest process. You can also run explicitly:

```bash
QT_QPA_PLATFORM=minimal python -m pytest -q
```

Coverage includes LOR \(\leftrightarrow (\theta,s)\) conversion, sinogram binning, deterministic seeds, empty data, FBP shape/finiteness, an **analytic disk / point-source** projection test (independent of the PET geometry converter), tumor-location sensitivity, and UI state management.

---

## Known limitations

- Educational model only — not clinically accurate  
- Coarse 48-detector angular sampling → streaks / limited resolution  
- No attenuation correction, scatter, randoms, or iterative methods (OSEM / ML-EM)  
- Comparison phantom is the activity map frozen at acquisition start  

---

## License

MIT (see `pyproject.toml`).
