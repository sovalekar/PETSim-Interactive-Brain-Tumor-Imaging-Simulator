# PETSim — Interactive Brain Tumor Imaging Simulator

Educational desktop workstation for **computational simulation of brain-tumor PET imaging**.  
Built with **Python 3.11+**, **PySide6**, **NumPy**, and embedded **Matplotlib**.

> **Not a clinical diagnostic tool.** Stage 4 metrics are educational reconstruction-quality measures, not calibrated clinical PET quantities (SUV, Bq/mL, or validated detectability indices).

---

## Stages

| Stage | Status | Contents |
| --- | --- | --- |
| **1** | Done | Dark three-panel UI, adjustable brain/tumor phantom, 48-crystal ring |
| **2** | Done | Annihilation sampling, 511 keV back-to-back photons, efficiency, coincidences, LOR animation |
| **3** | Done | Measured sinogram from coincidences + filtered backprojection (FBP) |
| **4** | **This release (v0.4.0)** | Tumor localization, TBR / contrast / background CV, coincidence-count comparison |

---

## Stage 4 overview

After Stage 3 reconstruction, PETSim can quantitatively evaluate the reconstructed image against the phantom settings that generated the acquisition:

1. **Tumor localization accuracy** — estimate the tumor center from the reconstructed image and compare it with the known phantom center.  
2. **Tumor-to-background contrast** — ROI means, TBR, relative contrast, and background coefficient of variation.  
3. **Image quality vs coincidence count** — rebuild nested prefixes of the same coincidence list at 1k / 3k / 5k / 10k detected coincidences and tabulate metrics.

Workflow:

1. Enable a tumor and set a known center.  
2. Run an acquisition (prefer enough events to reach several thousand coincidences).  
3. **Run Reconstruction**.  
4. In **Quantitative Analysis**, click **Run Analysis**.  
5. Optionally click **Count Comparison** and inspect the table / CV plot.  
6. Resetting acquisition, clearing reconstruction, changing the phantom, or changing the FBP filter clears stale metrics.

### Tumor localization

Method: **constrained intensity-weighted centroid** on nonnegative reconstructed intensities.

1. Restrict the search to the circular brain disk (phantom *radius* only — not the tumor center).  
2. Find the brightest pixel in that region (coarse peak).  
3. Compute an intensity-weighted centroid in a local neighborhood around the peak.  
4. Convert the centroid to FOV coordinates using the same pixel-center grid as FBP (`linspace(-fov, fov, N)`, row↔y, col↔x).

Localization error:

\[
\sqrt{(x_\mathrm{est}-x_\mathrm{true})^2 + (y_\mathrm{est}-y_\mathrm{true})^2}
\]

reported in FOV units. The estimate comes from reconstructed image data only; ground-truth coordinates are used for error evaluation (and ROI placement below), never to invent the estimate. If the tumor is disabled or the search region has no positive intensity, the GUI shows an explanatory message instead of a fabricated measurement.

### Tumor / background ROIs and contrast

Intensity basis (documented and consistent): **nonnegative display-clipped** reconstructed values `max(image, 0)`. Raw FBP negatives are excluded so background means are not artificially lowered. These are relative reconstruction units — **not** clinical activity concentrations.

* **Tumor ROI**: circle at the documented ground-truth tumor center with radius `tumor_radius`.  
* **Background ROI**: annulus around the same center with inner radius `1.5 × tumor_radius` and width `0.12` FOV units, clipped to the brain disk and excluding tumor pixels.

Metrics:

* \(\mathrm{TBR} = \mu_\mathrm{tumor} / \mu_\mathrm{background}\)  
* \(\mathrm{Contrast} = (\mu_\mathrm{tumor} - \mu_\mathrm{background}) / \mu_\mathrm{background}\)  
* \(\mathrm{Background\ CV} = \sigma_\mathrm{background} / \mu_\mathrm{background}\)

Background CV is a variability measure; it is **not** automatically labeled tumor SNR. Zero / negative / NaN background means yield invalid ratios with a clear message. Tiny or empty ROIs are flagged.

### Coincidence-count comparison

Target levels are **detected coincidence counts**: 1000, 3000, 5000, 10000.

PETSim takes nested prefixes of the current measured coincidence list (the first N events), rebuilds the sinogram, runs FBP with the same settings, and recomputes localization error, TBR, and background CV. Levels above the available coincidence total are marked unavailable — events are never fabricated. Individual random realizations need not improve monotonically.

---

## Stage 3 overview (still included)

Workflow:

1. Adjust brain / tumor parameters.  
2. Run a Stage 2 acquisition (Start Acquisition).  
3. Click **Run Reconstruction**.  
4. PETSim bins measured LORs into a sinogram \(p(\theta, s)\).  
5. PETSim applies **filtered backprojection** to form a \(128\times 128\) activity image.  
6. View **Reconstructed Image**, **Sinogram**, **Comparison**, or **Count Comparison Plot**.

Reconstruction uses **only coincidence endpoints** (detector-ring hits). True annihilation positions are never reconstruction inputs.

### Sinogram

Each accepted coincidence defines a line of response (LOR). In normal form:

\[
x\cos\theta + y\sin\theta = s
\]

- \(\theta \in [0,\pi)\): angle of the LOR normal  
- \(s\): signed perpendicular distance from the scanner center  

Default educational grid: **96 angular × 128 radial** bins (ring radius \(s_{\max}=0.92\)), with mild Gaussian pre-smoothing before FBP.

### Filtered backprojection (FBP)

Default Hann-windowed ramp filter; negatives from ringing are kept in the raw array and clipped only for display (percentile windowing ≈99.5%).

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
├── analysis_worker.py            # QThread Stage 4 analysis worker
├── models/                       # Phantom + detector geometry
├── physics/                      # Stage 2 acquisition engine
├── reconstruction/               # Stage 3 (Qt-free)
│   ├── lor.py
│   ├── sinogram.py
│   ├── fbp.py
│   └── pipeline.py
├── analysis/                     # Stage 4 (Qt-free)
│   ├── coordinates.py            # Pixel ↔ FOV
│   ├── localization.py           # Constrained centroid
│   ├── roi.py                    # TBR / contrast / CV
│   ├── count_comparison.py       # Nested count levels
│   └── pipeline.py               # analyze_reconstruction
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
# or
./.venv/bin/python -m petsim
```

### Acquisition + reconstruction + analysis

1. Set tumor / brain parameters (enable tumor; note the true center).  
2. Choose event count, detector efficiency, optional seed; click **Start Acquisition**.  
3. Wait until acquisition completes (or pause — reconstruction uses recorded coincidences so far).  
4. In the right panel: choose Filter (**Ramp** / **Hann**), Display mode, colormap.  
5. Click **Run Reconstruction**.  
6. Click **Run Analysis** — inspect localization, TBR, contrast, and background CV.  
7. Optionally enable **Overlay tumor / background ROIs**.  
8. Click **Count Comparison** for the 1k–10k table and CV plot.  
9. Changing the phantom **clears** acquisition, reconstruction, and analysis so datasets are never mixed.

---

## Tests

```bash
pip install -e ".[dev]"

# Linux / CI (headless):
QT_QPA_PLATFORM=offscreen python -m pytest -q

# Project interpreter:
./.venv/bin/python -m pytest -q

# macOS (use cocoa — matches a working PETSim GUI):
QT_QPA_PLATFORM=cocoa python -m pytest -q
```

**macOS note:** ``offscreen`` and ``minimal`` are often broken with PySide6 even when
plugin files exist on disk (abort / exit 134), while **``cocoa`` works** and is what
the desktop app uses. ``tests/conftest.py`` therefore defaults Darwin to ``cocoa`` and
remaps ``offscreen``/``minimal`` → ``cocoa`` (an explicit ``cocoa`` choice is preserved).
``MainWindow`` checks still run in a subprocess so a plugin abort cannot kill parent pytest.

Coverage includes LOR \(\leftrightarrow (\theta,s)\) conversion, sinogram binning, FBP,
pixel↔FOV coordinates, synthetic-blob localization, ROI / TBR / CV edge cases,
coincidence-count comparison, analysis invalidation, and Stage 2–3 regression tests.

---

## Known limitations

- Educational model only — not clinically accurate or validated for patient imaging  
- Coarse 48-detector geometry + finite counts → residual streaks and localization scatter  
- No attenuation correction, scatter, randoms, or iterative methods (OSEM / ML-EM)  
- ROI geometry uses phantom ground truth for evaluation; localization estimates do not  
- Count-comparison curves are single-realization nested prefixes — not ensemble averages  
- Very low counts remain noisy; prefer ≥5k coincidences for stable demos  

---

## License

MIT (see `pyproject.toml`).
