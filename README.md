# PETSim — Interactive Brain Tumor Imaging Simulator

Dark-themed desktop workstation scaffold for interactive PET brain-tumor **visualization**.  
Built with **Python 3.11+**, **PySide6**, **NumPy**, and embedded **Matplotlib**.

> **Scope of this release:** adjustable synthetic brain phantom + tumor display, a 48-element circular detector ring, and a reserved reconstruction panel.  
> **Not included yet:** PET physics, coincidence detection, projection/sinogram formation, or image reconstruction (and nothing is faked in their place).

---

## Features

| Panel | Role |
| --- | --- |
| **Simulation Controls** | Sliders/toggles for brain size & intensity, tumor position/size/intensity, colormap, detector labels |
| **Acquisition Geometry** | 48-crystal circular detector ring around a rasterized brain phantom (Matplotlib) |
| **Reconstruction** | Reserved placeholder — controls disabled until a real reconstructor lands |

---

## Project layout

```
petsim/
├── __init__.py
├── __main__.py          # python -m petsim
├── app.py               # QApplication bootstrap + dark theme
├── main_window.py       # Three-panel workstation shell
├── theme.py             # Dark clinical stylesheet + palette
├── models/
│   ├── phantom.py       # Synthetic brain + tumor raster (display only)
│   └── detector.py      # 48-element ring geometry (display only)
└── widgets/
    ├── controls_panel.py
    ├── detector_view.py
    └── reconstruction_panel.py
requirements.txt
pyproject.toml
README.md
```

---

## Requirements

- Python **3.11** or newer (3.12 tested)
- A display server for the GUI (native desktop, or X11/Wayland; see headless note below)
- Dependencies listed in `requirements.txt` / `pyproject.toml`:
  - `PySide6`
  - `numpy`
  - `matplotlib`

---

## Setup

```bash
# Clone
git clone <repo-url> PETSim-Interactive-Brain-Tumor-Imaging-Simulator
cd PETSim-Interactive-Brain-Tumor-Imaging-Simulator

# Recommended: virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional editable install (exposes the `petsim` console script)
pip install -e .
```

---

## Run

```bash
python -m petsim
```

Or, after an editable install:

```bash
petsim
```

### Headless / CI smoke check

The GUI needs a display. For a quick import/widget construction check without opening a window:

```bash
QT_QPA_PLATFORM=offscreen python -c "
from PySide6.QtWidgets import QApplication
import sys
app = QApplication(sys.argv)
from petsim.main_window import MainWindow
w = MainWindow()
print('ok', w.windowTitle())
"
```

---

## Using the controls

1. **Brain phantom** — radius and background intensity; optional outline.
2. **Tumor** — enable/disable, shift center (X/Y), change radius and hot-spot intensity.
3. **Display** — activity colormap (`hot`, `inferno`, …); toggle crystal arcs and every-8th detector labels.
4. **Reset defaults** — restore the starting phantom and display options.
5. **Reconstruction panel** — intentionally inactive; reserved for a future FBP/OSEM module.

---

## Design notes

- Coordinates are normalized FOV units with the detector ring near radius `0.92` and the FOV circle at radius `1.0`.
- The phantom is a simple elliptical brain silhouette plus a circular tumor; it is a **display raster**, not a physical activity distribution for forward projection.
- Modules under `petsim/models/` are kept free of Qt so they can later feed a physics engine without UI coupling.

---

## License

MIT (see `pyproject.toml`).
