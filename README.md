# PETSim — Interactive Brain Tumor Imaging Simulator

Dark-themed desktop workstation shell for exploring a synthetic brain phantom inside a circular PET detector ring. Built with **Python 3.11+**, **PySide6**, **NumPy**, and embedded **Matplotlib**.

> **Scope of this release:** UI + phantom visualization only.  
> No PET physics simulation and no image reconstruction (the right-hand panel is intentionally reserved / empty).

## Features

- Three-panel workstation layout
  1. **Simulation controls** — brain size/intensity, tumor size/position/intensity, show/hide tumor
  2. **Detector ring & phantom** — 48 discrete detector elements around a synthetic activity map
  3. **Reconstruction** — placeholder panel reserved for future algorithms
- Dark clinical theme throughout
- Modular package layout ready for later `physics/` and `reconstruction/` modules

## Project layout

```
petsim/
  __main__.py          # python -m petsim
  app.py               # QApplication bootstrap
  theme.py             # dark palette + stylesheet
  models/
    phantom.py         # synthetic brain + tumor parameters / rasterization
    scanner.py         # 48-element ring geometry
  ui/
    main_window.py     # three-panel shell
    controls_panel.py
    detector_panel.py
    reconstruction_panel.py
  viz/
    detector_canvas.py # Matplotlib FOV canvas
  physics/             # reserved (empty)
  reconstruction/      # reserved (empty)
```

## Requirements

- Python 3.11 or newer
- See `requirements.txt` / `pyproject.toml` for `numpy`, `matplotlib`, `PySide6`

On Linux you may also need OS packages for Qt (e.g. `libegl1`, `libxkbcommon0`, and a working display or `QT_QPA_PLATFORM=offscreen` for headless smoke tests).

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
# or: pip install -r requirements.txt
```

## Run

```bash
python -m petsim
# or, after editable install:
petsim
```

Use the left-hand sliders to reshape the phantom; the center FOV updates immediately. The reconstruction panel stays empty by design.

## Development notes

- Phantom coordinates are normalized FOV units in `[-1, 1]`.
- Detector count defaults to **48** via `DetectorRingGeometry`.
- Future coincidence / LoR physics belongs in `petsim.physics`.
- Future reconstruction solvers belong in `petsim.reconstruction` and should drive the reserved UI panel — do not fake reconstructed images in the FOV view.

## License

MIT (see `pyproject.toml`).
