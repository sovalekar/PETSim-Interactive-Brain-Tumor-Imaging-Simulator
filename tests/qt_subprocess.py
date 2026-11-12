"""Run Qt/GUI snippets in a child process so a native abort cannot kill pytest.

``MainWindow`` pulls in Matplotlib ``FigureCanvasQTAgg``. If the chosen Qt
platform plugin aborts during widget construction, isolating that work in a
subprocess keeps the parent pytest session alive.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class QtSubprocessResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and "PETSIM_QT_SUBPROCESS_OK" in self.stdout


def qt_subprocess_env() -> dict[str, str]:
    """Environment for child Qt scripts (same policy as ``tests.conftest``)."""
    env = os.environ.copy()
    if sys.platform == "darwin":
        # cocoa works when the PETSim GUI launches; offscreen/minimal often do not.
        current = env.get("QT_QPA_PLATFORM", "cocoa")
        if current in {"offscreen", "minimal", ""}:
            env["QT_QPA_PLATFORM"] = "cocoa"
        else:
            env["QT_QPA_PLATFORM"] = current
    else:
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env["PETSIM_PYTEST_CHILD"] = "1"
    return env


def run_qt_script(script: str, *, timeout: float = 180.0) -> QtSubprocessResult:
    """Execute ``script`` with ``python -c`` under a working Qt platform."""
    wrapped = (
        "import sys\n"
        + script
        + "\nprint('PETSIM_QT_SUBPROCESS_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", wrapped],
        env=qt_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    )
    return QtSubprocessResult(proc.returncode, proc.stdout, proc.stderr)
