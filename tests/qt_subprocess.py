"""Run Qt/GUI snippets in a child process so a native abort cannot kill pytest.

On macOS, ``QApplication`` + Matplotlib ``FigureCanvasQTAgg`` (used by
``MainWindow``) can still ``abort()`` inside the Qt platform plugin.  That
kills the whole interpreter.  Executing those checks in a subprocess lets
the parent pytest session continue and exit with a normal status code.
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
    env = os.environ.copy()
    if sys.platform == "darwin":
        # Never pass offscreen into the child on macOS.
        env["QT_QPA_PLATFORM"] = "minimal"
    else:
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
    # Hint for child scripts (optional).
    env["PETSIM_PYTEST_CHILD"] = "1"
    return env


def run_qt_script(script: str, *, timeout: float = 180.0) -> QtSubprocessResult:
    """Execute ``script`` with ``python -c`` under a headless Qt platform."""
    # Ensure the child prints a sentinel on success.
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
