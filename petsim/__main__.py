"""Entry point: ``python -m petsim`` or the ``petsim`` console script."""

from __future__ import annotations

import sys


def main() -> int:
    from petsim.app import run

    return run()


if __name__ == "__main__":
    sys.exit(main())
