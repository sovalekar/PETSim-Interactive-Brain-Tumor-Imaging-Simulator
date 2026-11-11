"""Allow ``python -m petsim`` to launch the workstation."""

from petsim.app import main

if __name__ == "__main__":
    raise SystemExit(main())
