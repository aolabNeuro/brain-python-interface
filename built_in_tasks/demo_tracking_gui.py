"""Compatibility shim for the relocated demo launcher module.

Use `python -m demo_tracking.demo_tracking_gui` for the canonical entrypoint.
"""

from demo_tracking.demo_tracking_gui import main


if __name__ == "__main__":
    main()
