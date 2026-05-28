"""Compatibility shim for the relocated demo launcher module.

Use `python -m demo_tasks.demo_tracking_task` for the canonical entrypoint.
"""

from demo_tasks.demo_tracking_task import main


if __name__ == "__main__":
    main()
