# DemoTracking GUI and Standalone Binary

This repository now includes a small GUI launcher with three buttons for the DemoTracking variants:

- Moon Ref (No Disturbance)
- Moon Disturbance Only
- Moon Combined

## Run from source

From the repository root:

```bash
python -m demo_tracking.demo_tracking_gui
```

To run one demo directly (without the launcher GUI):

```bash
python -m demo_tracking.demo_tracking_gui --demo moon_ref
python -m demo_tracking.demo_tracking_gui --demo moon_disturbance
python -m demo_tracking.demo_tracking_gui --demo moon_combined
```

## Build standalone binary (PyInstaller)

From the repository root:

```bash
chmod +x build-demo-tracking-binary.sh
./build-demo-tracking-binary.sh
```

If your shell `python` is not the environment where PyInstaller is installed, pass it explicitly:

```bash
PYTHON_BIN=/Users/leoscholl/miniconda3/envs/leo-analysis/bin/python ./build-demo-tracking-binary.sh
```

The built artifact is:

```bash
dist/demo-tracking-launcher
```

## Notes

- The binary bundles `features/images/moon.png` and `features/images/ship.png` for the textured demo visuals.
- The launcher starts each demo in a subprocess, so the GUI remains responsive.
- The build script uses `demo-tracking-launcher.spec` (with a higher recursion limit) to avoid PyInstaller recursion errors in this codebase.
- If you need a macOS `.app` bundle instead of the single-file binary, change the PyInstaller options to remove `--onefile`.