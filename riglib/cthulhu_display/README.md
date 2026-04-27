# Cthulhu TDU serial demo

This folder contains an Arduino sketch (`cthulhu_display.ino`) for driving a Cthulhu tongue display unit (TDU) from serial image frames.

## 1) Flash the Arduino sketch

Flash [cthulhu_display.ino](cthulhu_display.ino) to the Arduino attached to the Cthulhu shield.

Protocol from host to Arduino (newline terminated):

- `G <w> <h> <p0> ... <pN-1>`: frame, row-major, pixel values `0..255`
- `S <alpha> <threshold>`: smoothing and activation threshold
- `I <gain>`: global intensity gain `0..1`
- `Z`: clear
- `?`: print status

The sketch applies:

- temporal exponential smoothing on incoming pixel values
- bilinear interpolation from input grid to electrode map
- 4-phase subpixel scanning to increase perceived spatial detail
- automatic stimulation timeout: if no new `G` frame arrives for about 250 ms, the Arduino clears the display and stops stimulating
- reduced default intensity with programmable global gain

## 2) Run the Python serial demo

Use:

```bash
python -m riglib.cthulhu_display.cthulhu_display_tdu_demo --port /dev/cthulhu_display_tdu
```

or directly:

```bash
python riglib/cthulhu_display/cthulhu_display_tdu_demo.py --port /dev/cthulhu_display_tdu
```

## 3) Use with BMI3D

A new feature mixin is available:

- `features.peripheral_device_features.CthulhuTDUFeedback`
- built-in feature key: `cthulhu_display_tdu`

This feature streams a low-res world representation (cursor + target blobs) to the TDU during task cycles.
It is designed for `ScreenTargetCapture`-style tasks that expose `plant` and `target_location`.
