"""
Small demo script for driving the Cthulhu TDU Arduino serial sketch.
"""

import argparse
import time
import numpy as np

from riglib.cthulhu_display.cthulhu_display_tdu import CthulhuTDU


def moving_dot_frame(rows, cols, phase):
    frame = np.zeros((rows, cols), dtype=float)
    r = int((rows - 1) * (0.5 + 0.45 * np.sin(phase * 0.07)))
    c = int((cols - 1) * (0.5 + 0.45 * np.cos(phase * 0.11)))
    frame[r, c] = 1.0

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = np.clip(r + dr, 0, rows - 1)
            cc = np.clip(c + dc, 0, cols - 1)
            frame[rr, cc] = max(frame[rr, cc], 0.4)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None, help="Serial port, e.g. /dev/cthulhu_display_tdu. If omitted, auto-detects Arduino.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--threshold", type=float, default=0.04)
    parser.add_argument("--intensity", type=float, default=0.45)
    args = parser.parse_args()

    client = CthulhuTDU(port=args.port, baudrate=args.baud)
    client.set_smoothing(args.alpha, args.threshold)
    client.set_intensity(args.intensity)

    dt = 1.0 / max(1.0, args.fps)
    phase = 0

    print("Streaming TDU demo frames. Press Ctrl+C to stop.")
    try:
        while True:
            frame = moving_dot_frame(args.rows, args.cols, phase)
            client.send_grid(frame)
            phase += 1
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        client.clear()
        client.close()


if __name__ == "__main__":
    main()
