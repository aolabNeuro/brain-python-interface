"""
Utilities for driving a Cthulhu tongue display unit (TDU) over serial.
"""

import time
import numpy as np

try:
    import serial
    import serial.tools.list_ports
except ImportError:  # pragma: no cover
    serial = None


def normalize_grid(grid):
    """
    Convert an input 2D array to uint8 [0, 255].

    Accepts float/int inputs. Float inputs in [0, 1] are scaled to [0, 255].
    Other float ranges are clipped to [0, 255].
    """
    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError("Grid must be a 2D array")

    if arr.size == 0:
        raise ValueError("Grid must not be empty")

    if np.issubdtype(arr.dtype, np.floating):
        finite_max = np.nanmax(arr)
        finite_min = np.nanmin(arr)
        if finite_min >= 0.0 and finite_max <= 1.0:
            arr = arr * 255.0

    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return arr


def build_grid_command(grid):
    """
    Build an ASCII command for the Arduino protocol.
    """
    grid_u8 = normalize_grid(grid)
    h, w = grid_u8.shape
    payload = " ".join(str(int(v)) for v in grid_u8.ravel(order="C"))
    return f"G {w} {h} {payload}\n".encode("ascii")


def gaussian_world_grid(
    cursor_pos,
    target_pos,
    bounds,
    grid_shape=(8, 8),
    axes=(0, 2),
    cursor_sigma_cm=1.2,
    target_sigma_cm=1.8,
    cursor_weight=1.0,
    target_weight=0.8,
):
    """
    Create an intensity grid from cursor and target positions.

    Parameters
    ----------
    cursor_pos, target_pos : array-like or None
        Position vectors from the task state. Only selected ``axes`` are used.
    bounds : tuple
        ``(x_min, x_max, y_min, y_max)`` in task/world units (typically cm).
    grid_shape : tuple
        Output ``(rows, cols)``.
    axes : tuple
        Which vector entries map to the 2D world plane.
    """
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    if rows < 1 or cols < 1:
        raise ValueError("grid_shape must have positive dimensions")

    x_min, x_max, y_min, y_max = [float(x) for x in bounds]
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("bounds must have increasing min/max")

    yy, xx = np.mgrid[0:rows, 0:cols]
    x_centers = x_min + (xx + 0.5) * (x_max - x_min) / cols
    y_centers = y_max - (yy + 0.5) * (y_max - y_min) / rows

    img = np.zeros((rows, cols), dtype=float)

    def add_blob(pos, sigma_cm, weight):
        if pos is None:
            return
        vec = np.asarray(pos, dtype=float).ravel()
        if vec.size <= max(axes):
            return

        px = vec[axes[0]]
        py = vec[axes[1]]
        sigma2 = max(1e-6, float(sigma_cm) ** 2)
        dist2 = (x_centers - px) ** 2 + (y_centers - py) ** 2
        img[:] += float(weight) * np.exp(-0.5 * dist2 / sigma2)

    add_blob(target_pos, target_sigma_cm, target_weight)
    add_blob(cursor_pos, cursor_sigma_cm, cursor_weight)

    max_val = np.max(img)
    if max_val > 0:
        img = img / max_val
    return (img * 255.0).astype(np.uint8)


class CthulhuTDU:
    """
    Serial client for the Arduino sketch in ``riglib/cthulhu_display/cthulhu_display.ino``.
    """

    @staticmethod
    def find_arduino_port():
        if serial is None:
            raise ImportError("pyserial is required for CthulhuTDU")

        ports = list(serial.tools.list_ports.comports())
        if len(ports) == 0:
            raise RuntimeError("No serial ports found; cannot auto-detect Cthulhu Arduino")

        keyword_priority = [
            "cthulhu",
            "arduino",
            "ttyacm",
            "ttyusb",
            "usb serial",
            "wchusbserial",
            "usbmodem",
            "ch340",
            "cp210",
        ]

        scored_ports = []
        for port_info in ports:
            device = (port_info.device or "").lower()
            description = (port_info.description or "").lower()
            manufacturer = (port_info.manufacturer or "").lower()
            haystack = f"{device} {description} {manufacturer}"

            score = 0
            for idx, keyword in enumerate(keyword_priority):
                if keyword in haystack:
                    score += (len(keyword_priority) - idx)

            if score > 0:
                scored_ports.append((score, port_info.device))

        if len(scored_ports) == 0:
            if len(ports) == 1:
                return ports[0].device

            port_names = ", ".join(p.device for p in ports)
            raise RuntimeError(
                "Unable to auto-detect Cthulhu Arduino port from available ports: %s" % port_names
            )

        scored_ports.sort(reverse=True)
        return scored_ports[0][1]

    def __init__(self, port=None, baudrate=115200, timeout=0.05, handshake_wait_s=2.0):
        if serial is None:
            raise ImportError("pyserial is required for CthulhuTDU")

        if port is None:
            port = self.find_arduino_port()

        self.port_name = port

        try:
            self.port = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        except Exception as exc:
            raise RuntimeError("Failed to open Cthulhu serial port %s: %s" % (port, exc)) from exc

        self._last_write_time = 0.0

        end = time.time() + float(handshake_wait_s)
        saw_ready = False
        while time.time() < end:
            line = self.port.readline()
            if not line:
                continue
            if (b"CTHULHU_DISPLAY_TDU_READY" in line) or (b"CTHULHU_TDU_READY" in line):
                saw_ready = True
                break

        if not saw_ready:
            for _ in range(3):
                self.port.write(b"?\n")
                line = self.port.readline()
                if (b"CTHULHU_DISPLAY_TDU" in line) or (b"CTHULHU_TDU" in line):
                    saw_ready = True
                    break

        self.ready = saw_ready

        if not self.ready:
            self.close()
            raise RuntimeError(
                "Connected to %s but did not receive CTHULHU_DISPLAY_TDU_READY within %.2f s"
                % (self.port_name, float(handshake_wait_s))
            )

    def close(self):
        if self.port is not None and self.port.is_open:
            self.port.close()

    def clear(self):
        self.port.write(b"Z\n")
        self._last_write_time = time.time()

    def set_smoothing(self, alpha=0.35, threshold=0.04):
        alpha = float(np.clip(alpha, 0.01, 1.0))
        threshold = float(np.clip(threshold, 0.0, 1.0))
        cmd = f"S {alpha:.3f} {threshold:.3f}\n".encode("ascii")
        self.port.write(cmd)
        self._last_write_time = time.time()

    def set_intensity(self, gain=0.45):
        gain = float(np.clip(gain, 0.0, 1.0))
        cmd = f"I {gain:.3f}\n".encode("ascii")
        self.port.write(cmd)
        self._last_write_time = time.time()

    def send_grid(self, grid):
        cmd = build_grid_command(grid)
        self.port.write(cmd)
        self._last_write_time = time.time()

    def send_world(
        self,
        cursor_pos,
        target_pos,
        bounds,
        grid_shape=(8, 8),
        axes=(0, 2),
        cursor_sigma_cm=1.2,
        target_sigma_cm=1.8,
        cursor_weight=1.0,
        target_weight=0.8,
    ):
        frame = gaussian_world_grid(
            cursor_pos=cursor_pos,
            target_pos=target_pos,
            bounds=bounds,
            grid_shape=grid_shape,
            axes=axes,
            cursor_sigma_cm=cursor_sigma_cm,
            target_sigma_cm=target_sigma_cm,
            cursor_weight=cursor_weight,
            target_weight=target_weight,
        )
        self.send_grid(frame)
        return frame
