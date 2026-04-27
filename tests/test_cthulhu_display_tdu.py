import numpy as np
from types import SimpleNamespace

import riglib.cthulhu_display.cthulhu_display_tdu as cthulhu_display_tdu
from riglib.cthulhu_display.cthulhu_display_tdu import CthulhuTDU, normalize_grid, build_grid_command, gaussian_world_grid


def test_normalize_grid_float_unit_interval():
    grid = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=float)
    out = normalize_grid(grid)
    assert out.dtype == np.uint8
    assert out.shape == (2, 2)
    assert out[0, 0] == 0
    assert out[1, 0] == 255


def test_build_grid_command_format():
    grid = np.array([[0, 10], [20, 30]], dtype=np.uint8)
    cmd = build_grid_command(grid)
    assert cmd.startswith(b"G 2 2 ")
    assert cmd.endswith(b"\n")
    assert b"0 10 20 30" in cmd


def test_gaussian_world_grid_shape_and_range():
    grid = gaussian_world_grid(
        cursor_pos=np.array([0.0, 0.0, 0.0]),
        target_pos=np.array([3.0, 0.0, 3.0]),
        bounds=(-10.0, 10.0, -10.0, 10.0),
        grid_shape=(8, 8),
        axes=(0, 2),
    )
    assert grid.shape == (8, 8)
    assert grid.dtype == np.uint8
    assert np.max(grid) <= 255
    assert np.min(grid) >= 0
    assert np.max(grid) > 0


def test_find_arduino_port_prefers_arduino_keyword(monkeypatch):
    fake_ports = [
        SimpleNamespace(device="/dev/tty.usbmodemA", description="USB modem", manufacturer="ACME"),
        SimpleNamespace(device="/dev/tty.usbmodemB", description="Arduino Mega 2560", manufacturer="Arduino"),
    ]

    monkeypatch.setattr(cthulhu_display_tdu.serial.tools.list_ports, "comports", lambda: fake_ports)
    detected = CthulhuTDU.find_arduino_port()
    assert detected == "/dev/tty.usbmodemB"


def test_find_arduino_port_single_port_fallback(monkeypatch):
    fake_ports = [
        SimpleNamespace(device="/dev/tty.single", description="Unknown Device", manufacturer="Unknown"),
    ]

    monkeypatch.setattr(cthulhu_display_tdu.serial.tools.list_ports, "comports", lambda: fake_ports)
    detected = CthulhuTDU.find_arduino_port()
    assert detected == "/dev/tty.single"


def test_set_intensity_writes_command():
    writes = []

    class FakeSerial:
        def __init__(self):
            self.is_open = True

        def write(self, data):
            writes.append(data)

    client = CthulhuTDU.__new__(CthulhuTDU)
    client.port = FakeSerial()
    client._last_write_time = 0.0
    client.set_intensity(0.45)
    assert writes == [b"I 0.450\n"]
