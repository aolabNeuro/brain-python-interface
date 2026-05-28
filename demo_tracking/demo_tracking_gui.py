"""Small launcher GUI for DemoTracking task variants.

This module supports two modes:
1) GUI mode (default): opens a window with 3 demo buttons.
2) CLI mode: run a single demo directly with --demo <name>.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox

from built_in_tasks.manualcontrolmultitasks import TrackingTask
from demo_tracking.demo_features import MouseControl, ProgressBar, ScoreRewards, SpheresToImages
from riglib import experiment
from riglib.stereo_opengl.window import Window2D


DEMOS = {
    "moon_ref": {
        "label": "Moon Ref (No Disturbance)",
        "trajectory_amplitude": 6,
        "disturbance_amplitude": 0,
        "score_display_location": (-200, 0, 7),
    },
    "moon_disturbance": {
        "label": "Moon Disturbance Only",
        "trajectory_amplitude": 0,
        "disturbance_amplitude": 2,
        "score_display_location": (-200, 0, 7),
    },
    "moon_combined": {
        "label": "Moon Combined",
        "trajectory_amplitude": 6,
        "disturbance_amplitude": 2,
        "score_display_location": (-2, 0, 7),
    },
}


def init_exp(base_class, feats, seq=None, **kwargs):
    hostname = socket.gethostname()
    if hostname in ["pagaiisland2", "human-bmi"]:
        os.environ["DISPLAY"] = ":0.1"
    exp_type = experiment.make(base_class, feats=feats)
    exp = exp_type(seq, **kwargs) if seq is not None else exp_type(**kwargs)
    exp.init()
    return exp


def run_tracking_demo(name: str) -> None:
    if name not in DEMOS:
        valid = ", ".join(sorted(DEMOS.keys()))
        raise ValueError(f"Unknown demo '{name}'. Valid options: {valid}")

    demo = DEMOS[name]
    seq = TrackingTask.tracking_target_chain(
        nblocks=1,
        ntrials=2,
        time_length=9,
        ramp=1,
        ramp_down=0,
        num_primes=10,
        seed=42,
        sample_rate=60,
        dimensions=2,
        disturbance=True,
        boundaries=(-10, 10, -10, 10),
        decay_rate=None,
    )

    exp = init_exp(
        TrackingTask,
        [Window2D, MouseControl, SpheresToImages, ProgressBar, ScoreRewards],
        seq,
        window_size=(1280, 720),
        fullscreen=True,
        limit1d=False,
        trajectory_amplitude=demo["trajectory_amplitude"],
        disturbance_amplitude=demo["disturbance_amplitude"],
        lookahead_time=1,
        reward_time=4,
        score_display_location=demo["score_display_location"],
        score_multiplier=10000,
        cursor_radius=1,
        tracking_out_time=8,
        cursor_color="black",
        target_color="black",
    )
    exp.stereo_mode = "projection"
    exp.rotation = "xzy"
    exp.trajectory_type = "2d"
    exp.run()


class DemoTrackingLauncher:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Demo Tracking Launcher")
        self.root.geometry("420x240")
        self.proc: subprocess.Popen | None = None

        self.status_var = tk.StringVar(value="Ready")
        self._build_ui()
        self._poll_process()

    def _build_ui(self) -> None:
        frame = tk.Frame(self.root, padx=16, pady=16)
        frame.pack(fill=tk.BOTH, expand=True)

        title = tk.Label(frame, text="DemoTracking Demos", font=("Helvetica", 14, "bold"))
        title.pack(anchor="w", pady=(0, 12))

        self.buttons = []
        for key in ("moon_ref", "moon_disturbance", "moon_combined"):
            btn = tk.Button(
                frame,
                text=DEMOS[key]["label"],
                width=32,
                command=lambda demo_key=key: self.launch_demo(demo_key),
            )
            btn.pack(anchor="w", pady=4)
            self.buttons.append(btn)

        self.stop_btn = tk.Button(frame, text="Stop Current Demo", width=32, command=self.stop_demo, state=tk.DISABLED)
        self.stop_btn.pack(anchor="w", pady=(12, 6))

        status = tk.Label(frame, textvariable=self.status_var)
        status.pack(anchor="w", pady=(8, 0))

    def launch_demo(self, name: str) -> None:
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showinfo("Demo running", "A demo is already running. Stop it before starting another.")
            return

        try:
            if getattr(sys, "frozen", False):
                cmd = [sys.executable, "--demo", name]
            else:
                cmd = [sys.executable, "-m", "demo_tracking.demo_tracking_gui", "--demo", name]

            self.proc = subprocess.Popen(cmd)
            self.status_var.set(f"Running: {DEMOS[name]['label']}")
            self._set_running(True)
        except Exception as exc:
            self.proc = None
            self._set_running(False)
            messagebox.showerror("Launch failed", str(exc))

    def stop_demo(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            self._set_running(False)
            self.status_var.set("Ready")
            return

        self.proc.terminate()
        self.status_var.set("Stopped demo")
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        for button in self.buttons:
            button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_btn.configure(state=tk.NORMAL if running else tk.DISABLED)

    def _poll_process(self) -> None:
        if self.proc is not None:
            code = self.proc.poll()
            if code is not None:
                self.status_var.set("Ready" if code == 0 else f"Demo exited with code {code}")
                self.proc = None
                self._set_running(False)
        self.root.after(800, self._poll_process)

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DemoTracking launcher")
    parser.add_argument(
        "--demo",
        choices=sorted(DEMOS.keys()),
        help="Run a single demo directly instead of opening the GUI",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo:
        run_tracking_demo(args.demo)
    else:
        app = DemoTrackingLauncher()
        app.run()


if __name__ == "__main__":
    main()