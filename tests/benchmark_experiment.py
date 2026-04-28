import argparse
import sys
import time
from dataclasses import dataclass

from built_in_tasks.passivetasks import TargetCaptureVisualFeedback
from built_in_tasks.target_capture_task import ScreenTargetCapture
from features.simulation_features import SimClock
from riglib.stereo_opengl.window import Window2D


@dataclass
class Sample:
    t: float
    elapsed: float
    cycles: int
    hz: float


class VFBWindow2DRegression(TargetCaptureVisualFeedback, Window2D):
    """Minimal high-throughput graphics benchmark used as a slowdown regression check."""

    def __init__(
        self,
        gen,
        fps=600,
        duration_s=30.0,
        report_interval_s=15.0,
        use_sim_clock=True,
        **kwargs,
    ):
        self.fps = int(fps)
        self.duration_s = float(duration_s)
        self.report_interval_s = float(report_interval_s)
        self.use_sim_clock = bool(use_sim_clock)

        self.t_start = None
        self.last_report_t = None
        self.last_report_cycle = 0
        self.samples = []
        self.wall_t0 = time.perf_counter()

        super().__init__(gen, **kwargs)
        self.assist_level = (1, 1)

    def init(self):
        if self.use_sim_clock:
            self.clock = SimClock()
        super().init()

    def _get_event(self):
        return None

    def _start_wait(self):
        self.wait_time = 0.0
        super()._start_wait()

    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause

    def _test_stop(self, ts):
        return super()._test_stop(ts)

    def _start_target(self):
        super()._start_target()
        if self.t_start is None:
            now = time.perf_counter()
            self.t_start = now
            self.last_report_t = now
            self.last_report_cycle = self.cycle_count

    def _cycle(self):
        super()._cycle()

        if self.t_start is None:
            return

        now = time.perf_counter()
        elapsed = now - self.last_report_t
        if elapsed >= self.report_interval_s:
            cycles = self.cycle_count - self.last_report_cycle
            hz = cycles / elapsed if elapsed > 0 else 0.0
            sample = Sample(t=now - self.t_start, elapsed=elapsed, cycles=cycles, hz=hz)
            self.samples.append(sample)
            print(
                f"t={sample.t:8.1f}s | window={sample.elapsed:6.2f}s | "
                f"cycles={sample.cycles:8d} | effective_hz={sample.hz:8.2f}"
            )
            self.last_report_t = now
            self.last_report_cycle = self.cycle_count

        if (time.perf_counter() - self.wall_t0) >= self.duration_s:
            self.end_task()


def summarize(samples):
    if not samples:
        return None

    hz_vals = [s.hz for s in samples]
    first_hz = hz_vals[0]
    last_hz = hz_vals[-1]
    mean_hz = sum(hz_vals) / len(hz_vals)
    drift_pct = ((last_hz - first_hz) / first_hz * 100.0) if first_hz else 0.0
    return dict(
        samples=len(samples),
        first_hz=first_hz,
        last_hz=last_hz,
        mean_hz=mean_hz,
        drift_pct=drift_pct,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal Window2D slowdown regression benchmark for visual-feedback target capture"
    )
    parser.add_argument("--fps", type=int, default=600)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--report-interval", type=float, default=15.0)
    parser.add_argument("--max-negative-drift-pct", type=float, default=10.0)
    parser.add_argument("--window-width", type=int, default=640)
    parser.add_argument("--window-height", type=int, default=480)
    parser.add_argument("--nblocks", type=int, default=500)
    parser.add_argument("--distance", type=float, default=2.5)
    parser.add_argument("--assist-speed", type=float, default=25.0)
    parser.add_argument("--target-radius", type=float, default=4.0)
    parser.add_argument("--hold-time", type=float, default=0.0)
    parser.add_argument("--delay-time", type=float, default=0.0)
    parser.add_argument("--timeout-time", type=float, default=1.0)
    parser.add_argument("--reward-time", type=float, default=0.02)
    parser.add_argument("--penalty-time", type=float, default=0.02)
    parser.add_argument("--sim-clock", action="store_true", help="Use SimClock to remove loop sleep pacing")
    return parser.parse_args()


def main():
    args = parse_args()
    print(
        f"Starting minimal VFB Window2D regression benchmark: fps={args.fps}, duration={args.duration:.1f}s, "
        f"report_interval={args.report_interval}s"
    )

    gen = ScreenTargetCapture.centerout_2D(
        nblocks=args.nblocks,
        ntargets=8,
        distance=args.distance,
        origin=(0, 0, 0),
    )

    bench = VFBWindow2DRegression(
        gen,
        fps=args.fps,
        duration_s=args.duration,
        report_interval_s=args.report_interval,
        use_sim_clock=args.sim_clock,
        assist_speed=args.assist_speed,
        target_radius=args.target_radius,
        hold_time=args.hold_time,
        delay_time=args.delay_time,
        timeout_time=args.timeout_time,
        reward_time=args.reward_time,
        hold_penalty_time=args.penalty_time,
        delay_penalty_time=args.penalty_time,
        timeout_penalty_time=args.penalty_time,
        max_attempts=1,
        num_targets_per_attempt=1,
        window_size=(args.window_width, args.window_height),
        fullscreen=False,
        screen_dist=50,
        screen_half_height=22.5,
    )

    wall_t0 = time.perf_counter()
    bench.run_sync()
    wall_elapsed = time.perf_counter() - wall_t0

    print("\nBenchmark complete")
    print(f"wall_elapsed_s={wall_elapsed:.3f}")
    print(f"total_cycles={bench.cycle_count}")
    print(f"overall_hz={bench.cycle_count / wall_elapsed:.2f}")
    print(f"rewards={bench.calc_state_occurrences('reward')}")
    print(f"trials={bench.calc_trial_num()}")

    summary = summarize(bench.samples)
    if summary is None:
        print("No throughput samples collected")
        return 2

    print("\nWindowed throughput summary")
    print(f"samples={summary['samples']}")
    print(f"first_hz={summary['first_hz']:.2f}")
    print(f"last_hz={summary['last_hz']:.2f}")
    print(f"mean_hz={summary['mean_hz']:.2f}")
    print(f"drift_pct={summary['drift_pct']:.2f}%")

    if summary['drift_pct'] < -abs(args.max_negative_drift_pct):
        print(
            f"FAIL: negative throughput drift {summary['drift_pct']:.2f}% exceeded "
            f"threshold {-abs(args.max_negative_drift_pct):.2f}%"
        )
        return 1

    print("PASS: throughput drift stayed within threshold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
