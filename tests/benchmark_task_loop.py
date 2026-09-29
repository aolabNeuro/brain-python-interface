'''
Benchmark of the per-cycle cost of ManualControl (ScreenTargetCapture) with the frame
rate cap removed, broken down by phase of the cycle.

With no DISPLAY set it runs headless via SDL's offscreen (EGL) driver, so there is no
vsync and the numbers are the time the task *needs* per frame. On a rig (DISPLAY set) it
opens a real window, so 'display.flip' includes any vsync wait -- if flip takes ~1/refresh
per cycle, the display, not the CPU, is what limits the frame rate. Cursor input is
synthetic and chases the current target so the FSM goes through real trials.

Usage:
    python tests/benchmark_task_loop.py                       # all modes
    python tests/benchmark_task_loop.py --mode window2d --profile
    python tests/benchmark_task_loop.py --sink                # also pickle task_data to a pipe like SaveHDF
    python tests/benchmark_task_loop.py --finish              # glFinish after draw (GPU time)
    python tests/benchmark_task_loop.py --feats window2d saveHDF   # add features by built_in_features name

Modes:
    logic     no rendering at all (FakeWindow-style), pure task CPU cost
    window2d  Window2D (orthographic Renderer2D)
    window3d  default Window (stereo_mode='hmd' -> ShadowMapper, two passes)
'''
import os
if not os.environ.get('DISPLAY'):
    os.environ.setdefault('SDL_VIDEODRIVER', 'offscreen')
if os.environ.get('SDL_VIDEODRIVER') == 'offscreen':
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')  # offscreen contexts are EGL, not GLX
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import argparse
import cProfile
import collections
import multiprocessing as mp
import pstats
import subprocess
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from built_in_tasks.manualcontrolmultitasks import ManualControl, rotations, baseline_rotations, exp_rotations
from built_in_tasks.target_capture_task import ScreenTargetCapture
from riglib import experiment
from riglib.stereo_opengl.window import Window2D
import OpenGL
from OpenGL.GL import glFinish  # after riglib, which sets OpenGL.ERROR_CHECKING


class _NoClock:
    '''Replaces pygame.time.Clock so the loop runs uncapped'''
    def tick(self, fps):
        pass
    def get_fps(self):
        return 0.


class _ChaseTarget:
    '''Fake joystick that moves a fixed step per cycle toward the current target'''
    def __init__(self, task, step=0.3):
        self.task = task
        self.step = step
        self.pos = np.array(task.starting_pos, dtype=float)
        M = np.linalg.multi_dot((rotations[task.rotation], baseline_rotations[task.baseline_rotation],
                                 exp_rotations[task.exp_rotation]))
        self.M_inv = np.linalg.inv(M * np.r_[[task.scale]*3, 1][:, None])

    def get(self):
        t = self.task
        if t.state in ('target', 'hold', 'delay', 'targ_transition') and t.target_index >= 0:
            goal = np.asarray(t.targs[t.target_index], dtype=float)
            d = goal - self.pos
            n = np.linalg.norm(d)
            self.pos = goal if n < self.step else self.pos + d / n * self.step
        raw = np.r_[self.pos, 1] @ self.M_inv
        return [raw[:3]]


class SyntheticInput:
    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.joystick = _ChaseTarget(self)


class NoRender:
    '''Same as FakeWindow's graphics stubs, but keeps the rest of the Window behavior'''
    def screen_init(self):
        from riglib.stereo_opengl.models import Group
        self.world = Group(self.models)
    def draw_world(self):
        pass
    def requeue(self):
        pass
    def _get_event(self):
        pass


def _sink_proc(conn):
    while True:
        msg = conn.recv()
        if msg is None:
            return


class PipeSink:
    '''Mimics a DataSink (e.g. SaveHDF): pickles each packet into a pipe drained by another process'''
    def __init__(self):
        self.parent, child = mp.Pipe()
        self.proc = mp.Process(target=_sink_proc, args=(child,), daemon=True)
        self.proc.start()
    def send(self, system, data):
        self.parent.send((system, data))
    def close(self):
        self.parent.send(None)
        self.proc.join()


def build(mode, window_size, extra_feats=()):
    feats = [SyntheticInput]
    if extra_feats:
        import features  # needs the rig's hardware modules (e.g. hid) to import
        feats += [features.built_in_features[f] for f in extra_feats]
    if mode == 'logic':
        feats.append(NoRender)
    elif mode == 'window2d':
        feats.append(Window2D)
    Exp = experiment.make(ManualControl, feats=feats)
    seq = ScreenTargetCapture.centerout_2D(nblocks=1000)
    exp = Exp(seq, fullscreen=False, window_size=window_size, fps=120, verbose=False)
    exp.init()
    exp.clock = _NoClock()
    return exp


class Timers:
    def __init__(self):
        self.t = collections.defaultdict(list)

    def wrap(self, name, fn):
        t = self.t[name]
        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                t.append(time.perf_counter() - t0)
        return wrapped


def instrument(exp, timers, finish):
    exp.move_effector = timers.wrap('move_effector', exp.move_effector)
    exp.sinks.send = timers.wrap('sinks.send', exp.sinks.send)
    exp.update_report_stats = timers.wrap('update_report_stats', exp.update_report_stats)
    if hasattr(exp, 'renderer'):
        exp.requeue = timers.wrap('requeue', exp.requeue)
        draw = exp.renderer.draw
        if finish:
            def draw(*a, _draw=draw, **k):
                _draw(*a, **k)
                glFinish()
        exp.renderer.draw = timers.wrap('renderer.draw' + (' (+glFinish)' if finish else ''), draw)
        exp.renderer.draw_done = timers.wrap('draw_done', exp.renderer.draw_done)
        pygame.display.flip = timers.wrap('display.flip', pygame.display.flip)
        exp._get_event = timers.wrap('_get_event', exp._get_event)


def run(mode, seconds, n_warmup, window_size, finish=False, profile=False, sink=False, extra_feats=()):
    exp = build(mode, window_size, extra_feats)
    exp.screen_init()
    if sink:
        ps = PipeSink()
        exp.sinks.sinks.append(ps)
    exp.set_state('wait')

    for _ in range(n_warmup):
        exp.fsm_tick()

    timers = Timers()
    orig_flip = pygame.display.flip
    instrument(exp, timers, finish)
    states = collections.Counter()
    cycle_t = []
    prof = cProfile.Profile() if profile else None
    if prof:
        prof.enable()
    t_end = time.perf_counter() + seconds
    while True:
        t0 = time.perf_counter()
        if t0 > t_end:
            break
        exp.fsm_tick()
        cycle_t.append(time.perf_counter() - t0)
        states[exp.state] += 1
    if prof:
        prof.disable()
    pygame.display.flip = orig_flip
    cycle_t = np.array(cycle_t)
    n_cycles = len(cycle_t)

    n_trials = exp.calc_trial_num()
    n_rewards = exp.calc_state_occurrences('reward')
    if sink:
        exp.sinks.sinks.remove(ps)
        ps.close()
    driver = pygame.display.get_driver() if mode != 'logic' else 'none'
    pygame.display.quit()

    print(f'\n=== mode={mode}  window={window_size}  video={driver}'
          f'{"  feats=" + ",".join(extra_feats) if extra_feats else ""}  gl_errcheck={OpenGL.ERROR_CHECKING}  cycles={n_cycles}'
          f'{"  glFinish" if finish else ""}{"  +pipe sink" if sink else ""}{"  (cProfile on)" if profile else ""} ===')
    mean = cycle_t.mean()
    print(f'cycle: mean {mean*1e3:.3f} ms  median {np.median(cycle_t)*1e3:.3f}  p99 {np.percentile(cycle_t, 99)*1e3:.3f}'
          f'  max {cycle_t.max()*1e3:.3f}  -> {1/mean:.0f} fps uncapped')
    print(f'trials completed {n_trials}, rewards {n_rewards}, states {dict(states)}')
    accounted = 0.
    print(f'  {"phase":<28}{"ms/cycle":>10}{"% cycle":>9}{"calls":>8}{"ms/call":>9}')
    for name, t in sorted(timers.t.items(), key=lambda kv: -sum(kv[1])):
        per_cycle = sum(t) / n_cycles
        if name != 'update_report_stats':  # nested inside move_effector/_cycle, don't double count
            accounted += per_cycle
        print(f'  {name:<28}{per_cycle*1e3:>10.3f}{100*per_cycle/mean:>8.1f}%{len(t):>8}{np.mean(t)*1e3:>9.3f}')
    other = mean - accounted
    print(f'  {"fsm + task _cycle (rest)":<28}{other*1e3:>10.3f}{100*other/mean:>8.1f}%')

    if prof:
        print()
        pstats.Stats(prof).sort_stats('tottime').print_stats(25)
    return cycle_t


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['logic', 'window2d', 'window3d', 'all'], default='all')
    p.add_argument('-t', '--seconds', type=float, default=20, help='wall time per mode (trials need a few s each)')
    p.add_argument('--warmup', type=int, default=300)
    p.add_argument('--size', type=int, nargs=2, default=(1920, 1080))
    p.add_argument('--finish', action='store_true', help='glFinish after each draw to include GPU execution time')
    p.add_argument('--profile', action='store_true', help='also print the top cProfile entries')
    p.add_argument('--sink', action='store_true', help='send task_data through a pipe to another process, like SaveHDF')
    p.add_argument('--feats', nargs='*', default=[], help='extra features, by features.built_in_features key')
    args = p.parse_args()

    if args.mode == 'all':
        # separate processes: the sink manager and pygame display are process-global
        for m in ['logic', 'window2d', 'window3d']:
            subprocess.run([sys.executable, __file__, '--mode', m] + [a for a in sys.argv[1:] if a not in ('--mode', 'all')])
    else:
        run(args.mode, args.seconds, args.warmup, tuple(args.size), finish=args.finish, profile=args.profile,
            sink=args.sink, extra_feats=args.feats)
