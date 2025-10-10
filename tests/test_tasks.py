import time
from built_in_tasks.force_task import DiskMatching
from built_in_tasks.manualcontrolmultitasks import TrackingTask, rotations, ManualControl, ScreenTargetTracking, ReadySetGoTask
from built_in_tasks.othertasks import Conditions, LaserConditions, SweptLaserConditions
from built_in_tasks.target_capture_task import ScreenTargetCapture
from built_in_tasks.passivetasks import YouTube
from built_in_tasks.example_task import ExampleSequenceTask
from features.generator_features import Autostart
from features.hdf_features import SaveHDF
from riglib.stereo_opengl.environment import Grid
from riglib.stereo_opengl.window import WindowDispl2D
from riglib import experiment
from features.peripheral_device_features import ForceControl, MouseControl
from features.optitrack_features import OptitrackSimulate, Optitrack, SpheresToCylinders
from features.reward_features import ProgressBar, ScoreRewards
import cProfile
import pstats
from riglib.stereo_opengl.window import Window, Window2D
import unittest
import numpy as np
import matplotlib.pyplot as plt
import os
import socket

def init_exp(base_class, feats, seq=None, **kwargs):
    hostname = socket.gethostname()
    if hostname in ['pagaiisland2', 'human-bmi']:
        os.environ['DISPLAY'] = ':0.1'
    Exp = experiment.make(base_class, feats=feats)
    if seq is not None:
        exp = Exp(seq, **kwargs)
    else:
        exp = Exp(**kwargs)
    exp.init()
    return exp

class TestManualControlTasks(unittest.TestCase):

    @unittest.skip("")
    def test_readysetgo(self):
        seq = ManualControl.centerout_2D()
        exp = init_exp(ReadySetGoTask, [MouseControl, Window2D], seq, window_size=(1200,800), fullscreen=False)
        exp.rotation = 'xzy'
        exp.ready_set_sound = 'tones.wav'
        exp.run()

    @unittest.skip("")
    def test_exp(self):
        seq = ManualControl.centerout_2D()
        exp = init_exp(ManualControl, [MouseControl, Window2D, ScoreRewards], seq, window_size=(1200,800), fullscreen=False)
        exp.rotation = 'xzy'
        exp.stereo_mode = 'projection'
        exp.run()

    @unittest.skip("")
    def test_example_task(self):
        seq = ExampleSequenceTask.example_generator()
        print('Testing example task')
        print(seq)
        print(hasattr(seq, '__next__'))
        exp = init_exp(ExampleSequenceTask, [], seq, window_size=(1200,800), fullscreen=False)
        exp.run()
    
    @unittest.skip("")
    def test_tracking(self):
        print("Running tracking task test")
        seq = TrackingTask.tracking_target_debug(nblocks=1, ntrials=6, time_length=5, seed=40, sample_rate=60, ramp=1) # sample_rate needs to match fps in ScreenTargetTracking
        exp = init_exp(TrackingTask, [MouseControl, Window2D], seq, window_size=(1000,800), fullscreen=False)
        exp.rotation = 'xzy'
        # exp.trajectory_type = 'space'
        exp.trajectory_amplitude = 5
        # exp.lookahead_time = 1
        exp.run()

    @unittest.skip("")
    def test_tracking_2d(self):
        print("Running tracking task test")
        seq = TrackingTask.tracking_target_chain(nblocks=1, ntrials=2, time_length=20, ramp=1, ramp_down=1, 
                                                 num_primes=10, seed=42, sample_rate=60, dimensions=2, 
                                                 disturbance=False, boundaries=(-10,10,-10,10))
        exp = init_exp(TrackingTask, [Window2D, MouseControl], seq, window_size=(1000,800), fullscreen=False, 
                       limit1d=False, trajectory_amplitude=5, lookahead_time=1)
        exp.stereo_mode = 'projection'
        exp.rotation = 'xzy'
        exp.trajectory_type = 'space'
        exp.run()

    @unittest.skip("")
    def test_sequence(self):
        print("Running sequence task test")
        seq = ScreenTargetCapture.sequence_2D(nblocks=1, distance=5)
        exp = init_exp(ManualControl, [MouseControl, Window2D], seq) # , window_size=(1000,800)
        exp.rotation = 'xzy'
        exp.run()

    @unittest.skip("")
    def test_force_task(self):
        print("Running force task test")
        exp = init_exp(DiskMatching, [ForceControl, Window2D, Autostart], None) # , window_size=(1000,800)
        exp.rand_start = (0.5,1)
        exp.run()
        t0 = time.time()
        while time.time() - t0 < 10:
            pass
        exp.end_task()

    @unittest.skip("only to test progress bar")
    def test_progress_bar(self):
        seq = TrackingTask.tracking_target_debug(nblocks=1, ntrials=6, time_length=5, seed=40, sample_rate=60, ramp=1) # sample_rate needs to match fps in ScreenTargetTracking
        exp = init_exp(TrackingTask, [MouseControl, Window2D, ProgressBar], seq)
        exp.rotation = 'xzy'
        exp.run()

    @unittest.skip("only to test 3d window")
    def test_3d(self):
        seq = ManualControl.centerout_2D()
        exp = init_exp(ManualControl, [MouseControl, SpheresToCylinders], seq, stereo_mode='projection',
                       rotation='xyz', window_size=(1000,800), fullscreen=False, limit2d=False)
        exp.run()


class TestSeqGenerators(unittest.TestCase):

    @unittest.skip("")
    def test_gen_ascending(self):
        seq = Conditions.gen_conditions(3, [1, 2], ascend=True)
        self.assertSequenceEqual(seq[0], [0, 0, 0, 1, 1, 1])

    @unittest.skip("")
    def test_gen_out_2D(self):
        seq = ScreenTargetCapture.out_2D(nblocks=1, )
        seq = list(seq)
        idx = np.array([s[0][0] for s in seq])
        loc = np.array([s[1][0] for s in seq])
        print(idx)
        print(loc)
        self.assertCountEqual(idx, [1, 2, 3, 4, 5, 6, 7, 8])

        # Target 1 should be 12 o'clock
        self.assertAlmostEqual(loc[idx == 1, 0][0], 0)
        self.assertAlmostEqual(loc[idx == 1, 2][0], 10)

        # Target 3 should be 3 o'clock
        self.assertAlmostEqual(loc[idx == 3, 0][0], 10)
        self.assertAlmostEqual(loc[idx == 3, 2][0], 0)

    @unittest.skip("")
    def test_dual_laser_wave(self):
        seq = LaserConditions.dual_laser_square_wave(duty_cycle_1=0.025, duty_cycle_2=0.025, phase_delay_2=0.1)
        print(seq[0])

    @unittest.skip("")
    def test_swept_laser_pulse(self):
        seq = SweptLaserConditions.single_laser_pulse()
        print(seq[0])

    @unittest.skip("")
    def test_corners(self):
        seq = ScreenTargetCapture.corners_2D(chain_length=3)
        seq = list(seq)

        idx = np.array([s[0][0] for s in seq])
        loc = np.array([s[1][0] for s in seq])
        print("corners---------------")
        print(idx)
        print(loc)
        print("---------------corners")

    #@unittest.skip("")
    def test_tracking_2d(self):
        seq = TrackingTask.tracking_target_chain(nblocks=1, ntrials=2, time_length=20, ramp=0, ramp_down=0, 
                                                 num_primes=12, seed=42, sample_rate=60, dimensions=2, 
                                                 disturbance=False, boundaries=(-10,10,-10,10))
        trajectories = [t[1][0] for t in seq]
        print("2D Test-------")
        print(np.shape(trajectories))
        print("2D Test-------")
        fig, axs = plt.subplots(2,1, figsize=(10,8))
        for idx, trial in enumerate(trajectories): 
            ax = axs[idx]
            trialx = np.fft.fft(trial[:,0])
            trial_length = np.shape(trialx)[0]
            freq = np.fft.fftfreq(trial_length, d=1./60)
            non_neg_freq = freq[freq >= 0] #get positive frequencies 
            non_neg_x = trialx[freq >= 0] / complex(trial_length, 0) #normalize 
            non_neg_x[1:] = 2*non_neg_x[1:] #account for negative frequencies
            trialy = np.fft.fft(trial[:,2])
            non_neg_y = trialy[freq >= 0] / complex(trial_length, 0) #normalize 
            non_neg_y[1:] = 2*non_neg_y[1:] #account for negative frequencies
            ax.plot(non_neg_freq, np.abs(non_neg_x), 'o-', label = 'X')
            ax.plot(non_neg_freq, np.abs(non_neg_y), 'o-', label = 'Y')
            ax.set_title(f'Trial {idx}')
            ax.set_xlim(0, 3)
            ax.set_xlabel('Frequency (Hz)')
        plt.legend()
        plt.tight_layout()
        plt.show()

class TestYouTube(unittest.TestCase):

    @unittest.skip("")
    def test_youtube_exp(self):

        exp = init_exp(YouTube, [], youtube_url="https://www.youtube.com/watch?v=Qe9ansjvF7M")
        exp.run()

if __name__ == '__main__':
    unittest.main()