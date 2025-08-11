'''
Eye tracking calibration task for head-mounted display
'''
import numpy as np

from riglib.experiment import traits, Sequence
from riglib.stereo_opengl.openxr import WindowVR
from riglib.stereo_opengl.window import Window
from .target_graphics import *

import zmq, msgpack, time


class CalibrateHMD(WindowVR, Sequence):
    """
    Show one target after another and send target locations to pupil-labs
    """

    status = dict(
        wait = dict(start_trial="target", start_pause="pause"),
        target = dict(fixation="target_calibrate", start_pause="pause"),
        target_calibrate = dict(target_done="wait", start_pause="pause", end_state=True), # end_state is just for counting trials
        pause = dict(end_pause="wait", end_state=True),
    )

    # initial state
    state = "wait"

    sequence_generators = ['target_generator']

    fixation_time = traits.Float(1.0, desc="Time in seconds to display target before calibration")
    calibration_time = traits.Float(1.0, desc="Duration in seconds of each target calibration")
    startup_time = traits.Float(5.0, desc="Time to wait before starting first trial")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = VirtualCircularTarget(target_radius=1, target_color=[1,1,1,1])
        for model in self.target.graphics_models:
            self.add_model(model)

        print("Connecting to Pupil Service/Capture...")
        ctx = zmq.Context()

        # create a zmq REQ socket to talk to Pupil Service/Capture
        self.req = ctx.socket(zmq.REQ)
        self.req.connect("tcp://128.95.215.191:50020") # to-do don't hard code

        # set start eye windows
        n = {"subject": "eye_process.should_start.0", "eye_id": 0, "args": {}} # delay?
        print(self.send_recv_notification(n))
        n = {"subject": "eye_process.should_start.1", "eye_id": 1, "args": {}}
        print(self.send_recv_notification(n))
        time.sleep(2)

        n = {"subject": "set_detection_mapping_mode", "mode": "3d"}
        print(self.send_recv_notification(n))

        # set calibration method to hmd calibration
        n = {"subject": "start_plugin", "name": "HMD3DChoreographyPlugin", "args": {}}
        print(self.send_recv_notification(n))

        # start caliration routine with params. This will make pupil start sampeling pupil data.
        # the eye-translations have to be in mm, these here are default values from Unity XR
        n = {
            "subject": "calibration.should_start",
            "translation_eye0": [30., 0.0, 0.0],
            "translation_eye1": [-30., 0.0, 0.0],
            "record": True,
        }
        print(self.send_recv_notification(n))
        self.ref_data = []

    # pupil-labs convenience functions
    def send_recv_notification(self, n):
        # REQ REP requirese lock step communication with multipart msg (topic,msgpack_encoded dict)
        self.req.send_string("notify.%s" % n["subject"], flags=zmq.SNDMORE)
        self.req.send(msgpack.dumps(n, use_bin_type=True))
        return self.req.recv_string()

    def get_pupil_timestamp(self):
        self.req.send_string("t")  # see Pupil Remote Plugin for details
        return float(self.req.recv_string())

    def init(self):
        self.trial_dtype = np.dtype([('trial', 'u4'), ('index', 'u4'), ('target', 'f8', (3,))])
        super().init()

    def _cycle(self):
        '''
        Calls any update functions necessary and redraws screen
        '''
        super()._cycle()

    #### TEST FUNCTIONS ####
    def _test_start_trial(self, ts):
        if self.target_idx == 0:
            return ts > self.startup_time
        return not self.pause

    def _test_start_pause(self, ts):
        return self.pause

    def _test_fixation(self, ts):
        return ts > self.fixation_time

    def _test_target_done(self, ts):
        return ts > self.calibration_time
    
    def _test_end_pause(self, ts):
        return not self.pause

    def _parse_next_trial(self):
        '''Check that the generator has the required data'''
        self.target_idx, self.target_location = self.next_trial
        self.target_location = np.array(self.target_location).astype(float)
        
        # Update the data sinks with trial information
        self.trial_record['trial'] = self.calc_trial_num()
        self.trial_record['index'] = self.target_idx
        self.trial_record['target'] =  self.target_location
        self.sinks.send("trials", self.trial_record)

    ### STATE FUNCTIONS ###
    def _start_wait(self):
        self.sync_event('TARGET_OFF')
        self.target.hide()
        super()._start_wait()

    def _start_target(self):
        self.sync_event('TARGET_ON', 0xd)
        self.target.move_to_position(self.target_location)
        self.target.show()

    def _while_target_calibrate(self):
        # get the current pupil time (pupil uses CLOCK_MONOTONIC with adjustable timebase).
        # You can set the pupil timebase to another clock and use that.
        
        if self.cycle_count % 3 == 0:
            t = self.get_pupil_timestamp()

            # in this mockup  the left and right screen marker positions are identical.
            datum0 = {"mm_pos": (self.target_location[0]*10, self.target_location[2]*10, self.target_location[1]*10), "timestamp": t}
            self.ref_data.append(datum0)

    def _end_target_calibrate(self):
        # Send ref data to Pupil Capture/Service:
        # This notification can be sent once at the end or multiple times.
        # During one calibraiton all new data will be appended.
        n = {
            "subject": "calibration.add_ref_data",
            "ref_data": self.ref_data,
            "record": True,
        }
        print(self.send_recv_notification(n))
        self.ref_data = []

    def _start_pause(self):
        self.sync_event('PAUSE_START')
        self.target.hide()

    def _end_pause(self):
        self.sync_event('PAUSE_END')

    def _start_None(self):
        # stop calibration
        # Pupil will correlate pupil and ref data based on timestamps,
        # compute the gaze mapping params, and start a new gaze mapper.
        n = {
            "subject": "calibration.should_stop",
            "record": True,
        }
        print(self.send_recv_notification(n))


    @classmethod
    def get_desc(cls, params, log_summary):
        duration = round(log_summary['runtime'] / 60, 1)
        return "{}/{} succesful trials in {} min".format(
            log_summary['n_success_trials'], log_summary['n_trials'], duration)

    @staticmethod
    def target_generator(size=10):
        '''
        Generates a sequence of targets

        Parameters
        ----------
        size : int, optional
            Distance between targets, by default 10

        Returns
        -------
        [15 x 1] array of tuples containing trial index and [1 x 3] target coordinates

        '''
        pos = [
            (0.0, 0.0, 600.0),
            (0.0, 0.0, 1000.0),
            (0.0, 0.0, 2000.0),
            (180.0, 0.0, 600.0),
            (240.0, 0.0, 1000.0),
            (420.0, 0.0, 2000.0),
            (55.62306, 195.383, 600.0),
            (74.16407, 260.5106, 1000.0),
            (129.7871, 455.8936, 2000.0),
            (-145.6231, 120.7533, 600.0),
            (-194.1641, 161.0044, 1000.0),
            (-339.7872, 281.7577, 2000.0),
            (-145.6231, -120.7533, 600.0),
            (-194.1641, -161.0044, 1000.0),
            (-339.7872, -281.7577, 2000.0),
            (55.62306, -195.383, 600.0),
            (74.16407, -260.5106, 1000.0),
            (129.7871, -455.8936, 2000.0),
        ]
        pos = np.array(pos)/300
        pos = pos[:,[0,2,1]] - np.array([0,3,0])
        for idx in range(len(pos)):
            yield idx, np.array(pos[idx])*size

