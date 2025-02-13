import zmq
from riglib import experiment
from features.eyetracker_features import PupilLabStreaming
from riglib.pupillabs import System, NoSurfaceTracking
from riglib.pupillabs.pupillab_timesync import setup_pupil_remote_connection, request_pupil_time
from datetime import datetime
import time
import numpy as np
import os
import natnet

import unittest


class TestPupillabs(unittest.TestCase):

    # def test_client(self):
    #     socket = setup_pupil_remote_connection(ip_adress='128.95.215.191')
    #     time = request_pupil_time(socket)
    #     print(time)

    @unittest.skip("")
    def test_datasource(self):
        eyedata = NoSurfaceTracking()
        eyedata.start()
        time.sleep(0.5)

        data = eyedata.get()
        print(data)

        time.sleep(0.5)

        data = eyedata.get()
        print(data)

        eyedata.stop()

    def test_datasource_system(self):
        from riglib import source
        motiondata = source.DataSource(NoSurfaceTracking)
        motiondata.start()
        time.sleep(2)
        
        # Count packet rate
        count = 0
        duration = 5
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            data = motiondata.get()
            if len(data) > 0:
                count += 1
            time.sleep(0.001)

        print(f"Packet rate: {count/duration} Hz")

        motiondata.stop()

        print(data)

if __name__ == '__main__':
    unittest.main()