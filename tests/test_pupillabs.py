import zmq
from riglib import experiment
from features.eyetracker_features import PupilLabStreaming
from riglib.pupillabs import System
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

    # def test_datasource(self):
    #     eyedata = System()
    #     eyedata.start()
    #     time.sleep(0.5)

    #     data = eyedata.get()
    #     print(data)

    #     time.sleep(0.5)

    #     data = eyedata.get()
    #     print(data)

    #     eyedata.stop()


    def test_datasource(self):
        from riglib import source
        motiondata = source.DataSource(System)
        motiondata.start()
        time.sleep(0.5)

        data = motiondata.get()
        motiondata.stop()

        print(data)

if __name__ == '__main__':
    unittest.main()