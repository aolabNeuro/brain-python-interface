# test file for new optitrack integration

import time
from riglib import source
from riglib.optitrack_client_update.PythonSample import OptitrackStreamingClient
from riglib.bmi import state_space_models, train, extractor
import numpy as np
import unittest

STREAMING_DURATION = 3

class TestOptiTrackStreaming(unittest.TestCase):

    #@unittest.
    def test_direct(self):
        opt = OptitrackStreamingClient()
        opt.setup()
        print('Data socket: ' + str(opt.streaming_client.data_port))
                # check sockets
        print('Command socket: ' + str(opt.streaming_client.command_port))
        opt.read_frame()
        opt.stop()

if __name__ == '__main__':
    unittest.main()