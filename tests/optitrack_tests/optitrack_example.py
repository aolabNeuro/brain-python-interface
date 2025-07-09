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
        print('Starting optitrack streaming client...')
        opt.start()
        print("Successfully connected to OptiTrack, requesting Frame")
        frame = opt.get()
        if frame is None:
            print('No data recieved')
        else:
            print('Received frame data:')
            if hasattr(frame, 'rigid_body_data'):
                print('Rigid Body Data:', frame.rigid_body_data.get_rigid_body_count())
            print(np.shape(frame))
        opt.stop()

if __name__ == '__main__':
    unittest.main()