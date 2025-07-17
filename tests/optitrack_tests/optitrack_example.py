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
        
        for i in range(STREAMING_DURATION):
            frame = opt.get()
            if frame is None:
                print('No data recieved')
            else:
                frame_number = frame.prefix_data.frame_number
                print('Received frame data:')
                rigid_bodies = frame.rigid_body_data.rigid_body_list
                #How should I format it? position 0 is timestamp, 1 id, 2-4 x,y,z; 5-8 orientation x,y,z,w, repeat
                print('BOOP')
                data_out = np.zeros(1 + len(rigid_bodies)*8)
                data_out[0] = frame_number
                print(data_out.shape)
                for i,body in enumerate(rigid_bodies):
                    
                    strt = i*8 + 1
                    id = body.id_num
                    pos = body.pos
                    rot = body.rot
                    data_out[strt] = id
                    data_out[strt + 1:strt + 4] = pos
                    data_out[strt + 4:strt + 8] = rot
                print(data_out)
            time.sleep(1)
        print('Stopping optitrack streaming client...')
                
        opt.stop()

if __name__ == '__main__':
    unittest.main()