import time
import traceback
import numpy as np
import zmq
from msgpack import loads
import msgpack
import time
from .surface_no_delay import NoDelaySurfaceGazeMapper, RadialDistortionCamera
from .pupillab_timesync import measure_clock_offset

from riglib.source import DataSourceSystem

class System(DataSourceSystem):
    '''
    
    '''
    dtype = np.dtype((float, (12,)))
    update_freq = 120
    
    def __init__(self, ip="128.95.215.191", port="50020", confidence_threshold=0.0):
        '''
        For eye tracking, need Pupil Capture running in the background (after calibration in Pupil Capture)
        '''
        # define a surface AOI
        
        # open a req port to talk to pupil
        self.ip = ip  # remote ip or localhost
        self.port = port # same as in the pupil remote gui
        
        # # matrix for camera distortion
        camera = RadialDistortionCamera(
            resolution=(1280, 720),
            cam_matrix=[
                [794.3311439869655, 0.0, 633.0104437728625],
                [0.0, 793.5290139393004, 397.36927353414865],
                [0.0, 0.0, 1.0],
            ],
            dist_coefs=[
                [
                    -0.3758628065070806,
                    0.1643326166951343,
                    0.00012182540692089567,
                    0.00013422608638039466,
                    0.03343691733865076,
                    0.08235235770849726,
                    -0.08225804883227375,
                    0.14463365333602152,
                ]
            ],
        )

        self.mapper = NoDelaySurfaceGazeMapper(camera)
        self.mapped_points = []
        self.confidence_threshold = confidence_threshold
    
    def start(self):
        '''
        
        '''
        self.ctx = zmq.Context()
        self.pupil_remote = self.ctx.socket(zmq.REQ)
        # The REQ talks to Pupil remote and receives the session unique IPC SUB PORT
        self.pupil_remote.connect(f'tcp://{self.ip}:{self.port}') #connect to eye tracker

        # rec_name = 'test'
        # req.send_string(f'R {rec_name}') # start recording
        # print('pupillab starting recording in Pupil Capture')

        # # sync pupil internal clock with the local time 
        # local_clock = time.time  # Unix time, less accurate 

        # # Measure clock offset once
        # self.offset = measure_clock_offset(pupil_remote, clock_function=local_clock)
        # print(f"\n Pupillab Clock offset (1 measurement): {self.offset} seconds")

        self.pupil_remote.send_string('SUB_PORT') # Request 'SUB_PORT' for reading data
        sub_port = self.pupil_remote.recv_string()

        # open sub ports to listen to pupil; sub: subport that receives surface data
        self.sub = self.ctx.socket(zmq.SUB) # open a sub port to listen to pupil surface topic
        self.sub.connect(f'tcp://{self.ip}:{sub_port}')
        # self.sub.setsockopt_string(zmq.SUBSCRIBE, f"surfaces.{surface_name}")
        self.sub.subscribe(f"surfaces")  # receive all surface messages
        self.sub.subscribe("gaze")  # receive all gaze messages
        self.sub.subscribe('pupil.0.2d')  # receive all pupil0 messages, right eye
        self.sub.subscribe('pupil.1.2d')  # receive all pupil1 messages, left eye

    def stop(self):
        self.sub.close()
        # req.send_string('r') # stop recording
        # print('pupillab stopped recording in Pupil Capture')
        self.pupil_remote.close()
        self.ctx.term()

    def get(self):
        """
        read in a batch of eye data and retun x, y on surface & pupil diameters for each eye
        """
        gaze = np.full((6,), np.nan)
        left_pupil = np.full((3,), np.nan)
        right_pupil = np.full((3,), np.nan)
        coords = np.hstack([gaze, left_pupil, right_pupil])

        while np.any(np.isnan(coords)):
            if not self.sub.poll(0) == zmq.POLLIN:
                time.sleep(0.001)  # sleep for 1 ms to avoid busy waiting
                continue

            topic, payload = self.sub.recv_multipart(flags=zmq.NOBLOCK)
            message = msgpack.loads(payload, raw=False)

            if message["topic"].startswith("surfaces"):
                self.mapper.update_homography(message["img_to_surf_trans"])

            if message["topic"].startswith("gaze.3d.") and message["confidence"] >= self.confidence_threshold:
                    
                    gaze[:2] = message["norm_pos"]
                    gaze[2:5] = message["gaze_point_3d"]

                    if self.mapper is not None:
                        mapped_gaze = self.mapper.gaze_to_surface(message["norm_pos"])
                        if mapped_gaze is not None:
                            gaze[:2] = np.array(mapped_gaze.norm_x, mapped_gaze.norm_y)

                    gaze[5] = message["timestamp"]

            if message["topic"] == "pupil.1.2d" and message["confidence"] >= self.confidence_threshold:
                left_pupil[:2] = message["norm_pos"]
                left_pupil[2] = float(message["diameter"]) # pupil 1 diamter, left eye, unit: pixel
            
            elif message["topic"] == "pupil.0.2d" and message["confidence"] >= self.confidence_threshold:
                right_pupil[:2] = message["norm_pos"] 
                right_pupil[2] = float(message["diameter"]) # pupil 0 diamter, right eye, unit: pixel

            coords = np.hstack([gaze, left_pupil, right_pupil])
            
        coords = np.expand_dims(coords, axis=0)
        return coords

    
class NoSurfaceTracking(System):

    def __init__(self, ip="128.95.215.191", port="50020", confidence_threshold=0.0):
        '''
        For eye tracking, need Pupil Capture running in the background (after calibration in Pupil Capture)
        '''
        # define a surface AOI
        
        # open a req port to talk to pupil
        self.ip = ip  # remote ip or localhost
        self.port = port # same as in the pupil remote gui
        
        self.mapper = None
        self.mapped_points = []

        self.confidence_threshold = confidence_threshold
