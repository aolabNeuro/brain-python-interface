import time
import numpy as np
import zmq
# from msgpack import loads
import msgpack
import time
# from surface_no_delay import NoDelaySurfaceGazeMapper, RadialDistortionCamera
# from pupillab_timesync import measure_clock_offset

from riglib.source import DataSourceSystem

class System(DataSourceSystem):
    '''
    
    '''
    dtype = np.dtype(('float', (8,)))
    update_freq = 200
    
    def __init__(self, ip="127.0.0.1", port="50020"):
        '''
        For eye tracking, need Pupil Capture running in the background (after calibration in Pupil Capture)
        '''
        # define a surface AOI
        
        # open a req port to talk to pupil
        self.ip = ip  # remote ip or localhost
        self.port = port # same as in the pupil remote gui
        
        # # matrix for camera distortion
        # camera = RadialDistortionCamera(
        #     resolution=(1280, 720),
        #     cam_matrix=[
        #         [794.3311439869655, 0.0, 633.0104437728625],
        #         [0.0, 793.5290139393004, 397.36927353414865],
        #         [0.0, 0.0, 1.0],
        #     ],
        #     dist_coefs=[
        #         [
        #             -0.3758628065070806,
        #             0.1643326166951343,
        #             0.00012182540692089567,
        #             0.00013422608638039466,
        #             0.03343691733865076,
        #             0.08235235770849726,
        #             -0.08225804883227375,
        #             0.14463365333602152,
        #         ]
        #     ],
        # )

        # self.mapper = NoDelaySurfaceGazeMapper(camera)
        # self.mapped_points = []
    
    def start(self):
        '''
        
        '''
        ctx = zmq.Context()
        pupil_remote = ctx.socket(zmq.REQ)
        # The REQ talks to Pupil remote and receives the session unique IPC SUB PORT
        pupil_remote.connect(f'tcp://{self.ip}:{self.port}') #connect to eye tracker

        # rec_name = 'test'
        # req.send_string(f'R {rec_name}') # start recording
        # print('pupillab starting recording in Pupil Capture')

        # # sync pupil internal clock with the local time 
        # local_clock = time.time  # Unix time, less accurate 

        # # Measure clock offset once
        # self.offset = measure_clock_offset(pupil_remote, clock_function=local_clock)
        # print(f"\n Pupillab Clock offset (1 measurement): {self.offset} seconds")

        pupil_remote.send_string('SUB_PORT') # Request 'SUB_PORT' for reading data
        sub_port = pupil_remote.recv_string()

        # open sub ports to listen to pupil; sub: subport that receives surface data
        self.sub = ctx.socket(zmq.SUB) # open a sub port to listen to pupil surface topic
        self.sub.connect(f'tcp://{self.ip}:{sub_port}')
        # self.sub.setsockopt_string(zmq.SUBSCRIBE, f"surfaces.{surface_name}")
        self.sub.subscribe(f"surfaces")  # receive all surface messages
        self.sub.subscribe("gaze")  # receive all gaze messages
        self.sub.subscribe('pupil.0.2d')  # receive all pupil0 messages, right eye
        self.sub.subscribe('pupil.1.2d')  # receive all pupil1 messages, left eye
        pupil_remote.close()

    def stop(self):
        self.sub.close()
        
        ctx = zmq.Context()
        pupil_remote = ctx.socket(zmq.REQ)
        # The REQ talks to Pupil remote and receives the session unique IPC SUB PORT
        pupil_remote.connect(f'tcp://{self.ip}:{self.port}') #connect to eye tracker

        # req.send_string('r') # stop recording
        # print('pupillab stopped recording in Pupil Capture')
        pupil_remote.close()

    def get(self):
        """
        read in a batch of eye data and retun x, y on surface & pupil diameters for each eye
        """
        raw0, raw1 = (np.nan, np.nan)
        confidence = np.nan
        diameter0, diameter1 = np.nan
        timestamp = np.nan

        try:  # load messages from the surface subport, this updates 200Hz
            while self.sub.poll(0) == zmq.POLLIN: # clear the buffer after
                topic, payload = self.sub.recv_multipart(flags=zmq.NOBLOCK) #noblock for recv(), unless it will wait until messages come in
                message = msgpack.loads(payload, raw=False)
            
                if topic.startswith(b"surfaces"): # get the surface datum when gaze in on the surface
                    # self.mapper.update_homography(message["img_to_surf_trans"])
                    #  "gaze_on_surfaces": (
                    #     {
                    #         "topic": "gaze.3d.1._on_surface",
                    #         "norm_pos": (-0.6709809899330139, 0.41052111983299255),
                    #         "confidence": 0.5594810076623645,
                    #         "on_surf": False,
                    #         "base_data": ("gaze.3d.1.", 714040.132285),
                    #         "timestamp": 714040.132285,
                    #     },
                    if len(message["gaze_on_surfaces"]) == 2:
                        raw0 = message["gaze_on_surfaces"][0]["norm_pos"]
                        raw1 = message["gaze_on_surfaces"][0]["norm_pos"]
                        timestamp = message["timestamp"]

                elif topic.startswith(b"pupil.0.2d"):
                    diameter0 = float(message["diameter"]) # pupil 0 diamter, right eye, unit: pixel
                elif topic.startswith(b"pupil.1.2d"):
                    diameter1 = float(message["diameter"]) # pupil 1 diamter, left eye, unit: pixel

                # elif topic.startswith(b"gaze"):
                #     mapped_gaze = self.mapper.gaze_to_surface(message["norm_pos"])
                #     if mapped_gaze is None: 
                #         continue

                #     #raw x, y coordinates from surface data, (0,0) is the bottom left corner of the surface and (1,1) is the top right corner
                #     raw_x = float(mapped_gaze.norm_x)
                #     raw_y = float(mapped_gaze.norm_y)
                #     confidence = float(message["confidence"])  # quaulity of measurements, [0,1]            

        except: # mannually set x and y to be nan when no pupil data is updated
            pass

        coords = [raw0[0], raw0[1], raw1[0], raw1[1], diameter0, diameter1, confidence, timestamp]
        coords = np.expand_dims(coords, axis=0)
        return coords


    
