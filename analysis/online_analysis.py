'''
Functions to analyze data in real-time. 
Useful for debugging and monitoring the system on a separate machine connected over the network.
'''

import json
import socket
import select
from threading import Thread, Event
from queue import Queue
import numpy as np
import matplotlib.pyplot as plt
import aopy
from riglib.ecube import MultiSource, map_channels_for_multisource
from riglib.source import MultiChanDataSource

class OnlineDataWorker(Thread):
    '''
    Worker thread to receive data from BMI3D.
    '''

    def __init__(self, socket, result_queue):
        '''
        Start the server on the specified IP address and port number.
        '''
        self.sock = socket
        self._stop_event = Event()
        self.result_queue = result_queue
        Thread.__init__(self)

    def run(self):
        while not self._stop_event.is_set():
            ready = select.select([self.sock], [], [], 0.1) # 100ms timeout to check _stop_event
            if ready[0]:
                data = self.sock.recv(4096)
                key, value = data.decode('utf-8').split(':')
                self.result_queue.put((key, [json.loads(v) for v in value.split('#')]))

    def stop(self):
        self._stop_event.set()


class OnlineDataServer:
    '''
    Interface for ingesting and accumulating BMI3D data in real-time.
    '''

    def __init__(self, host_ip, port=5000):
        '''
        Initialize the server on the specified IP address and port number.

        Args:
            host_ip (str): IP address of the machine running the online analysis
            port (int): Port number for the online analysis server
        '''
        self.sock = socket.socket(
            socket.AF_INET, # Internet
              socket.SOCK_DGRAM) # UDP 
        self.sock.bind((host_ip, port))
        self.sock.setblocking(0)
        self.reset()

    def reset(self):
        self.result_queue = Queue()
        self.worker = OnlineDataWorker(self.sock, self.result_queue)
        self.worker.start()
        self.is_running = False
        self.is_completed = False
        self.task_params = {}
        self.cycle_count = 0
        self.state = None
        self.sync_events = []
        self.cursor_pos = np.zeros(2)
        self.eye_pos = np.zeros(2)
        self.target_pos = {}

    def init(self):
        '''
        Code that runs once the experiment is initialized but before it starts
        '''
        pass

    def update_sync_events(self):
        '''
        Code that runs when a sync event is detected
        '''
        pass

    def _stop(self):
        if self.worker.is_alive():
            self.worker.stop()
            self.worker.join()
        
    def update(self):
        '''
        Get the latest data from the online server
        '''
        while not self.result_queue.empty():
            key, values = self.result_queue.get()
            print(key, values)
            if key == 'state':
                self.state = values[0]
                if values[0] is None:
                    print('Experiment finished')
                    self._stop()
                    self.is_running = False
                    self.is_completed = True
            elif key == 'sync_event':
                event_name, event_data = values
                self.sync_events.append((event_name, int(event_data)))
                self.update_sync_events()
            elif key == 'cursor':
                self.cursor_pos = np.array(values[0])[0,2]
            elif key == 'eye_pos':
                self.eye_pos = np.array(values[0])[:2]
            elif key == 'target_location':
                target_idx, target_location = values
                self.target_pos[int(target_idx)] = target_location[0,2]
            elif key == 'init':
                self.is_running = True
                self.init()
            else:
                # Any other incoming data is a parameter update
                self.task_params[key] = values[0]
    
    def close(self):
        self._stop()
        self.sock.close()

class OnlineEyeCursorTarget(OnlineDataServer):
    '''
    Buffers eye, cursor, and target data from experiments that have them. Performs automatic
    calibration of eye data to target locations when the cursor enters the target.
    '''
    def reset(self):
        super().reset()
        self.sync_events_checked = 0
        self.eye_coeff = np.zeros((2, 2))
        self.calibration_data = []

    def update_sync_events(self):
        '''
        Look at the sync events to determine which target is active
        '''
        while self.sync_events_checked < len(self.sync_events):
            event_name, event_data = self.sync_events[self.sync_events_checked]
            if event_name == 'TARGET_ON':
                self.targets[event_data] = True
            elif event_name == 'TARGET_OFF':
                self.targets[event_data] = False
            elif event_name == 'TRIAL_END':
                self.targets = {} # Clear targets at the end of the trial
            elif event_name == 'CURSOR_ENTER_TARGET' and event_data > 0:
                self.calibration_data.append((self.eye_pos, self.target_pos[event_data]))

            self.sync_events_checked += 1

    def update_eye_calibration(self):
        '''
        Update the eye calibration coefficients using the collected data
        '''
        if len(self.calibration_data) > 0:
            eye_data, target_data = zip(*self.calibration_data)
            eye_data = np.array(eye_data)
            target_data = np.array(target_data)
            self.eye_coeff = np.linalg.lstsq(eye_data, target_data, rcond=None)[0]
        
    def get_cursor(self):
        self.update()
        return self.cursor_pos

    def get_eye(self):
        self.update()
        self.update_eye_calibration()
        calibrated_eye_pos = np.dot(self.eye_pos, self.eye_coeff)
        return calibrated_eye_pos
    
    def get_targets(self):
        self.update()
        try:
            radius = self.task_params['target_radius']
            color = self.task_params['target_color']
            return [(self.target_pos[k], radius, color) for k, v in self.targets.items() if v]
        except:
            return []

class OnlineECoG244ERP(OnlineDataServer):
    '''
    Buffers ERP data from experiments with a ECoG244 array. Automatically calculates 
    ERPs for flash, movement, or laser events depending on the task.
    '''
    def reset(self):
        super().reset()

    def init(self):
        self.elec_pos, self.acq_ch, _ = aopy.data.load_chmap('ECoG244')
        channels = map_channels_for_multisource(headstage_channels=self.acq_ch)
        self.ds = MultiChanDataSource(MultiSource, channels=channels, bufferlen=bufferlen)
        self.ds.start()

        
    def get_erp(self, channels):
        data = self.ds.get_new(channels)
