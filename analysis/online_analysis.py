import time
import json
import socket
import select
import multiprocessing as mp
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
import aopy
from riglib.ecube import MultiSource, map_channels_for_multisource
from riglib.source import MultiChanDataSource

class OnlineDataWorker(threading.Thread):
    '''
    Worker process to receive data from BMI3D.
    '''

    def __init__(self, socket, result_queue):
        '''
        Start the server on the specified IP address and port number.
        '''
        self.sock = socket
        self._stop_event = threading.Event()
        self.result_queue = result_queue
        super().__init__()

    def run(self):
        while not self._stop_event.is_set():
            ready = select.select([self.sock], [], [], 0.1) # 100ms timeout to check _stop_event
            if ready[0]:
                data = self.sock.recv(4096)
                key, value = data.decode('utf-8').split(':')
                self.result_queue.put((key, [json.loads(v) for v in value.split('#')]))

    def stop(self):
        self._stop_event.set()

class AnalysisWorker(mp.Process):
    '''
    Plots eye, cursor, and target data from experiments that have them. Performs automatic
    calibration of eye data to target locations when the cursor enters the target.
    '''

    def __init__(self, task_params, data_queue, figsize=(8,10)):
        self.task_params = task_params
        self._stop_event = mp.Event()
        self.data_queue = data_queue
        self.figsize = figsize
        self.sync_events = []
        self.cursor_pos = np.zeros(2)
        self.eye_pos = np.zeros(2)
        self.target_pos = {}
        self.sync_events_checked = 0
        self.eye_coeff = np.zeros((2, 2))
        self.calibration_data = []
        super().__init__()
   
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
        
    def get_current_pos(self):
        '''
        Get the current cursor, eye, and target positions

        Returns:
            cursor_pos ((2,) tuple): Current cursor position
            eye_pos ((2,) tuple): Current eye position
            targets (list): List of active targets in (position, radius, color) format
        '''
        self.update_eye_calibration()
        calibrated_eye_pos = np.dot(self.eye_pos, self.eye_coeff)

        try:
            radius = self.task_params['target_radius']
            color = self.task_params['target_color']
            targets = [(self.target_pos[k], radius, color) for k, v in self.targets.items() if v]
        except:
            targets = []

        return self.cursor_pos, calibrated_eye_pos, targets

    def init(self):
        # Initialize figure
        self.fig = plt.figure(figsize=self.figsize)
        plt.show(block=False)

        self.ax = self.fig.add_subplot(221)
        self.text = self.ax.text(0.5, 0.5, '', ha='center', va='center', fontsize=12)
        self.text.set_text('Waiting for data...')
        self.fig.canvas.draw()

    def update(self):
        try:
            key, values = self.data_queue.get(timeout=0.1)
        except queue.Empty:
            return
        if key == 'cycle_count':
            self.text.set_text(values[0])
        if key == 'sync_event':
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

    def run(self):
        print('Starting analysis worker:', self.__class__.__name__)
        self.init()
        while not self._stop_event.is_set():
            self.update()
            plt.pause(0.1)

        plt.close(self.fig) # add saving feature here instead of just closing the figure

    def stop(self):
        self._stop_event.set()

class ERPAnalysisWorker(AnalysisWorker):
    '''
    Plots ERP data from experiments with a ECoG244 array. Automatically calculates 
    ERPs for flash, movement, or laser events depending on the task.
    '''

    bufferlen = 5 # seconds of data to keep in the buffer

    def init(self):
        self.elec_pos, self.acq_ch, _ = aopy.data.load_chmap('ECoG244')
        if hasattr(self.task_params, 'qwalor_trigger_dch'):
            self.trigger_ch = self.task_params['qwalor_trigger_dch']
            self.channels = map_channels_for_multisource(headstage_channels=self.acq_ch, digital_channels=[self.trigger_ch])
        self.ds = MultiChanDataSource(MultiSource, channels=self.channels, bufferlen=self.bufferlen)
        self.ds.start()

    def get_erp(self, channels):
        data = self.ds.get_new(channels)


class OnlineDataServer(threading.Thread):
    '''
    Interface for accumulating and analyzing BMI3D data in real-time.
    '''

    def __init__(self, host_ip, port=5000):
        '''
        Initialize the server on the specified IP address and port number.

        Args:
            host_ip (str): IP address of the machine running the online analysis
            port (int): Port number for the online analysis server
        '''

        # Initialize socket
        self.sock = socket.socket(
            socket.AF_INET, # Internet
              socket.SOCK_DGRAM) # UDP 
        self.sock.bind((host_ip, port))
        self.sock.setblocking(0)
                
        # Initialize workers
        self.data_worker = None
        self.analysis_workers = []

        # Initialize the server
        self._stop_event = threading.Event()
        self.reset()
        self.is_completed = False
        super().__init__()

    def _stop(self):
        # Stop all the workers
        for worker, _ in self.analysis_workers:
            if worker.is_alive():
                worker.stop()
                worker.join()
        self.analysis_workers = []
        if self.data_worker and self.data_worker.is_alive():
            self.data_worker.stop()
            self.data_worker.join()
        self.data_worker = None

    def reset(self):
        self._stop()

        # Start new workers
        self.result_queue = queue.Queue()
        self.data_worker = OnlineDataWorker(self.sock, self.result_queue)
        self.data_worker.start()
        self.is_running = False
        self.is_completed = True
        self.task_params = {}
        self.state = None

    def init(self):
        '''
        Once the experiment is initialized but before it starts, we spin up the analysis processes
        based on what kind of experiment is running.
        '''
        # Always start with the basic analysis worker
        print('init in state', self.state)
        data_queue = mp.Queue()
        self.analysis_workers.append((AnalysisWorker(self.task_params, data_queue), data_queue))

        # # Is there an ECoG array?
        # if hasattr(self.task_params, 'record_headstage') and self.task_params['record_headstage']:
        #     queue = Queue()
        #     self.analysis_workers.append((ERPAnalysisWorker(self.task_params, queue), queue))

        # Start all the workers
        for worker, _ in self.analysis_workers:
            worker.start()
        
    def update(self):
        '''
        Get the latest data from the online server
        '''
        try:
            key, values = self.result_queue.get(timeout=0.1)
        except queue.Empty:
            time.sleep(0.1)
            return False
        if key == 'state':
            self.state = values[0]
            if values[0] is None:
                print('Experiment finished')
                self.reset()
                self.is_running = False
                self.is_completed = True
            elif not self.is_running and not self.is_completed: # check in case we missed the init message
                self.is_running = True
                self.init()
            for _, data_queue in self.analysis_workers:
                data_queue.put((key, values))
        elif key == 'init':
            self.is_running = True
            self.init()
        elif key == 'param':
            name, value = values
            self.task_params[name] = value
        else:
            # Send everything else back onto the queues for the analysis workers
            for _, data_queue in self.analysis_workers:
                data_queue.put((key, values))
        return True

    def run(self):
        '''
        Main loop to run the server
        '''
        while True:
            self.update()
            if self._stop_event.is_set():
                break

        self._stop()
        self.sock.close()

    def stop(self):
        self._stop_event.set()
