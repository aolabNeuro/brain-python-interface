'''
Functions to analyze data in real-time. 
Useful for debugging and monitoring the system on a separate machine connected over the network.
'''

import socket
import select
from threading import Thread, Event
from queue import Queue
import numpy as np
import matplotlib.pyplot as plt
from riglib.ecube import Digital, Analog, LFP, LFP_Plus_Trigger
from riglib.source import MultiChanDataSource

class OnlineDataWorker(Thread):
    '''
    Interface for ingesting and accumulating BMI3D data in real-time.
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
            ready = select.select([self.sock], [], [], self.timeout)
            if ready[0]:
                data = self.sock.recv(4096)
                key, value = data.split(':')
                self.result_queue.put((key, value))

    def stop(self):
        self._stop_event.set()


class OnlineDataServer:

    def __init__(self, host_ip, port=5000, timeout=10):
        self.sock = socket.socket(
            socket.AF_INET, # Internet
              socket.SOCK_DGRAM) # UDP 
        self.sock.bind((host_ip, port))
        self.sock.setblocking(0)
        self.result_queue = Queue()
        self.worker = OnlineDataWorker(self.sock, self.result_queue, timeout)
        self.worker.start()
        self.reset_experiment()

    def reset_experiment(self):
        self.running = False
        self.completed = False
        self.experiment = {}
        self.cycle_count = 0
        self.state = None
        self.sync_events = []

    def _stop(self):
        self.worker.stop()
        self.sock.close()
        print('Experiment stopped')
        
    def update(self):
        '''
        Get the latest data from the online server
        '''
        while not self.result_queue.empty():
            key, value = self.result_queue.get()
            if key == 'cycle':
                self.cycle_count = value
            elif key == 'state':
                self.state = value
                if value == 'None':
                    self._stop()
                    self.is_running = False
                    self.completed = True
            elif key == 'sync_event':
                event_name, event_data = value.split('/')
                self.sync_events.append((event_name, event_data))
            elif key == 'init':
                self.running = True
            else:
                # Any other incoming data is a parameter update
                self.experiment[key] = value

    @property
    def is_running(self):
        return self.running

    @property
    def experiment(self):
        return self.experiment

    @property
    def cycle_count(self):
        return self.cycle_count

    @property
    def state(self):
        return self.state
    
    @property
    def sync_events(self):
        return self.sync_events


class OnlineDataWorker(Thread):
    '''
    Interface for ingesting and accumulating BMI3D data in real-time.
    '''

    def __init__(self, socket, result_queue, timeout):
        '''
        Start the server on the specified IP address and port number.
        '''
        self.sock = socket
        self._stop_event = Event()
        self.result_queue = result_queue
        self.timeout = timeout
        Thread.__init__(self)

    def run(self):
        while not self._stop_event.is_set():
            ready = select.select([self.sock], [], [], self.timeout)
            if ready[0]:
                data = self.sock.recv(4096)
                key, value = data.split(':')
                self.result_queue.put((key, value))

    def stop(self):
        self._stop_event.set()


class OnlineERP(OnlineDataServer):
    '''
    
    '''

    def __init__(self, source, channels, bufferlen, host_ip, port=5000, timeout=10):
        super().__init__(host_ip, port, timeout)
        self.ds = MultiChanDataSource(source, channels=channels, bufferlen=bufferlen)
        self.ds.start()
        
    def get_erp(self, channels):
        data = self.ds.get_new(channels)
