'''
Functions to analyze data in real-time. 
Useful for debugging and monitoring the system on a separate machine connected over the network.
'''

import ast
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

    def __init__(self, socket, result_queue, timeout):
        '''
        Start the server on the specified IP address and port number.
        '''
        self.timeout = timeout
        self.sock = socket
        self._stop_event = Event()
        self.result_queue = result_queue
        Thread.__init__(self)

    def run(self):
        while not self._stop_event.is_set():
            ready = select.select([self.sock], [], [], self.timeout)
            if ready[0]:
                data = self.sock.recv(4096)
                key, value = data.decode('utf-8').split(':')
                self.result_queue.put((key, value))
            else:
                self.result_queue.put(('timeout', self.timeout))

    def stop(self):
        self._stop_event.set()


class OnlineDataServer:

    def __init__(self, host_ip, port=5000, timeout=1):
        self.sock = socket.socket(
            socket.AF_INET, # Internet
              socket.SOCK_DGRAM) # UDP 
        self.sock.bind((host_ip, port))
        self.sock.setblocking(0)
        self.timeout = timeout
        self.reset()

    def reset(self):
        self.result_queue = Queue()
        self.worker = OnlineDataWorker(self.sock, self.result_queue, self.timeout)
        self.worker.start()
        self.is_running = False
        self.is_completed = False
        self.task_params = {}
        self.cycle_count = 0
        self.state = None
        self.sync_events = []

    def _stop(self):
        if self.worker.is_alive():
            self.worker.stop()
            self.worker.join()
        
    def update(self):
        '''
        Get the latest data from the online server
        '''
        while not self.result_queue.empty():
            key, value = self.result_queue.get()
            print(key, value)
            if key == 'cycle':
                self.cycle_count = int(value)
            elif key == 'state':
                self.state = value
                if value == 'None':
                    print('Experiment finished')
                    self._stop()
                    self.is_running = False
                    self.is_completed = True
            elif key == 'sync_event':
                event_name, event_data = value.split('/')
                self.sync_events.append((event_name, int(event_data)))
            elif key == 'init':
                self.is_running = True
            else:
                # Any other incoming data is a parameter update
                self.task_params[key] = value
    
    def close(self):
        self._stop()
        self.sock.close()


class OnlineERP(OnlineDataServer):
    '''
    
    '''

    def __init__(self, source, channels, bufferlen, host_ip, port=5000, timeout=10):
        super().__init__(host_ip, port, timeout)
        self.ds = MultiChanDataSource(source, channels=channels, bufferlen=bufferlen)
        self.ds.start()
        
    def get_erp(self, channels):
        data = self.ds.get_new(channels)
