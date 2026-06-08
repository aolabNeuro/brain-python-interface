import numpy as np
import time 

from ..source import DataSourceSystem
from .usb_comms import SpikerBox

class LFP(DataSourceSystem):
    '''
    SpikerBox DataSourceSystem collects EMG data from USB packets. Compatible with riglib.source.MultiChanDataSource
    '''
    update_freq = 10000
    dtype = np.dtype('int16')

    def __init__(self, channels=[1,2]):
        self.b = SpikerBox()
        # Remove any channels that aren't 1 or 2, since those are the only ones that exist on the SpikerBox
        self.channels = [ch for ch in channels if ch in [1,2]]

    def start(self):
        self.b.start()

    def stop(self):
        self.b.stop()
        self.b.close()
    
    def get(self):
        while True:
            ch, data = self.b.get_next_ch()
            if ch in self.channels:
                return ch, data

