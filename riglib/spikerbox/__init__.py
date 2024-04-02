import numpy as np
import time 

from ..source import DataSourceSystem
from .usb_comms import SpikerBox

class LFP(DataSourceSystem):
    '''
    SpikerBox DataSourceSystem collects EMG data from USB packets. Compatible with riglib.source.MultiChanDataSource
    '''
    update_freq = 10000
    dtype = np.dtype('float')

    def __init__(self, channels=[1,2]):
        self.b = SpikerBox() # assume we asked for the right channels!!
        self.channels = channels

    def start(self):
        self.b.start()

    def stop(self):
        self.b.stop()
        self.b.close()
    
    def get(self):
        return self.b.get_next_ch()

