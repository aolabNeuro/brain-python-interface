from .comms import QuattroOtlight
import numpy as np
from riglib.source import DataSourceSystem

'''
quattrocento streaming sources
'''

def multi_chan_generator(data_block, channels, downsample=1):
    for idx in range(len(channels)):
        yield (channels[idx], data_block[::downsample, channels[idx]-1]) # yield one channel at a time

class EMG(DataSourceSystem):
    '''
    Wrapper class for QuattroOtlight compatible with using in DataSource for
    buffering neural data. Compatible with riglib.source.MultiChanDataSource
    '''
    # Required by DataSourceSystem: update_freq and dtype (see make() below)
    update_freq = 2048.
    dtype = np.dtype('float')

    def __init__(self, channels=[1]):
        self.channels = channels
        self.qt = QuattroOtlight(host='128.95.215.191', refresh_freq=32)

    def start(self):
        self.qt.setup()
        
        # Start with an empty generator
        self.gen = iter(())

    def stop(self):
        self.qt.tear_down()
        
    def get(self):
        '''
        Retrieve a packet from the host and split it into individual channels
        '''
        try:
            return next(self.gen)
        except StopIteration:
            emg, aux, samples = self.qt.read_emg()
            self.gen = multi_chan_generator(emg, self.channels)
            return next(self.gen)