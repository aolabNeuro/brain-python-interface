from .comms import QuattroOtlight
import numpy as np
from riglib.source import DataSourceSystem

'''
quattrocento streaming sources
'''

def multi_chan_generator(data_block, downsample=1):
    for idx in range(data_block.shape[1]):
        yield (idx+1, data_block[::downsample, idx]) # yield one channel at a time

class EMG(DataSourceSystem):
    '''
    Wrapper class for QuattroOtlight compatible with using in DataSource for
    buffering neural data. Compatible with riglib.source.MultiChanDataSource
    '''
    update_freq = 2048.
    dtype = np.dtype('int16')
    emg_refresh_freq = 32
    n_arrays = 1

    def __init__(self, *args, **kwargs):

        # Configure channels based on how many 64-channel arrays are connected to the MULTI IN ports
        channels = [16 * 8, 16 * 8 + 64 * self.n_arrays]
        self.qt = QuattroOtlight(host='128.95.215.191', refresh_freq=self.emg_refresh_freq, 
                                 emg_ch_range=channels, sample_freq=self.update_freq)

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
            data = np.hstack([emg, aux, samples])
            self.gen = multi_chan_generator(data)
            return next(self.gen)