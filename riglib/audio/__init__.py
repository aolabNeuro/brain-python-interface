from .audio import *
from riglib.source import DataSourceSystem

import numpy as np
import pyaudio

'''
Audio streaming sources
'''
class MonoAudio(DataSourceSystem):
    '''
    Wrapper class for pyaudio compatible with using in DataSource for
    buffering audio data.
    '''
    update_freq = 44100
    dtype = np.dtype('int16')

    def start(self):
        print("Initializing audio input stream")
        self.p_audio = pyaudio.PyAudio()

        print("Starting audio input stream")
        self.stream = self.p_audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024,
            input_device_index=2
        )

        self.gen = iter(())

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p_audio.terminate()

    def get(self):
        '''
        Retrieve a packet from the host
        '''
        try:
            return next(self.gen)
        except StopIteration:

            data = self.stream.read(1024, exception_on_overflow=False)
            self.gen = iter(np.frombuffer(data, dtype=self.dtype))
            return next(self.gen) 