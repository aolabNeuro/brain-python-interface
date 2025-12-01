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
    update_freq = 44100./1024  # Approx 43 Hz
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
        )

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p_audio.terminate()

    def get(self):
        '''
        Retrieve a packet from the host
        '''
        data = self.stream.read(1024, exception_on_overflow=False)
        return np.expand_dims(np.frombuffer(data, dtype=self.dtype), axis=1)