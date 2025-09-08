import pygame
import os
import numpy as np
import sounddevice as sd
import time
audio_path = os.path.dirname(__file__)

class AudioPlayer():

    def __init__(self, filename='click.wav'):
        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(44100, -16, 2, 2048)
            pygame.mixer.init()
        self.effect = pygame.mixer.Sound(os.path.join(audio_path, filename))

    def get_length(self):
        return self.effect.get_length()

    def play(self):
        self.effect.play()

    def stop(self):
        self.effect.stop()

class ToneGenerator():
    def __init__(self, freq=440, volume=1.0, sample_rate=44100):
        self.freq = freq
        self.volume = volume
        self.sample_rate = sample_rate
        self.phase = 0.0

    def callback(self, outdata, frames, time_info, status):
        t = (np.arange(frames) + 0) / self.sample_rate
        phase_inc = 2 * np.pi * self.freq / self.sample_rate
        phases = self.phase + phase_inc * np.arange(frames)
        self.phase = (phases[-1] + phase_inc) % (2*np.pi)
        samples = np.sin(phases).astype(np.float32) * self.volume
        outdata[:, 0] = samples
        outdata[:, 1] = samples

    def start(self):
        self.stream = sd.OutputStream(
            channels=2,
            callback=self.callback,
            samplerate=self.sample_rate,
            blocksize=1024
        )
        self.stream.start()

    def change_freq(self, freq):
        self.freq = freq

    def change_volume(self, volume):
        self.volume = max(0.0, min(1.0, volume))

    def stop(self):
        self.stream.stop()
        self.stream.close()