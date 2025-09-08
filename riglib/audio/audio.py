import pygame
import os
import numpy as np

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

class AudioGenerator():

    def __init__(self):
        self.sample_rate = 44100
        self.max_sample = 32767
        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(44100, -16, 2, 2048)
            pygame.mixer.init()
        self.sound = None
        self.phase = 0

    def get_tone(self, freq, duration):
        n_samples = int(self.sample_rate * duration)
        buf = np.zeros((n_samples, 2), dtype=np.int16)

        # update phase accumulator
        phase_inc = 2 * np.pi * freq / self.sample_rate
        phases = self.phase + phase_inc * np.arange(n_samples)
        self.phase = (phases[-1] + phase_inc) % (2*np.pi)

        # generate samples
        samples = np.sin(phases)
        buf[:,0] = np.round(self.max_sample * samples).astype(np.int16)
        buf[:,1] = np.round(self.max_sample * samples).astype(np.int16)
        return buf

    def play_tone(self, duration, freq):
        self.phase = 0
        buf = self.get_tone(freq, duration)
        sound = pygame.sndarray.make_sound(buf)
        sound.play()

    def play_tone_continuous(self, freq, buffer_len=0.1):
        self.phase = 0
        buf = self.get_tone(freq, buffer_len)
        sound = pygame.sndarray.make_sound(buf)
        sound.play(-1)
        self.sound = sound

    def update_tone(self, freq):
        if self.sound is None:
            return
        buf = pygame.sndarray.samples(self.sound) # This is required because make_sound copies the array, but we want a reference
        new_buf = self.get_tone(freq, buf.shape[0]/self.sample_rate)
        np.copyto(buf, new_buf)

    def stop_tone(self):
        if self.sound is not None:
            self.sound.stop()
            self.sound = None

    def play_white_noise(self, duration):
        n_samples = int(self.sample_rate * duration)
        buf = np.zeros((n_samples, 2), dtype=np.int16)

        buf[:,0] = np.round(self.max_sample*np.random.normal(0, self.max_sample/2., (n_samples, ))).astype(int)
        buf[:,1] = np.round(self.max_sample*np.random.normal(0, self.max_sample/2., (n_samples, ))).astype(int)
        sound = pygame.sndarray.make_sound(buf)
        sound.play()
