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

class TonePlayer():

    def __init__(self, frequency = 440, duration = 0.1, sample_rate = 44100):
        self.frequency = frequency
        self.duration = duration
        self.sample_rate = sample_rate
        self.sound = self.generate_tone()

    def generate_tone(self):
        # Generate a tone using numpy
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), False)
        tone = 0.5 * np.sin(2 * np.pi * self.frequency * t)
        stereo_tone = np.column_stack((tone, tone))  # Make it stereo
        sound = pygame.sndarray.make_sound((stereo_tone * 32767).astype(np.int16))
        return sound

    def play(self):
        self.sound.play()
    
    def stop(self):
        self.sound.stop()