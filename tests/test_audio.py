import unittest
from riglib.audio import ToneGenerator
import time
import numpy as np
import matplotlib.pyplot as plt

class TestDIO(unittest.TestCase):

    def test_audio_continuous(self):
        tg = ToneGenerator()
        freqs = np.linspace(440, 880, 20)
        tg.start()
        for t in range(len(freqs)):
            print(freqs[t])
            tg.change_freq(freqs[t])
            tg.change_volume(0.1*(t+1)/len(freqs))
            time.sleep(0.1)
        tg.stop()

if __name__ == '__main__':
    unittest.main()