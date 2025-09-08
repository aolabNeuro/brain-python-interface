import unittest
from riglib.audio import AudioGenerator
import time
import numpy as np
import matplotlib.pyplot as plt

class TestDIO(unittest.TestCase):

    @unittest.skip("")
    def test_audio_gen(self):
        ag = AudioGenerator()
        
        freqs = [440, 158, 440]
        for t in range(len(freqs)):
            tone = ag.get_tone(freqs[t], 0.01)

            time = np.linspace(0.01*t, 0.01*(t+1), len(tone), endpoint=False)
            plt.plot(1000*time, tone[:,0])
        plt.show()

    @unittest.skip("")
    def test_audio_tone(self):
        ag = AudioGenerator()
        # tone = ag.get_tone(440)
        # plt.plot(tone[:,0])
        # plt.show()

        ag.play_tone(1, 400)
        time.sleep(1.5)

    # @unittest.skip("")
    def test_audio_continuous(self):
        ag = AudioGenerator()
        
        freqs = np.linspace(440, 880, 10)
        ag.play_tone_continuous(freqs[0])
        for t in range(len(freqs)):
            print(freqs[t])
            ag.change_freq(freqs[t])
            time.sleep(0.1)
        ag.stop_tone()

if __name__ == '__main__':
    unittest.main()