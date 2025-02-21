import time
from riglib import source
from riglib.quattrocento import EMG, comms
from riglib.bmi import state_space_models, train, extractor
import numpy as np
import unittest

STREAMING_DURATION = 3

class TestStreaming(unittest.TestCase):

    @unittest.skip('works')
    def test_direct(self):
        qt = comms.QuattroOtlight(host='128.95.215.191', refresh_freq=32)
        qt.setup()
        data = qt.read_emg()
        for d in data:
            print(d.shape)
        qt.tear_down()

    @unittest.skip('works')
    def test_quattrocento_stream(self):
        channels = [1, 3]
        bb = EMG(channels=channels) # Note: this is not how this is normally used, just testing
        bb.start()
        data = bb.qt.read_emg()
        bb.stop()
        print(data[2].shape)
    
    @unittest.skip('works')
    def test_broadband_class(self):
        channels = [1, 3]
        bb = EMG(channels=channels)
        bb.start()
        ch_data = []
        for d in range(2):
            for i in range(len(channels)):
                ch, data = bb.get()
                ch_data.append(data)
                print(f"Got channel {ch} with {data.shape} samples")
        bb.stop()

    @unittest.skip('works')
    def test_broadband_datasource(self):
        channels = [1, 62]
        ds = source.MultiChanDataSource(EMG, channels=channels)
        ds.start()
        time.sleep(STREAMING_DURATION)
        data = ds.get_new(channels)
        ds.stop()
        data = np.array(data)

        n_samples = int(EMG.update_freq * STREAMING_DURATION)

        self.assertEqual(data.shape[0], len(channels))
        self.assertEqual(data.shape[1], n_samples)

    @unittest.skip('works')
    def test_update_frequency(self):
        channels = [1, 62]
        ds = source.MultiChanDataSource(EMG, channels=channels, send_data_to_sink_manager=True)
        ds.start()
        t0 = time.perf_counter()
        n_packets = 0
        while time.perf_counter() - t0 < STREAMING_DURATION:
            data = ds.get_new(channels)
            if len(data[0]) > 0 or len(data[1]) > 0:
                n_packets += 1
            time.sleep(0.001)
        ds.stop()

        expected_packets = 32 * int(STREAMING_DURATION)

        self.assertAlmostEqual(n_packets, expected_packets, delta=10)

    @unittest.skip('works')
    def test_ds_with_extractor(self):
        # Create the datasource
        channels = [1, 62]
        ds = source.MultiChanDataSource(EMG, channels=channels)

        # Make a feature extractor
        extr = extractor.LFPMTMPowerExtractor(ds, channels=channels, bands=[(90,110)], win_len=0.1, fs=2048)

        # Run the feature extractor
        extract_rate = 1/60
        feats = []
        ds.start()
        t0 = time.perf_counter()
        while (time.perf_counter() - STREAMING_DURATION < t0):
            neural_features_dict = extr(time.perf_counter())
            feats.append(neural_features_dict['lfp_power'])
            time.sleep(extract_rate)
        
        ds.stop()
        data = np.array(feats)
        print(feats)
        print(data.shape)
        self.assertEqual(data.shape[1], len(channels))
        self.assertEqual(data.shape[0], STREAMING_DURATION/extract_rate)

if __name__ == '__main__':
    unittest.main()