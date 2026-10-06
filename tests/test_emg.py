import serial
import pyfirmata, time
import serial.tools.list_ports
import numpy as np
import re

# The following command can be used for finding available ports
# python -m serial.tools.list_ports

import unittest

from riglib import source
import hid
from riglib.spikerbox.usb_comms import SpikerBox
from riglib.spikerbox import EMG

class EMGTests(unittest.TestCase):

    #@unittest.skip("")
    def test_connect(self):
        b = SpikerBox()

        print("Manufacturer: %s" % b.manufacturer)
        print("Product: %s" % b.product)
        print("Serial No: %s" % b.serial)
        print("Firmware version: %s" % b.fw_ver)
        print("Hardware type: %s" % b.hw_type)
        print("Hardware version: %s" % b.hw_ver)
        print("Samplerate: %s" % b.samplerate)
        print("Number of channels: %s" % b.n_channels)

        b.start()

        samples = []
        for _ in range(4):
            samples.append(b.get_next_ch())

        print(samples)

        b.stop()
        b.close()

    #@unittest.skip("")
    def test_source(self):
        b = SpikerBox()
        b.start()
        t0 = time.time()
        while time.time() - t0 < 0.1:
            print(b.get_next_ch(), end="")

        b.stop()
        print('stop')
        time.sleep(0.5)

        while time.time() - t0 < 0.1:
            print(b.get_next_ch(), end="")
        print('done')

    #@unittest.skip("")
    def test_datasource(self):
        channels = [1, 2]
        ds = source.MultiChanDataSource(EMG, channels=channels)
        ds.start()
        time.sleep(4)
        if ds.status.value <= 0:
            ds.stop()
            self.skipTest("SpikerBox datasource did not start (device unavailable or metadata read failed)")
        data = ds.get_new(channels)
        ds.stop()

        print(data[0].shape)
        print(data[1].shape)

        print(data)
        self.assertEqual(len(data), len(channels))
        self.assertTrue(all(ch_data is not None for ch_data in data))


if __name__ == '__main__':
    unittest.main()