import serial
import pyfirmata, time
import serial.tools.list_ports
import numpy as np
import hid
import re

# The following command can be used for finding available ports
# python -m serial.tools.list_ports

import unittest


class EMGTests(unittest.TestCase):

    def test_connect(self):
        
        h = hid.device()
        h.open(0x2e73, 0x0001)  # Muscle SpikerBox Pro VendorID/ProductID

        print("Manufacturer: %s" % h.get_manufacturer_string())
        print("Product: %s" % h.get_product_string())
        print("Serial No: %s" % h.get_serial_number_string())

        # enable non-blocking mode
        # h.set_nonblocking(1)

        # write version query data to the device
        print("Write the data")
        data = [0x3f, 0x3e] + list(bytearray("?:;".ljust(62, "\0").encode("utf-8")))
        print(data)
        h.write(data)

        # read back the answer
        print("Read the data")
        d = h.read(64, 1000)
        if d:
            length = d[1]
            print(bytes(d[2:length]))
            payload = re.search(b'\xff\xff\x01\x01\x80\xff(.*?)\xff\xff\x01\x01\x81\xff', bytes(d[2:length])).group(1)
            print(payload.decode('utf-8'))
        else:
            print("timeout")

        # Ask for max samplerate and channels
        data = [0x3f, 0x3e] + list(bytearray("max:;".ljust(62, "\0").encode("utf-8")))
        print(data)
        h.write(data)
        d = h.read(64, 1000)
        print(d)

        # Start streaming data
        data = [0x3f, 0x3e] + list(bytearray("start:;".ljust(62, "\0").encode("utf-8")))
        print(data)
        h.write(data)

        data_ch1 = []
        data_ch2 = []
        next_ch1 = 0
        next_ch2 = 0
        frame_counter = 0
        for _ in range(2):
            d = h.read(64, 1000)
            for i in d[2:]:
                if i >> 7:
                    frame_counter = 0 # start of frame
                    data_ch1.append(next_ch1)
                    data_ch2.append(next_ch2)
                    next_ch1 = 0
                    next_ch2 = 0
                else:
                    frame_counter += 1 # continuation

                if frame_counter == 0:
                    next_ch1 = i & 0x7
                    print('ch1:', next_ch1, end=" ")
                elif frame_counter == 1:
                    next_ch1 = (next_ch1 << 7) | (i & 0x7f)
                    print(i & 0x7f, end=" ")
                elif frame_counter == 2:
                    next_ch2 = i & 0x7
                    print('ch2:', next_ch1, end=" ")
                elif frame_counter == 3:
                    next_ch2 = (next_ch2 << 7) | (i & 0x7f)
                    print(i & 0x7f)
                elif frame_counter > 3:
                    break
            



        print(data_ch1, data_ch2)

        # Stop streaming data
        data = [0x3f, 0x3e] + list(bytearray("h:;".ljust(62, "\0").encode("utf-8")))
        print(data)
        h.write(data)
        d = h.read(64, 1000)
        print(d)


        print("Closing the device")
        h.close()

    # def test_source(self):
    #     f = ForceSensorControl()
    #     f.init()
    #     t0 = time.time()
    #     while time.time() - t0 < 5:
    #         print(f.force_sensor.get())
    #         time.sleep(0.1)


if __name__ == '__main__':
    unittest.main()