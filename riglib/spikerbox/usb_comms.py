import numpy as np
import hid
import re


class SpikerBox:

    def __init__(self):
        
        self.h = hid.device()
        self.h.open(0x2e73, 0x0001)  # Muscle SpikerBox Pro VendorID/ProductID

        # write version query data to the device
        self.send_cmd("?:;")
        
        # Ask for max samplerate and channels
        self.send_cmd("max:;")

        self.next_ch1 = 0
        self.next_ch2 = 0
        self.frame_counter = 0

    def start(self):
        return self.send_cmd("start:;")

    def stop(self):
        return self.send_cmd("start:;")

    def send_cmd(self, cmd):
        '''
        Command packet is always 64 bytes long; starts with 0x3f and 0x3e, then command null padded
        Response always 64 bytes long; 1st byte constant (ignored), 2nd byte payload length, then 
            the data, which is escaped with 
            start signal: \xff\xff\x01\x01\x80\xff and 
            stop signal: \xff\xff\x01\x01\x81\xff
            in the case of a command
        '''
        data = [0x3f, 0x3e] + list(bytearray(cmd.ljust(62, "\0").encode("utf-8")))
        self.h.write(data)

        d = self.h.read(64, 1000)
        if d:
            length = d[1]
            print(bytes(d[2:length]))
            payload = re.search(b'\xff\xff\x01\x01\x80\xff(.*?)\xff\xff\x01\x01\x81\xff', bytes(d[2:length])).group(1)
            return payload.decode('utf-8')
        else:
            return None

    def read_data(self):
        '''
        Data packets always 64 bytes long; 1st byte constant (ignored), 2nd byte payload length, then 
        Data is always 2 channels of 10 bits each encoded in frames of 4 bytes.
        '''
        data_ch1 = []
        data_ch2 = []
        for _ in range(2):
            d = self.h.read(64, 1000)
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
        
    def close(self):
        self.h.close()