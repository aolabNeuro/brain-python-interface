import numpy as np
import hid
import re


class SpikerBox:

    def __init__(self):
        
        self.h = hid.device()
        self.h.open(0x2e73, 0x0001)  # Muscle SpikerBox Pro VendorID/ProductID

        # write version query data to the device
        self.send_cmd("?:;")
        self.fw_ver, self.hw_type, self.hw_ver = self.parse_response("FWV", "HWT", "HWV")
        print("Firmware version:", self.fw_ver)
        print("Hardware type:", self.hw_type)
        print("Hardware version:", self.hw_ver)

        # Ask for max samplerate and channels
        self.send_cmd("max:;")
        samplerate, n_channels = self.parse_response("MSF", "MNC")
        self.samplerate = float(samplerate)
        self.n_channels = int(n_channels)
        print("Samplerate:", self.samplerate, "hz")
        print("Number of channels:", self.n_channels)

        # Some attributes to keep track of the continuous data
        self.data = None
        self.idx = 0
        self.ch = 1

    def start(self):
        self.send_cmd("start:;")

    def stop(self):
        self.send_cmd("h:;")

    def send_cmd(self, cmd):
        '''
        Command packet is always 64 bytes long; starts with 0x3f and 0x3e, then command null padded
        '''
        data = [0x3f, 0x3e] + list(bytearray(cmd.ljust(62, "\0").encode("utf-8")))
        self.h.write(data)

    def parse_response(self, *keys):
        '''
        Response always 64 bytes long; 1st byte constant (ignored), 2nd byte payload length, then 
            the data, which is escaped with 
            start signal: \xff\xff\x01\x01\x80\xff and 
            stop signal: \xff\xff\x01\x01\x81\xff
        '''
        d = self.h.read(64, 10) # 10 ms timeout
        if d:
            length = d[1]
            payload = re.search(b'\xff\xff\x01\x01\x80\xff(.*?)\xff\xff\x01\x01\x81\xff', bytes(d[2:length])).group(1)
            msg = payload.decode('utf-8')
            response = []
            for key in keys:
                response.append(re.search(f'{key}:(.*?);', msg).group(1))
            return tuple(response)
        else:
            return None

    def get_next_ch(self):
        '''
        Data packets always 64 bytes long; 1st byte constant (ignored), 2nd byte payload length, then 
        Data is always 2 channels of 10 bits each encoded in frames of 4 bytes with data in the first 
            7 bits of each byte.
        '''
        MSB = 0
        LSB = 0
        frame_counter = 0
        while frame_counter < 2:

            if self.data is None or self.idx >= len(self.data):
                d = self.h.read(64, 10) # 10 ms timeout
                if d is None or len(d) == 0:
                    return (self.ch, [0]) # no data to read
                self.data = d[2:]
                self.idx = 0

            i = self.data[self.idx]
            if frame_counter == 0:
                if self.ch == 1 and not i > 127: # frame error
                    frame_counter += 1
                    break
                MSB  = i & 0x7F

            elif frame_counter == 1:
                if i > 127: # frame error
                    frame_counter = 0
                    break # continue as if we have new frame
                
                LSB = i & 0x7F
            
            self.idx += 1
            frame_counter += 1

        out = (self.ch, [LSB | MSB<<7])
        self.ch = (self.ch % self.n_channels) + 1
        return out
        
    def close(self):
        self.h.close()