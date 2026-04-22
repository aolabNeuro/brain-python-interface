import numpy as np
import hid
import re
import time

class SpikerBox:

    def __init__(self, timeout=0.05):

        self.timeout = int(timeout*1000) # convert s to ms
        self.h = hid.device()
        self.h.open(0x2e73, 0x0001)  # Muscle SpikerBox Pro VendorID/ProductID

        # Ensure device starts from a known non-streaming state
        self.send_cmd("h:;")
        time.sleep(0.02)
        self._drain_input()

        # write version query data to the device
        self.fw_ver, self.hw_type, self.hw_ver = self._query_with_retry("?:;", "FWV", "HWT", "HWV")
        print("Firmware version:", self.fw_ver)
        print("Hardware type:", self.hw_type)
        print("Hardware version:", self.hw_ver)

        # Ask for max samplerate and channels
        samplerate, n_channels = self._query_with_retry("max:;", "MSF", "MNC")
        if samplerate is None or n_channels is None:
            print("SpikerBox metadata missing; using defaults samplerate=10000, channels=2")
            self.samplerate = 10000.0
            self.n_channels = 2
        else:
            self.samplerate = float(samplerate)
            self.n_channels = int(n_channels)
        print("Samplerate:", self.samplerate, "hz")
        print("Number of channels:", self.n_channels)

        # Some attributes to keep track of the continuous data
        self.data = None
        self.idx = 0
        self.ch = 1
        self.pending_samples = []

    def _query_with_retry(self, cmd, *keys, n_tries=5, retry_delay=0.02):
        resp = tuple(None for _ in keys)
        for _ in range(n_tries):
            self._drain_input()
            self.send_cmd(cmd)
            resp = self.parse_response(*keys)
            if all(v is not None for v in resp):
                return resp
            time.sleep(retry_delay)
        return resp

    def _drain_input(self):
        # Remove stale packets before sending a command so parse_response
        # reads the corresponding reply.
        for _ in range(10):
            d = self.h.read(64, 1)
            if d is None or len(d) == 0:
                break

    def start(self):
        self.send_cmd("start:;")

    def stop(self):
        self.send_cmd("h:;")
        
        # Get some packets until there is no data
        while len(self.h.read(64, self.timeout)) > 0:
            time.sleep(float(self.timeout)/1000)


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
        response = {key: None for key in keys}
        deadline = time.time() + max(0.1, 5 * float(self.timeout) / 1000.0)

        while time.time() < deadline:
            d = self.h.read(64, self.timeout)
            if not d:
                continue

            length = d[1] if len(d) > 1 else len(d)
            data = bytes(d[2:length]) if length > 2 else bytes(d[2:])

            payload_match = re.search(b'\xff\xff\x01\x01\x80\xff(.*?)\xff\xff\x01\x01\x81\xff', data)
            payload = payload_match.group(1) if payload_match is not None else data
            msg = payload.decode('utf-8', errors='ignore')

            for key in keys:
                if response[key] is None:
                    key_match = re.search(rf'{re.escape(key)}:(.*?);', msg)
                    if key_match is not None:
                        response[key] = key_match.group(1)

            if all(response[key] is not None for key in keys):
                break

        return tuple(response[key] for key in keys)

    def get_next_ch(self):
        '''
        Data packets always 64 bytes long; 1st byte constant (ignored), 2nd byte payload length, then 
        Data is always 2 channels of 10 bits each encoded in frames of 4 bytes with data in the first 
            7 bits of each byte.
        '''
        if self.pending_samples:
            return self.pending_samples.pop(0)

        while True:
            d = self.h.read(64, self.timeout)
            if d is None or len(d) == 0:
                continue

            length = d[1] if len(d) > 1 else len(d)
            payload = d[2:length] if length > 2 else d[2:]

            i = 0
            while i + 3 < len(payload):
                b0 = payload[i]
                if not (b0 >> 7):
                    i += 1
                    continue

                b1 = payload[i + 1]
                b2 = payload[i + 2]
                b3 = payload[i + 3]

                if (b1 >> 7) or (b2 >> 7) or (b3 >> 7):
                    i += 1
                    continue

                ch1 = ((b0 & 0x07) << 7) | (b1 & 0x7F)
                ch2 = ((b2 & 0x07) << 7) | (b3 & 0x7F)

                self.pending_samples.append((1, [ch1]))
                if self.n_channels >= 2:
                    self.pending_samples.append((2, [ch2]))

                i += 4

            if self.pending_samples:
                return self.pending_samples.pop(0)
        
    def close(self):
        self.h.close()