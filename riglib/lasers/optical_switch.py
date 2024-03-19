import serial
import pyfirmata, time
import serial.tools.list_ports
import numpy as np

class LFiberOpticalSwitch:
    '''
    Communication to the LFiber 1x16 optical switch
    '''

    def __init__(self, port='/dev/lfiberswitch', channels=16, timeout=3):
        self.conn = serial.Serial(port, 115200, timeout=timeout)
        self.channels = channels
            
    def _send_and_receive(self, int):
        print(bytearray([int]))
        self.conn.write(bytearray([int]))
        ret = list(self.conn.read())
        if len(ret) == 0:
            return False
        return bool(ret[0])

    def reset(self):
        return self._send_and_receive(16)

    def check_status(self):
        return self._send_and_receive(17)

    def set_channel(self, idx):
        '''Set the 0-indexed channel by idx'''
        return self._send_and_receive(idx)
    
    def exit(self):
        self.conn.close()

