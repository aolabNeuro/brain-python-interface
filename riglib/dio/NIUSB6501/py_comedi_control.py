import subprocess
from shlex import split
import sys, time
import os

COMEDI_DIO_WRITE = 0 #this writes to a channel
COMEDI_DIO_BITFIELD = 1 #tis writes to multiple channel
MODULE_FOLDER_PATH = os.path.dirname(__file__)

def write_to_comedi(data, mask = 255, debug = False):
    '''
    data: an integer bet. 0 and 255 (inclusive) that maps to the 8 bit channels. 
    mask: an integer that marks which channels to write e.g. 0x3 means 0b11 writes to channels 0 and 1
    debug: decoding switch, if
        True: print out the command
        False (default): do nothing. 
    eg. write_to_comedi(3) sets first two channels to high. 

    note: for efficiency, this function assumes data is in integer in [0, 255], 

    '''
    
    cmd = f'{MODULE_FOLDER_PATH}/control_comedi {COMEDI_DIO_BITFIELD} {hex(mask)} {hex(data)}'

    if debug: print(cmd)

    proc = subprocess.Popen(split(cmd), 
                        stderr= subprocess.PIPE,
                        stdout= subprocess.PIPE,
                        stdin= subprocess.PIPE,
                        shell = False)
    retval = (proc.stdout.readline())
    proc.stdout.flush()
    return retval

def time_write_to_comedi():
    '''
    this function calculates how long one write takes, 
    but does not test how long the signal appears on the port. 
    return the time difference in s
    '''
    data = 255

    t1 = time.perf_counter()
    write_to_comedi(data, mask = 255, debug = True)
    t2 = time.perf_counter()

    return t2 - t1



if __name__ == "__main__":

    print(f'calling c process takes: {time_write_to_comedi()} s')

