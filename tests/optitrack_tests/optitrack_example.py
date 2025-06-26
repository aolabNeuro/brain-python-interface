# test file for new optitrack integration

import time
from riglib import source
from riglib.optitrack_client_update.PythonSample import OptitrackStreamingClient
from riglib.bmi import state_space_models, train, extractor
import numpy as np
import unittest

STREAMING_DURATION = 3

class TestOptiTrackStreaming(unittest.TestCase):

    #@unittest.
    def test_direct(self):
        opt = OptitrackStreamingClient()
        opt.setup()
        print(opt.streaming_client.connected())
                # check sockets
        print(opt.streaming_client.command_socket)
'''            ret_value = False
        elif self.data_socket ==None:
            ret_value = False
        # check versions
        elif self.get_application_name() == "Not Set":
            ret_value = False
        elif (self.__server_version[0] == 0) and\
            (self.__server_version[1] == 0) and\
            (self.__server_version[2] == 0) and\
            (self.__server_version[3] == 0):
            ret_value = False
        return ret_value'''
        #opt.start(opt.optionsDict)
        #time.sleep(STREAMING_DURATION)
        #opt.stop()

if __name__ == '__main__':
    unittest.main()