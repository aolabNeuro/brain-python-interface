"""
Code for reward system used in Amy Orsborn lab
"""

# import functions
import pyfirmata
import time

class Basic(object):
    com_port = '/dev/ttyACM0'  # specify the port, based on windows/Unix, can find it on IDE or terminal
    board = pyfirmata.Arduino(com_port)

    def __init__(self):
        com_port = '/dev/ttyACM0'  # specify the port, based on windows/Unix, can find it on IDE or terminal
        board = pyfirmata.Arduino(com_port)

    def reward(self, rewardtime):
        board.digital[13].write(1) #send a high signal to Pin 13 on the arduino which should be connected to the reward system
        time.sleep(rewardtime)  # in second
        print('ON')
        board.digital[13].write(0)
        print('OFF')

    def calibrate(self):
        board.digital[13].write(1)
        time.sleep(72)  # it takes around 72 seconds to drain 200 ml of fluid - Flow rate: 2.8 mL/s
        board.digital[13].write(0)
        print('Check the breaker for calibration. You should notice 200 ml of fluid')

    def drain(self, drain_time = 1200): #call this function to drain the reward system
        """
        this function is called from the webserver in ajax.reward_drain
        """
       #if cmd == 'ON':
        board.digital[13].write(1)
        time.sleep(drain_time)
        #   cmd = 'OFF'
        #if cmd == 'OFF':
        board.digital[13].write(0)

def open():
    try:
        reward = Basic()
        return reward
    except:
        print("Reward system not found/ not active")
        import traceback
        import os
        import builtins
        traceback.print_exc(file=builtins.open(os.path.expanduser('~/code/bmi3d/log/reward.log'), 'w'))