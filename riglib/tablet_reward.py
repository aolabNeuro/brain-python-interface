from .gpio import ArduinoGPIO
from multiprocessing import Process
from riglib import singleton
import traceback
import time
import os
import requests
import threading

log_path = os.path.join(os.path.dirname(__file__), '../log/reward.log')

class Basic(singleton.Singleton):

    __instance = None

    def __init__(self):
        super().__init__()
        self.board = ArduinoGPIO() # let the computer find the arduino. this won't work with more than one arduino!
        self.board.write(12,1)

    def dispense(self):  # call this function to drain the reward system
        """
        this function is called from the webserver in ajax.reward_drain
        """
        self.board.write(12, 0)          #low
        time.sleep(0.02)
        self.board.write(12, 1)            #high

def open():
    try:
        reward = Basic.get_instance()
        return reward
    except:
        print("Reward system not found/ not active")
        import traceback
        import os
        import builtins
        traceback.print_exc()

def send_request(url, n_trigger):
    for i in range(n_trigger):
        try:
            response = requests.post(url, timeout=3)
            print(f"Request to {url} completed: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error sending request to {url}: {e}") # error occurs even when pellet dispenses
        time.sleep(0.5)

def send_nonblocking_request(url, n_trigger):
    thread = threading.Thread(target=send_request, args=(url, n_trigger))
    thread.daemon = True
    thread.start()
    print("Request initiated")

class RemoteReward():

    def __init__(self):

        self.hostName = "192.168.0.150"
        self.serverPort = 8080
      
    def trigger(self, ip_address, n_trigger): # set some default so the manual reward button works
        url = f"http://{ip_address}:{self.serverPort}"
        send_nonblocking_request(url, n_trigger)