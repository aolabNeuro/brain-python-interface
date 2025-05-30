'''
Updated optitrack client code compatible with new NatNet SDK
'''
import time
import numpy as np
import threading
import copy
from ..source import DataSourceSystem

# Try to import the new NatNet SDK
try:
    from NatNetClient import NatNetClient
    import MoCapData
    SDK_AVAILABLE = True
except ImportError:
    print("Warning: New NatNet SDK not available, using simulation mode")
    SDK_AVAILABLE = False


class System(DataSourceSystem):
    '''
    Updated Optitrack DataSourceSystem that works with the new NatNet SDK
    Maintains compatibility with existing BMI3D architecture
    '''
    update_freq = 240  # Hz - OptiTrack typically runs at 120-240 Hz
    
    def __init__(self, client, feature="rigid body", n_features=1):
        '''
        Initialize with the new NatNet client
        '''
        self.client = client
        self.feature = feature  # rigid body, skeleton, marker
        self.n_features = n_features
        
        # Data storage - thread-safe access
        self.data_lock = threading.Lock()
        self.rigid_bodies = []
        self.skeletons = []
        self.markers = []
        self.timing = []
        self.frame_number = 0
        
        # Set up the client callbacks if it's a real NatNet client
        if hasattr(self.client, 'rigid_body_listener'):
            self.client.rigid_body_listener = self._rigid_body_callback
            self.client.new_frame_listener = self._new_frame_callback
    
    def start(self):
        '''
        Start the NatNet client connection
        '''
        if hasattr(self.client, 'run'):
            # For the new NatNet client, we need to start it in a separate thread
            # since run() is blocking
            #if not hasattr(self.client, '_client_thread') or not self.client._client_thread.is_alive():
            #    self.client_thread = threading.Thread(target=self.client.run, daemon=True)
            if not hasattr(self, 'client_thread') or not self.client_thread.isalive():
                self.client_thread = threading.Thread(target=self.client.run, daemon=True)
                self.client_thread.start()
                
                # Give it a moment to connect
                time.sleep(0.5)
        elif hasattr(self.client, 'set_callback'):
            # For compatibility with old-style clients (simulation, etc.)
            self.client.set_callback(
                lambda rb, s, m, t: self._update(rb, s, m, t))

    def stop(self):
        '''
        Stop the NatNet client
        '''
        if hasattr(self.client, 'stop_threads'):
            self.client.stop_threads = True
    
    def get(self):
        '''
        Main logic -- parse the motion tracking data into a defined datatype
        Returns data in the format expected by BMI3D
        '''
        
        # For old-style clients, run once to get data
        if hasattr(self.client, 'run_once'):
            self.client.run_once(timeout=0.1)
        
        # Extract coordinates from feature
        coords = np.empty((self.n_features, 3))
        coords[:] = np.nan
        
        with self.data_lock:
            if self.feature == "rigid body":
                for i in range(min(self.n_features, len(self.rigid_bodies))):
                    if hasattr(self.rigid_bodies[i], 'tracking_valid') and self.rigid_bodies[i].tracking_valid:
                        # Convert from meters to centimeters for BMI3D compatibility
                        position = np.array(self.rigid_bodies[i].position)
                        if hasattr(self.rigid_bodies[i], '_already_converted_to_cm'):
                            coords[i] = position  # Already in cm
                        else:
                            coords[i] = position * 100  # Convert m to cm
            elif self.feature == "marker":
                for i in range(min(self.n_features, len(self.markers))):
                    position = np.array(self.markers[i].position)
                    coords[i] = position * 100  # Convert m to cm
            elif self.feature == "skeleton":
                raise NotImplementedError("Skeleton feature not yet implemented for new SDK")
            else:
                raise AttributeError(f"Feature type '{self.feature}' unknown!")

        # For HDFWriter we need a dim 0
        coords = np.expand_dims(coords, axis=0)
        return coords
    
    def _rigid_body_callback(self, id, position, rotation):
        '''
        Callback for new NatNet SDK rigid body data
        '''
        with self.data_lock:
            # Create a RigidBody object compatible with the old interface
            rb = RigidBody(position, rotation, id)
            rb.tracking_valid = True
            rb._already_converted_to_cm = False  # Mark that this needs conversion
            
            # Update or add the rigid body
            found = False
            for i, existing_rb in enumerate(self.rigid_bodies):
                if hasattr(existing_rb, 'id') and existing_rb.id == id:
                    self.rigid_bodies[i] = rb
                    found = True
                    break
            
            if not found:
                self.rigid_bodies.append(rb)
    
    def _new_frame_callback(self, frame_number):
        '''
        Callback for new frame from NatNet SDK
        '''
        with self.data_lock:
            self.frame_number = frame_number
    
    def _update(self, rigid_bodies, skeletons, markers, timing):
        '''
        Callback for old-style natnet client (for backwards compatibility)
        '''
        with self.data_lock:
            self.rigid_bodies = rigid_bodies
            self.skeletons = skeletons
            self.markers = markers
            self.timing = timing


#################
# Data structures compatible with both old and new systems
#################
class RigidBody():
    '''
    Enhanced RigidBody class that works with both old and new SDK
    '''
    
    def __init__(self, position, rotation=None, id=None):
        self.position = np.array(position) if position is not None else np.array([0, 0, 0])
        self.rotation = np.array(rotation) if rotation is not None else np.array([0, 0, 0, 1])
        self.id = id
        self.tracking_valid = True
        self.error = 0.0


class Marker():
    '''
    Marker class for individual marker data
    '''
    
    def __init__(self, position, id=None):
        self.position = np.array(position) if position is not None else np.array([0, 0, 0])
        self.id = id


#################
# Updated NatNet Client Wrapper
#################
class NatNetClientWrapper():
    '''
    Wrapper for the new NatNet SDK that provides the interface expected by the System class
    '''
    
    def __init__(self, server_ip="127.0.0.1", local_ip="127.0.0.1"):
        self.server_ip = server_ip
        self.local_ip = local_ip
        self.client = None
        self.connected = False
        
        if SDK_AVAILABLE:
            self.client = NatNetClient()
            self.client.set_server_address(server_ip)
            self.client.set_client_address(local_ip)
        else:
            raise ImportError("NatNet SDK not available")
    
    def run(self):
        '''
        Start the NatNet client - this is blocking so should be run in a thread
        '''
        if self.client:
            self.client.run()
            self.connected = self.client.connected()
    
    def start_recording(self):
        '''
        Start recording in Motive
        '''
        if self.client and self.connected:
            result = self.client.send_command("StartRecording")
            return result >= 0
        return False
    
    def stop_recording(self):
        '''
        Stop recording in Motive
        '''
        if self.client and self.connected:
            result = self.client.send_command("StopRecording")
            return result >= 0
        return False
    
    def send_command(self, command):
        '''
        Send a command to Motive
        '''
        if self.client and self.connected:
            return self.client.send_command(command)
        return -1
    
    def set_take(self, take_name):
        '''
        Set the take name in Motive
        '''
        return self.send_command(f"SetCurrentTake,{take_name}")
    
    def set_session(self, session_name):
        '''
        Set the session path in Motive
        '''
        return self.send_command(f"SetCurrentSession,{session_name}")
    
    @property
    def rigid_body_listener(self):
        return getattr(self.client, 'rigid_body_listener', None)
    
    @rigid_body_listener.setter
    def rigid_body_listener(self, callback):
        if self.client:
            self.client.rigid_body_listener = callback
    
    @property
    def new_frame_listener(self):
        return getattr(self.client, 'new_frame_listener', None)
    
    @new_frame_listener.setter
    def new_frame_listener(self, callback):
        if self.client:
            self.client.new_frame_listener = callback
    
    @property
    def stop_threads(self):
        return getattr(self.client, 'stop_threads', False)
    
    @stop_threads.setter
    def stop_threads(self, value):
        if self.client:
            self.client.stop_threads = value


#################
# Simulated data (unchanged for compatibility)
#################
class SimulatedClient():
    '''
    Simulated client for testing without OptiTrack hardware
    '''

    def __init__(self, n=1, radius=(0.2, 0.04, 0.1), speed=(0.5, 1, 2)):
        self.stime = time.time()
        self.n = n
        self.radius = radius
        self.speed = speed

    def set_callback(self, callback):
        self.callback = callback

    def run_once(self, timeout=None):
        '''
        Generate fake motion data
        '''
        time.sleep(1./240)
        ts = (time.time() - self.stime)
        coords = np.multiply(self.radius, np.cos(np.divide(ts, self.speed) * 2 * np.pi))
        data = [RigidBody(coords)]
        data[0]._already_converted_to_cm = True  # Mark as already in cm units
        self.callback(data, [], [], [])

    def start_recording(self):
        print("Simulation: Start recording")
        return True

    def stop_recording(self):
        print("Simulation: Stop recording")
        return True

    def set_take(self, take_name):
        print(f"Simulation: Setting take_name: {take_name}")
        return 0

    def set_session(self, session_name):
        print(f"Simulation: Setting session_name: {session_name}")
        return 0


########################
# Playback from csv file (updated)
########################

class PlaybackClient(SimulatedClient):
    '''
    Client for playing back recorded CSV data
    '''

    def __init__(self, filename):
        import pandas as pd
        self.stime = time.time()
        self.current_index = 0
        
        try:
            csv = pd.read_csv(filename, header=[1, 4, 5])
            self.motiondata = csv['Rigid Body']['Position']
            self.time = csv['Type'].iloc[:, 0]
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            # Fallback to simulation
            super().__init__()
            self.motiondata = None

    def run_once(self, timeout=None):
        '''
        Read one line of motion data from the csv file
        '''
        read_freq = 240
        time.sleep(1./read_freq)
        
        coords = np.empty((3,))
        coords[:] = np.nan
        
        if self.motiondata is not None:
            ts = (time.time() - self.stime)
            
            # Find the appropriate row based on timestamp
            try:
                while (self.current_index < len(self.time) and 
                       self.time.iloc[self.current_index] < ts):
                    self.current_index += 1
                
                if self.current_index < len(self.motiondata):
                    row = self.motiondata.iloc[self.current_index]
                    coords[0] = row.X
                    coords[1] = row.Y
                    coords[2] = row.Z
            except Exception:
                pass
        
        data = [RigidBody(coords)]
        data[0]._already_converted_to_cm = True  # CSV data assumed to be in cm
        self.callback(data, [], [], [])


# System definition function (updated)
def make(cls, client, feature, num_features=1, **kwargs):
    """
    Dynamically creates a System class with the specified parameters
    """
    def init(self):
        super(self.__class__, self).__init__(client, feature, num_features, **kwargs)
    
    dtype = np.dtype((float, (num_features, 3)))
    return type(cls.__name__, (cls,), dict(dtype=dtype, __init__=init))


# Factory function for creating clients
def create_client(client_type="natnet", **kwargs):
    '''
    Factory function to create the appropriate client type
    '''
    if client_type == "natnet" and SDK_AVAILABLE:
        server_ip = kwargs.get('server_ip', '127.0.0.1')
        local_ip = kwargs.get('local_ip', '127.0.0.1')
        return NatNetClientWrapper(server_ip, local_ip)
    elif client_type == "simulation":
        return SimulatedClient(**kwargs)
    elif client_type == "playback":
        filename = kwargs.get('filename')
        if filename:
            return PlaybackClient(filename)
        else:
            raise ValueError("Playback client requires 'filename' parameter")
    else:
        # Fallback to simulation if SDK not available
        print("Falling back to simulation client")
        return SimulatedClient(**kwargs)