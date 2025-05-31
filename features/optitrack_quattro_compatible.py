'''
OptiTrack data source that mimics Quattrocento streaming format for BMI3D decoder compatibility
Updated to work with optitrack_client_updated.py
'''
import sys
sys.path.append('/home/aolab/NatNet_SDK_4.2_ubuntu/samples/PythonClient')
import numpy as np
import threading
import time
from riglib.source import DataSourceSystem
#from ..source import DataSourceSystem

try:
    from NatNetClient import NatNetClient
    SDK_AVAILABLE = True
except ImportError:
    print("Warning: NatNet SDK not available, using simulation")
    SDK_AVAILABLE = False


class OptiTrackQuattroStream(DataSourceSystem):
    """
    OptiTrack data source formatted like Quattrocento for decoder compatibility
    Streams rigid body positions as multi-channel data array
    """
    update_freq = 240  # Hz
    
    def __init__(self, client, n_rigid_bodies=1, channels_per_body=3):
        """
        Initialize OptiTrack streaming source
        
        Parameters:
        -----------
        client : NatNet client or simulation client (from optitrack_client_updated.py)
        n_rigid_bodies : int, number of rigid bodies to track
        channels_per_body : int, channels per rigid body (3 for x,y,z position)
        """
        self.client = client
        self.n_rigid_bodies = n_rigid_bodies
        self.channels_per_body = channels_per_body
        self.n_channels = n_rigid_bodies * channels_per_body
        
        # Thread-safe data storage
        self.data_lock = threading.Lock()
        self.rigid_bodies = []
        self.latest_data = np.zeros((1, self.n_channels))  # Shape: (1, n_channels) like Quattrocento
        
        # Set up callbacks based on client type
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Setup appropriate callbacks based on client type"""
        if hasattr(self.client, 'rigid_body_listener'):
            # New NatNetClientWrapper from optitrack_client_updated.py
            self.client.rigid_body_listener = self._rigid_body_callback
        elif hasattr(self.client, 'set_callback'):
            # Legacy clients (SimulatedClient, PlaybackClient)
            self.client.set_callback(self._legacy_callback)
    
    def start(self):
        """Start the client connection"""
        if hasattr(self.client, 'run'):
            # New SDK - run in separate thread
            self.client_thread = threading.Thread(target=self.client.run, daemon=True)
            self.client_thread.start()
            time.sleep(0.5)
    
    def stop(self):
        """Stop the client connection"""
        if hasattr(self.client, 'stop_threads'):
            self.client.stop_threads = True
    
    def get(self):
        """
        Get current data in Quattrocento format
        Returns: np.array of shape (1, n_channels) - compatible with decoder input
        """
        # For legacy clients, trigger data update
        if hasattr(self.client, 'run_once'):
            self.client.run_once(timeout=0.01)
        
        with self.data_lock:
            return self.latest_data.copy()
    
    def _rigid_body_callback(self, id, position, rotation):
        """Callback for new NatNet SDK (NatNetClientWrapper)"""
        with self.data_lock:
            # Convert position from meters to centimeters
            pos_cm = np.array(position) * 100
            
            # Update rigid body data (organize by ID)
            if id < self.n_rigid_bodies:
                start_idx = id * self.channels_per_body
                end_idx = start_idx + self.channels_per_body
                self.latest_data[0, start_idx:end_idx] = pos_cm
    
    def _legacy_callback(self, rigid_bodies, skeletons, markers, timing):
        """Callback for legacy clients (SimulatedClient, PlaybackClient)"""
        with self.data_lock:
            for i, rb in enumerate(rigid_bodies[:self.n_rigid_bodies]):
                if hasattr(rb, 'position') and rb.position is not None:
                    # Handle position data - check if already converted to cm
                    pos = np.array(rb.position)
                    if not getattr(rb, '_already_converted_to_cm', False):
                        pos *= 100  # Convert m to cm
                    
                    start_idx = i * self.channels_per_body
                    end_idx = start_idx + self.channels_per_body
                    self.latest_data[0, start_idx:end_idx] = pos


class SimulatedOptiTrackClient:
    """Simplified simulation client - kept for backward compatibility"""
    
    def __init__(self, n_bodies=1):
        self.n_bodies = n_bodies
        self.start_time = time.time()
        self.callback = None
    
    def set_callback(self, callback):
        self.callback = callback
    
    def run_once(self, timeout=None):
        time.sleep(1/240)  # Simulate 240Hz
        
        # Generate circular motion for each rigid body
        t = time.time() - self.start_time
        rigid_bodies = []
        
        for i in range(self.n_bodies):
            # Each body gets slightly different motion
            phase = i * np.pi / 2
            radius = 10 + i * 5  # cm
            
            pos = np.array([
                radius * np.cos(t + phase),
                5 * np.sin(2 * t + phase),
                radius * np.sin(t + phase)
            ])
            
            rb = type('RigidBody', (), {
                'position': pos,
                '_already_converted_to_cm': True
            })()
            rigid_bodies.append(rb)
        
        if self.callback:
            self.callback(rigid_bodies, [], [], [])


def create_optitrack_quattro_source(n_rigid_bodies=1, client_type="natnet", **client_kwargs):
    """
    Factory function to create OptiTrack source compatible with Quattrocento decoders
    Uses the updated optitrack client system
    
    Parameters:
    -----------
    n_rigid_bodies : int, number of rigid bodies to track  
    client_type : str, type of client ("natnet", "simulation", "playback")
    **client_kwargs : additional arguments for client creation
    
    Returns:
    --------
    OptiTrackQuattroStream instance ready for decoder input
    """
    
    # Import the client creation function from optitrack_client_updated
    try:
        from riglib.optitrack_client_update_natnet.optitrack_client_updated import create_client
        client = create_client(client_type, **client_kwargs)
        print(f"Using {client_type} client with {n_rigid_bodies} rigid bodies")
    except ImportError as e:
        # Fallback to local simulation if import fails
        print(f"Error importing client: {e}. Using simulated client instead.")
        client = SimulatedOptiTrackClient(n_rigid_bodies)
        
        print(f"Using fallback simulation with {n_rigid_bodies} rigid bodies")
    
    return OptiTrackQuattroStream(client, n_rigid_bodies)


# Example usage:
if __name__ == "__main__":
    # Example 1: Using real OptiTrack hardware
    print("Testing with real NatNet client:")
    try:
        optitrack_source = create_optitrack_quattro_source(
            n_rigid_bodies=2, 
            client_type="natnet",
            server_ip="192.168.1.100",  # Your OptiTrack server IP
            local_ip="192.168.1.101"    # Your local IP
        )
    except:
        print("Real client failed, using simulation")
        optitrack_source = create_optitrack_quattro_source(
            n_rigid_bodies=2, 
            client_type="simulation"
        )
    
    optitrack_source.start()
    
    try:
        print("\nStreaming data (Quattrocento format):")
        for i in range(10):
            data = optitrack_source.get()
            print(f"Frame {i}: Shape {data.shape}, Data: {data}")
            time.sleep(0.1)
    finally:
        optitrack_source.stop()
    
    # Example 2: Using playback from CSV
    print("\nTesting with playback client:")
    try:
        playback_source = create_optitrack_quattro_source(
            n_rigid_bodies=1,
            client_type="playback",
            filename="your_recorded_data.csv"
        )
        
        playback_source.start()
        
        for i in range(5):
            data = playback_source.get()
            print(f"Playback Frame {i}: {data}")
            time.sleep(0.1)
            
        playback_source.stop()
        
    except Exception as e:
        print(f"Playback test failed: {e}")
