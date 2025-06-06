"""
OptiTrack motion capture system interface for BMI3D
Provides data source for streaming rigid body and skeleton position data
"""

import numpy as np
import time
import threading
from collections import deque
from riglib.optitrack_client_update.NatNetClient import NatNetClient
import traceback


# Centralized IP configuration
DEFAULT_OPTITRACK_SERVER_IP = "128.95.215.191"
DEFAULT_OPTITRACK_CLIENT_IP = "128.95.215.213"

"""
Simplified OptiTrack system class that follows the same pattern as Quattrocento
Remove manual HDF5 handling and rely on sink manager
"""

class OptiTrackData(object):
    """
    Data source for OptiTrack motion capture system using NatNet SDK
    Streams rigid body and skeleton position data at specified frame rate
    """
    
    # Class variables for sink manager (like Quattrocento)
    subj = None
    saveid = None
    
    update_freq = 120.0  # Default update frequency - BMI3D expects this as class attribute
    dtype = np.dtype(np.float32) # bmi3d expects this to have an itemsize attribute
    
    def __init__(self, server_ip=DEFAULT_OPTITRACK_SERVER_IP, client_ip=DEFAULT_OPTITRACK_CLIENT_IP, 
                 use_multicast=False, update_freq=120.0, buffer_size=1000,
                 send_data_to_sink_manager=False, channels=None, **kwargs):
        """
        Initialize OptiTrack data source
        """
        self.server_ip = server_ip
        self.client_ip = client_ip
        self.use_multicast = use_multicast
        self.update_freq = update_freq
        self.send_data_to_sink_manager = send_data_to_sink_manager
        self.channels = channels or []
        
        # Data storage
        self.buffer_size = buffer_size
        self.data_buffer = deque(maxlen=buffer_size)
        self.timestamp_buffer = deque(maxlen=buffer_size)
        
        # Threading and control
        self.running = False
        self.data_thread = None
        self.data_lock = threading.Lock()
        
        # NatNet client
        self.streaming_client = None
        
        # Data tracking
        self.rigid_bodies = {}
        self.skeletons = {}
        self.frame_number = 0
        self.last_timestamp = 0
        
        # Channel mapping
        self.channel_map = {}
        self.total_channels = 0
        
        # REMOVED: All HDF5-related code - let sink manager handle it
        
    def _frame_callback(self, data_dict):
        """Callback function called for each new frame from OptiTrack"""
        try:
            with self.data_lock:
                self.frame_number = data_dict.get('frameNumber', 0)
                self.last_timestamp = data_dict.get('timestamp', time.time())
                
                # Extract data from mocap_data if available
                if 'mocap_data' in data_dict:
                    mocap_data = data_dict['mocap_data']
                    self._process_mocap_data(mocap_data)
                
                # Convert current state to channel data
                channel_data = self._convert_to_channels()
                
                if len(channel_data) > 0:
                    # Store data with timestamp
                    self.data_buffer.append(channel_data)
                    self.timestamp_buffer.append(self.last_timestamp)
                    
        except Exception as e:
            print(f"Error in OptiTrack frame callback: {e}")
            traceback.print_exc()
    
    def _process_mocap_data(self, mocap_data):
        """Process motion capture data and update internal state"""
        # Update rigid bodies
        if hasattr(mocap_data, 'rigid_body_data') and mocap_data.rigid_body_data:
            for rb in mocap_data.rigid_body_data.rigid_body_list:
                self.rigid_bodies[rb.id_num] = {
                    'position': [rb.pos[0], rb.pos[1], rb.pos[2]],
                    'rotation': [rb.rot[0], rb.rot[1], rb.rot[2], rb.rot[3]],
                    'tracking_valid': rb.tracking_valid
                }
        
        # Update skeletons
        if hasattr(mocap_data, 'skeleton_data') and mocap_data.skeleton_data:
            for skel in mocap_data.skeleton_data.skeleton_list:
                skeleton_rbs = {}
                for rb in skel.rigid_body_list:
                    skeleton_rbs[rb.id_num] = [rb.pos[0], rb.pos[1], rb.pos[2]]
                
                self.skeletons[skel.id_num] = {
                    'rigid_bodies': skeleton_rbs
                }
    
    def _convert_to_channels(self):
        """Convert current OptiTrack data to channel format for BMI3D"""
        channel_data = []
        
        # Add rigid body positions (x, y, z for each)
        for rb_id in sorted(self.rigid_bodies.keys()):
            rb_data = self.rigid_bodies[rb_id]
            if rb_data['tracking_valid']:
                channel_data.extend(rb_data['position'])
            else:
                channel_data.extend([0.0, 0.0, 0.0])  # Use zeros for invalid tracking
        
        # Add skeleton rigid body positions
        for skel_id in sorted(self.skeletons.keys()):
            skel_data = self.skeletons[skel_id]
            for rb_id in sorted(skel_data['rigid_bodies'].keys()):
                channel_data.extend(skel_data['rigid_bodies'][rb_id])
        
        return np.array(channel_data, dtype=np.float32)
    
    def _update_channel_count(self):
        """Update total channel count based on current tracked objects"""
        # Count channels: 3 per rigid body + 3 per skeleton rigid body
        rb_count = len(self.rigid_bodies)
        skel_rb_count = sum(len(skel['rigid_bodies']) for skel in self.skeletons.values())
        self.total_channels = (rb_count + skel_rb_count) * 3  # 3 channels (x,y,z) per object
        
        # Update channels list if not provided
        if not self.channels:
            self.channels = list(range(self.total_channels))
    def start(self):
        """Start the OptiTrack data streaming"""
        if self.running:
            return
        
        try:
            # REMOVED: HDF5 setup - sink manager handles this
            
            # Initialize NatNet client
            self.streaming_client = NatNetClient()
            self.streaming_client.set_client_address(self.client_ip)
            self.streaming_client.set_server_address(self.server_ip)
            self.streaming_client.set_use_multicast(self.use_multicast)
            
            # Set up callbacks
            self.streaming_client.new_frame_with_data_listener = self._frame_callback
            
            # Start streaming
            is_running = self.streaming_client.run('d')
            if not is_running:
                raise Exception("Could not start OptiTrack streaming client")
            
            # Wait for connection
            time.sleep(1)
            if not self.streaming_client.connected():
                raise Exception("Could not connect to OptiTrack server")
            
            # Request data descriptions
            self.streaming_client.send_request(
                self.streaming_client.command_socket, 
                self.streaming_client.NAT_REQUEST_MODELDEF, 
                "", 
                (self.streaming_client.server_ip_address, self.streaming_client.command_port)
            )
            
            self.running = True
            print(f"OptiTrack streaming started - Server: {self.server_ip}, Client: {self.client_ip}")
            
            # Give some time for data descriptions and initial frames
            time.sleep(2)
            self._update_channel_count()
            
        except Exception as e:
            print(f"Error starting OptiTrack streaming: {e}")
            traceback.print_exc()
            self.stop()
            raise
    
    def stop(self):
        """Stop the OptiTrack data streaming"""
        self.running = False
        
        if self.streaming_client:
            try:
                self.streaming_client.shutdown()
            except:
                pass
            self.streaming_client = None
        
        # REMOVED: Manual HDF5 writing - sink manager handles this
        print("OptiTrack streaming stopped")
    
    def get_new_data(self):
        """Get new data from the buffer - BMI3D interface method"""
        with self.data_lock:
            if len(self.data_buffer) == 0:
                return np.array([]), np.array([])
            
            # Get all available data
            data_list = list(self.data_buffer)
            timestamp_list = list(self.timestamp_buffer)
            
            # Clear buffers
            self.data_buffer.clear()
            self.timestamp_buffer.clear()
            
            if len(data_list) == 0:
                return np.array([]), np.array([])
            
            # Stack data
            data_array = np.vstack(data_list) if len(data_list) > 1 else data_list[0].reshape(1, -1)
            timestamp_array = np.array(timestamp_list)
            
            return data_array, timestamp_array
    
    def get_data(self):
        """Alternative interface for getting data"""
        return self.get_new_data()
    
#    @property
#    def dtype(self):
#        """Data type for compatibility with BMI3D"""
#        return np.float32
    
    def __str__(self):
        return f"OptiTrackData"  # Simplified, matches Quattrocento pattern