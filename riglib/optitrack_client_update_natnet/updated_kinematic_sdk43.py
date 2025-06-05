'''
Updated Kinematic System for Optitrack motion capture data with NatNet SDK 4.3 compatibility
'''
import time
import numpy as np
from datetime import datetime
from ..source import DataSourceSystem
from ..source import MultiChanDataSource
import os

class KinematicSystem(DataSourceSystem):
    '''
    Optitrack Kinematic DataSourceSystem - formats motion capture data like EMG data
    for compatibility with BMI3D decoder infrastructure using NatNet SDK 4.3
    '''
    update_freq = 240  # Hz
    
    # Class attributes (similar to quattrocento.EMG)
    subj = None
    saveid = None
    n_features = 1
    feature_type = "rigid body"
    server_address = "127.0.0.1"
    client_address = "127.0.0.1"
    use_multicast = True
    
    def __init__(self, **kwargs):
        '''
        Initialize the kinematic system with NatNet SDK 4.3 client
        '''
        super().__init__()
        
        # Update settings from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Initialize client connection
        self._init_client()
        
        # Calculate data dimensions
        self.n_kinematic_channels = self.n_features * 9  # x,y,z, vx,vy,vz, ax,ay,az per feature
        
        # Data buffers for derivative calculations
        self.position_history = []
        self.velocity_history = []
        self.max_history = 3  # Keep last 3 samples for derivative calculation
        
        # Last valid data for NaN handling
        self.last_valid_data = np.zeros(self.n_kinematic_channels, dtype=np.float64)
        
        # Set up data structure
        self.dtype = np.dtype([
            ('data', np.float64, (self.n_kinematic_channels,)),
            ('timestamp', np.float64)
        ])
        
        # Initialize data storage
        self.current_data = np.zeros(self.n_kinematic_channels, dtype=np.float64)
        self.frame_data = None
        self.data_lock = False
        self.data_ready = False
        
    def _init_client(self):
        '''Initialize the NatNet SDK 4.3 client connection'''
        try:
            # Import the new NatNet SDK 4.3 client
            from NatNetClient import NatNetClient
            
            # Create NatNet client instance
            self.streaming_client = NatNetClient()
            
            # Configure client addresses
            self.streaming_client.set_client_address(self.client_address)
            self.streaming_client.set_server_address(self.server_address)
            self.streaming_client.set_use_multicast(self.use_multicast)
            
            # Set up callbacks for data reception
            self.streaming_client.new_frame_listener = self._frame_callback
            self.streaming_client.rigid_body_listener = self._rigid_body_callback
            
            # Optionally set up frame with data listener for skeleton data
            if hasattr(self.streaming_client, 'new_frame_with_data_listener'):
                self.streaming_client.new_frame_with_data_listener = self._frame_with_data_callback
            
            # Start the streaming client
            stream_type = 'd'  # datastream
            is_running = self.streaming_client.run(stream_type)
            
            if not is_running:
                raise Exception("Could not start NatNet streaming client")
            
            # Wait a moment for connection
            time.sleep(1)
            
            if not self.streaming_client.connected():
                raise Exception("Could not connect to Motive. Check that Motive streaming is enabled.")
            
            # Set up recording if saveid is provided
            if self.saveid is not None:
                now = datetime.now()
                take_name = f"Take {now.strftime('%Y-%m-%d %H:%M:%S')} ({self.saveid})"
                
                # Send commands to start recording
                self.streaming_client.send_command("TimelineStop")
                time.sleep(0.1)
                # Note: Recording control may require additional Motive setup
                
            self.optitrack_status = 'streaming'
            print("NatNet SDK 4.3 client connected successfully")
            
        except ImportError as e:
            print(f"NatNet SDK 4.3 import error: {e}")
            print("Make sure NatNetClient.py is in your Python path")
            self.optitrack_status = 'NatNet SDK 4.3 not found'
            # Fall back to simulated client
            self._init_simulated_client()
            
        except Exception as e:
            print(f"Optitrack connection error: {e}")
            self.optitrack_status = f'Connection failed: {str(e)}'
            # Fall back to simulated client
            self._init_simulated_client()
            
    def _init_simulated_client(self):
        '''Initialize simulated client for testing'''
        try:
            from simulated_client import SimulatedClient
            self.streaming_client = SimulatedClient()
            self.optitrack_status = 'simulated'
            print("Using simulated Optitrack client")
        except ImportError:
            # Create a minimal mock client
            self.streaming_client = type('MockClient', (), {
                'connected': lambda: False,
                'run': lambda x: False,
                'shutdown': lambda: None
            })()
            self.optitrack_status = 'disconnected'
            
    def start(self):
        '''Start data collection - callbacks are already set up'''
        self.frame_data = None
        self.data_lock = False
        self.data_ready = True
        
        # If using simulated client, start the simulation loop
        if hasattr(self.streaming_client, 'simulate_frame'):
            import threading
            self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
            self.simulation_thread.start()
        
    def stop(self):
        '''Clean up callbacks and stop streaming'''
        self.data_ready = False
        
        if hasattr(self.streaming_client, 'shutdown'):
            self.streaming_client.shutdown()
            
    def _simulation_loop(self):
        '''Run simulation loop for simulated client'''
        while self.data_ready:
            if hasattr(self.streaming_client, 'simulate_frame'):
                self.streaming_client.simulate_frame()
            time.sleep(1.0 / self.update_freq)
    
    def get(self):
        '''
        Get kinematic data formatted as multi-channel array (like EMG)
        Returns data in format compatible with BMI decoders
        '''
        # Wait for new frame data with timeout
        timeout_counter = 0
        max_timeout = int(self.update_freq * 0.1)  # 0.1 second timeout
        
        while self.frame_data is None and timeout_counter < max_timeout:
            time.sleep(1/self.update_freq)
            timeout_counter += 1
            
        if self.frame_data is None:
            # Return last known data or zeros
            timestamp = time.time()
            result = np.array([(self.last_valid_data, timestamp)], dtype=self.dtype)
            return result
        
        # Extract position data
        positions = self._extract_positions()
        if positions is None:
            timestamp = time.time()
            result = np.array([(self.last_valid_data, timestamp)], dtype=self.dtype)
            return result
            
        # Calculate derivatives (velocity, acceleration)
        kinematic_data = self._calculate_kinematics(positions)
        
        # Handle NaN values
        kinematic_data = self._handle_nan_data(kinematic_data)
        
        # Format as structured array with timestamp
        timestamp = time.time()
        result = np.array([(kinematic_data, timestamp)], dtype=self.dtype)
        
        # Clear frame data to wait for next frame
        self.frame_data = None
        
        return result
    
    def get_data_len(self):
        '''Return the number of channels available'''
        return self.n_kinematic_channels
    
    def get_channels(self):
        '''Return list of available channel numbers'''
        return list(range(1, self.n_kinematic_channels + 1))
    
    def is_available(self):
        '''Check if the system is ready to provide data'''
        return self.optitrack_status in ['recording', 'streaming', 'simulated']
    
    def cleanup_data_source(self):
        '''Clean shutdown of the data source'''
        self.stop()
    
    def _handle_nan_data(self, kinematic_data):
        '''Replace NaN values with last valid values or zeros'''
        if np.isnan(kinematic_data).any():
            nan_mask = np.isnan(kinematic_data)
            
            # Use last valid values where available
            kinematic_data[nan_mask] = self.last_valid_data[nan_mask]
            
            # If still NaN (no previous valid data), use zeros
            remaining_nan = np.isnan(kinematic_data)
            kinematic_data[remaining_nan] = 0.0
        
        # Update last valid data
        valid_mask = ~np.isnan(kinematic_data)
        self.last_valid_data[valid_mask] = kinematic_data[valid_mask]
        
        return kinematic_data
    
    def _extract_positions(self):
        '''Extract position data from frame data'''
        positions = np.full((self.n_features, 3), np.nan)
        
        if not self.data_lock and self.frame_data is not None:
            self.data_lock = True
            try:
                if self.feature_type == "rigid body":
                    # Handle both direct MoCap data and data_dict formats
                    if hasattr(self.frame_data, 'rigid_body_data'):
                        # Direct MoCap data format
                        rigid_body_data = self.frame_data.rigid_body_data
                    elif isinstance(self.frame_data, dict) and 'mocap_data' in self.frame_data:
                        # Data dict format
                        rigid_body_data = self.frame_data['mocap_data'].rigid_body_data
                    else:
                        # Try to access as dict
                        rigid_body_data = getattr(self.frame_data, 'rigid_body_data', None)
                    
                    if rigid_body_data and hasattr(rigid_body_data, 'rigid_body_list'):
                        rb_list = rigid_body_data.rigid_body_list
                        for i in range(min(self.n_features, len(rb_list))):
                            rb = rb_list[i]
                            if hasattr(rb, 'tracking_valid') and rb.tracking_valid:
                                positions[i] = [rb.pos_x, rb.pos_y, rb.pos_z]
                                
                elif self.feature_type == "skeleton":
                    skeleton_data = None
                    if hasattr(self.frame_data, 'skeleton_data'):
                        skeleton_data = self.frame_data.skeleton_data
                    elif isinstance(self.frame_data, dict) and 'mocap_data' in self.frame_data:
                        skeleton_data = self.frame_data['mocap_data'].skeleton_data
                        
                    if skeleton_data and hasattr(skeleton_data, 'skeleton_list'):
                        skeleton_list = skeleton_data.skeleton_list
                        bone_count = 0
                        for skeleton in skeleton_list:
                            if hasattr(skeleton, 'rigid_body_list'):
                                for bone in skeleton.rigid_body_list:
                                    if bone_count < self.n_features and hasattr(bone, 'tracking_valid') and bone.tracking_valid:
                                        positions[bone_count] = [bone.pos_x, bone.pos_y, bone.pos_z]
                                        bone_count += 1
                                    if bone_count >= self.n_features:
                                        break
                            if bone_count >= self.n_features:
                                break
                                
                elif self.feature_type == "marker":
                    marker_data = None
                    if hasattr(self.frame_data, 'labeled_marker_data'):
                        marker_data = self.frame_data.labeled_marker_data
                    elif isinstance(self.frame_data, dict) and 'mocap_data' in self.frame_data:
                        marker_data = self.frame_data['mocap_data'].labeled_marker_data
                        
                    if marker_data and hasattr(marker_data, 'labeled_marker_list'):
                        marker_list = marker_data.labeled_marker_list
                        for i in range(min(self.n_features, len(marker_list))):
                            marker = marker_list[i]
                            positions[i] = [marker.pos_x, marker.pos_y, marker.pos_z]
                            
            except Exception as e:
                print(f"Error extracting positions: {e}")
            finally:
                self.data_lock = False
                
        return positions
    
    def _calculate_kinematics(self, positions):
        '''Calculate position, velocity, and acceleration for each feature'''
        # Store current positions
        self.position_history.append(positions)
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
            
        # Calculate velocity (first derivative)
        if len(self.position_history) >= 2:
            dt = 1.0 / self.update_freq
            velocity = (self.position_history[-1] - self.position_history[-2]) / dt
        else:
            velocity = np.zeros_like(positions)
            
        # Store velocity history
        self.velocity_history.append(velocity)
        if len(self.velocity_history) > self.max_history:
            self.velocity_history.pop(0)
            
        # Calculate acceleration (second derivative)
        if len(self.velocity_history) >= 2:
            dt = 1.0 / self.update_freq
            acceleration = (self.velocity_history[-1] - self.velocity_history[-2]) / dt
        else:
            acceleration = np.zeros_like(velocity)
            
        # Combine all kinematic data into single array
        # Format: [x1, y1, z1, vx1, vy1, vz1, ax1, ay1, az1, x2, y2, z2, ...]
        kinematic_data = np.zeros(self.n_kinematic_channels)
        
        for i in range(self.n_features):
            base_idx = i * 9
            kinematic_data[base_idx:base_idx+3] = positions[i]      # position
            kinematic_data[base_idx+3:base_idx+6] = velocity[i]     # velocity
            kinematic_data[base_idx+6:base_idx+9] = acceleration[i] # acceleration
            
        return kinematic_data
    
    def _frame_callback(self, data_dict):
        '''Basic frame callback for NatNet SDK 4.3'''
        if not self.data_lock:
            self.frame_data = data_dict
            
    def _frame_with_data_callback(self, data_dict):
        '''Frame callback with full mocap data for skeleton tracking'''
        if not self.data_lock:
            self.frame_data = data_dict
                
    def _rigid_body_callback(self, new_id, position, rotation):
        '''Individual rigid body callback'''
        # This gets called for each rigid body per frame
        # Could be used for more specific rigid body handling
        pass
    
    def __str__(self):
        '''String representation for file naming'''
        return f"OptitrackKinematic_{self.feature_type}_{self.n_features}features"


# Simulated client remains the same for testing without hardware
class SimulatedRigidBody:
    """Simulated rigid body data"""
    def __init__(self, id_num=0):
        self.id = id_num
        self.tracking_valid = True
        self.pos_x = np.random.uniform(-1, 1)
        self.pos_y = np.random.uniform(-1, 1) 
        self.pos_z = np.random.uniform(-1, 1)
        self.quat_x = 0.0
        self.quat_y = 0.0
        self.quat_z = 0.0
        self.quat_w = 1.0
        
    def update(self):
        """Update position with small random movements"""
        self.pos_x += np.random.uniform(-0.01, 0.01)
        self.pos_y += np.random.uniform(-0.01, 0.01)
        self.pos_z += np.random.uniform(-0.01, 0.01)

class SimulatedRigidBodyData:
    """Container for rigid body data"""
    def __init__(self, n_bodies=2):
        self.rigid_body_list = [SimulatedRigidBody(i) for i in range(n_bodies)]
        
    def update(self):
        for rb in self.rigid_body_list:
            rb.update()

class SimulatedFrameData:
    """Simulated frame data from Optitrack"""
    def __init__(self, n_rigid_bodies=2):
        self.frame_number = 0
        self.timestamp = time.time()
        self.rigid_body_data = SimulatedRigidBodyData(n_rigid_bodies)
        
    def update(self):
        """Update all data with new values"""
        self.frame_number += 1
        self.timestamp = time.time()
        self.rigid_body_data.update()

class SimulatedClient:
    """
    Simulated Optitrack client for testing without hardware
    Compatible with NatNet SDK 4.3 interface
    """
    
    def __init__(self):
        self.connected_status = True
        self.recording = False
        self.frame_data = SimulatedFrameData()
        
        # Callback functions (SDK 4.3 style)
        self.new_frame_listener = None
        self.new_frame_with_data_listener = None
        self.rigid_body_listener = None
        
        # Simulation control
        self.running = False
        self.frame_rate = 240  # Hz
        self.last_frame_time = 0
        
    def connected(self):
        """Check if connected (SDK 4.3 method)"""
        return self.connected_status
        
    def run(self, stream_type):
        """Start the client (SDK 4.3 method)"""
        self.running = True
        return True
        
    def shutdown(self):
        """Shutdown the client (SDK 4.3 method)"""
        self.connected_status = False
        self.running = False
        
    def send_command(self, command):
        """Send command to Motive (SDK 4.3 method)"""
        print(f"Simulated command: {command}")
        return 0  # Success
        
    def simulate_frame(self):
        """Generate a new simulated frame"""
        current_time = time.time()
        if current_time - self.last_frame_time >= (1.0 / self.frame_rate):
            # Update frame data
            self.frame_data.update()
            
            # Call callbacks if they exist
            if self.new_frame_listener:
                try:
                    self.new_frame_listener(self.frame_data)
                except Exception as e:
                    print(f"Frame callback error: {e}")
                    
            if self.new_frame_with_data_listener:
                try:
                    self.new_frame_with_data_listener({'mocap_data': self.frame_data})
                except Exception as e:
                    print(f"Frame with data callback error: {e}")
                    
            if self.rigid_body_listener:
                for rb in self.frame_data.rigid_body_data.rigid_body_list:
                    try:
                        self.rigid_body_listener(
                            rb.id, 
                            [rb.pos_x, rb.pos_y, rb.pos_z], 
                            [rb.quat_x, rb.quat_y, rb.quat_z, rb.quat_w]
                        )
                    except Exception as e:
                        print(f"Rigid body callback error: {e}")
                        
            self.last_frame_time = current_time
            return True
        return False