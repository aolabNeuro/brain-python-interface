'''
Features for the Optitrack motiontracker - Updated for new NatNet SDK
'''
from riglib.experiment import traits
from riglib.optitrack_client_update_natnet import optitrack_client_update
from datetime import datetime
import time
import numpy as np
import os
from config.rig_defaults import optitrack as defaults
from riglib.stereo_opengl.primitives import Cylinder, Sphere
import threading
import queue
import copy

# Import the new NatNet SDK components
# Note: You'll need to make sure these are in your Python path
try:
    from NatNetClient import NatNetClient
    import MoCapData
    SDK_AVAILABLE = True
except ImportError:
    print("Warning: New NatNet SDK not available, falling back to simulation")
    SDK_AVAILABLE = False

########################################################################################################
# Updated Optitrack datasources
########################################################################################################

class OptitrackNatNetClient:
    """
    Wrapper class to interface the new NatNet SDK with BMI3D's data architecture
    """
    
    def __init__(self, server_ip="127.0.0.1", local_ip="127.0.0.1", feature_type="rigid body"):
        self.server_ip = server_ip
        self.local_ip = local_ip
        self.feature_type = feature_type
        self.client = None
        self.data_queue = queue.Queue(maxsize=1000)
        self.latest_data = None
        self.is_connected = False
        self.is_recording = False
        self.data_lock = threading.Lock()
        
    def connect(self):
        """Initialize connection to OptiTrack system"""
        if not SDK_AVAILABLE:
            raise Exception("NatNet SDK not available")
            
        self.client = NatNetClient()
        self.client.set_server_address(self.server_ip)
        self.client.set_client_address(self.local_ip)
        
        # Set up callbacks based on feature type
        if self.feature_type == "rigid body":
            self.client.rigid_body_listener = self._rigid_body_callback
        
        self.client.new_frame_listener = self._new_frame_callback
        
        try:
            # Connect to the server
            self.client.run()
            
            # Wait a moment for connection to establish
            time.sleep(0.5)
            
            if self.client.connected():
                self.is_connected = True
                print(f"Connected to OptiTrack server at {self.server_ip}")
                return True
            else:
                print("Failed to connect to OptiTrack server")
                return False
                
        except Exception as e:
            print(f"Error connecting to OptiTrack: {e}")
            return False
    
    def _rigid_body_callback(self, id, position, rotation):
        """Callback for rigid body data"""
        with self.data_lock:
            # Store the latest data - convert to centimeters for BMI3D compatibility
            self.latest_data = {
                'id': id,
                'position': [position[0] * 100, position[1] * 100, position[2] * 100],  # m to cm
                'rotation': rotation,
                'timestamp': time.time()
            }
    
    def _new_frame_callback(self, frame_number):
        """Callback for new frame data"""
        # Can be used for frame-based synchronization if needed
        pass
    
    def get_latest_data(self):
        """Get the most recent data"""
        with self.data_lock:
            if self.latest_data is not None:
                return copy.deepcopy(self.latest_data)
            return None
    
    def start_recording(self, session_path=None, take_name=None):
        """Start recording in Motive"""
        if not self.is_connected:
            return False
            
        try:
            # Send recording commands to Motive
            if session_path:
                self.client.send_command(f"SetCurrentSession,{session_path}")
            if take_name:
                self.client.send_command(f"SetCurrentTake,{take_name}")
            
            result = self.client.send_command("StartRecording")
            if result >= 0:
                self.is_recording = True
                print("Started OptiTrack recording")
                return True
            else:
                print("Failed to start OptiTrack recording")
                return False
        except Exception as e:
            print(f"Error starting recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop recording in Motive"""
        if not self.is_connected or not self.is_recording:
            return True
            
        try:
            result = self.client.send_command("StopRecording")
            if result >= 0:
                self.is_recording = False
                print("Stopped OptiTrack recording")
                return True
            else:
                print("Failed to stop OptiTrack recording")
                return False
        except Exception as e:
            print(f"Error stopping recording: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from OptiTrack system"""
        if self.client and self.is_connected:
            try:
                if self.is_recording:
                    self.stop_recording()
                self.client.stop_threads = True
                self.is_connected = False
                print("Disconnected from OptiTrack")
            except Exception as e:
                print(f"Error disconnecting: {e}")


class OptitrackUpdated(traits.HasTraits):
    '''
    Updated Optitrack feature using the new NatNet SDK
    Maintains compatibility with existing BMI3D architecture
    '''

    optitrack_feature = traits.OptionsList(("rigid body", "skeleton", "marker"))
    smooth_features = traits.Int(1, desc="How many features to average")
    scale = traits.Float(defaults['scale'], desc="Control scale factor")
    offset = traits.Array(value=defaults['offset'], desc="Control offset")
    optitrack_server_ip = traits.String(defaults.get('server_address', '127.0.0.1'), desc="OptiTrack server IP")
    optitrack_local_ip = traits.String(defaults.get('local_address', '127.0.0.1'), desc="Local machine IP")
    optitrack_save_path = defaults['save_path']

    hidden_traits = ['optitrack_feature', 'smooth_features']

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up the connection to OptiTrack
        '''
        
        # Initialize the OptiTrack client
        self.optitrack_client = OptitrackNatNetClient(
            server_ip=self.optitrack_server_ip,
            local_ip=self.optitrack_local_ip,
            feature_type=self.optitrack_feature
        )
        
        # Attempt to connect
        if self.optitrack_client.connect():
            self.optitrack_status = 'connected'
            
            # Start recording if we have a save ID
            if hasattr(self, 'saveid') and self.saveid is not None:
                now = datetime.now()
                session = f"OptiTrackSession_{now.strftime('%Y-%m-%d')}"
                take = f"Take_{now.strftime('%Y-%m-%d_%H-%M-%S')}_{self.saveid}"
                
                session_path = os.path.join(self.optitrack_save_path, session)
                
                if self.optitrack_client.start_recording(session_path, take):
                    self.optitrack_status = 'recording'
                    self.recording_filename = os.path.join(session_path, take + '.tak')
                else:
                    self.optitrack_status = 'streaming'
            else:
                self.optitrack_status = 'streaming'
                
        else:
            self.optitrack_status = 'Failed to connect to OptiTrack'
            # Fall back to simulation
            self.optitrack_client = None
        
        # Initialize data storage for smoothing
        self.position_history = []
        
        super().init()

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing
        '''
        if self.optitrack_status not in ['recording', 'streaming', 'connected']:
            import io
            self.terminated_in_error = True
            self.termination_err = io.StringIO()
            self.termination_err.write(self.optitrack_status)
            self.termination_err.seek(0)
            self.state = None
            super().run()
        else:
            try:
                super().run()
            finally:
                print("Stopping OptiTrack connection")
                if self.optitrack_client:
                    self.optitrack_client.disconnect()

    def _start_None(self):
        '''
        Code to run before the 'None' state starts (i.e., the task stops)
        '''
        if self.optitrack_client:
            self.optitrack_client.stop_recording()
        super()._start_None()

    def join(self):
        '''
        Clean up OptiTrack connection
        '''
        if self.optitrack_client and self.optitrack_client.is_connected:
            print("Disconnecting from OptiTrack")
            self.optitrack_client.disconnect()
        super().join()

    def cleanup(self, database, saveid, **kwargs):
        '''
        Save the optitrack recorded file into the database
        '''
        super_result = super().cleanup(database, saveid, **kwargs)
        
        if hasattr(self, 'recording_filename') and self.optitrack_status == 'recording':
            print("Saving OptiTrack file to database...")
            try:
                database.save_data(self.recording_filename, "optitrack", saveid, False, False)
                print("OptiTrack file saved successfully.")
            except Exception as e:
                print(f"Error saving OptiTrack file: {e}")
                return False
        
        return super_result

    def _get_manual_position(self):
        ''' 
        Overridden method to get input coordinates based on motion data
        Returns position in centimeters for BMI3D compatibility
        '''
        
        if not self.optitrack_client or not self.optitrack_client.is_connected:
            return None
        
        # Get latest data from OptiTrack
        data = self.optitrack_client.get_latest_data()
        
        if data is None:
            return None
        
        # Extract position (already converted to cm in the client)
        position = np.array(data['position'])
        
        # Apply smoothing by maintaining a history
        self.position_history.append(position)
        
        # Keep only the most recent positions for smoothing
        if len(self.position_history) > self.smooth_features:
            self.position_history = self.position_history[-self.smooth_features:]
        
        # Return smoothed position
        if len(self.position_history) > 0:
            smoothed_position = np.nanmean(self.position_history, axis=0)
            if not np.isnan(smoothed_position).any():
                # Apply scale and offset
                return smoothed_position * self.scale + np.array(self.offset)
        
        return None


class OptitrackSimulateUpdated(OptitrackUpdated):
    '''
    Simulation version for testing without OptiTrack hardware
    '''

    def init(self):
        '''
        Initialize simulation client
        '''
        self.optitrack_client = None
        self.optitrack_status = 'simulation'
        self.position_history = []
        
        # Create fake data for testing
        self.sim_position = np.array([0.0, 0.0, 0.0])
        self.sim_time = time.time()
        
        super(OptitrackUpdated, self).init()

    def _get_manual_position(self):
        '''
        Generate simulated position data for testing
        '''
        # Generate some simple movement pattern
        current_time = time.time()
        dt = current_time - self.sim_time
        self.sim_time = current_time
        
        # Simple circular motion for testing
        t = current_time * 0.5  # slow motion
        self.sim_position[0] = 10 * np.cos(t)  # 10cm radius
        self.sim_position[2] = 10 * np.sin(t)
        self.sim_position[1] = 5 * np.sin(t * 2)  # vertical component
        
        return self.sim_position * self.scale + np.array(self.offset)


# Keep the existing helper classes for compatibility
class HidePlantOnPause():
    '''
    Makes the cursor hidden when the game is paused.
    '''

    def _cycle(self):
        self.plant_visible = not self.pause
        super()._cycle()


class SpheresToCylinders():
    '''
    Convert spheres to cylinders pointing in and out of the screen up to the bounds.
    '''
    
    def add_model(self, model):
        '''
        Hijack spheres and switch them for cylinders along the y-axis
        '''
        if isinstance(model, Sphere) and model.radius > 0.5:
            height = self.cursor_bounds[3] - self.cursor_bounds[2]
            tmp_model = Cylinder(height, model.radius, color=model.color)
            print('Switched sphere for cylinder')
            model.verts, model.polys, model.tcoords, model.normals = tmp_model.verts, tmp_model.polys, tmp_model.tcoords, tmp_model.normals
            model.rotate_x(90)
        super().add_model(model)

    def _test_enter_target(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos[[0,2]] - self.targs[self.target_index][[0,2]])
        return d <= (self.target_radius - self.cursor_radius)

    def _test_leave_target(self, ts):
        '''
        return true if cursor moves outside the exit radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos[[0,2]] - self.targs[self.target_index][[0,2]])
        rad = self.target_radius - self.cursor_radius
        return d > rad


# Keep original classes for backwards compatibility
Optitrack = OptitrackUpdated  # Alias for drop-in replacement
OptitrackSimulate = OptitrackSimulateUpdated  # Alias for drop-in replacement


# Helper class for logging (kept for compatibility)
import logging
log_path = os.path.join(os.path.dirname(__file__), '../log/optitrack.log')

class Logger(object):

    def __init__(self, msg="", log_filename=log_path):
        self.log_filename = log_filename
        self.reset(msg)

    def log_str(self, s, mode="a", newline=True):
        if self.log_filename != '':
            if newline and not s.endswith("\n"):
                s += "\n"
            with open(self.log_filename, mode) as fp:
                fp.write(s)
    
    def _log(self, msg, *args):
        self.log_str(msg % args)

    def reset(self, s="Logger"):
        with open(self.log_filename, "w") as fp:
            fp.write(s + "\n\n")

    debug = _log
    info = _log
    warning = _log
    error = _log
    fatal = _log