'''
Features for the Optitrack motiontracker
Updated for NatNet SDK 4.3 compatibility and BMI/decoder integration
'''
from riglib.experiment import traits
from riglib.optitrack_client_update_natnet import updated_kinematic_sdk43 as optitrack
from datetime import datetime
import time
import numpy as np
import os
from config.rig_defaults import optitrack as defaults
from riglib.stereo_opengl.primitives import Cylinder, Sphere
from riglib import source
from features.neural_sys_features import CorticalBMI, CorticalData

########################################################################################################
# Optitrack datasources
########################################################################################################

class OptitrackBMI(CorticalBMI):
    '''
    BMI using Optitrack motion capture as the datasource.
    Streams hand kinematic data in the same format as EMG data for decoder compatibility.
    Uses NatNet SDK 4.3 compatible client.
    '''
    optitrack_feature = traits.OptionsList(("rigid body", "skeleton", "marker"))
    n_features = traits.Int(1, desc="Number of features to track (e.g., number of bones for skeleton)")
    optitrack_sampling_rate = traits.Float(240, desc="Sampling rate of the motion capture data")
    scale = traits.Float(defaults['scale'], desc="Control scale factor")
    offset = traits.Array(value=defaults['offset'], desc="Control offset")
    
    # NatNet SDK 4.3 connection parameters
    server_address = traits.String(defaults.get('server_address', '127.0.0.1'), 
                                 desc="IP address of Motive server")
    client_address = traits.String(defaults.get('client_address', '127.0.0.1'), 
                                 desc="IP address of this client")
    use_multicast = traits.Bool(defaults.get('use_multicast', True), 
                              desc="Use multicast vs unicast")
    
    optitrack_save_path = defaults['save_path']
    optitrack_sync_dch = defaults['sync_dch']
    
    # Define which features to use as "neural channels" for the decoder
    kinematic_channels = traits.Array(desc="Indices of kinematic features to use as decoder inputs")
    
    def __init__(self, *args, **kwargs):
        '''
        Set up the motion capture data to stream like EMG data.
        Each feature (rigid body, bone, marker) becomes a "channel" that the decoder can use.
        '''
        super().__init__(*args, **kwargs)
        
        # Calculate total number of channels based on features
        # Each feature has 9 channels: position (3), velocity (3), acceleration (3)
        total_channels = self.n_features * 9
        
        # All channels available to the decoder
        self.cortical_channels = np.arange(1, total_channels + 1)
        
        # If not specified, use position channels as decoder inputs
        if len(self.kinematic_channels) == 0:
            # Use just position channels (first 3 of each feature)
            pos_channels = []
            for i in range(self.n_features):
                pos_channels.extend([i*9 + 1, i*9 + 2, i*9 + 3])  # x, y, z for each feature
            self.kinematic_channels = np.array(pos_channels)
        
        # Set up the KinematicSystem parameters (like Quattrocento setup)
        optitrack.KinematicSystem.subj = getattr(self, 'subject_name', 'default')
        optitrack.KinematicSystem.saveid = getattr(self, 'saveid', None)
        optitrack.KinematicSystem.n_features = self.n_features
        optitrack.KinematicSystem.feature_type = self.optitrack_feature
        optitrack.KinematicSystem.update_freq = self.optitrack_sampling_rate
        optitrack.KinematicSystem.server_address = self.server_address
        optitrack.KinematicSystem.client_address = self.client_address
        optitrack.KinematicSystem.use_multicast = self.use_multicast
        
        # Configure the neural source (same pattern as EMG)
        self._neural_src_type = source.MultiChanDataSource
        self._neural_src_kwargs = dict(
            send_data_to_sink_manager=True,
            channels=self.cortical_channels
        )
        self._neural_src_system_type = optitrack.KinematicSystem
        
    def init(self):
        '''
        Initialize the Optitrack connection and data streaming using NatNet SDK 4.3.
        '''
        # Let the parent class handle the neural source setup
        super().init()
        
        # The KinematicSystem is now accessible through self.neural_sys
        self.optitrack_status = getattr(self.neural_sys, 'optitrack_status', 'unknown')
        
        print(f"OptitrackBMI initialized with status: {self.optitrack_status}")
        print(f"Tracking {self.n_features} {self.optitrack_feature}(s)")
        print(f"Total channels: {len(self.cortical_channels)}")
        print(f"Decoder input channels: {self.kinematic_channels}")
        
    def cleanup(self, database, saveid, **kwargs):
        '''
        Save the motion capture data to the database.
        '''
        super_result = super().cleanup(database, saveid, **kwargs)
        time.sleep(2)  # Allow time for file to save cleanly
        
        dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
        subject_name = getattr(self, 'subject_name', 'default')
        filename = f'/var/tmp/tmp_{str(optitrack.KinematicSystem())}_{subject_name}_{saveid}.hdf'
        print(f"Saving {filename} to database {dbname}")
        
        if saveid is not None:
            try:
                if dbname == 'default':
                    database.save_data(filename, "kinematic", saveid, True, False)
                else:
                    database.save_data(filename, "kinematic", saveid, True, False, dbname=dbname)
                print("Kinematic data saved successfully")
            except Exception as e:
                print(f"Error saving kinematic data: {e}")
        else:
            print('\n\nOptitrack file not found properly! It will have to be manually linked!\n\n')
            
        return super_result
    
    @property
    def sys_module(self):
        return optitrack


class Optitrack(traits.HasTraits):
    '''
    Enable reading of raw motiontracker data from Optitrack system using NatNet SDK 4.3
    Compatible with the updated_kinematic_sdk43 module
    '''

    optitrack_feature = traits.OptionsList(("rigid body", "skeleton", "marker"))
    smooth_features = traits.Int(1, desc="How many features to average")
    scale = traits.Float(defaults['scale'], desc="Control scale factor")  
    offset = traits.Array(value=defaults['offset'], desc="Control offset")
    
    # NatNet SDK 4.3 connection parameters
    server_address = traits.String(defaults.get('server_address', '127.0.0.1'),
                                 desc="IP address of Motive server")
    client_address = traits.String(defaults.get('client_address', '127.0.0.1'),
                                 desc="IP address of this client")
    use_multicast = traits.Bool(defaults.get('use_multicast', True),
                              desc="Use multicast vs unicast")
    
    optitrack_save_path = defaults['save_path']
    optitrack_sync_dch = defaults['sync_dch']
    
    # New traits for SDK 4.3 features
    n_features = traits.Int(1, desc="Number of features to track")

    hidden_traits = ['optitrack_feature', 'smooth_features', 'n_features']

    def init(self):
        '''
        Secondary init function using NatNet SDK 4.3 compatible client.
        Sets up the DataSource for interacting with the motion tracker system.
        '''

        # Initialize the KinematicSystem with NatNet SDK 4.3
        try:
            # Create KinematicSystem instance with our parameters
            kinematic_kwargs = {
                'n_features': self.n_features,
                'feature_type': self.optitrack_feature,
                'server_address': self.server_address,
                'client_address': self.client_address,
                'use_multicast': self.use_multicast,
                'subj': getattr(self, 'subject_name', 'default'),
                'saveid': getattr(self, 'saveid', None)
            }
            
            self.kinematic_system = optitrack.KinematicSystem(**kinematic_kwargs)
            self.optitrack_status = self.kinematic_system.optitrack_status
            
            # Create filename for saving if we have a saveid
            if getattr(self, 'saveid', None) is not None:
                now = datetime.now()
                session = "OptiTrack/Session " + now.strftime("%Y-%m-%d")
                take = now.strftime("Take %Y-%m-%d %H:%M:%S") + f" ({self.saveid})"
                self.filename = os.path.join(session, take + '.hdf')  # Changed to .hdf for consistency
            
        except Exception as e:
            print(f"Optitrack connection error: {e}")
            self.optitrack_status = f'Connection failed: {str(e)}'
            # Fall back to simulated system
            kinematic_kwargs = {
                'n_features': self.n_features,
                'feature_type': self.optitrack_feature,
            }
            self.kinematic_system = optitrack.KinematicSystem(**kinematic_kwargs)
            self.kinematic_system.optitrack_status = 'simulated'
            self.optitrack_status = 'simulated'

        # Create a DataSource to buffer the motion tracking data
        self.motiondata = source.DataSource(self.kinematic_system)

        # Register with sink manager for data saving
        from riglib import sink
        sink_manager = sink.SinkManager.get_instance()
        sink_manager.register(self.motiondata)
        
        super().init()

    def run(self):
        '''
        Start the motion data source before FSM execution, stop it after.
        '''
        if not self.optitrack_status in ['recording', 'streaming', 'simulated']:
            import io
            self.terminated_in_error = True
            self.termination_err = io.StringIO()
            self.termination_err.write(self.optitrack_status)
            self.termination_err.seek(0)
            self.state = None
            super().run()
        else:
            print(f"Starting optitrack with status: {self.optitrack_status}")
            self.motiondata.start()
            try:
                super().run()
            finally:
                print("Stopping optitrack")
                self.motiondata.stop()

    def _start_None(self):
        '''
        Code to run before the 'None' state starts (i.e., the task stops)
        '''
        self.motiondata.stop()
        super()._start_None()

    def join(self):
        '''
        Re-join the motiondata source process before cleaning up the experiment thread
        '''
        if self.optitrack_status in ['recording', 'streaming', 'simulated']:
            print("Joining optitrack datasource")
            self.motiondata.join()
        super().join()

    def cleanup(self, database, saveid, **kwargs):
        '''
        Save the optitrack recorded file into the database
        '''
        super_result = super().cleanup(database, saveid, **kwargs)
        print("Saving optitrack file to database...")
        try:
            if hasattr(self, 'filename') and saveid is not None:
                database.save_data(self.filename, "optitrack", saveid, False, False)
                print("Optitrack file saved successfully")
            else:
                print("No filename or saveid available for saving")
        except Exception as e:
            print(f"Error saving optitrack file: {e}")
            return False
        print("...done.")
        return super_result

    def _get_manual_position(self):
        ''' 
        Overridden method to get input coordinates based on motion data
        Uses the new KinematicSystem data format
        '''
        # Get data from optitrack datasource
        try:
            data = self.motiondata.get()  # Returns structured array with 'data' and 'timestamp'
            if data is None or len(data) == 0:
                return None
                
            # Extract kinematic data from structured array
            kinematic_data = data[0]['data']  # Shape: (n_features * 9,)
            
            if np.isnan(kinematic_data).any():
                return None
                
            # Extract position data (first 3 channels of each feature)
            n_features = self.n_features
            positions = []
            
            for i in range(n_features):
                base_idx = i * 9
                pos = kinematic_data[base_idx:base_idx+3]  # x, y, z
                if not np.isnan(pos).any():
                    positions.append(pos)
            
            if len(positions) == 0:
                return None
                
            # Convert to numpy array
            positions = np.array(positions)
            
            # For single feature, return the position directly
            if n_features == 1:
                return positions[0] * 100  # convert meters to centimeters
            else:
                # For multiple features, return centroid or first valid position
                if self.smooth_features > 1:
                    # Average the specified number of features
                    features_to_avg = min(self.smooth_features, len(positions))
                    return np.mean(positions[:features_to_avg], axis=0) * 100
                else:
                    return positions[0] * 100  # Just return first feature
                    
        except Exception as e:
            print(f"Error getting manual position: {e}")
            return None


class OptitrackSimulate(Optitrack):
    '''
    Fake optitrack data for testing with NatNet SDK 4.3 compatibility
    '''

    def init(self):
        '''
        Initialize with simulated KinematicSystem for testing
        '''
        print("Initializing simulated Optitrack system")
        
        # Create simulated KinematicSystem
        kinematic_kwargs = {
            'n_features': self.n_features,
            'feature_type': self.optitrack_feature,
            'subj': getattr(self, 'subject_name', 'simulated'),
            'saveid': getattr(self, 'saveid', None)
        }
        
        self.kinematic_system = optitrack.KinematicSystem(**kinematic_kwargs)
        # Force simulated status
        self.kinematic_system.optitrack_status = 'simulated'
        self.kinematic_system._init_simulated_client()
        self.optitrack_status = 'simulated'

        # Create a DataSource to buffer the motion tracking data
        self.motiondata = source.DataSource(self.kinematic_system)

        # Register with sink manager
        from riglib import sink
        sink_manager = sink.SinkManager.get_instance()
        sink_manager.register(self.motiondata)
        
        # Call HasTraits init directly to skip Optitrack.init()
        super(Optitrack, self).init()


class OptitrackPlayback(Optitrack):
    '''
    Read a CSV/HDF file back into BMI3D as if it were live data
    Updated for NatNet SDK 4.3 compatibility
    '''

    filepath = traits.String("", desc="path to optitrack data file for playback")
    
    # TODO: Implement playback client in updated_kinematic_sdk43.py
    def init(self):
        '''
        Initialize with playback data source
        '''
        print(f"Initializing Optitrack playback from: {self.filepath}")
        
        # This would require implementing a PlaybackClient in updated_kinematic_sdk43.py
        # For now, fall back to simulation
        print("Playback not yet implemented, using simulation")
        super(OptitrackSimulate, self).init()


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


# Helper class for logging - simplified for NatNet SDK 4.3
class Logger(object):
    def __init__(self, msg="", log_filename=None):
        if log_filename is None:
            log_filename = os.path.join(os.path.dirname(__file__), '../log/optitrack.log')
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