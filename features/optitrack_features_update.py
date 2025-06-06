import time
import os
import numpy as np
from riglib.experiment import traits
from riglib import source
from features.neural_sys_features import CorticalBMI, CorticalData
import riglib.optitrack_client_update.optitrack_system
import traceback


class OptiTrackBMI(CorticalBMI):
    """
    BMI using OptiTrack motion capture as the data source.
    Streams rigid body and skeleton position data for use with decoders.
    """
    
    # OptiTrack configuration traits
    optitrack_server_ip = traits.String("128.95.215.191", desc="IP address of OptiTrack/Motive server")
    optitrack_client_ip = traits.String("128.95.215.213", desc="IP address of this client machine")
    optitrack_use_multicast = traits.Bool(False, desc="Use multicast streaming (vs unicast)")
    optitrack_update_freq = traits.Float(120.0, desc="Expected OptiTrack frame rate (Hz)")
    optitrack_buffer_size = traits.Int(1000, desc="Size of data buffer")
    
    def __init__(self, *args, **kwargs):
        """
        Initialize OptiTrack BMI feature
        """
        super().__init__(*args, **kwargs)
        
        # Set up OptiTrack-specific parameters
        optitrack_system.OptiTrackData.server_ip = self.optitrack_server_ip
        optitrack_system.OptiTrackData.client_ip = self.optitrack_client_ip
        optitrack_system.OptiTrackData.use_multicast = self.optitrack_use_multicast
        optitrack_system.OptiTrackData.update_freq = self.optitrack_update_freq
        optitrack_system.OptiTrackData.buffer_size = self.optitrack_buffer_size
        optitrack_system.OptiTrackData.subj = getattr(self, 'subject_name', 'unknown')
        optitrack_system.OptiTrackData.saveid = getattr(self, 'saveid', 0)
        
        # Estimate initial channel count (will be updated when streaming starts)
        # This is a rough estimate - actual count depends on tracked objects
        estimated_rigid_bodies = 10  # Estimate for typical setup
        estimated_skeleton_rbs = 20   # Estimate for skeleton rigid bodies
        estimated_channels = (estimated_rigid_bodies + estimated_skeleton_rbs) * 3
        self.cortical_channels = np.arange(1, estimated_channels + 1)
        
        # Set up neural source parameters for CorticalData initialization
        self._neural_src_type = source.MultiChanDataSource
        self._neural_src_kwargs = dict(
            server_ip=self.optitrack_server_ip,
            client_ip=self.optitrack_client_ip,
            use_multicast=self.optitrack_use_multicast,
            update_freq=self.optitrack_update_freq,
            buffer_size=self.optitrack_buffer_size,
            send_data_to_sink_manager=True,
            channels=self.cortical_channels
        )
        self._neural_src_system_type = optitrack_system.OptiTrackData
    
    def init(self):
        """
        Initialize the OptiTrack data source
        """
        # Call parent init first
        super().init()
        
        # Update channel count after OptiTrack connection is established
        if hasattr(self.neurondata, 'source') and hasattr(self.neurondata.source, 'total_channels'):
            actual_channels = self.neurondata.source.total_channels
            if actual_channels > 0:
                self.cortical_channels = np.arange(1, actual_channels + 1)
                print(f"Updated OptiTrack channels to {actual_channels} based on tracked objects")
    
    def cleanup(self, database, saveid, **kwargs):
        """
        Cleanup function to save OptiTrack data to HDF5 file
        """
        super_result = super().cleanup(database, saveid, **kwargs)
        
        # Sleep to ensure data is fully written
        time.sleep(2)
        
        dbname = kwargs.get('dbname', 'default')
        subject_name = getattr(self, 'subject_name', 'unknown')
        
        # Create filename following BMI3D conventions
        filename = f'/var/tmp/tmp_OptiTrack_{subject_name}_{saveid}.hdf'
        
        print(f"Saving OptiTrack data {filename} to database {dbname}")
        
        try:
            if saveid is not None:
                if dbname == 'default':
                    database.save_data(filename, "optitrack", saveid, True, False)
                else:
                    database.save_data(filename, "optitrack", saveid, True, False, dbname=dbname)
                print(f"Successfully saved OptiTrack data to database")
            else:
                print('\n\nOptiTrack file not found properly! It will have to be manually linked!\n\n')
        except Exception as e:
            print(f"Error saving OptiTrack data to database: {e}")
            traceback.print_exc()
        
        return super_result
    
    @property
    def sys_module(self):
        """Return the OptiTrack system module"""
        return optitrack_system


class OptiTrackData(CorticalData):
    """
    Feature for streaming OptiTrack motion capture data without BMI/decoder integration
    """
    
    # OptiTrack configuration traits
    optitrack_server_ip = traits.String("127.0.0.1", desc="IP address of OptiTrack/Motive server")
    optitrack_client_ip = traits.String("127.0.0.1", desc="IP address of this client machine") 
    optitrack_use_multicast = traits.Bool(False, desc="Use multicast streaming (vs unicast)")
    optitrack_update_freq = traits.Float(120.0, desc="Expected OptiTrack frame rate (Hz)")
    optitrack_buffer_size = traits.Int(1000, desc="Size of data buffer")
    
    # Override cortical_channels to be set automatically
    cortical_channels = None
    
    def __init__(self, *args, **kwargs):
        """
        Initialize OptiTrack data streaming feature
        """
        super().__init__(*args, **kwargs)
        
        # Set up OptiTrack-specific parameters
        optitrack_system.OptiTrackData.server_ip = self.optitrack_server_ip
        optitrack_system.OptiTrackData.client_ip = self.optitrack_client_ip
        optitrack_system.OptiTrackData.use_multicast = self.optitrack_use_multicast
        optitrack_system.OptiTrackData.update_freq = self.optitrack_update_freq
        optitrack_system.OptiTrackData.buffer_size = self.optitrack_buffer_size
        
        # Estimate channel count (will be updated when streaming starts)
        estimated_rigid_bodies = 10
        estimated_skeleton_rbs = 20
        estimated_channels = (estimated_rigid_bodies + estimated_skeleton_rbs) * 3
        self.cortical_channels = np.arange(1, estimated_channels + 1)
    
    def init(self):
        """
        Initialize the OptiTrack data source
        """
        # Set up data source parameters
        kwargs = dict(
            server_ip=self.optitrack_server_ip,
            client_ip=self.optitrack_client_ip,
            use_multicast=self.optitrack_use_multicast,
            update_freq=self.optitrack_update_freq,
            buffer_size=self.optitrack_buffer_size,
            send_data_to_sink_manager=self.send_data_to_sink_manager,
            channels=self.cortical_channels
        )
        
        # Create the data source
        self.neurondata = source.MultiChanDataSource(optitrack_system.OptiTrackData, **kwargs)
        
        # Register with sink manager if requested
        if self.register_with_sink_manager:
            from riglib import sink
            sink_manager = sink.SinkManager.get_instance()
            sink_manager.register(self.neurondata)
        
        # Call parent init
        super(CorticalData, self).init()  # Skip CorticalData.init() to avoid sys_module error
    
    def cleanup(self, database, saveid, **kwargs):
        """
        Cleanup function to save OptiTrack data
        """
        # Sleep to ensure data is fully written
        time.sleep(2)
        
        dbname = kwargs.get('dbname', 'default')
        subject_name = getattr(self, 'subject_name', 'unknown')
        
        # Create filename
        filename = f'/var/tmp/tmp_OptiTrack_{subject_name}_{saveid}.hdf'
        
        print(f"Saving OptiTrack data {filename} to database {dbname}")
        
        try:
            if saveid is not None:
                if dbname == 'default':
                    database.save_data(filename, "optitrack", saveid, True, False)
                else:
                    database.save_data(filename, "optitrack", saveid, True, False, dbname=dbname)
                print(f"Successfully saved OptiTrack data to database")
            else:
                print('\n\nOptiTrack file not found properly! It will have to be manually linked!\n\n')
        except Exception as e:
            print(f"Error saving OptiTrack data to database: {e}")
            traceback.print_exc()
    
    @property
    def sys_module(self):
        """Return the OptiTrack system module"""
        return optitrack_system
