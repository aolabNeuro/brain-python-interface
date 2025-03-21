'''
Features for the Optitrack motiontracker
'''
from riglib.experiment import traits
from riglib.optitrack_client import optitrack
from datetime import datetime
import time
import numpy as np
import os
from config.rig_defaults import optitrack as defaults
from riglib.stereo_opengl.primitives import Cylinder, Sphere

########################################################################################################
# Optitrack datasources
########################################################################################################
class Optitrack(traits.HasTraits):
    '''
    Enable reading of raw motiontracker data from Optitrack system
    Requires the natnet library from https://github.com/leoscholl/python_natnet
    To be used as a feature with the ManualControl task for the time being. However,
    ideally this would be implemented as a decoder :)
    '''

    optitrack_feature = traits.OptionsList(("rigid body", "skeleton", "marker"))
    smooth_features = traits.Int(1, desc="How many features to average")
    scale = traits.Float(defaults['scale'], desc="Control scale factor")
    offset = traits.Array(value=defaults['offset'], desc="Control offset")
    optitrack_ip = defaults['address']
    optitrack_save_path = defaults['save_path']
    optitrack_sync_dch = defaults['sync_dch']

    hidden_traits = ['optitrack_feature', 'smooth_features']

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up the DataSource for interacting with the 
        motion tracker system and registers the source with the SinkRegister so that the data gets saved to file as it is collected.
        '''

        # Start the natnet client and recording
        import natnet
        now = datetime.now()
        session = "OptiTrack/Session " + now.strftime("%Y-%m-%d")
        take = now.strftime("Take %Y-%m-%d %H:%M:%S")
        logger = Logger(take)
        try:
            client = natnet.Client.connect(server=self.optitrack_ip, logger=logger, timeout=1)
            if self.saveid is not None:
                take += " (%d)" % self.saveid
                client.set_session(os.path.join(self.optitrack_save_path, session))
                client.set_take(take)
                self.filename = os.path.join(session, take + '.tak')
                client._send_command_and_wait("LiveMode")
                time.sleep(0.1)
                if client.start_recording():
                    self.optitrack_status = 'recording'
            else:
                self.optitrack_status = 'streaming'
        except natnet.DiscoveryError:
            self.optitrack_status = 'Optitrack couldn\'t be started, make sure Motive is open!'
            client = optitrack.SimulatedClient()
        self.client = client

        # Create a source to buffer the motion tracking data
        from riglib import source
        self.motiondata = source.DataSource(optitrack.make(optitrack.System, self.client, self.optitrack_feature, 1))

        # Save to the sink
        from riglib import sink
        sink_manager = sink.SinkManager.get_instance()
        sink_manager.register(self.motiondata)
        super().init()

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the motiondata source and stops it after the FSM has finished running
        '''
        if not self.optitrack_status in ['recording', 'streaming']:
            import io
            self.terminated_in_error = True
            self.termination_err = io.StringIO()
            self.termination_err.write(self.optitrack_status)
            self.termination_err.seek(0)
            self.state = None
            super().run()
        else:
            self.motiondata.start()
            try:
                super().run()
            finally:
                print("Stopping optitrack")
                self.client.stop_recording()
                self.motiondata.stop()

    def _start_None(self):
        '''
        Code to run before the 'None' state starts (i.e., the task stops)
        '''
        #self.client.stop_recording()
        self.motiondata.stop()
        super()._start_None()

    def join(self):
        '''
        See riglib.experiment.Experiment.join(). Re-join the motiondata source process before cleaning up the experiment thread
        '''
        if self.optitrack_status in ['recording', 'streaming']:
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
            database.save_data(self.filename, "optitrack", saveid, False, False) # Make sure you actually have an "optitrack" system added!
        except Exception as e:
            print(e)
            return False
        print("...done.")
        return super_result

    def _get_manual_position(self):
        ''' Overridden method to get input coordinates based on motion data'''

        # Get data from optitrack datasource
        data = self.motiondata.get() # List of (list of features)
        if len(data) == 0: # Data is not being streamed
            return
        recent = data[-self.smooth_features:] # How many recent coordinates to average
        averaged = np.nanmean(recent, axis=0) # List of averaged features
        if np.isnan(averaged).any(): # No usable coords
            return
        return averaged*100 # convert meters to centimeters

class OptitrackSimulate(Optitrack):
    '''
    Fake optitrack data for testing
    '''

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up the DataSource for interacting with the 
        motion tracker system and registers the source with the SinkRegister so that the data gets saved to file as it is collected.
        '''

        # Start the fake natnet client
        self.client = optitrack.SimulatedClient()
        self.optitrack_status = 'streaming'

        # Create a source to buffer the motion tracking data
        from riglib import source
        self.motiondata = source.DataSource(optitrack.make(optitrack.System, self.client, self.optitrack_feature, 1))

        # Save to the sink
        from riglib import sink
        sink_manager = sink.SinkManager.get_instance()
        sink_manager.register(self.motiondata)
        super(Optitrack, self).init()

class OptitrackPlayback(Optitrack):
    '''
    Read a csv file back into BMI3D as if it were live data
    '''

    filepath = traits.String("", desc="path to optitrack csv file for playback")
    
    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up the DataSource for interacting with the 
        motion tracker system and registers the source with the SinkRegister so that the data gets saved to file as it is collected.
        '''

        # Start the fake natnet client
        self.client = optitrack.PlaybackClient(self.filepath)
        self.optitrack_status = 'streaming'

        # Create a source to buffer the motion tracking data
        from riglib import source
        self.motiondata = source.DataSource(optitrack.make(optitrack.System, self.client, self.optitrack_feature, 1))

        # Save to the sink
        from riglib import sink
        sink_manager = sink.SinkManager.get_instance()
        sink_manager.register(self.motiondata)
        super(Optitrack, self).init()

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
    
# Helper class for natnet logging
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