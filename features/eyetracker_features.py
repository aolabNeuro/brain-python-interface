'''
Features for the eyetracker system
'''

import tempfile
import time
import traceback
import numpy as np
import tables
from traits.trait_types import self
from db import dbfunctions as db
from riglib import calibrations
from riglib.experiment import traits
from riglib.gpio import ArduinoGPIO
from riglib.oculomatic import oculomatic
from riglib.pupillabs import utils, eye_labels, gaze_options, gaze_options_idx
from built_in_tasks.target_graphics import *
from built_in_tasks.target_capture_task import ScreenTargetCapture
from riglib.stereo_opengl.primitives import AprilTag
from riglib.stereo_opengl.xfm import Quaternion
from .peripheral_device_features import *
from riglib import plants
from collections import deque

import aopy
import glob
import os

###### CONSTANTS
sec_per_min = 60

class EyeCalibration(traits.HasTraits):
    '''
    Calculates 'calibrated_eye_pos' from 'eye_pos' (added by EyeStreaming feature) by regressing previous
    recorded cursor position and eye position when cursor enters targets. Must specify taskid for calibration. 
    '''

    taskid_for_eye_calibration = traits.Int(0, desc="directory where hdf file lives")
    show_eye_pos = traits.Bool(False, desc="Whether to show eye positions")
    eye_target_calibration = traits.Bool(False, desc="Whether to regress eye positions against target positions")
    center_eye_data = traits.Bool(False, desc="Whether to demean eye data with eye position for the center target")
    offset_time_eye_calibration = traits.Float(0.1, desc="Data after this offset_time is only used for eye calibration")
    duration_eye_calibration = traits.Float(0.2, desc="Data within this duration after offset_time is only used for eye calibration")

    def __init__(self, *args, **kwargs): #, start_pos, calibration):
        super(EyeCalibration,self).__init__(*args, **kwargs)
        
        # proc_exp # preprocess cursor data only
        taskid = self.taskid_for_eye_calibration
        try:
            entry = db.get_task_entries_by_id([taskid])[0]
        except:
            traceback.print_exc()
            raise ValueError(f"Taskid {taskid} not found in database")
        files = entry.get_data_files_dict_absolute()
        print(files)
        
        self.eye_center = np.zeros((4,))
        _, metadata_hdf = aopy.data.load_bmi3d_hdf_table('', files['hdf'], 'task')
        # Check if coefficients are already calculated
        if 'eye_coeff' in metadata_hdf:
            self.eye_coeff = metadata_hdf['eye_coeff']
            self.eye_center = metadata_hdf['eye_center']
            print("Calibration data have been loaded:", self.eye_coeff)
            return

        if not self.keyboard_control:
            try:
                bmi3d_data, bmi3d_metadata = aopy.preproc.proc_exp('', files, '', '', overwrite=True, save_res=False)
            except:
                traceback.print_exc()
                raise ValueError(f"Error processing hdf file for taskid {taskid}")

            # load raw eye data
            # raw_eye_data, raw_eye_metadata = aopy.preproc.parse_oculomatic(hdf_dir, files, debug=False)
            samplerate = 1000
            eye_interp = aopy.data.get_interp_kinematics(bmi3d_data, bmi3d_metadata, datatype='eye',
                                                        samplerate=samplerate)[:,:4]

            # calculate coefficients to calibrate eye data
            events = bmi3d_data['events']

            if not self.eye_target_calibration:
                cursor_interp = aopy.data.get_interp_kinematics(bmi3d_data, bmi3d_metadata, datatype='cursor',
                                                        samplerate=samplerate)
                self.eye_coeff,_,_,_ = aopy.preproc.calc_eye_calibration(
                    cursor_interp, samplerate, eye_interp, samplerate, 
                    events['timestamp'], events['code'], return_datapoints=True
                )


            # Calculate coefficients
            else:
                def get_target_locations(data, target_indices):
                    try:
                        trials = data['trials']
                    except:
                        trials = data['bmi3d_trials']
                    locations = np.nan*np.zeros((len(target_indices), 3))
                    for i in range(len(target_indices)):
                        trial_idx = np.where(trials['index'] == target_indices[i])[0]
                        if len(trial_idx) > 0:
                            locations[i,:] = trials['target'][trial_idx[0]][[0,2,1]] # use x,y,z format
                        else:
                            raise ValueError(f"Target index {target_indices[i]} not found")
                    return np.round(locations,4)

                target_pos = get_target_locations(bmi3d_data, [1,2,3,4,5,6,7,8])
                
                # Get eye_pos data when subjects gaze at the center. Target position doesn't matter for this computation
                if self.center_eye_data:
                    _, _, eye_center = aopy.preproc.calc_eye_target_calibration(
                        eye_interp, samplerate, events['timestamp'], events['code'], target_pos,
                        offset=self.offset_time_eye_calibration, duration=self.duration_eye_calibration, 
                        align_events=80, return_datapoints=True
                    )
                    self.eye_center = np.nanmedian(eye_center, axis=0)
                else:
                    self.eye_center = np.zeros((eye_interp.shape[1],))


                # Calculate coefficient by linear regression between targets and centered eye positions
                self.eye_coeff, _ = aopy.preproc.calc_eye_target_calibration(
                    eye_interp-self.eye_center, samplerate, events['timestamp'], events['code'], target_pos,
                    offset=self.offset_time_eye_calibration, duration=self.duration_eye_calibration
                )
            
            print("Calibration complete:", self.eye_coeff)

        # Set up eye cursor
        self.eye_cursor = VirtualCircularTarget(target_radius=.25, target_color=(0., 1., 0., 0.5))
        self.target_location = np.array(self.starting_pos).copy()
        self.calibrated_eye_pos = np.zeros((2,))*np.nan
        for model in self.eye_cursor.graphics_models:
            self.add_model(model)
        
    def init(self):
        self.add_dtype('calibrated_eye', 'f8', (2,))
        super().init()

    #### STATE FUNCTIONS ####
    def _start_wait(self):
        super()._start_wait()

        if self.calc_trial_num() == 0:
            if self.show_eye_pos:
                self.eye_cursor.show()
            else:
                self.eye_cursor.hide()

    def _cycle(self):
        self._update_eye_pos()

        # Do calibration
        ave_pos = self.eye_pos
        if len(self.eye_pos) > 2 and len(self.eye_center) > 2:
            calibrated_pos = aopy.postproc.get_calibrated_eye_data(self.eye_pos[:4]-self.eye_center, self.eye_coeff)
            ave_pos = np.array([(calibrated_pos[0] + calibrated_pos[2])/2, (calibrated_pos[1] + calibrated_pos[3])/2])
        elif not self.keyboard_control:
            ave_pos = aopy.postproc.get_calibrated_eye_data(self.eye_pos-self.eye_center[:len(self.eye_pos)], self.eye_coeff[:len(self.eye_pos)])

        # Save calibration
        self.calibrated_eye_pos = ave_pos
        self.task_data['calibrated_eye'] = ave_pos

        super()._cycle()

        # Move the eye cursor
        if np.any(np.isnan(self.calibrated_eye_pos)):
            pass
        else:
            self.eye_cursor.move_to_position([self.calibrated_eye_pos[0],0,self.calibrated_eye_pos[1]])
            if self.show_eye_pos:
                self.eye_cursor.show()

class AutomaticEyeCalibration(traits.HasTraits):
    
    trial_numbers_for_calibration = traits.Int(16, desc="how many trials are used in each calibration")
    trial_numbers_for_auto_reward = traits.Int(8, desc="how many trials are automatic rewards")
    offset_time_eye_calibration = traits.Float(0.2, desc="Data after this offset_time is only used for eye calibration")
    duration_eye_calibration = traits.Float(0.2, desc="Data within this duration after offset_time is only used for eye calibration")
    show_eye_pos = traits.Bool(False, desc="Whether to show eye positions")

    def __init__(self, *args, **kwargs): #, start_pos, calibration):
        super(AutomaticEyeCalibration,self).__init__(*args, **kwargs)

        self.m_eye_pos = deque(maxlen = self.trial_numbers_for_calibration)
        self.m_center_eye_pos = deque(maxlen = self.trial_numbers_for_calibration)
        self.target_pos_calibration = deque(maxlen = self.trial_numbers_for_calibration)

        self.eye_center = np.zeros((4,))
        self.eye_coeff = np.vstack(([1,1,1,1], [0,0,0,0])).T

        # Set up eye cursor
        self.eye_cursor = VirtualCircularTarget(target_radius=.25, target_color=(0., 1., 0., 0.5))
        self.calibrated_eye_pos = np.zeros((2,))*np.nan
        for model in self.eye_cursor.graphics_models:
            self.add_model(model)

    def init(self):
        self.add_dtype('calibrated_eye', 'f8', (2,))
        super().init()

    def _start_hold(self):
        super()._start_hold()
        self.start_hold_time = self.get_time()

    def _while_hold(self):
        super()._while_hold()

        # Only store eye pos between offset ~ offset + duration
        elapsed_time = self.get_time() - self.start_hold_time
        if elapsed_time > self.offset_time_eye_calibration and elapsed_time < self.offset_time_eye_calibration + self.duration_eye_calibration:
            if self.target_index == 0:
                self.eye_pos_tmp0.append(self.eye_pos[:4])
            elif self.target_index == 1:
                self.eye_pos_tmp1.append(self.eye_pos[:4])
        
    def _end_reward(self):
        super()._end_reward()

        # Store data only in rewarded trials
        self.m_center_eye_pos.append(np.nanmean(self.eye_pos_tmp0, axis=0)) # eye pos for the center target
        self.m_eye_pos.append(np.nanmean(self.eye_pos_tmp1, axis=0)) # eye pos for the peripheral target
        self.target_pos_calibration.append(np.array(self.targs)[-1,[0,2]]) # peripheral target pos

    def _start_wait(self):
        super()._start_wait()

        self.eye_pos_tmp0 = [] # to store eye pos when target_index == 0
        self.eye_pos_tmp1 = [] # to store eye pos when target_index == 1

        if self.calc_trial_num() == 0:
            if self.show_eye_pos:
                self.eye_cursor.show()
            else:
                self.eye_cursor.hide()

        if self.calc_state_occurrences('reward') < self.trial_numbers_for_auto_reward:
            self.automatic_reward = True

        else:
            self.automatic_reward = False
            
            # Perform regression
            if not self.keyboard_control:
                if self.tries == 0:
                
                    self.eye_center = np.nanmean(self.m_center_eye_pos, axis=0)
                    if len(self.eye_center) == 4:
                        target_pos_tile = np.tile(np.array(self.target_pos_calibration), (1,2))
                    else:
                        target_pos_tile = np.array(self.target_pos_calibration)

                    slopes, intercepts, _ = aopy.analysis.fit_linear_regression(np.array(self.m_eye_pos)-self.eye_center, target_pos_tile)
                    #intercepts = np.array([0,0,0,0]) # Don't need this intercept because eye data is already centered

                    # Update eye coefficients
                    self.eye_coeff = np.vstack((slopes, intercepts)).T

    def _cycle(self):
        self._update_eye_pos()

        # Do calibration
        ave_pos = self.eye_pos
        if len(self.eye_pos) > 2 and len(self.eye_center) > 2:
            calibrated_pos = aopy.postproc.get_calibrated_eye_data(self.eye_pos[:4]-self.eye_center, self.eye_coeff)
            ave_pos = np.array([(calibrated_pos[0] + calibrated_pos[2])/2, (calibrated_pos[1] + calibrated_pos[3])/2])
        elif not self.keyboard_control:
            ave_pos = aopy.postproc.get_calibrated_eye_data(self.eye_pos-self.eye_center[:2], self.eye_coeff[:2])
        
        # Save calibration
        self.calibrated_eye_pos = ave_pos
        self.task_data['calibrated_eye'] = ave_pos

        super()._cycle()

        # Move the eye cursor
        if np.any(np.isnan(self.calibrated_eye_pos)):
            pass
        else:
            self.eye_cursor.move_to_position([self.calibrated_eye_pos[0],0,self.calibrated_eye_pos[1]])
            if self.show_eye_pos:
                self.eye_cursor.show()
    
    def cleanup_hdf(self):
        super().cleanup_hdf()
        if hasattr(self, "h5file"):
            h5file = tables.open_file(self.h5file.name, mode='a')
            h5file.root.task.attrs['eye_coeff'] = self.eye_coeff
            h5file.root.task.attrs['eye_center'] = self.eye_center
            h5file.close()

class EyeStreaming(traits.HasTraits):
    '''
    Adds eye_data streamed from oculomatic.
    '''

    keyboard_control = traits.Bool(False, desc="Whether to replace eye control with keyboard control")
    binocular = traits.Bool(True, desc="Whether to stream binocular eye data (4D) or just left eye data (2D)")
    eye_pixels_per_cm = traits.Float(51.67, desc="Conversion from eye diameter to cm")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Visualize eye positions
        if self.keyboard_control:
            self.eye_data = Eye([0,0])
            self.eye_pos = np.zeros((2,))*np.nan
            self.eye_diam = np.zeros((2,))*np.nan
        else:
            from riglib import source
            from riglib.oculomatic import System
            self.eye_data = source.DataSource(System)
            from riglib import sink
            sink_manager = sink.SinkManager.get_instance()
            sink_manager.register(self.eye_data) # register to the sink so it can save data
            self.eye_pos = np.zeros((4,))*np.nan if self.binocular else np.zeros((2,))*np.nan
            self.eye_diam = np.zeros((2,))*np.nan

    def init(self):
        if self.keyboard_control:
            self.add_dtype('eye', 'f8', (2,))
            self.add_dtype('eye_diam', 'f8', (1,))
        else:
            self.add_dtype('eye', 'f8', (4,)) if self.binocular else self.add_dtype('eye', 'f8', (2,))
            self.add_dtype('eye_diam', 'f8', (2,)) if self.binocular else self.add_dtype('eye_diam', 'f8', (1,))
        super().init()

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the motiondata source and stops it after the FSM has finished running
        '''
        if not self.keyboard_control:
            self.eye_data.start()
        try:
            super().run()
        finally:
            if not self.keyboard_control:
                print("Stopping streaming eye data")
                self.eye_data.stop()

    def _update_eye_pos(self):
        if not self.keyboard_control:
            eye_data = self.eye_data.get(all=True) # This is (n,4) array of new values since we last checked
            if eye_data.ndim < 2 or eye_data.size == 0:
                eye_pos = np.zeros((4,))*np.nan if self.binocular else np.zeros((2,))*np.nan
                eye_diam = np.zeros((2,))*np.nan if self.binocular else np.zeros((1,))*np.nan
            else:
                eye_pos = eye_data[-1,:4] if self.binocular else eye_pos[-1,:2] # the most recent position
                eye_diam = eye_data[-1,4:6] if self.binocular else eye_pos[-1,4:5]
            eye_diam = np.array(eye_diam)/self.eye_pixels_per_cm
        else:
            eye_pos = self.eye_data.get() # A list of lists of of x,y keyboard pos
            eye_pos = eye_pos[0]
            eye_diam = 0
        self.eye_pos = eye_pos
        self.eye_diam = eye_diam
        self.task_data['eye'] = eye_pos
        self.task_data['eye_diam'] = eye_diam

    def _cycle(self):
        if not hasattr(self, 'calibrated_eye_pos'):
            self._update_eye_pos()
        super()._cycle()

class EyeConstrained(ScreenTargetCapture):
    '''
    Add a penalty state when subjects looks away. Only tested in center-out task.
    '''

    fixation_dist = traits.Float(2.5, desc="Distance from center that is considered a broken fixation")
    fixation_penalty_time = traits.Float(0., desc="Time in fixation penalty state")
    fixation_target_color = traits.OptionsList("cyan", *target_colors, desc="Color of the center target under fixation state", bmi3d_input_options=list(target_colors.keys()))
    
    status = dict(
        wait = dict(start_trial="target", start_pause="pause"),
        target = dict(timeout="timeout_penalty",gaze_target="fixation", start_pause="pause"),
        fixation = dict(enter_target="hold", fixation_break="target", start_pause="pause"),
        hold = dict(leave_target="hold_penalty", hold_complete="delay", fixation_break="fixation_penalty", start_pause="pause"),
        delay = dict(leave_target="delay_penalty", delay_complete="targ_transition", fixation_break="fixation_penalty", start_pause="pause"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target", start_pause="pause"),
        timeout_penalty = dict(timeout_penalty_end="wait", start_pause="pause", end_state=True),
        hold_penalty = dict(hold_penalty_end="wait", start_pause="pause", end_state=True),
        delay_penalty = dict(delay_penalty_end="wait", start_pause="pause", end_state=True),
        fixation_penalty = dict(fixation_penalty_end="wait", start_pause="pause", end_state=True),
        reward = dict(reward_end="wait", start_pause="pause", stoppable=False, end_state=True),
        pause = dict(end_pause="wait", end_state=True),
    )
 
    def _test_gaze_target(self,ts):
        '''
        Check whether eye positions are within the fixation distance
        Only apply this to the first target (1st target)
        '''
        if self.target_index <= 0:     
            d = np.linalg.norm(self.calibrated_eye_pos)
            return d < self.fixation_dist
        else:
            return True
        
    def _test_fixation_break(self,ts):
        '''
        Triggers the fixation_penalty state when eye positions are outside fixation distance
        Only apply this to the first hold and delay period
        '''
        if self.target_index <= 0:   
            d = np.linalg.norm(self.calibrated_eye_pos)
            return (d > self.fixation_dist)
        
    def _test_fixation_penalty_end(self,ts):
        # d = np.linalg.norm(self.calibrated_eye_pos)
        return (ts > self.fixation_penalty_time) # (d < self.fixation_dist) and 
    
    def _start_wait(self):
        super()._start_wait()
        self.num_fixation_state = 0 # Initializa fixation state

    def _start_target(self):
        if self.num_fixation_state == 0:
            super()._start_target() # target index shouldn't be incremented after fixation break loop
        else:
            self.sync_event('FIXATION', 0)
            self.targets[0].reset() # reset target color after fixation break

    def _start_fixation(self):
        self.num_fixation_state = 1
        self.targets[0].sphere.color = target_colors[self.fixation_target_color] # change target color in fixation state
        if self.target_index == 0:
            self.sync_event('FIXATION', 1)
    
    def _start_timeout_penalty(self):
        super()._start_timeout_penalty()
        self.num_fixation_state = 0

    def _start_hold(self):
        super()._start_hold()
        self.num_fixation_state = 0 # because target state comes again after hold state in a trial

    def _start_fixation_penalty(self):
        if hasattr(super(), '_start_fixation_penalty'):
            super()._start_fixation_penalty()

        self._increment_tries()
        self.sync_event('FIXATION_PENALTY') 
        self.penalty_index = 1
        self.num_fixation_state = 0
        
        # Hide targets
        for target in self.targets:
            target.hide()
            target.reset()

    def _end_fixation_penalty(self):
        self.sync_event('TRIAL_END')

def _latest_value(buffer_data):
    if np.ndim(buffer_data) == 1:
        buffer_data = buffer_data[:,None]
    not_nan = np.all(~np.isnan(buffer_data), axis=1)
    if np.any(not_nan):
        return buffer_data[not_nan][-1]
    else:
        return buffer_data[-1]
  
class PupilLabStreaming(EyeStreaming):
    '''
    Adds eye_data from pupil labs. Optionally displays AprilTag markers on the screen for
    surface tracking. Requires a task with the Window feature enabled.
    '''

    surface_marker_size = traits.Float(6., desc="Size in cm of apriltag surface markers")
    surface_marker_count = traits.Int(0, desc="How many surface markers to draw")
    pupillabs_gaze = traits.OptionsList(gaze_options, desc="Which gaze option to use for eye position", bmi3d_options_list=gaze_options)
    pupillabs_confidence_threshold = traits.Float(0.5, desc="Minimum confidence for gaze position to be used")
    pupillabs_debug = traits.Bool(False, desc="Show debug info about eye streaming")

    exclude_parent_traits = ['binocular']
    hidden_traits = ['pupillabs_confidence_threshold', 'pupillabs_debug']

    def __init__(self, *args, **kwargs):
        if self.keyboard_control:
            super().__init__(*args, **kwargs)
            return
        
        super(EyeStreaming, self).__init__(*args, **kwargs)
    
        # Add apriltag models
        centers = utils.calculate_square_positions(self.screen_half_height, self.window_size, 
                                                   self.surface_marker_count, self.surface_marker_size)
        for id, (x, z) in enumerate(centers):
            tag = AprilTag(id, self.surface_marker_size).translate(x, 0, z)
            self.add_model(tag)

        # Visualize eye positions
        from riglib import source
        from riglib.pupillabs import System, NoSurfaceTracking
        if self.surface_marker_count > 0:
            self.eye_data = source.DataSource(System)
        else:
            self.eye_data = source.DataSource(NoSurfaceTracking)
        
        # Register to the sink
        from riglib import sink
        sink_manager = sink.SinkManager.get_instance()
        sink_manager.register(self.eye_data)

    def init(self):
        if self.keyboard_control:
            super().init()
            return
    
        eye_idx = gaze_options_idx[gaze_options.index(self.pupillabs_gaze)]
        self.eye_pos = np.zeros((len(eye_idx),))*np.nan
        self.eye_diam = np.zeros((2,))*np.nan
        self.add_dtype('eye', 'f8', (len(eye_idx),))
        self.add_dtype('eye_diam', 'f8', (2,))
        super(EyeStreaming, self).init()

    def _update_eye_pos(self):
        '''
        Set self.eye_pos and self.eye_diam to the most recent non-empty eye positions and diameters.
        '''
        if self.keyboard_control:
            super()._update_eye_pos()
            return

        eye = self.eye_data.get(all=True)
        if eye.ndim < 2 or eye.size == 0:
            return
        
        eye_idx = gaze_options_idx[gaze_options.index(self.pupillabs_gaze)]
        eye_pos = eye[:, eye_idx] # get only the columns corresponding to the selected gaze option
        eye_pos_confidence = eye[:, np.where(np.isin(eye_labels, ['gaze_confidence']))[0][0]] # get gaze confidence columns
        eye_diam = eye[:, np.where(np.isin(eye_labels, ['le_diam', 're_diam']))[0]] # get diameter columns
        eye_diam_confidence = eye[:, np.where(np.isin(eye_labels, ['le_diam_confidence', 're_diam_confidence']))[0][0]] # get diameter confidence columns

        # Find the last non-nan value of eye position
        eye_pos = _latest_value(eye_pos[eye_pos_confidence > self.pupillabs_confidence_threshold]) # only consider positions with high confidence
        eye_diam = _latest_value(eye_diam[eye_diam_confidence > self.pupillabs_confidence_threshold]) / self.eye_pixels_per_cm

        # Prepare the gaze position depending on its source
        if self.pupillabs_gaze == 'gaze3d':
            eye_pos = self.convert_gaze3d_to_screen(eye_pos)[:2]
        elif self.pupillabs_gaze == 'gaze2d':
            eye_pos = self.convert_gaze2d_to_screen(eye_pos)[:2]

        self.eye_pos = eye_pos
        self.eye_diam = eye_diam
        self.task_data['eye'] = eye_pos
        self.task_data['eye_diam'] = eye_diam

        if self.pupillabs_debug:
            if hasattr(self, 'debug_text'):
                self.remove_model(self.debug_text.model)
                self.debug_text.model.release()
            screen_half_width = self.screen_half_height * self.window_size[0] / self.window_size[1]
            gaze_ts = eye[:, np.where(np.isin(eye_labels, ['gaze_timestamp']))[0]]
            gaze_conf = eye[:, np.where(np.isin(eye_labels, ['gaze_confidence']))[0]]
            eye_conf = eye[:, np.where(np.isin(eye_labels, ['le_diam_confidence', 're_diam_confidence']))[0]]
            gaze_ts = _latest_value(gaze_ts)[0]
            gaze_conf = _latest_value(gaze_conf)[0]
            eye_conf = _latest_value(eye_conf)
            self.reportstats['eye pos'] = f"({eye_pos[0]:0.2f}, {eye_pos[1]:0.2f})"
            self.reportstats['gaze_ts'] = f"{gaze_ts:0.2f}"
            self.reportstats['gaze_conf'] = f"{gaze_conf:0.2f}"
            self.reportstats['le_conf'] = f"{eye_conf[0]:0.2f}"
            self.reportstats['re_conf'] = f"{eye_conf[1]:0.2f}"
            # text = f"LE: {eye_conf[0]:0.2f}"
            # self.debug_text = TextTarget(text, color=(1,1,1,1), height=2, 
            #                              starting_pos=(-screen_half_width+1, 0, 0))
            # self.add_model(self.debug_text.model)

    def _end_reward(self):
        if hasattr(super(), '_end_reward'):
            super()._end_reward()

    def convert_gaze2d_to_screen(self, gaze2d, z=0):
        '''
        Convert from [0,1] norm_pos to cm with center at (0,0)
        '''
        x, y = (gaze2d - 0.5) * self.screen_half_height * 2
        aspect_ratio = self.window_size[0] / self.window_size[1]
        x *= aspect_ratio
        xyz = np.array([x, y, z, 1])  # Convert to x, z, y format
        return xyz[:3]
        modelview = self.modelview.copy()
        modelview[0,3] = 0  # Set x translation to 0
        # modelview[3,3] *= -1  # Invert z-axis translation
        # xyz = modelview @ xyz  # Apply modelview transformation
        return xyz[:3]

    def convert_gaze3d_to_screen(self, gaze3d):
        '''
        Convert from gaze3d (vector from eye to gaze point in cm) to 
        screen coordinates in cm with center at (0,0).
        '''
        x, y, z = (gaze3d)
        y = -y  # Invert y-axis
        xyz = np.array([x, z, y])  # Convert to x, z, y format
        cylinder_start = np.array(self.camera_position)[[0, 2, 1]]
        cylinder_start[0] *= -1
        cylinder_start[2] *= -1
        w, i, j, k = self.camera_orientation
        camera_rotation = Quaternion(w, i, j, k).to_mat() # 4,4
        rot = np.array([[1, 0, 0, 0],
               [0, 0, 1, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 1]])
        camera_rotation = rot @ camera_rotation @ rot.T  # Apply the rotation to swap y and z axes
        eye_pos = cylinder_start + np.dot(camera_rotation[:3,:3], xyz/10)
        return eye_pos[[0,2,1]] # convert back to x,y,z format

    def cleanup_hdf(self):
        super().cleanup_hdf()
        if hasattr(self, "h5file"):
            h5file = tables.open_file(self.h5file.name, mode='a')
            h5file.root.pupillabs.attrs['eye_labels'] = eye_labels
            h5file.close()

class EyeCursor(traits.HasTraits):
    '''
    Adds a virtual eye cursor to the task, which can be used to visualize eye positions.
    '''
    eye_cursor_color = traits.OptionsList("green", *target_colors, desc="Color of the eye cursor", bmi3d_input_options=list(target_colors.keys()))
    eye_cursor_radius = traits.Float(0.5, desc="Radius of the eye cursor in cm")

    def init(self):
        super().init()
        self.eye_plant = plants.CursorPlant()
        self.eye_plant.set_color(target_colors[self.eye_cursor_color])
        self.eye_plant.set_cursor_radius(self.eye_cursor_radius)
        self.eye_plant.set_endpoint_pos(np.array(self.starting_pos))
        for model in self.eye_plant.graphics_models:
            self.add_model(model)

    def _cycle(self):
        super()._cycle()

        if hasattr(self, 'calibrated_eye_pos') and not np.any(np.isnan(self.calibrated_eye_pos)):
            eye = self.calibrated_eye_pos
        elif hasattr(self, 'eye_pos') and not np.any(np.isnan(self.eye_pos)):
            eye = self.eye_pos
        else:
            return
        self.eye_plant.set_endpoint_pos(np.array([eye[0], 0, eye[1]]))

'''
Old code not currently used in aolab
'''
class EyeData(traits.HasTraits):
    '''
    Pulls data from the eyetracking system and make it available on self.eyedata
    '''
    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up the 'eyedata' DataSource and registers it with the 
        SinkRegister so that the data gets saved to file as it is collected.
        '''
        from riglib import source
        from riglib import sink
        sink_manager = sink.SinkManager.get_instance()

        src, ekw = self.eye_source
        #f = open('/home/helene/code/bmi3d/log/eyetracker', 'a')
        self.eyedata = source.DataSource(src, **ekw)
        sink_manager.register(self.eyedata)
        f.write('instantiated source\n')
        super(EyeData, self).init()
        #f.close()
    
    @property
    def eye_source(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        from riglib import eyetracker
        return eyetracker.System, dict()

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the 'eyedata' source and stops it after the FSM has finished running
        '''
        #f = open('/home/helene/code/bmi3d/log/eyetracker', 'a')
        self.eyedata.start()
        #f.write('started eyedata\n')
        #f.close()
        try:
            super(EyeData, self).run()
        finally:
            self.eyedata.stop()
    
    def join(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.eyedata.join()
        super(EyeData, self).join()
    
    def _start_None(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.eyedata.pause()
        self.eyefile = tempfile.mktemp()
        print("retrieving data from eyetracker...")
        self.eyedata.retrieve(self.eyefile)
        print("Done!")
        self.eyedata.stop()
        super(EyeData, self)._start_None()
    
    def set_state(self, state, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.eyedata.sendMsg(state)
        super(EyeData, self).set_state(state, **kwargs)

    def cleanup(self, database, saveid, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        
        super(EyeData, self).cleanup(database, saveid, **kwargs)
        dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
        if dbname == 'default':
            database.save_data(self.eyefile, "eyetracker", saveid)
        else:
            database.save_data(self.eyefile, "eyetracker", saveid, dbname=dbname)

class SimulatedEyeData(EyeData):
    '''Simulate an eyetracking system using a series of fixations, with saccades interpolated'''
    fixations = traits.Array(value=[(0,0), (-0.6,0.3), (0.6,0.3)], desc="Location of fixation points")
    fixation_len = traits.Float(0.5, desc="Length of a fixation")

    @property
    def eye_source(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        from riglib import eyetracker
        return eyetracker.Simulate, dict(fixations= self.fixations)

    def _cycle(self):
        '''
        Docstring
        basically, extract the data and do something with it


        Parameters
        ----------

        Returns
        -------
        '''
        #retrieve data
        data_temp = self.eyedata.get()

        #send the data to sinks
        if data_temp is not None:
            self.sinks.send(self.eyedata.name, data_temp)

        super(SimulatedEyeData, self)._cycle()

class CalibratedEyeData(EyeData):
    '''Filters eyetracking data with a calibration profile'''
    cal_profile = traits.Instance(calibrations.EyeProfile)

    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(CalibratedEyeData, self).__init__(*args, **kwargs)
        self.eyedata.set_filter(self.cal_profile)

class FixationStart(CalibratedEyeData):
    '''Triggers the start_trial event whenever fixation exceeds *fixation_length*'''
    fixation_length = traits.Float(2., desc="Length of fixation required to start the task")
    fixation_dist = traits.Float(50., desc="Distance from center that is considered a broken fixation")

    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(FixationStart, self).__init__(*args, **kwargs)
        self.status['wait']['fixation_break'] = "wait"
        self.log_exclude.add(("wait", "fixation_break"))
    
    def _start_wait(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.eyedata.get()
        super(FixationStart, self)._start_wait()

    def _test_fixation_break(self, ts):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        return (np.sqrt((self.eyedata.get()**2).sum(1)) > self.fixation_dist).any()
    
    def _test_start_trial(self, ts):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        return ts > self.fixation_length
