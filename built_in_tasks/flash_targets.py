'''
Tasks which control a plant under pure machine control. Used typically for initializing BMI decoder parameters.
'''
import numpy as np
import os
import tables
import time
import subprocess
import tempfile
from OpenGL.GL import GL_RGB, GL_RGB8, GL_UNSIGNED_BYTE
import pygame

from riglib.experiment import traits, Experiment
from riglib.bmi.state_space_models import StateSpaceEndptVel2D
from riglib.bmi.bmi import Decoder, BMILoop, MachineOnlyFilter
from riglib.bmi.extractor import DummyExtractor
from riglib.stereo_opengl.primitives import TexPlane
from riglib.stereo_opengl.textures import Texture
from riglib.stereo_opengl.window import Window, WindowDispl2D, Window2D
from config.rig_defaults import window as window_defaults

from built_in_tasks.manualcontrolmultitasks import ScreenTargetCapture
from built_in_tasks.bmimultitasks import BMIControlMulti
from built_in_tasks.target_capture_task_eye import ScreenTargetCapture_Saccade

from .target_graphics import *

from .bmimultitasks import BMIControlMultiEyeConstrained

class FlashTargets(ScreenTargetCapture_Saccade):
    "This task recquires a central fixation and then shows different peripheral targets. "

    #Step 1: Make the center eye target come on, hold, reward
        #Don't use a generator to make the center target, it's always at the center!
    #Step 2: Add peripheral target flashes
        #Use the out_2d generator to make targets
    #Step 3: Make blink_time_threshold logic into a feature so taht we can use it for multiple tasks

    #Recquired Features
        #Autostart selected 

    blink_time_threshold = traits.Float(0.1, desc="The amount of time in seconds that " \
    "the eyes can be closed before triggering a fixation break, measured by eye_diam=0")
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.most_recent_open_eye = 0
        #This is the variable that sets the total flash on and flash off time
        self.target_flash_time_s = 0.2
        #this is the variable that sets the buffer windown at the beginning and end of the hold period where no peripheral target will be shown
        self.flash_buffer_time_s = self.target_flash_time_s
        #Here we calculate the total number of flash targets that can be shown in a single trial
        self.total_flashes = np.floor((self.hold_time - 2*(self.flash_buffer_time_s))/(2*self.target_flash_time_s))
        
        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)

        self.peripheral_target_radius = traits.Float(2, desc="Radius of targets in cm")
        self.peripheral_target_color = traits.OptionsList("yellow", *target_colors, desc="Color of the target", bmi3d_input_options=list(target_colors.keys()))

        if instantiate_targets:

            # Control transparency of targets
            new_color1 = list(target_colors[self.target_color])
            new_color1[3] = self.init_eye_target_alpha
            new_color2 = list(target_colors[self.target_color])
            new_color2[3] = self.goal_eye_target_alpha

            # 2 targets for delay
            #Target 1 is the central fixation target we are using as the eye target
            target1 = VirtualRectangularTarget(target_width=self.target_radius, target_height=self.target_radius/2, target_color=new_color1)
            #Target 2 will be the peripheral 'flash' target; needs to be modified to regular center out target
            target2 = VirtualCircularTarget(target_radius=self.peripheral_target_radius, target_color=target_colors[self.peripheral_target_color])

            self.targets = [target1, target2]

            self.offset_cube = np.array([0,0,self.target_radius/2]) # To center the cube target
            self.center_target_index = 0
            self.center_target_position = np.array([0,0,0] )

    
    

    status = dict(
        wait = dict(start_trial="target", start_pause="pause"),
        target = dict(enter_target="hold_start_buffer", start_pause="pause"),
        hold_start_buffer = dict(leave_target="hold_penalty", start_flash_cycle='hold_flash_on', start_pause="pause"),
        hold_flash_on = dict(leave_target="hold_penalty", flash_complete="hold_flash_off", flash_cycle_complete='hold_end_buffer', start_pause="pause"),
        hold_flash_off = dict(leave_target="hold_penalty", flash_interval_complete = "hold_flash_on", flash_cycle_complete='hold_end_buffer', start_pause="pause"),
        hold_end_buffer = dict(leave_target="hold_penalty", hold_complete='reward', start_pause="pause"),
        hold_penalty = dict(hold_penalty_end="wait", start_pause="pause", end_state=True),
        reward = dict(reward_end="wait", start_pause="pause", stoppable=False, end_state=True),
        pause = dict(end_pause="wait", end_state=True),
    )

    def _test_hold_complete(self, time_in_state):
        time_since_hold_start = self.get_time() - self.most_recent_hold_start_time
        return self.hold_time<=(time_since_hold_start)
    
    def _start_hold_flash_off(self):
        #Turn off the target
        #Which actual animatino blob do I hide?
        target = self.targets[1]
        #Show the target
        target.hide()

    def _test_flash_interval_complete(self, time_in_state):
        #wait until the inter flash interval has completed before turning the flash back on
        return time_in_state>=self.target_flash_time_s

    def _test_flash_cycle_complete(self, time_in_state):
        #make sure we have not encroached on the end hold buffer time
        time_since_hold_start = self.get_time() - self.most_recent_hold_start_time
        return self.hold_time<=(time_since_hold_start + self.flash_buffer_time_s)

    def _test_start_flash_cycle(self, time_in_state):
        #Test to see if the buffer period at the beginningof the hold has been completed so that we can start showing peripheral targets
        return time_in_state > self.flash_buffer_time_s
    
    def _test_flash_complete(self, time_in_state):
        return time_in_state>self.target_flash_time_s

    def _start_hold_start_buffer(self):
        #Save the start time of the hold for future reference and checking
        self.most_recent_hold_start_time = self.get_time()

    def _start_hold_flash_on(self):
        #We want to grab the first peripheral target and turn it on
        #Which actual animatino blob do I move? The second one
        target = self.targets[1]

        #Incrementing to the next target in the generator
        self.target_index += 1
        #move the animation object to the correct location
        print(f'The peripheral target object is {target}')
        print(f'The target index is {self.target_index}')
        print(f'The shape of targs is {self.targs}')
        target_position = self.targs[self.target_index] - self.offset_cube
        print(f'The peripheral target position {target_position}')
        target.move_to_position(target_position)

        #Show the target
        target.show()
        #Sync to ecube
        self.sync_event('TARGET_ON', self.gen_indices[1])
        #Save the target location
        self.target_location = self.targs[self.target_index]

    def _start_target(self):
        #self.target_index += 1 #I'm not sure if this makes sense to increment here; may

        # Show target if it is hidden (this is the first target, or previous state was a penalty)
        #UPdate!
        target = self.targets[self.center_target_index]
        #if self.target_index == 0:
        target.move_to_position(self.center_target_position - self.offset_cube) #Check the formating on these indicies; self.targs[self.target_index] - self.offset_cube)
        target.show()
        self.sync_event('TARGET_ON', self.center_target_index) #Double check that 0 corresponds correctly to the center target
        self.target_location = self.center_target_position # save for BMILoop

    def _test_enter_target(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        eye_pos = self.calibrated_eye_pos
        target_pos = self.center_target_position[0:2] #Only grab two values from the target position to match the shape of the eye position
        d_eye = np.linalg.norm(eye_pos - target_pos)
        return (d_eye <= self.target_radius + self.fixation_radius_buffer)

    def _test_leave_target(self, ts):
        eye_pos = self.calibrated_eye_pos
        target_pos = self.center_target_position[0:2] #Only grab two values from the target position to match the shape of the eye position
        d_eye = np.linalg.norm(eye_pos - target_pos)
        return not (d_eye <= self.target_radius + self.fixation_radius_buffer)