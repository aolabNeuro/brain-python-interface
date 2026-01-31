'''
Target capture tasks with eye position requirement
'''
import numpy as np
import random
import os

from .target_graphics import *
from .target_capture_task import ScreenTargetCapture
from riglib.experiment import traits

class EyeConstrainedTargetCapture(ScreenTargetCapture):
    '''
    Fixation requirement is added before go cue
    '''

    fixation_penalty_time = traits.Float(0., desc="Time in fixation penalty state")
    fixation_target_color = traits.OptionsList("fixation_color", *target_colors, desc="Color of the center target under fixation state", bmi3d_input_options=list(target_colors.keys()))
    eye_target_color = traits.OptionsList("eye_color", *target_colors, desc="Color of the center target under fixation state", bmi3d_input_options=list(target_colors.keys()))
    fixation_radius_buffer = traits.Float(.5, desc="additional radius for eye target")
    hand_target_color = traits.OptionsList("yellow", *target_colors, desc="Color for the hand-only target", bmi3d_input_options=list(target_colors.keys()))
    fixation_radius = traits.Float(.5, desc="additional radius for eye target")

    status = dict(
        wait = dict(start_trial="target", start_pause="pause"),
        target = dict(timeout="timeout_penalty", gaze_enter_target="hold", start_pause="pause"),
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:

            target3 = VirtualRectangularTarget(target_width=self.fixation_radius, target_height=self.fixation_radius/2, target_color=target_colors[self.eye_target_color])
            self.targets_eye = [target3]
            self.offset_cube = np.array([0,10,self.fixation_radius/2]) # To center the cube target

    def _test_gaze_enter_target(self,ts):
        '''
        Check whether eye positions and hand cursor are within the target radius
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.targs[self.target_index,[0,2]])

        cursor_pos = self.plant.get_endpoint_pos()
        hand_d = np.linalg.norm(cursor_pos - self.targs[self.target_index])
        
        # Fixation requirement is only for the center target
        if self.target_index == 0:
            return (eye_d <= self.target_radius + self.fixation_radius_buffer) and (hand_d <= self.target_radius - self.cursor_radius)
        else:
            return hand_d <= self.target_radius - self.cursor_radius
        
    def _test_fixation_break(self,ts):
        '''
        Triggers the fixation_penalty state when eye positions are outside fixation distance
        Only apply this to the first hold and delay period
        '''
        if self.target_index <= 0:   
            eye_d = np.linalg.norm(self.calibrated_eye_pos)
            return (eye_d > self.target_radius + self.fixation_radius_buffer)
    
    def _test_fixation_penalty_end(self,ts):
        return (ts > self.fixation_penalty_time) 

    def _start_wait(self):
        super()._start_wait()

        if self.calc_trial_num() == 0:

            # Instantiate the targets here so they don't show up in any states that might come before "wait"
            for target in self.targets_eye:
                for model in target.graphics_models:
                    self.add_model(model)
                    target.hide()

    def _start_target(self):
        super()._start_target()

        if self.target_index == 0:
            self.targets_eye[0].move_to_position(self.targs[self.target_index] - self.offset_cube)
            self.targets_eye[0].show()

    def _while_target(self):
        if self.target_index == 0:
            eye_pos = self.calibrated_eye_pos
            eye_d = np.linalg.norm(eye_pos - self.targs[self.target_index,[0,2]])
            if eye_d <= (self.target_radius + self.fixation_radius_buffer):
                self.targets_eye[0].cube.color = target_colors[self.fixation_target_color] # chnage color in fixating center
            else:
                self.targets_eye[0].reset()

    def _start_delay(self):
        next_idx = (self.target_index + 1)
        if next_idx < self.chain_length: # This is for hand target in the second delay
            self.targets[next_idx].move_to_position(self.targs[next_idx])
            self.targets[next_idx].sphere.color = target_colors[self.hand_target_color]
            self.targets[next_idx].show()
            self.sync_event('TARGET_ON', self.gen_indices[next_idx])

    def _start_targ_transition(self):
        super()._start_targ_transition()

        if self.target_index + 1 < self.chain_length:
            # Hide the current target if there are more
            self.targets_eye[0].hide()

    def _start_hold_penalty(self):
        super()._start_hold_penalty()
        self.targets_eye[0].hide()
        self.targets_eye[0].reset()

    def _start_delay_penalty(self):
        super()._start_delay_penalty()
        self.targets_eye[0].hide()
        self.targets_eye[0].reset()

    def _start_timeout_penalty(self):
        super()._start_timeout_penalty()
        self.targets_eye[0].hide()
        self.targets_eye[0].reset()

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

        self.targets_eye[0].hide()
        self.targets_eye[0].reset()

    def _end_fixation_penalty(self):
        self.sync_event('TRIAL_END')

class HandConstrainedEyeCapture(ScreenTargetCapture):
    '''
    Saccade task with holding another target with hand. Subjects need to hold an initial target with their hand. 
    Then they need to fixate the first eye target and make a saccade for the second eye target. 2 of chain_length is only tested.
    '''

    fixation_radius = traits.Float(2.5, desc="Distance from center that is considered a broken fixation")
    fixation_penalty_time = traits.Float(1.0, desc="Time in fixation penalty state")
    fixation_target_color = traits.OptionsList("fixation_color", *target_colors, desc="Color of the eye target under fixation state", bmi3d_input_options=list(target_colors.keys()))
    eye_target_color = traits.OptionsList("eye_color", *target_colors, desc="Color of the eye target", bmi3d_input_options=list(target_colors.keys()))
    fixation_radius_buffer = traits.Float(.5, desc="additional radius for eye target")
    fixation_time = traits.Float(.2, desc="additional radius for eye target")
    incorrect_target_radius_buffer = traits.Float(.5, desc="additional radius for eye target")
    incorrect_target_penalty_time = traits.Float(1, desc="Length of penalty time for acquiring an incorrect target")
    exclude_parent_traits = ['hold_time']

    status = dict(
        wait = dict(start_trial="init_target", start_pause="pause"),
        init_target = dict(enter_target="target", start_pause="pause"),
        target = dict(start_pause="pause", timeout="timeout_penalty", return_init_target='init_target', gaze_enter_target="fixation"),
        target_eye = dict(start_pause="pause", timeout="timeout_penalty", leave_target='hold_penalty', gaze_target="fixation", gaze_incorrect_target="incorrect_target_penalty"),
        fixation = dict(start_pause="pause", leave_target="hold_penalty", fixation_complete="delay", fixation_break="fixation_penalty"), 
        delay = dict(leave_target="delay_penalty", delay_complete="targ_transition", fixation_break="fixation_penalty", start_pause="pause"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target_eye", start_pause="pause"),
        timeout_penalty = dict(timeout_penalty_end="wait", start_pause="pause", end_state=True),
        hold_penalty = dict(hold_penalty_end="wait", start_pause="pause", end_state=True),
        delay_penalty = dict(delay_penalty_end="wait", start_pause="pause", end_state=True),
        fixation_penalty = dict(fixation_penalty_end="wait", start_pause="pause", end_state=True),
        incorrect_target_penalty = dict(incorrect_target_penalty_end="wait", start_pause="pause", end_state=True),
        reward = dict(reward_end="wait", start_pause="pause", stoppable=False, end_state=True),
        pause = dict(end_pause="wait", end_state=True),
    )
 
    sequence_generators = ['row_target','sac_hand_2d']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:

            # Target 1 and 2 are for saccade. Target 3 is for hand
            target1 = VirtualRectangularTarget(target_width=self.fixation_radius, target_height=self.fixation_radius/2, target_color=target_colors[self.eye_target_color])
            target2 = VirtualRectangularTarget(target_width=self.fixation_radius, target_height=self.fixation_radius/2, target_color=target_colors[self.eye_target_color])
            target3 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])

            self.targets = [target1, target2]
            self.targets_hand = [target3]

            self.offset_cube = np.array([0,0,self.fixation_radius/2]) # To center the cube target
        
    def _parse_next_trial(self):
        '''Check that the generator has the required data'''
        # 2 target positions for hand and eye. The first and second target index is for eye, and the third one is for hand
        self.gen_indices, self.targs = self.next_trial 

        # Update the data sinks with trial information
        self.trial_record['trial'] = self.calc_trial_num()
        for i in range(len(self.gen_indices)):
            self.trial_record['index'] = self.gen_indices[i]
            self.trial_record['target'] = self.targs[i]
            self.sinks.send("trials", self.trial_record)

    def _test_gaze_enter_target(self,ts):
        '''
        Check whether eye positions and hand cursor are within the target radius
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.targs[self.target_index,[0,2]])

        cursor_pos = self.plant.get_endpoint_pos()
        hand_d = np.linalg.norm(cursor_pos - self.targs[-1])
        
        return (eye_d <= self.fixation_radius + self.fixation_radius_buffer) and (hand_d <= self.target_radius - self.cursor_radius)
    
    def _test_gaze_target(self, ts):
        '''
        Check whether eye position is within the target radius
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.targs[self.target_index,[0,2]])

        return eye_d <= self.fixation_radius + self.fixation_radius_buffer   
    
    def _test_gaze_incorrect_target(self, ts):
        '''
        Check whether eye position is within the different target (hand target)
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.targs[-1,[0,2]])

        return eye_d <= self.target_radius + self.incorrect_target_radius_buffer
    
    def _test_fixation_break(self,ts):
        '''
        Triggers the fixation_penalty state when eye positions are outside fixation distance
        '''
        # Distance of an eye position from a target position
        eye_pos = self.calibrated_eye_pos
        d_eye = np.linalg.norm(eye_pos - self.targs[self.target_index,[0,2]])
        return d_eye > self.fixation_radius + self.fixation_radius_buffer
    
    def _test_fixation_complete(self,ts):
        return ts > self.fixation_time
    
    def _test_fixation_penalty_end(self,ts):
        return ts > self.fixation_penalty_time
    
    def _test_enter_target(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.targs[-1]) # hand must be within the initial target
        return d <= self.target_radius - self.cursor_radius

    def _test_leave_target(self, ts):
        '''
        return true if cursor moves outside the exit radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.targs[-1]) # hand must be within the initial target
        return d > self.target_radius - self.cursor_radius

    def _test_return_init_target(self, ts):
        '''
        return true if cursor moves outside the exit radius, but only applied when the target index is 0.
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.targs[-1])
        return (d > self.target_radius - self.cursor_radius) and self.target_index == 0
    
    def _test_trial_incomplete(self, ts):
        return self.target_index < self.chain_length
    
    def _test_incorrect_target_penalty_end(self, ts):
        return ts > self.incorrect_target_penalty_time
    
    def _start_wait(self):
        super()._start_wait()
        # Redefine chain length because targs in this task has both eye and hand targets
        self.chain_length = len(self.targets)
        self.isfixation_state = False

        if self.calc_trial_num() == 0:

            # Instantiate the targets here so they don't show up in any states that might come before "wait"
            for target in self.targets:
                for model in target.graphics_models:
                    self.add_model(model)
                    target.hide()

            for target in self.targets_hand:
                for model in target.graphics_models:
                    self.add_model(model)
                    target.hide()

    def _start_init_target(self):
        # Only show the hand target
        if self.target_index == -1:
            target_hand = self.targets_hand[0]
            target_hand.move_to_position(self.targs[-1])
            target_hand.show()
            self.sync_event('TARGET_ON', self.gen_indices[-1]) # the hand target is on  

        elif self.target_index == 0: # this is from the target state
            target = self.targets[self.target_index]
            target.hide()
            self.sync_event('EYE_TARGET_OFF', self.gen_indices[self.target_index])

    def _start_target(self):
        if self.target_index == -1 and not self.isfixation_state:
            self.target_index += 1

        if self.isfixation_state:
            self.target_index += 1

        # Show the eye target
        if self.target_index == 0:
            target = self.targets[self.target_index]
            target.move_to_position(self.targs[self.target_index] - self.offset_cube)
            target.show()
            self.sync_event('EYE_TARGET_ON', self.gen_indices[self.target_index])

    def _start_target_eye(self):
        self.target_index += 1

    def _start_fixation(self):
        self.isfixation_state = True
        self.targets[self.target_index].cube.color = target_colors[self.fixation_target_color] # change target color in fixation state
        self.sync_event('FIXATION', self.gen_indices[self.target_index])

    def _start_delay(self):
        # Make next target visible unless this is the final target in the trial
        next_idx = (self.target_index + 1)
        if next_idx < self.chain_length:
            target = self.targets[next_idx]
            target.move_to_position(self.targs[next_idx] - self.offset_cube)
            target.show()
            self.sync_event('EYE_TARGET_ON', self.gen_indices[next_idx])

    def _start_targ_transition(self):
        if self.target_index + 1 < self.chain_length:

            # Hide the current target if there are more
            self.targets[self.target_index].hide()
            self.sync_event('EYE_TARGET_OFF', self.gen_indices[self.target_index])

    def _start_timeout_penalty(self):
        super()._start_timeout_penalty()
        for target in self.targets_hand:
            target.hide()
            target.reset()
            
    def _start_hold_penalty(self):
        super()._start_hold_penalty()
        # Hide targets
        for target in self.targets_hand:
            target.hide()
            target.reset()

    def _start_delay_penalty(self):
        super()._start_delay_penalty()
        # Hide targets
        for target in self.targets_hand:
            target.hide()
            target.reset()

    def _start_fixation_penalty(self):
        self._increment_tries()
        self.sync_event('FIXATION_PENALTY') 
        self.penalty_index = 1

        # Hide targets
        for target in self.targets:
            target.hide()
            target.reset()

        for target in self.targets_hand:
            target.hide()
            target.reset()

    def _end_fixation_penalty(self):
        self.sync_event('TRIAL_END')

    def _start_incorrect_target_penalty(self):
        self._increment_tries()
        self.sync_event('OTHER_PENALTY')
        self.penalty_index = 1

        # Hide targets
        for target in self.targets:
            target.hide()
            target.reset()

        for target in self.targets_hand:
            target.cue_trial_end_failure()
            target.show()

    def _end_incorrect_target_penalty(self):
        self.sync_event('TRIAL_END')

        for target in self.targets_hand:
            target.hide()
            target.reset()

    def _start_reward(self):
        super()._start_reward()
        for target in self.targets_hand:
            target.cue_trial_end_success()

    def _end_reward(self):
        super()._end_reward()

        # Hide targets
        for target in self.targets_hand:
            target.hide()
            target.reset()        

    def _start_pause(self):
        super()._start_pause()

        # Hide targets
        for target in self.targets_hand:
            target.hide()
            target.reset()

    # Generator functions
    @staticmethod
    def row_target(nblocks=20, ntargets=3, dx=5.,offset1=(0,0,-2),offset2=(0,0,6.),offset3=(0,0,-7.5),origin=(0,0,0),seed=0):
        '''
        Generates a sequence of 3D for 2 eye targets and 1 hand target at a given distance from the origin

        Parameters
        ----------
        nblocks : int
            The number of ntarget pairs in the sequence.
        ntargets : int
            The number of equally spaced targets
        distance : float
            The distance in cm between targets
        offset1 : 3-tuple
            y location of the first eye target
        offset2 : 3-tuple
            y location of the second eye target
        offset3 : 3-tuple
            y location of the hand target
        origin : 3-tuple
            Location of the central targets

        Returns
        -------
        [nblocks*ntargets x 1] array of tuples containing trial indices and [1 x 3] target coordinates

        '''
        rng = np.random.default_rng(seed=seed)
        for _ in range(nblocks):
            order = np.arange(ntargets**3)
            rng.shuffle(order)
            x_pos_candidate = [-dx,0,dx]
            for t in range(ntargets**3):
                idx = np.base_repr(order[t],3).zfill(3) # convert a decimal number to ternary

                # Target index for hand target, initial eye target, final eye target
                idx1 = int(idx[0])
                idx2 = int(idx[1])
                idx3 = int(idx[2])

                # Get positions for each target
                x_pos1 = x_pos_candidate[idx1]
                x_pos2 = x_pos_candidate[idx2]
                x_pos3 = x_pos_candidate[idx3]
                pos1 = np.array([x_pos1,0,0]).T
                pos2 = np.array([x_pos2,0,0]).T
                pos3 = np.array([x_pos3,0,0]).T

                yield [idx1],[idx2],[idx3],[pos1+offset1+origin],[pos2+offset2+origin],[pos3+offset3+origin]

    @staticmethod
    def sac_hand_2d(nblocks=20, ntargets=3, dx=10,offset1=(0,0,-2),offset2=(0,0,6.),offset3=(0,0,-7.5),origin=(0,0,0),seed=0):
        '''
        Pairs of hand targets and eye targets

        Returns
        -------
        [nblocks*ntargets x 1] array of tuples containing trial indices and [3 x 3] target coordinates
        '''

        gen = HandConstrainedEyeCapture.row_target(nblocks=nblocks,ntargets=ntargets,dx=dx,offset1=offset1,offset2=offset2,offset3=offset3,origin=origin,seed=seed)
        for _ in range(nblocks*(ntargets**3)):
            idx1,idx2,idx3,pos1,pos2,pos3 = next(gen)

            targs = np.zeros([3, 3])
            targs[0,:] = pos1[0]
            targs[1,:] = pos2[0]
            targs[2,:] = pos3[0]

            indices = np.zeros([3,1])
            indices[0] = idx1[0]
            indices[1] = idx2[0] + ntargets
            indices[2] = idx3[0]

            yield indices, targs

class EyeConstrainedHandCapture(HandConstrainedEyeCapture):

    status = dict(
        wait = dict(start_trial="init_target", start_pause="pause"),
        init_target = dict(enter_target="target", start_pause="pause"),
        target = dict(start_pause="pause", timeout="timeout_penalty", return_init_target='init_target', gaze_enter_target="fixation", gaze_incorrect_target="incorrect_target_penalty"),
        fixation = dict(start_pause="pause", leave_target="hold_penalty", fixation_hold_complete="delay", fixation_break="fixation_penalty"), 
        delay = dict(leave_target="delay_penalty", delay_complete="targ_transition", fixation_break="fixation_penalty", start_pause="pause"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target", start_pause="pause"),
        timeout_penalty = dict(timeout_penalty_end="wait", start_pause="pause", end_state=True),
        hold_penalty = dict(hold_penalty_end="wait", start_pause="pause", end_state=True),
        delay_penalty = dict(delay_penalty_end="wait", start_pause="pause", end_state=True),
        incorrect_target_penalty = dict(incorrect_target_penalty_end="wait", start_pause="pause", end_state=True),
        fixation_penalty = dict(fixation_penalty_end="wait", start_pause="pause", end_state=True),
        reward = dict(reward_end="wait", start_pause="pause", stoppable=False, end_state=True),
        pause = dict(end_pause="wait", end_state=True),
    )
 
    sequence_generators = ['row_target','sac_hand_2d']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:

            # Target 1 and 2 are for saccade. Target 3 is for hand
            target1 = VirtualRectangularTarget(target_width=self.fixation_radius, target_height=self.fixation_radius/2, target_color=target_colors[self.eye_target_color])
            target2 = VirtualRectangularTarget(target_width=self.fixation_radius, target_height=self.fixation_radius/2, target_color=target_colors[self.eye_target_color])
            target3 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])
            target4 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])

            self.targets = [target1, target2]
            self.targets_hand = [target3, target4]

            self.offset_cube = np.array([0,10,self.fixation_radius/2]) # To center the cube target

    def _test_gaze_enter_target(self,ts):
        '''
        Check whether eye positions and hand cursor are within the target radius
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.targs[self.target_index,[0,2]])

        cursor_pos = self.plant.get_endpoint_pos()
        hand_d = np.linalg.norm(cursor_pos - self.targs[-1-self.target_index]) #targs[-1] is the first hand target, targ[-2] is the second target
        
        return (eye_d <= self.fixation_radius + self.fixation_radius_buffer) and (hand_d <= self.target_radius - self.cursor_radius)

    def _test_gaze_incorrect_target(self, ts):
        '''
        Check whether eye position is within the different target (hand target). This is only applied to the second target
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.targs[-1,[0,2]])

        return eye_d <= (self.target_radius + self.incorrect_target_radius_buffer) and self.target_index == 1
    
    def _test_leave_target(self, ts):
        '''
        return true if cursor moves outside the exit radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.targs[-1-self.target_index])
        return d > self.target_radius - self.cursor_radius
    
    def _test_fixation_hold_complete(self,ts):
        return ts > self.fixation_time
    
    def _while_target(self):
        target = self.targets[self.target_index]
        target.move_to_position(self.targs[self.target_index] - self.offset_cube)

        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.targs[self.target_index,[0,2]])
        
        if eye_d <= self.fixation_radius + self.fixation_radius_buffer:
            self.targets[self.target_index].cube.color = target_colors[self.fixation_target_color] # change target color in fixation state
        else:
            self.targets[self.target_index].cube.color = target_colors[self.eye_target_color]

    def _start_delay(self):
        next_idx = (self.target_index + 1)
        if next_idx < self.chain_length:
            # Show hand target
            target = self.targets_hand[next_idx]
            target.move_to_position(self.targs[next_idx])
            target.show() # Don't have to sync event because the second target is shared between hand and eye.

            # Show eye target
            target = self.targets[next_idx]
            target.move_to_position(self.targs[next_idx] - self.offset_cube)
            target.show()
            self.sync_event('EYE_TARGET_ON', self.gen_indices[next_idx])

    def _start_targ_transition(self):
        super()._start_targ_transition()
        if self.target_index + 1 < self.chain_length:

            # Hide the current hand target
            self.targets_hand[self.target_index].hide()

class EyeHandSequenceCapture(EyeConstrainedTargetCapture):
    '''
    Subjects have to gaze at and reach to a target, responding to the eye or hand go cue indivisually in sequence trials.
    They need to simultaneously move eye and hand to the target in simultaneous trials.
    '''

    exclude_parent_traits = ['delay_time', 'rand_delay']
    rand_delay1 = traits.Tuple((0.4, 0.7), desc="Delay interval for eye")
    rand_delay2 = traits.Tuple((0., 0.7), desc="Delay interval for hand")
    rand_fixation1 = traits.Tuple((0., 0.7), desc='Length of fixation required at targets')
    rand_fixation2 = traits.Tuple((0., 0.7), desc='Length of 2nd fixation required at targets')
    sequence_ratio = traits.Float(0.5, desc='Ratio of sequence trials')

    status = dict(
        wait = dict(start_trial="target", start_pause="pause"),
        target = dict(timeout="timeout_penalty", gaze_enter_target='fixation', start_pause="pause"),
        target_eye = dict(timeout="timeout_penalty", gaze_target="fixation", leave_target="hold_penalty", start_pause="pause"),
        target_hand = dict(timeout="timeout_penalty", enter_target="hold", fixation_break="fixation_penalty", start_pause="pause"),
        target_eye_hand = dict(timeout="timeout_penalty", gaze_enter_target='fixation', start_pause="pause"),
        fixation = dict(fixation_complete="delay", leave_target="hold_penalty", fixation_break="fixation_penalty", start_pause="pause"),
        hold = dict(hold_complete="delay", leave_target="hold_penalty",  fixation_break="fixation_penalty", start_pause="pause"),
        delay = dict(delay_complete="targ_transition", leave_target="delay_penalty", fixation_break="fixation_penalty", start_pause="pause"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", targ_simultaneous="target_eye_hand",\
                               targ_first_sequence="target_eye", targ_second_sequence="target_hand", start_pause="pause"),
        timeout_penalty = dict(timeout_penalty_end="wait", start_pause="pause", end_state=True),
        hold_penalty = dict(hold_penalty_end="wait", start_pause="pause", end_state=True),
        delay_penalty = dict(delay_penalty_end="wait", start_pause="pause", end_state=True),
        fixation_penalty = dict(fixation_penalty_end="wait", start_pause="pause", end_state=True),
        reward = dict(reward_end="wait", start_pause="pause", stoppable=False, end_state=True),
        pause = dict(end_pause="wait", end_state=True),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:

            # Target 1 and 2 are for eye targets. Target 3 and 4 is for hand targets
            target1 = VirtualRectangularTarget(target_width=self.fixation_radius, target_height=self.fixation_radius/2, target_color=target_colors[self.eye_target_color])
            target2 = VirtualRectangularTarget(target_width=self.fixation_radius, target_height=self.fixation_radius/2, target_color=target_colors[self.eye_target_color])
            target3 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])
            target4 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])

            self.targets_eye = [target1, target2]
            self.targets_hand = [target3, target4]

            self.offset_cube = np.array([0,10,self.fixation_radius/2]) # To center the cube target

    def init(self):
        self.add_dtype('is_sequence', bool, (1,))
        super().init()

    def _test_gaze_enter_target(self,ts):
        '''
        Check whether eye positions and hand cursor are within the target radius
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.eye_targs[self.eye_target_index,[0,2]])

        cursor_pos = self.plant.get_endpoint_pos()
        hand_d = np.linalg.norm(cursor_pos - self.hand_targs[self.hand_target_index])
        
        return (eye_d <= self.fixation_radius + self.fixation_radius_buffer) and (hand_d <= self.target_radius - self.cursor_radius)

    def _test_gaze_target(self,ts):
        '''
        Check whether eye positions are within the fixation distance
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.eye_targs[self.eye_target_index,[0,2]])
        return eye_d <= self.fixation_radius + self.fixation_radius_buffer

    def _test_leave_target(self, ts):
        '''
        check whether the hand cursor is outside the target distance
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        hand_d = np.linalg.norm(cursor_pos - self.hand_targs[self.hand_target_index])
        return hand_d > (self.target_radius - self.cursor_radius)    
        
    def _test_enter_target(self, ts):
        '''
        check whether the hand cursor is within the target distance
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        hand_d = np.linalg.norm(cursor_pos - self.hand_targs[self.hand_target_index])
        return hand_d <= (self.target_radius - self.cursor_radius)
    
    def _test_fixation_break(self,ts):
        '''
        Triggers the fixation_penalty state when eye positions are outside fixation distance
        Only apply this to the first hold and delay period
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.eye_targs[self.eye_target_index,[0,2]])
        return eye_d > (self.fixation_radius + self.fixation_radius_buffer)

    def _test_fixation_complete(self, ts):
        if self.eye_target_index == 0:
            return ts > self.fixation_time1
        else:
            return ts > self.fixation_time2
    
    def _test_hold_complete(self, ts):
        return ts > self.hold_time
    
    def _test_delay_complete(self, ts):
        '''
        Test whether the delay period, when the cursor or eye or both must stay in place
        while another target is being presented, is over. 
        '''
        if self.target_index == 0:
            return ts > self.delay_time1
        elif self.target_index == 1:
            return ts > self.delay_time2
        else:
            return True
    
    def _test_trial_complete(self, ts):
        return self.eye_target_index == 1 and self.hand_target_index == 1
    
    def _test_targ_simultaneous(self,ts):
        return self.eye_target_index == 0 and self.hand_target_index == 0 and self.is_simultaneous
    
    def _test_targ_first_sequence(self, ts):
        return self.eye_target_index == 0 and self.hand_target_index == 0 and self.is_sequence
    
    def _test_targ_second_sequence(self, ts):
        return self.eye_target_index == 1 and self.hand_target_index == 0 and self.is_sequence
    
    def _start_wait(self):
        super()._start_wait()

        # Initialize target index and target positons for eye and hand
        self.eye_target_index = -1
        self.hand_target_index = -1
        self.eye_targs = np.copy(self.targs)
        self.hand_targs = np.copy(self.targs)
        self.eye_gen_indices = np.copy(self.gen_indices)
        self.hand_gen_indices = np.copy(self.gen_indices)

        if self.calc_trial_num() == 0:

            # Instantiate the targets here so they don't show up in any states that might come before "wait"
            for target in self.targets_eye:
                for model in target.graphics_models:
                    self.add_model(model)
                    target.hide()

            for target in self.targets_hand:
                for model in target.graphics_models:
                    self.add_model(model)
                    target.hide()

        if self.tries == 0: # Update delay_time only in the first attempt

            # Set delay time
            s, e = self.rand_delay1
            self.delay_time1 = random.random()*(e-s) + s
            s, e = self.rand_delay2
            self.delay_time2 = random.random()*(e-s) + s
            s, e = self.rand_fixation1
            self.fixation_time1 = random.random()*(e-s) + s
            s, e = self.rand_fixation2
            self.fixation_time2 = random.random()*(e-s) + s

            # Decide sequence or simultaneous trials
            self.is_sequence = False
            self.is_simultaneous = False

            a = random.random()
            if a < self.sequence_ratio:
                self.is_sequence = True
                self.chain_length = 3
            else:
                self.is_simultaneous = True
                self.chain_length = 2

            self.task_data['is_sequence'] =  self.is_sequence

    def _start_target(self):
        
        self.target_index += 1
        self.eye_target_index += 1
        self.hand_target_index += 1

        # Show eye hand target
        target_hand = self.targets_hand[self.hand_target_index]
        target_hand.move_to_position(self.hand_targs[self.hand_target_index])
        target_eye = self.targets_eye[self.eye_target_index]
        target_eye.move_to_position(self.eye_targs[self.eye_target_index] - self.offset_cube)
        target_hand.show()
        target_eye.show()
        self.sync_event('EYE_TARGET_ON', self.eye_gen_indices[self.eye_target_index])

    def _while_target(self):
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.eye_targs[self.eye_target_index,[0,2]])
        if eye_d <= (self.target_radius + self.fixation_radius_buffer):
            self.targets_eye[self.eye_target_index].cube.color = target_colors[self.fixation_target_color] # chnage color in fixating center
        else:
            self.targets_eye[self.eye_target_index].cube.color = target_colors[self.eye_target_color]
    
    def _start_target_eye(self):
        self.target_index += 1
        self.eye_target_index += 1

    def _while_target_eye(self):
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.eye_targs[self.eye_target_index,[0,2]])
        if eye_d <= (self.target_radius + self.fixation_radius_buffer):
            self.targets_eye[self.eye_target_index].cube.color = target_colors[self.fixation_target_color] # chnage color in fixating center
        else:
            self.targets_eye[self.eye_target_index].cube.color = target_colors[self.eye_target_color]

    def _start_target_hand(self):
        self.target_index += 1
        self.hand_target_index += 1

    def _start_target_eye_hand(self):
        self.target_index += 1
        self.eye_target_index += 1
        self.hand_target_index += 1

    def _while_target_eye_hand(self):
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.eye_targs[self.eye_target_index,[0,2]])
        if eye_d <= (self.target_radius + self.fixation_radius_buffer):
            self.targets_eye[self.eye_target_index].cube.color = target_colors[self.fixation_target_color] # chnage color in fixating center
        else:
            self.targets_eye[self.eye_target_index].cube.color = target_colors[self.eye_target_color]

    def _start_fixation(self):
        #if self.target_index != 0:
        self.sync_event('FIXATION', self.eye_gen_indices[self.eye_target_index])
        self.targets_eye[self.eye_target_index].cube.color = target_colors[self.fixation_target_color]

    def _start_hold(self):
        if self.target_index != 1: # when the state comes from target_eye, skip start_hold
            self.sync_event('CURSOR_ENTER_TARGET', self.hand_gen_indices[self.hand_target_index])

    def _start_delay(self):
        if self.target_index == 0 and self.is_simultaneous: # This is for both eye and hand targets
            next_idx = (self.eye_target_index + 1)
            self.targets_eye[next_idx].move_to_position(self.eye_targs[next_idx] - self.offset_cube)
            self.targets_hand[next_idx].move_to_position(self.hand_targs[next_idx])
            self.targets_eye[next_idx].show()
            self.targets_hand[next_idx].show()
            self.sync_event('EYE_TARGET_ON', self.eye_gen_indices[next_idx])

        elif self.target_index == 0 and self.is_sequence: # This is for eye target in the first delay
            next_eye_idx = (self.eye_target_index + 1)
            self.targets_eye[next_eye_idx].move_to_position(self.eye_targs[next_eye_idx] - self.offset_cube)
            self.targets_eye[next_eye_idx].show()
            
            next_hand_idx = (self.hand_target_index + 1)
            self.targets_hand[next_hand_idx].move_to_position(self.hand_targs[next_hand_idx]) # Target position is the same, but change color?
            self.targets_hand[next_hand_idx].show()

            self.sync_event('EYE_TARGET_ON', self.eye_gen_indices[next_eye_idx])

        elif self.target_index == 1 and self.is_sequence: # This is for hand target in the second delay

            pass
            #self.sync_event('TARGET_ON', self.hand_gen_indices[next_hand_idx])

    def _start_targ_transition(self):
        if self.target_index == 0 and self.is_simultaneous: # This is a go cue for both eye and hand
            self.targets_eye[self.eye_target_index].hide()
            self.targets_hand[self.hand_target_index].hide()
            self.sync_event('EYE_TARGET_OFF', self.eye_gen_indices[self.eye_target_index])

        elif self.target_index == 0 and self.is_sequence: # This is a go cue for eye
            self.targets_eye[self.eye_target_index].hide()
            self.sync_event('EYE_TARGET_OFF', self.eye_gen_indices[self.eye_target_index])
        
        elif self.target_index == 1 and self.is_sequence: # This is a go cue for hand
            self.targets_hand[self.hand_target_index].hide()
            self.sync_event('TARGET_OFF', self.hand_gen_indices[self.hand_target_index])   

    def _start_timeout_penalty(self):
        super()._start_timeout_penalty()
        for target_eye, target_hand in zip(self.targets_eye,self.targets_hand):
            target_eye.hide()
            target_eye.reset()
            target_hand.hide()
            target_hand.reset()
            
    def _start_hold_penalty(self):
        super()._start_hold_penalty()
        for target_eye, target_hand in zip(self.targets_eye,self.targets_hand):
            target_eye.hide()
            target_eye.reset()
            target_hand.hide()
            target_hand.reset()

    def _start_delay_penalty(self):
        super()._start_delay_penalty()
        for target_eye, target_hand in zip(self.targets_eye,self.targets_hand):
            target_eye.hide()
            target_eye.reset()
            target_hand.hide()
            target_hand.reset()

    def _start_fixation_penalty(self):
        super()._start_fixation_penalty()
        for target_eye, target_hand in zip(self.targets_eye,self.targets_hand):
            target_eye.hide()
            target_eye.reset()
            target_hand.hide()
            target_hand.reset()

    def _start_reward(self):
        super()._start_reward()
        for target in self.targets_eye:
            target.cue_trial_end_success()

    def _end_reward(self):
        super()._end_reward()
        for target_eye, target_hand in zip(self.targets_eye,self.targets_hand):
            target_eye.hide()
            target_eye.reset()
            target_hand.hide()
            target_hand.reset()     

    def _start_pause(self):
        super()._start_pause()
        for target_eye, target_hand in zip(self.targets_eye,self.targets_hand):
            target_eye.hide()
            target_eye.reset()
            target_hand.hide()
            target_hand.reset()

class ScreenTargetCapture_Saccade(ScreenTargetCapture):
    '''
    Center-out saccade task. The controller for the cursor position is eye position.
    Hand cursor is also visible. You should remove the hand cursor by setting cursor_radius to 0 as needed.
    '''

    fixation_radius_buffer = traits.Float(.5, desc="additional radius for eye target")
    target_color = traits.OptionsList("eye_color", *target_colors, desc="Color of the target", bmi3d_input_options=list(target_colors.keys()))
    fixation_target_color = traits.OptionsList("fixation_color", *target_colors, desc="Color of the eye target under fixation state", bmi3d_input_options=list(target_colors.keys()))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:

            # 2 targets for delay
            target1 = VirtualRectangularTarget(target_width=self.target_radius, target_height=self.target_radius/2, target_color=target_colors[self.target_color])
            target2 = VirtualRectangularTarget(target_width=self.target_radius, target_height=self.target_radius/2, target_color=target_colors[self.target_color])

            self.targets = [target1, target2]

            self.offset_cube = np.array([0,0,self.target_radius/2]) # To center the cube target

    def _test_enter_target(self, ts):
        '''
        Check whether eye positions from a target are within the fixation distance
        '''
        # Distance of an eye position from a target position
        eye_pos = self.calibrated_eye_pos
        target_pos = np.delete(self.targs[self.target_index],1)
        d_eye = np.linalg.norm(eye_pos - target_pos)
        return (d_eye <= self.target_radius + self.fixation_radius_buffer) or self.pause

    def _test_leave_target(self, ts):
        '''
        Check whether eye positions from a target are outside the fixation distance
        '''
        # Distance of an eye position from a target position
        eye_pos = self.calibrated_eye_pos
        target_pos = np.delete(self.targs[self.target_index],1)
        d_eye = np.linalg.norm(eye_pos - target_pos)
        return (d_eye > self.target_radius + self.fixation_radius_buffer) or self.pause

    def _start_target(self):
        self.target_index += 1

        # Show target if it is hidden (this is the first target, or previous state was a penalty)
        target = self.targets[self.target_index % 2]
        if self.target_index == 0:
            target.move_to_position(self.targs[self.target_index] - self.offset_cube)
            target.show()
            self.sync_event('TARGET_ON', self.gen_indices[self.target_index])
        self.target_location = self.targs[self.target_index] # save for BMILoop

    def _start_hold(self):
        super()._start_hold()
        self.targets[self.target_index].cube.color = target_colors[self.fixation_target_color] # change target color in fixating the target

    def _start_delay(self):
        # Make next target visible unless this is the final target in the trial
        next_idx = (self.target_index + 1)
        if next_idx < self.chain_length:
            target = self.targets[next_idx % 2]
            target.move_to_position(self.targs[next_idx] - self.offset_cube)
            target.show()
            self.sync_event('TARGET_ON', self.gen_indices[next_idx])
        else:
            # This delay state should only last 1 cycle, don't sync anything
            pass