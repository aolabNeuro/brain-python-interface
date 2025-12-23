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
    Add a penalty state when subjects looks away.
    '''

    fixation_penalty_time = traits.Float(0., desc="Time in fixation penalty state")
    fixation_target_color = traits.OptionsList("cyan", *target_colors, desc="Color of the center target under fixation state", bmi3d_input_options=list(target_colors.keys()))
    fixation_radius_buffer = traits.Float(.5, desc="additional radius for eye target")

    status = dict(
        wait = dict(start_trial="target"),
        target = dict(timeout="timeout_penalty",gaze_target="fixation"),
        fixation = dict(enter_target="hold", fixation_break="target"),
        hold = dict(leave_target="hold_penalty", hold_complete="delay", fixation_break="fixation_penalty"),
        delay = dict(leave_target="delay_penalty", delay_complete="targ_transition", fixation_break="fixation_penalty"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition", end_state=True),
        hold_penalty = dict(hold_penalty_end="targ_transition", end_state=True),
        delay_penalty = dict(delay_penalty_end="targ_transition", end_state=True),
        fixation_penalty = dict(fixation_penalty_end="targ_transition",end_state=True),
        reward = dict(reward_end="wait", stoppable=False, end_state=True)
    )
 
    def _test_gaze_target(self,ts):
        '''
        Check whether eye positions are within the fixation distance
        Only apply this to the first target (1st target)
        '''
        if self.target_index <= 0:     
            d = np.linalg.norm(self.calibrated_eye_pos)
            return d < self.target_radius + self.fixation_radius_buffer
        else:
            return True
        
    def _test_fixation_break(self,ts):
        '''
        Triggers the fixation_penalty state when eye positions are outside fixation distance
        Only apply this to the first hold and delay period
        '''
        if self.target_index <= 0:   
            d = np.linalg.norm(self.calibrated_eye_pos)
            return (d > self.target_radius + self.fixation_radius_buffer)
    
    def _test_fixation_penalty_end(self,ts):
        return (ts > self.fixation_penalty_time) 
    
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
        self._increment_tries()
        self.sync_event('FIXATION_PENALTY') 

        # Hide targets
        for target in self.targets:
            target.hide()
            target.reset()

    def _end_fixation_penalty(self):
        self.sync_event('TRIAL_END')

class HandConstrainedEyeCapture(ScreenTargetCapture):
    '''
    Saccade task with holding another target with hand. Subjects need to hold an initial target with their hand. 
    Then they need to fixate the first eye target and make a saccade for the second eye target 
    '''

    fixation_radius = traits.Float(2.5, desc="Distance from center that is considered a broken fixation")
    fixation_penalty_time = traits.Float(1.0, desc="Time in fixation penalty state")
    fixation_target_color = traits.OptionsList("cyan", *target_colors, desc="Color of the eye target under fixation state", bmi3d_input_options=list(target_colors.keys()))
    eye_target_color = traits.OptionsList("white", *target_colors, desc="Color of the eye target", bmi3d_input_options=list(target_colors.keys()))
    fixation_radius_buffer = traits.Float(.5, desc="additional radius for eye target")
    fixation_time = traits.Float(.2, desc="additional radius for eye target")
    exclude_parent_traits = ['hold_time']

    status = dict(
        wait = dict(start_trial="target", start_pause="pause"),
        target = dict(start_pause="pause", timeout="timeout_penalty", gaze_enter_target="fixation"),
        target_eye = dict(start_pause="pause", timeout="timeout_penalty", leave_target='hold_penalty', gaze_target="fixation"),
        fixation = dict(start_pause="pause", leave_target="hold_penalty", fixation_complete="delay", fixation_break="fixation_penalty"), 
        delay = dict(leave_target="delay_penalty", delay_complete="targ_transition", fixation_break="fixation_penalty", start_pause="pause"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target_eye", start_pause="pause"),
        timeout_penalty = dict(timeout_penalty_end="wait", start_pause="pause", end_state=True),
        hold_penalty = dict(hold_penalty_end="wait", start_pause="pause", end_state=True),
        delay_penalty = dict(delay_penalty_end="wait", start_pause="pause", end_state=True),
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
            target1 = VirtualCircularTarget(target_radius=self.fixation_radius, target_color=target_colors[self.eye_target_color])
            target2 = VirtualCircularTarget(target_radius=self.fixation_radius, target_color=target_colors[self.eye_target_color])
            target3 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])

            self.targets = [target1, target2]
            self.targets_hand = [target3]
        
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
        
        return (eye_d <= self.target_radius + self.fixation_radius_buffer) and (hand_d <= self.target_radius - self.cursor_radius)
    
    def _test_gaze_target(self, ts):
        '''
        Check whether eye positions and hand cursor are within the target radius
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.targs[self.target_index,[0,2]])

        return eye_d <= self.target_radius + self.fixation_radius_buffer   
    
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
    
    def _test_trial_incomplete(self, ts):
        return self.target_index < self.chain_length
    
    def _start_wait(self):
        super()._start_wait()
        # Redefine chain length because targs in this task has both eye and hand targets
        self.chain_length = len(self.targets)

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

    def _start_target(self):
        self.target_index += 1
        self.is_eye_target_on = False # this is for _while_target

        # Show the hand target
        target_hand = self.targets_hand[0]
        if self.target_index == 0:
            target_hand.move_to_position(self.targs[-1])
            target_hand.show()
            self.sync_event('TARGET_ON', self.gen_indices[-1]) # the hand target is on

    def _while_target(self):
        
        if self.target_index == 0:
            cursor_pos = self.plant.get_endpoint_pos()
            hand_d = np.linalg.norm(cursor_pos - self.targs[-1])

            target = self.targets[self.target_index]
            target.move_to_position(self.targs[self.target_index])

            # the eye target is on when the hand positon is within the hand target
            if hand_d <= self.target_radius - self.cursor_radius and not self.is_eye_target_on:
                target.show()
                self.sync_event('EYE_TARGET_ON', self.gen_indices[self.target_index]) # sync_event only when eye target is off
                self.is_eye_target_on = True

            elif hand_d > self.target_radius - self.cursor_radius and self.is_eye_target_on:
                target.hide()
                self.sync_event('EYE_TARGET_OFF', self.gen_indices[self.target_index]) # sync_event only when eye target is on
                self.is_eye_target_on = False

    def _start_target_eye(self):
        self.target_index += 1

    def _start_fixation(self):
        self.targets[self.target_index].sphere.color = target_colors[self.fixation_target_color] # change target color in fixation state
        self.sync_event('FIXATION', self.gen_indices[self.target_index])

    def _start_delay(self):
        # Make next target visible unless this is the final target in the trial
        next_idx = (self.target_index + 1)
        if next_idx < self.chain_length:
            target = self.targets[next_idx]
            target.move_to_position(self.targs[next_idx])
            target.show()
            self.sync_event('EYE_TARGET_ON', self.gen_indices[next_idx])
        else:
            # This delay state should only last 1 cycle, don't sync anything
            pass

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

    def _start_reward(self):
        super()._start_reward()
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

class ScreenTargetCapture_Saccade(ScreenTargetCapture):
    '''
    Center-out saccade task. The controller for the cursor position is eye position.
    Hand cursor is also visible. You should remove the hand cursor by setting cursor_radius to 0 as needed.
    '''

    fixation_radius_buffer = traits.Float(.5, desc="additional radius for eye target")
    target_color = traits.OptionsList("white", *target_colors, desc="Color of the target", bmi3d_input_options=list(target_colors.keys()))
    fixation_target_color = traits.OptionsList("cyan", *target_colors, desc="Color of the eye target under fixation state", bmi3d_input_options=list(target_colors.keys()))

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
    
    def _start_hold(self):
        super()._start_hold()
        self.targets[self.target_index].sphere.color = target_colors[self.fixation_target_color] # change target color in fixating the target