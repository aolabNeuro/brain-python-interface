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
    fixation_target_color = traits.OptionsList("cyan", *target_colors, desc="Color of the center target under fixation state", bmi3d_input_options=list(target_colors.keys()))
    fixation_radius_buffer = traits.Float(.5, desc="additional radius for eye target")
    hand_target_color = traits.OptionsList("yellow", *target_colors, desc="Color for the hand-only target", bmi3d_input_options=list(target_colors.keys()))

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
    
    def _while_target(self):
        if self.target_index == 0:
            eye_pos = self.calibrated_eye_pos
            eye_d = np.linalg.norm(eye_pos - self.targs[self.target_index,[0,2]])
            if eye_d <= (self.target_radius + self.fixation_radius_buffer):
                self.targets[self.target_index].sphere.color = target_colors[self.fixation_target_color] # chnage color in fixating center
            else:
                self.targets[self.target_index].reset()

    def _start_delay(self):
        next_idx = (self.target_index + 1)
        if next_idx < self.chain_length: # This is for hand target in the second delay
            self.targets[next_idx].move_to_position(self.targs[next_idx]) # Target position is the same, but change color?
            self.targets[next_idx].sphere.color = target_colors[self.hand_target_color]
            self.targets[next_idx].show()
            self.sync_event('TARGET_ON', self.gen_indices[next_idx])

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

class EyeHandSequenceCapture(EyeConstrainedTargetCapture):
    '''
    Subjects have to gaze at and reach to a target.
    '''

    status = dict(
        wait = dict(start_trial="target", start_pause="pause"),
        target = dict(timeout="timeout_penalty", gaze_enter_target='hold', start_pause="pause"),
        target_eye = dict(timeout="timeout_penalty", gaze_target="fixation", leave_target="hold_penalty", start_pause="pause"),
        target_hand = dict(timeout="timeout_penalty", enter_target="hold", fixation_break="fixation_penalty", start_pause="pause"),
        fixation = dict(fixation_complete="delay", leave_target="hold_penalty", fixation_break="fixation_penalty", start_pause="pause"),
        hold = dict(hold_complete="delay", leave_target="hold_penalty",  fixation_break="fixation_penalty", start_pause="pause"),
        delay = dict(delay_complete="targ_transition", leave_target="delay_penalty", fixation_break="fixation_penalty", start_pause="pause"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", first_targ_complete="target_eye", second_targ_complete="target_hand", start_pause="pause"),
        timeout_penalty = dict(timeout_penalty_end="wait", start_pause="pause", end_state=True),
        hold_penalty = dict(hold_penalty_end="wait", start_pause="pause", end_state=True),
        delay_penalty = dict(delay_penalty_end="wait", start_pause="pause", end_state=True),
        fixation_penalty = dict(fixation_penalty_end="wait", start_pause="pause", end_state=True),
        reward = dict(reward_end="wait", start_pause="pause", stoppable=False, end_state=True),
        pause = dict(end_pause="wait", end_state=True),
    )

    exclude_parent_traits = ['delay_time', 'rand_delay']
    rand_delay1 = traits.Tuple((0.4, 0.4), desc="Delay interval for eye")
    rand_delay2 = traits.Tuple((0.4, 0.4), desc="Delay interval for hand")
    fixation_time = traits.Float(0.2, desc='Length of fixation required at targets')
    
    def _test_gaze_enter_target(self,ts):
        '''
        Check whether eye positions and hand cursor are within the target radius
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.eye_targs[self.eye_target_index,[0,2]])

        cursor_pos = self.plant.get_endpoint_pos()
        hand_d = np.linalg.norm(cursor_pos - self.hand_targs[self.hand_target_index])
        
        return (eye_d <= self.target_radius + self.fixation_radius_buffer) and (hand_d <= self.target_radius - self.cursor_radius)

    def _test_gaze_target(self,ts):
        '''
        Check whether eye positions are within the fixation distance
        ''' 
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.eye_targs[self.eye_target_index,[0,2]])
        return eye_d <= self.target_radius + self.fixation_radius_buffer

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
        return eye_d > (self.target_radius + self.fixation_radius_buffer)

    def _test_fixation_complete(self, ts):
        return ts > self.fixation_time
    
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
    
    def _test_first_targ_complete(self, ts):
        return self.eye_target_index == 0 and self.hand_target_index == 0
    
    def _test_second_targ_complete(self, ts):
        return self.eye_target_index == 1 and self.hand_target_index == 0
    
    def _start_wait(self):
        super()._start_wait()

        # Initialize target index and target positons for eye and hand
        self.eye_target_index = -1
        self.hand_target_index = -1
        self.eye_targs = np.copy(self.targs)
        self.hand_targs = np.copy(self.targs)
        self.eye_gen_indices = np.copy(self.gen_indices)
        self.hand_gen_indices = np.copy(self.gen_indices)

        if self.tries == 0: # only update delay_time for the first attempt
            self.chain_length = 3
            
            # Set delay time
            s, e = self.rand_delay1
            self.delay_time1 = random.random()*(e-s) + s
            s, e = self.rand_delay2
            self.delay_time2 = random.random()*(e-s) + s

    def _start_target(self):
        
        self.target_index += 1
        self.eye_target_index += 1
        self.hand_target_index += 1

        # Show eye hand target
        target = self.targets[self.eye_target_index]
        target.move_to_position(self.eye_targs[self.eye_target_index])
        target.show()
        self.sync_event('EYE_TARGET_ON', self.eye_gen_indices[self.eye_target_index])

    def _while_target(self):
        eye_pos = self.calibrated_eye_pos
        eye_d = np.linalg.norm(eye_pos - self.eye_targs[self.eye_target_index,[0,2]])
        if eye_d <= (self.target_radius + self.fixation_radius_buffer):
            self.targets[self.eye_target_index].sphere.color = target_colors[self.fixation_target_color] # chnage color in fixating center
        else:
            self.targets[self.eye_target_index].reset()
    
    def _start_target_eye(self):
        self.target_index += 1
        self.eye_target_index += 1

    def _start_target_hand(self):
        self.target_index += 1
        self.hand_target_index += 1

    def _start_fixation(self):
        if self.target_index != 0:
            self.sync_event('FIXATION', self.eye_gen_indices[self.eye_target_index])
        self.targets[self.eye_target_index].sphere.color = target_colors[self.fixation_target_color]

    def _start_hold(self):
        if self.target_index != 1: # when the state comes from target_eye, skip start_hold
            self.sync_event('CURSOR_ENTER_TARGET', self.hand_gen_indices[self.hand_target_index])

    def _start_delay(self):
        if self.target_index == 0: # This is for eye target in the first delay
            next_idx = (self.eye_target_index + 1)
            self.targets[next_idx].move_to_position(self.eye_targs[next_idx])
            self.targets[next_idx].show()
            self.sync_event('EYE_TARGET_ON', self.eye_gen_indices[next_idx])

        elif self.target_index == 1: # This is for hand target in the second delay
            next_idx = (self.hand_target_index + 1)
            self.targets[next_idx].move_to_position(self.hand_targs[next_idx]) # Target position is the same, but change color?
            self.targets[next_idx].show()
            self.sync_event('TARGET_ON', self.hand_gen_indices[next_idx])

    def _start_targ_transition(self):
        if self.target_index == 0: # This is a go cue for eye target
            self.targets[self.eye_target_index].sphere.color = target_colors[self.hand_target_color]
            self.sync_event('EYE_TARGET_OFF', self.eye_gen_indices[self.eye_target_index])
        
        elif self.target_index == 1: # This is a go cue for hand target
            self.targets[self.hand_target_index].hide()
            self.sync_event('TARGET_OFF', self.hand_gen_indices[self.hand_target_index])            

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

    status = dict(
        wait = dict(start_trial="target", start_pause="pause"),
        target = dict(start_pause="pause", leave_target2="hold_penalty",timeout="timeout_penalty",enter_target="hold"),
        hold = dict(start_pause="pause", leave_target2="hold_penalty",leave_target="target", gaze_target="fixation"), # must hold an initial hand-target and eye-target
        fixation = dict(start_pause="pause", leave_target="delay_penalty",hold_complete="delay", fixation_break="fixation_penalty"), # must hold an initial hand-target and eye-target to initiate a trial
        delay = dict(leave_target="delay_penalty", delay_complete="targ_transition", fixation_break="fixation_penalty", start_pause="pause"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target", start_pause="pause"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition", start_pause="pause", end_state=True),
        hold_penalty = dict(hold_penalty_end="targ_transition", start_pause="pause", end_state=True),
        delay_penalty = dict(delay_penalty_end="targ_transition", start_pause="pause", end_state=True),
        fixation_penalty = dict(fixation_penalty_end="targ_transition", start_pause="pause", end_state=True),
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
        self.gen_indices, self.targs = self.next_trial # 2 target positions for hand and eye

        # Update the data sinks with trial information
        self.trial_record['trial'] = self.calc_trial_num() # TODO save both eye and hand target positions
        for i in range(len(self.gen_indices)):
            self.trial_record['index'] = self.gen_indices[i]
            self.trial_record['target'] = self.targs[i]
            self.sinks.send("trials", self.trial_record)

    def _test_gaze_target(self,ts):
        '''
        Check whether eye positions from a target are within the fixation distance
        '''
        # Distance of an eye position from a target position
        eye_pos = self.calibrated_eye_pos
        target_pos = np.delete(self.targs[self.target_index],1)
        d_eye = np.linalg.norm(eye_pos - target_pos)
        return (d_eye <= self.fixation_radius + self.fixation_radius_buffer) or self.pause
        
    def _test_fixation_break(self,ts):
        '''
        Triggers the fixation_penalty state when eye positions are outside fixation distance
        '''
        # Distance of an eye position from a target position
        eye_pos = self.calibrated_eye_pos
        target_pos = np.delete(self.targs[self.target_index],1)
        d_eye = np.linalg.norm(eye_pos - target_pos)
        return (d_eye > self.fixation_radius + self.fixation_radius_buffer) or self.pause
    
    def _test_fixation_penalty_end(self,ts):
        return (ts > self.fixation_penalty_time)
    
    def _test_enter_target(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.targs[-1]) # hand must be within the initial target
        return d <= (self.target_radius - self.cursor_radius) or self.pause

    def _test_leave_target(self, ts):
        '''
        return true if cursor moves outside the exit radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.targs[-1]) # hand must be within the initial target
        return d > (self.target_radius - self.cursor_radius) or self.pause

    def _test_leave_target2(self, ts):
        '''
        return true if cursor moves outside the exit radius (This is for the second target state)
        '''
        if self.target_index > 0:
            cursor_pos = self.plant.get_endpoint_pos()
            d = np.linalg.norm(cursor_pos - self.targs[-1]) # hand must be within the initial target
            return d > (self.target_radius - self.cursor_radius) or self.pause
    
    def _start_wait(self):
        super()._start_wait()
        # Redefine chain length because targs in this task has both eye and hand targets
        self.chain_length = len(self.targets)

        # Initialize fixation state
        self.num_hold_state = 0

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
        if self.num_hold_state == 0:
            self.target_index += 1 # target index shouldn't be incremented after hold break loop

            # Show target if it is hidden (this is the first target, or previous state was a penalty)
            target_hand = self.targets_hand[0]
            if self.target_index == 0:
                target_hand.move_to_position(self.targs[-1])
                target_hand.show()
                self.sync_event('TARGET_ON', self.gen_indices[-1])
                
        else:
            target = self.targets[self.target_index % 2]
            target.hide() # hide hand target
            self.sync_event('EYE_TARGET_OFF', self.gen_indices[self.target_index % 2])

    def _start_hold(self):
        #self.sync_event('CURSOR_ENTER_TARGET', self.gen_indices[self.target_index])
        self.num_hold_state = 1

        # Show target if it is hidden (this is the first target, or previous state was a penalty)
        target = self.targets[self.target_index % 2]
        if self.target_index == 0:
            target.move_to_position(self.targs[self.target_index])
            target.show()
            self.sync_event('EYE_TARGET_ON', self.gen_indices[self.target_index])

    def _start_fixation(self):
        self.num_hold_state = 0
        self.targets[self.target_index].sphere.color = target_colors[self.fixation_target_color] # change target color in fixation state
        self.sync_event('FIXATION', self.gen_indices[self.target_index])

    def _while_fixation(self):
        pass

    def _end_fixation(self):
        pass

    def _start_delay(self):
        # Make next target visible unless this is the final target in the trial
        next_idx = (self.target_index + 1)
        if next_idx < self.chain_length:
            target = self.targets[next_idx % 2]
            target.move_to_position(self.targs[next_idx % 2])
            target.show()
            self.sync_event('EYE_TARGET_ON', self.gen_indices[next_idx % 2])
        else:
            # This delay state should only last 1 cycle, don't sync anything
            pass

    def _start_targ_transition(self):
        if self.target_index == -1:

            # Came from a penalty state
            pass
        elif self.target_index + 1 < self.chain_length:

            # Hide the current target if there are more
            self.targets[self.target_index % 2].hide()
            self.sync_event('EYE_TARGET_OFF', self.gen_indices[self.target_index])

    def _start_timeout_penalty(self):
        super()._start_timeout_penalty()
        self.num_hold_state = 0
        for target in self.targets_hand:
            target.hide()
            target.reset()
            
    def _start_hold_penalty(self):
        super()._start_hold_penalty()
        self.num_hold_state = 0
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