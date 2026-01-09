
'''
Target capture tasks with additional features
'''
import numpy as np
import random
import os

from .target_graphics import *
from .target_capture_task import ScreenTargetCapture
from riglib.experiment import traits
from riglib.audio import AudioPlayer, TonePlayer

audio_path = os.path.join(os.path.dirname(__file__), '../riglib/audio')        

class ScreenReachAngle(ScreenTargetCapture):
    '''
    A modified task that requires the cursor to move in the right direction towards the target, 
    without actually needing to arrive at the target. If the maximum angle is exceeded, a reach 
    penalty is applied. No hold or delay period.

    Only works for sequences with 1 target in a chain. 
    '''

    status = dict(
        wait = dict(start_trial="target", start_pause="pause"),
        target = dict(reach_success="targ_transition", timeout="timeout_penalty", leave_bounds="reach_penalty", start_pause="pause"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target", start_pause="pause"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition", start_pause="pause", end_state=True),
        reach_penalty = dict(reach_penalty_end="targ_transition", start_pause="pause", end_state=True),
        reward = dict(reward_end="wait", start_pause="pause", stoppable=False, end_state=True),
        pause = dict(end_pause="wait", end_state=True),
    )

    sequence_generators = [
        'out_2D', 'rand_target_chain_2D', 'rand_target_chain_3D', 'discrete_targets_2D',
    ]

    max_reach_angle = traits.Float(90., desc="Angle defining the boundaries between the starting position of the cursor and the target")
    reach_penalty_time = traits.Float(1, desc="Length of penalty time for target hold error")
    reach_fraction = traits.Float(0.5, desc="Fraction of the distance between the reach start and the target before a reward")
    start_radius = traits.Float(1., desc="Buffer around reach start allowed in bounds (cm)")

    exclude_parent_traits = ['hold_time', 'hold_penalty_time', 'delay_time', 'delay_penalty_time']

    def _start_target(self):
        super()._start_target()

        # Define a reach start and reach target position whenever the target appears
        self.reach_start = self.plant.get_endpoint_pos().copy()
        self.reach_target = self.targs[self.target_index]

    def _test_leave_bounds(self, ts):
        '''
        Check whether the cursor is in the boundary defined by reach_start, target_pos,
        and max_reach_angle.
        '''

        # Calculate the angle between the vectors from the start pos to the current cursor and target
        a = self.plant.get_endpoint_pos() - self.reach_start
        b = self.reach_target - self.reach_start
        cursor_target_angle = np.arccos(np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b))

        # If that angle is more than half the maximum, we are outside the bounds
        out_of_bounds = np.degrees(cursor_target_angle) > self.max_reach_angle / 2

        # But also allow a target radius around the reach_start 
        away_from_start = np.linalg.norm(self.plant.get_endpoint_pos() - self.reach_start) > self.start_radius

        return away_from_start and out_of_bounds

    def _test_reach_success(self, ts):
        dist_traveled = np.linalg.norm(self.plant.get_endpoint_pos() - self.reach_start)
        dist_total = np.linalg.norm(self.reach_target - self.reach_start)
        dist_total -= (self.target_radius - self.cursor_radius)
        return dist_traveled/dist_total > self.reach_fraction

    def _start_reach_penalty(self):
        self.sync_event('OTHER_PENALTY')
        self._increment_tries()
        
        # Hide targets
        for target in self.targets:
            target.hide()
            target.reset()

    def _end_reach_penalty(self):
        self.sync_event('TRIAL_END')

    def _test_reach_penalty_end(self, ts):
        return ts > self.reach_penalty_time

    @staticmethod
    def discrete_targets_2D(nblocks=100, ntargets=3, boundaries=(-6,6,-3,3)):
        '''
        Generates a sequence of 2D (x and z) target pairs that don't overlap

        Parameters
        ----------
        nblocks : int
            The number of ntarget pairs in the sequence.
        ntargets : int
            The number of unique targets (up to 9 maximum)
        boundaries: 4 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)

        Returns
        -------
        [ntrials x ntargets x 3] array of target coordinates
        '''
        targets = np.array([
            [0, 0.5],
            [1, 0.5],
            [1, 0],
            [0, 0],
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.75, 0.75],
            [0.5, 1],
        ])
        rng = np.random.default_rng()
        for _ in range(nblocks):
            order = np.arange(ntargets) # target indices
            rng.shuffle(order)
            for t in range(ntargets):
                idx = order[t]
                pts = targets[idx]*((boundaries[1]-boundaries[0]),
                    (boundaries[3]-boundaries[2]))
                pts = pts+(boundaries[0], boundaries[2])
                pos = np.array([pts[0], 0, pts[1]])
                yield [idx], [pos]

class ScreenReachLine(ScreenTargetCapture):
    '''
    A modified task that requires the cursor must be within the straight area between the initial cursor position and the target, 
    Only works for sequences with 1 target in a chain (not tested for more than 2 chains). 
    '''

    status = dict(
        wait = dict(start_trial="target", start_pause="pause"),
        target = dict(leave_bounds="reach_penalty", enter_target="hold", timeout="timeout_penalty", start_pause="pause"),
        hold = dict(leave_target="hold_penalty", hold_complete="delay", start_pause="pause"),
        delay = dict(leave_target="delay_penalty", delay_complete="targ_transition", start_pause="pause"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target", start_pause="pause"),
        timeout_penalty = dict(timeout_penalty_end="wait", start_pause="pause", end_state=True),
        hold_penalty = dict(hold_penalty_end="wait", start_pause="pause", end_state=True),
        delay_penalty = dict(delay_penalty_end="wait", start_pause="pause", end_state=True),
        reach_penalty = dict(delay_penalty_end="wait", start_pause="pause", end_state=True),
        reward = dict(reward_end="wait", start_pause="pause", stoppable=False, end_state=True),
        pause = dict(end_pause="wait", end_state=True),
    )

    sequence_generators = [
        'centerout_2D', 'centerout_2D_select','out_2D', 'out_2D_select','rand_target_chain_2D', 'rand_target_chain_3D', 'discrete_targets_2D',
    ]

    reach_penalty_time = traits.Float(1, desc="Length of penalty time for target hold error")
    exclude_parent_traits = ['max_reach_angle','reach_fraction','start_radius']
    line_color = traits.OptionsList("white", *target_colors, desc="Color of the eye target", bmi3d_input_options=list(target_colors.keys()))
    line_width = traits.Float(3, desc="Line width where the cursor needs to stay")
    line_from_previous_target = traits.Bool(False, desc="Line is drew between the previous target and the pripheral target")
    remove_line_for_the_first_target = traits.Bool(False, desc="Remove the line for the first target, which is usually the center target")
    #line_for_target_index = traits.List([0,1], desc="Line is drawn for specified target index")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.penalty_index = 0
        self.pause_index = 0
          
    def _start_wait(self):
        if self.penalty_index == 0 and self.pause_index == 0:
            super()._start_wait() # skip _start_wait to make the same target appear even after penalty

        if self.tries==self.max_attempts: # The task goes to the next target after the number of reattempting is max attempts 
            super()._start_wait()
            self.tries = 0 # number of times this sequence of targets has been attempted
            
        # Delete bar because the shape of bar changes every trial
        if hasattr(self, 'bar'):
            for model in self.bar.graphics_models:
                self.remove_model(model)
            del self.bar

        self.penalty_index = 0
        self.pause_index = 0
        self.target_index = -1

    def _start_target(self):
        super()._start_target()
        
        # Define a reach start and reach target position whenever the target appears
        if self.target_index != 0:
            if self.line_from_previous_target:
                self.reach_start = self.targs[self.target_index-1] # Reach start position is defined as the previous target position
            else:
                self.reach_start = self.plant.get_endpoint_pos().copy() # Reach start position is defined as the initial cursor position

        else:
            self.reach_start = self.plant.get_endpoint_pos().copy()

        self.reach_target = self.targs[self.target_index]
  
        # slope between the reach start position and target position
        self.target_x = self.reach_target[0]
        self.target_y = self.reach_target[2]
        x0 = self.reach_start[0]
        y0 = self.reach_start[2]
        self.slope = (self.target_y-y0)/(self.target_x-x0)

        # Convert the slope to the angle
        bar_angle = np.degrees(np.arctan(self.slope))

        self.bar = VirtualRectangularTarget(target_width=self.line_width, target_height=50, 
                                                   target_color=target_colors[self.line_color],starting_pos=[0,0,0])
        for model in self.bar.graphics_models:
            self.add_model(model)

        # Rotate the rectangle
        self.bar.rotate_yaxis(-bar_angle, reset=True)

        # Compute offset because rotating the rectangle results in shifting the rectangle position
        offset_rectangle = [-np.sin(np.radians(-bar_angle))*self.line_width/2,0,-np.cos(np.radians(-bar_angle))*self.line_width/2]
        
        # Move the rectangle to the reach target, taking into account the offset
        self.bar.move_to_position(self.reach_target + offset_rectangle)

        # Show the rectangle
        if self.remove_line_for_the_first_target and self.target_index == 0:
            self.bar.hide()
        else:
            self.bar.show()

    def _test_leave_bounds(self, ts):
        '''
        Check whether the cursor is within the boundary
        '''

        # distance between the reach start positon and the parallel line to the bounday that passes through the curent cursor position
        current_pos = self.plant.get_endpoint_pos()
        X = current_pos[0]
        Y = current_pos[2]
        distance = np.abs(self.target_x*self.slope - self.target_y + (Y-self.slope*X))/np.sqrt(self.slope**2+1)

        if self.remove_line_for_the_first_target and self.target_index == 0:
            return False
        else:
            return distance > (self.line_width/2 - self.cursor_radius)

    def _start_targ_transition(self):
        super()._start_targ_transition()
        self.bar.hide()
        self.bar.reset()

    def _start_timeout_penalty(self):
        super()._start_timeout_penalty()
        self.bar.hide()
        self.bar.reset()
        self.penalty_index = 1

    def _start_delay_penalty(self):
        super()._start_delay_penalty()
        self.bar.hide()
        self.bar.reset()
        self.penalty_index = 1

    def _start_hold_penalty(self):
        super()._start_hold_penalty()
        self.bar.hide()
        self.bar.reset()
        self.penalty_index = 1

    def _start_reach_penalty(self):
        self.sync_event('OTHER_PENALTY')
        self._increment_tries()
        self.penalty_index = 1

        # Hide targets
        for target in self.targets:
            target.hide()
            target.reset()
        self.bar.hide()
        self.bar.reset()

    def _end_reach_penalty(self):
        self.sync_event('TRIAL_END')

    def _start_pause(self):
        super()._start_pause()
        self.pause_index = 1

        if hasattr(self, 'bar'): # bacause bar attribute doen't exist in the wait state
            self.bar.hide()
            self.bar.reset()

    def _test_reach_penalty_end(self, ts):
        return ts > self.reach_penalty_time

class SequenceCapture(ScreenTargetCapture):

    '''
    This is a sequence task in which a 2nd target appears after subjects acquire the 1st target, and
    a 3rd target (additional target) appears while they are moving the cursor to the 2nd target.
    '''

    status = dict(
        wait = dict(start_trial="target", start_pause="pause"),
        target = dict(enter_target="hold", timeout="timeout_penalty", show_additional_target='additional_target', start_pause="pause"),
        additional_target = dict(enter_target="hold", timeout="timeout_penalty", enter_incorrect_target="incorrect_target_penalty", start_pause="pause"),
        hold = dict(leave_target="hold_penalty", hold_complete="delay", start_pause="pause"),
        delay = dict(leave_target="delay_penalty", delay_complete="targ_transition", start_pause="pause"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target", start_pause="pause"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition", start_pause="pause", end_state=True),
        hold_penalty = dict(hold_penalty_end="targ_transition", start_pause="pause", end_state=True),
        delay_penalty = dict(delay_penalty_end="targ_transition", start_pause="pause", end_state=True),
        incorrect_target_penalty = dict(incorrect_target_penalty_end="targ_transition", start_pause="pause", end_state=True),
        reward = dict(reward_end="wait", start_pause="pause", stoppable=False, end_state=True),
        pause = dict(end_pause="wait", end_state=True),
    )

    sequence_generators = ['out_2D_sequence','sequence_2D', 'centerout_2D_different_center']
    incorrect_target_penalty_time = traits.Float(1, desc="Length of penalty time for acquiring an incorrect target")
    random_target_appearance_dist = traits.Tuple((3.5, 3.5), desc="Another target appear when the cursor passed a certain distance from the 1st target in cm")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:

            # Need three targets for sequence task
            target1 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])
            target2 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])
            target3 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])

            self.targets = [target1, target2, target3]

    #### STATE FUNCTIONS ####

    def _start_wait(self):
        '''
        Target apperance distance is sampled from the uniform distribution at the beginning of wait state.
        '''
        super()._start_wait()
        s, e = self.random_target_appearance_dist
        self.target_appearance_distance = random.uniform(s,e)

    def _start_delay(self):
        next_idx = (self.target_index + 1)
        if next_idx == 1 and next_idx < self.chain_length: # Delay period is only for the 2nd target
            target = self.targets[next_idx % self.chain_length]
            target.move_to_position(self.targs[next_idx])
            target.show()
            self.sync_event('TARGET_ON', self.gen_indices[next_idx])     
        else:
            # Don't sync anything when the 1st target and 3rd target appear
            pass

    def _start_targ_transition(self):
        if self.target_index == -1:

            # Came from a penalty state
            pass
        elif self.target_index + 1 < self.chain_length:

            # Hide the current target if there are more
            self.targets[self.target_index % self.chain_length].hide()
            self.sync_event('TARGET_OFF', self.gen_indices[self.target_index])

    def _start_additional_target(self):
        next_idx = (self.target_index + 1)
        target = self.targets[next_idx % self.chain_length]
        target.move_to_position(self.targs[next_idx])
        target.show()
        self.sync_event('TARGET_ON', self.gen_indices[next_idx])    

    def _start_incorrect_target_penalty(self):
        self._increment_tries()
        self.sync_event('OTHER_PENALTY') 
        # Hide targets
        for target in self.targets:
            target.hide()
            target.reset()

    def _end_incorrect_target_penalty(self):
        self.sync_event('TRIAL_END') 

    def _start_reward(self):
        self.targets[self.target_index % self.chain_length].cue_trial_end_success()
        self.sync_event('REWARD')

    #### TEST FUNCTIONS ####

    def _test_hold_complete(self, time_in_state):
        '''
        Test whether the target is held long enough to declare the
        trial a success

        Possible options
            - Target held for the minimum requred time (implemented here)
            - Sensorized object moved by a certain amount
            - Sensorized object moved to the required location
            - Manually triggered by experimenter
        '''
        if self.target_index == 0: # the 1st target
            hold_state = time_in_state > self.hold_time
        elif self.target_index == self.chain_length-1: # the last target
            hold_state = time_in_state > self.hold_time
        else:
            hold_state = True
        return hold_state

    def _test_delay_complete(self, time_in_state):
        '''
        Test whether the delay period, when the cursor must stay in place
        while another target is being presented, is over. There should be 
        no delay on the last target in a chain.
        '''
        if self.target_index == 0:
            delay_state = time_in_state > self.delay_time
        else:
            delay_state = True # No delay period for the 3rd target
        return delay_state

    def _test_enter_incorrect_target(self, ts):
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.targs[self.target_index+1]) # Measure distance between the cursor and the next target
        rad = self.target_radius - self.cursor_radius

        return d < rad   

    def _test_incorrect_target_penalty_end(self, time_in_state):
        return time_in_state > self.incorrect_target_penalty_time

    def _test_show_additional_target(self,ts):
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.targs[self.target_index-1]) # Measure distance between the cursor and the previous target
        rad = self.target_radius - self.cursor_radius

        if self.chain_length < 3: # The 3rd targets isn't shown if chain_length is less than 3 (ex. standard center out task)
           additional_target_state = False
        elif self.target_index == 1: # Target_index == 1 is when the subject is moving the cursor to the 2nd target 
            additional_target_state = d > self.target_appearance_distance
        else:
            additional_target_state = False
        return additional_target_state


    #### Generator functions ####
    @staticmethod
    def out_2D_sequence(nblocks=100, distance=6.8):
        '''
        Generates a sequence of 2D (x and z) targets at a given distance from the origin
        The center target positions change depnding on the peripheral target at each trial
        The number of targets are fixed to 8.

        Parameters
        ----------
        nblocks : int
            The number of ntarget pairs in the sequence.
        distance : float
            The distance in cm between the center and peripheral targets.

        Returns
        -------
        [nblocks*ntargets x 1] array of tuples containing trial indices and [1 x 3] target coordinates

        '''
        ntargets = 8
        rng = np.random.default_rng()
        for _ in range(nblocks):
            order = np.arange(ntargets) + 1 # target indices, starting from 1
            rng.shuffle(order)
            for t in range(ntargets):
                idx = order[t]
                theta = 2*np.pi*(1-idx)/4 + np.pi/2 # put idx 1 at 12 o'clock
                print(theta)
                if idx <= 4:
                    pos = np.array([distance*np.cos(theta)+distance/2,0,distance*np.sin(theta)]).T
                elif idx >=5:
                    pos = np.array([distance*np.cos(theta)-distance/2,0,distance*np.sin(theta)]).T
                yield [idx], [pos]

    @staticmethod
    def centerout_2D_different_center(nblocks=100, distance=6.8):
        '''
        Pairs of a center target and a peripheral target. 
        The center target position changes depending on the periphetal target at each trial.
        The number of targets is fixed to 8.

        Returns
        -------
        [nblocks*ntargets x 1] array of tuples containing trial indices and [2 x 3] target coordinates
        '''
        ntargets = 8
        gen = SequenceCapture.out_2D_sequence(nblocks=nblocks, distance=distance)
        for _ in range(nblocks*ntargets):
            idx, pos = next(gen)
            targs = np.zeros([2, 3])
            if idx[0] <= 4:
                targs[0,:] = np.array([distance/2,0,0])
            else:
                targs[0,:] = np.array([-distance/2,0,0])
            targs[1,:] = pos[0]
            indices = np.zeros([2,1])
            indices[1] = idx
            yield indices, targs

    @staticmethod
    def sequence_2D(nblocks=100, distance=6.8):
        '''
        Pairs of the 1st, 2nd, and 3rd target for the sequence task.

        Returns
        -------
        [nblocks*ntargets x 1] array of tuples containing trial indices and [3 x 3] target coordinates
        '''
        ntargets = 8
        gen = SequenceCapture.out_2D_sequence(nblocks=nblocks, distance=distance)
        for _ in range(nblocks*ntargets):
            idx, pos = next(gen)
            targs = np.zeros([3, 3])
            indices = np.zeros([3,1])
            idx = idx[0]
            pos = pos[0]

            if idx >= 5:
                targs[0,:] = np.array([distance/2,0,0]).T
                indices[0] = 0 # The target for trial initiation is 0

                targs[1,:] = np.array([-distance/2,0,0]).T
                indices[1] = 4

                targs[2,:] = pos
                indices[2] = idx
            
            elif idx <= 4:
                targs[0,:] = np.array([-distance/2,0,0]).T
                indices[0] = 0 #The target for trial initiation is 0

                targs[1,:] = np.array([distance/2,0,0]).T
                indices[1] = 6

                targs[2,:] = pos            
                indices[2] = idx

            yield indices, targs

class ScreenTargetCapture_ReadySet(ScreenTargetCapture):

    '''
    Center out task with ready set go auditory cues. Cues separated by 500 ms and participant is expected to move on final go cue. Additionally, participant must move out
    of center circle (mustmv_time) parameter or there will be an error. 
    '''
    
    status = dict(
        wait = dict(start_trial="target", start_pause="pause"),
        target = dict(enter_target="hold", timeout="timeout_penalty", start_pause="pause"),
        hold = dict(leave_target="hold_penalty", hold_complete_center="prepbuff", hold_complete_periph="reward", start_pause="pause"),
        prepbuff = dict(leave_target="hold_penalty", prepbuff_complete="delay", start_pause="pause"),
        delay = dict(leave_target="delay_penalty", delay_complete="leave_center", start_pause="pause"),
        leave_center = dict(leave_target="targ_transition", mustmv_complete="tooslow_penalty", start_pause="pause"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target", start_pause="pause"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition", start_pause="pause", end_state=True),
        hold_penalty = dict(hold_penalty_end="targ_transition", start_pause="pause", end_state=True),
        tooslow_penalty = dict(tooslow_penalty_end="targ_transition", start_pause="pause", end_state=True),
        delay_penalty = dict(delay_penalty_end="targ_transition", start_pause="pause", end_state=True),
        reward = dict(reward_end="wait", start_pause="pause", stoppable=False, end_state=True),
        pause = dict(end_pause="wait", end_state=True),
    )

    # the sum of the prepbuff & the delay time should be equal to the length of the ready_set_sound file 
    # the delay time corresponds to the amount of time the peripheral target is displayed, and the prepbuff time then makes up the difference 
    wait_time = traits.Float(1., desc="Length of time in wait state (inter-trial interval)")
    mustmv_time = traits.Float(.2, desc="Must leave center target within this time after auditory go cue.")
    tooslow_penalty_time = traits.Float(1, desc="Length of penalty time for too slow error")
    shadow_periph_radius = traits.Float(0.5, desc = 'additional radius for peripheral target')
    periph_hold = traits.Float(0.2, desc = "Hold time for peripheral target")
    ready_freq = traits.Float(320, desc="Frequency of the ready-set tone")
    set_freq = traits.Float(320, desc="Frequency of the set tone")
    go_freq = traits.Float(440, desc="Frequency of the go tone")
    tone_duration = traits.Float(0.1, desc="Duration of the ready-set and go tones")
    tone_space = traits.Float(0.5, desc="Time between start tone and start of next tone")
    early_move_time = traits.Float(0.0, desc = "Time prior to go cue that user is allowed to start moving") #difference between end of delay state and go tone 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepbuff_time = 2*self.tone_space - self.delay_time - self.early_move_time
        self.ready_tone = TonePlayer(frequency=self.ready_freq, duration=self.tone_duration)
        self.set_tone = TonePlayer(frequency=self.set_freq, duration=self.tone_duration)
        self.go_tone = TonePlayer(frequency=self.go_freq, duration=self.tone_duration)
        self.pseudo_reward = 0

        # Assert that parameters are set logically 
        assert self.tone_duration < self.tone_space, "Tone duration must be less than time between tones."
        assert self.delay_time + self.early_move_time <= 2*self.tone_space, "Time of peripheral target display (delay_time) plus early move allowance should be less than or equal to length of tone sequence"
        assert self.mustmv_time >= self.tone_duration, "Allow at least length of tone duration to move after the onset of the go cue"
        
    ###Test Functions ###

    def _test_start_trial(self, time_in_state):
        '''Start next trial automatically. You may want this to instead be
            - a random delay
            - require some initiation action
        '''
        return time_in_state > self.wait_time
    
    def _test_hold_complete_center(self, time_in_state):
        '''
        Test whether the center target is held long enough to declare the
        trial a success 

        Possible options
            - Target held for the minimum requred time (implemented here)
            - Sensorized object moved by a certain amount
            - Sensorized object moved to the required location
            - Manually triggered by experimenter
        '''
        return self.target_index == 0 and time_in_state > self.hold_time
    
    def _test_hold_complete_periph(self, time_in_state):
        '''
        Test whether the peripheral target is held long enough to declare the
        trial a success 

        Possible options
            - Target held for the minimum requred time (implemented here)
            - Sensorized object moved by a certain amount
            - Sensorized object moved to the required location
            - Manually triggered by experimenter
        '''
        return self.target_index == 1 and time_in_state > self.periph_hold
    
    def _test_prepbuff_complete(self, time_in_state):
        '''
        Test whether the center target is held long enough to transition from the prepbuff time to the delay state. 
        The delay state will display the peripheral target so this state just requires a center hold.  

        '''
        return time_in_state > self.prepbuff_time 
    
    def _test_mustmv_complete(self, time_in_state):
        '''
        Test whether the target is exited in time. Return of true for mustmv sends to penalty state.  

        Possible options
            - Target left before the minimum requred time (implemented here)
            - Sensorized object moved by a certain amount
            - Sensorized object moved to the required location
            - Manually triggered by experimenter
        '''
        return time_in_state > self.mustmv_time + self.early_move_time
    
    def _test_tooslow_penalty_end(self, time_in_state):
        return time_in_state > self.tooslow_penalty_time
    
    def update_report_stats(self): #add holds completed metric to report stats
        super().update_report_stats()
        self.reportstats['Audio Completed'] = self.calc_state_occurrences('leave_center') #count if delay state completed
        self.reportstats['Pseudo Reward'] = self.pseudo_reward + self.reward_count

    ### State Functions ###
    def _start_wait(self): #necessary reset so that these parameters exist at the beginning of each trial 
        super()._start_wait()
        self.ready_played = False
        self.set_played = False
        self.go_played = False
    
    def _start_hold(self): #this addresses a potential issue if the task is solved too quickly so that the go tone is not played after the trial has been rewarded 
        super()._start_hold()
        self.ready_played = False
        self.set_played = False
        self.go_played = False
        
    def _start_prepbuff(self):
        self.epsilon = 1/self.fps #small value to account for floating point precision errors. can set this to 1/(2*self.fps) to be more permissive 

        assert abs(self.prepbuff_time - (self.tone_space - 1/self.fps)) > self.epsilon, "Prep buffer time must not be within the time between tones minus one frame rate to avoid timing issues."
        self.sync_event('CUE') #integer code 112
        self.prep_start_time = self.get_time()
        self.ready_tone.play()
        self.ready_played = True

    def _cycle(self):
        super()._cycle()

        if not self.set_played and self.ready_played and (self.get_time() - self.prep_start_time) >= self.tone_space:
            self.set_tone.play()
            self.set_played = True
            self.sync_event('CUE', 1) #integer code 113
            self.color_set_cue() #placeholder function for the color change feature 

        if not self.go_played and self.set_played and (self.get_time() - self.prep_start_time) >= 2 * self.tone_space:
            self.go_tone.play()
            self.sync_event('CUE', 2) #integer code 114
            self.targets[0].hide()
            self.go_played = True
            self.color_go_cue() #placeholder function for the color change feature
    
    def color_set_cue(self): #do nothing in normal version (no color change)
        pass

    def color_go_cue(self): #do nothing in normal version (no color change)
        pass
    
    def _start_targ_transition(self):
        super()._start_targ_transition()
        if self.target_index == -1:   # Came from a penalty state
            pass

    def _start_hold_penalty(self):
        self.pseudo_success() #run before increment trials to prevent reseting of trial index 
        if hasattr(super(), '_start_hold_penalty'):
            super()._start_hold_penalty()
        self.ready_tone.stop()
        self.set_tone.stop()
        self.go_tone.stop()
        self.ready_played = False
        self.set_played = False
        self.go_played = False
    
    def _start_delay_penalty(self):
        if hasattr(super(), '_start_delay_penalty'):
            super()._start_delay_penalty()
        self.ready_tone.stop()
        self.set_tone.stop()
        self.go_tone.stop()
        self.ready_played = False
        self.set_played = False
        self.go_played = False
    
    def _start_timeout_penalty(self):
        self.pseudo_success() #run before increment trials to prevent reseting of trial index
        super()._start_timeout_penalty()
        self.ready_played = False
        self.set_played = False
        self.go_played = False

    def _start_tooslow_penalty(self):
        self._increment_tries()
        self.sync_event('OTHER_PENALTY') #integer code 79
        self.ready_tone.stop()
        self.set_tone.stop()
        self.go_tone.stop()
        self.jack_count = 0
        self.ready_played = False
        self.set_played = False
        self.go_played = False
        for target in self.targets: #Hide Targets 
            target.hide()
            target.reset()

    def _end_tooslow_penalty(self):
        self.sync_event('TRIAL_END')
    
    def pseudo_success(self): #function to measure almost success
        if self.target_index == 1: #if peripheral target is displayed 
            target_buffer_dist = self.target_radius + self.shadow_periph_radius - self.cursor_radius #combined radius 
            dist_from_targ = np.linalg.norm(self.plant.get_endpoint_pos() - self.targs[self.target_index]) #vector difference
            if dist_from_targ <= target_buffer_dist:
                self.pseudo_reward += 1 #increment if cursor position is less than the shadow radius plus radius 

    def _start_pause(self):
        super()._start_pause()
        self.ready_tone.stop()
        self.set_tone.stop()
        self.go_tone.stop()
        self.ready_played = False
        self.set_played = False
        self.go_played = False