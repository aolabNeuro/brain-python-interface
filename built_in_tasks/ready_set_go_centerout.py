from .target_capture_task import ScreenTargetCapture
from riglib.experiment import traits
import os
import numpy as np
from riglib.audio import AudioPlayer

audio_path = os.path.join(os.path.dirname(__file__), '../riglib/audio')

class ScreenTargetCapture_ReadySet(ScreenTargetCapture):

    '''
    Center out task with ready set go auditory cues. Cues separated by 500 ms and participant is expected to move on final go cue. Additionally, participant must move out
    of center circle (mustmv_time) parameter or there will be an error. 
    '''
    
    status = dict(
        wait = dict(start_trial="target"),
        target = dict(enter_target="hold", timeout="timeout_penalty"),
        hold = dict(leave_target="hold_penalty", hold_complete_center="prepbuff", hold_complete_periph='reward'),
        prepbuff = dict(leave_target="hold_penalty", prepbuff_complete="delay"),
        delay = dict(leave_target="delay_penalty", delay_complete="leave_center"),
        leave_center = dict(leave_target="targ_transition", mustmv_complete="tooslow_penalty"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition", end_state=True),
        hold_penalty = dict(hold_penalty_end="targ_transition", end_state=True),
        tooslow_penalty = dict(tooslow_penalty_end="targ_transition", end_state=True),
        delay_penalty = dict(delay_penalty_end="targ_transition", end_state=True),
        reward = dict(reward_end="wait", stoppable=False, end_state=True)
    )

    
    wait_time = traits.Float(1., desc="Length of time in wait state (inter-trial interval)")
    prepbuff_time = traits.Float(.2, desc="How long after completing center target hold before peripheral target appears")
    mustmv_time = traits.Float(.2, desc="Must leave center target within this time after auditory go cue.")
    tooslow_penalty_time = traits.Float(1, desc="Length of penalty time for too slow error")
    files = [f for f in os.listdir(audio_path) if '.wav' in f]
    ready_set_sound = traits.OptionsList(files, desc="File in riglib/audio to play on each trial for the go cue")
    tooslow_penalty_sound = traits.OptionsList(files, desc="File in riglib/audio to play on each must move penalty") #hold penalty is normally incorrect.wav
    shadow_periph_radius = traits.Float(0.5, desc = 'additional radius for peripheral target')
    periph_hold = traits.Float(0.2, desc = "Hold time for peripheral target")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ready_set_player = AudioPlayer(self.ready_set_sound)
        self.tooslow_penalty_player = AudioPlayer(self.tooslow_penalty_sound)
        self.pseudo_reward = 0
        #self.readyset_length = self.ready_set_player.get_length() #new to reduce number of features

        #self.prepbuff_time = self.readyset_length - self.delay_time #new
        #print(self.prepbuff_time) #new
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
        Test whether the target is held long enough to declare the
        trial a success

        Possible options
            - Target held for the minimum requred time (implemented here)
            - Sensorized object moved by a certain amount
            - Sensorized object moved to the required location
            - Manually triggered by experimenter
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
        return time_in_state > self.mustmv_time
    
    def _test_tooslow_penalty_end(self, time_in_state):
        return time_in_state > self.tooslow_penalty_time
    
    def update_report_stats(self): #add holds completed metric to report stats
        super().update_report_stats()
        self.reportstats['Audio Completed'] = self.calc_state_occurrences('leave_center') #count if delay state completed
        self.reportstats['Pseudo Reward'] = self.pseudo_reward + self.reward_count

    ### State Functions ###
    def _start_prepbuff(self):
        self.sync_event('CUE') #integer code 112
        self.ready_set_player.play() 

    def _start_leave_center(self):
        pass
        #if self.target_index == 0:   # hide center target 
            #self.targets[0].hide() 
        #     self.sync_event('TARGET_OFF', self.gen_indices[self.target_index])

    def _start_hold_penalty(self):
        self.pseudo_success() #run before increment trials to prevent reseting of trial index 
        if hasattr(super(), '_start_hold_penalty'):
            super()._start_hold_penalty()
        self.ready_set_player.stop()
    
    def _start_delay_penalty(self):
        if hasattr(super(), '_start_delay_penalty'):
            super()._start_delay_penalty()
        self.ready_set_player.stop()
    
    def _start_timeout_penalty(self):
        self.pseudo_success() #run before increment trials to prevent reseting of trial index
        super()._start_timeout_penalty()

    def _start_tooslow_penalty(self):
        self._increment_tries()
        self.sync_event('OTHER_PENALTY') #integer code 79
        self.tooslow_penalty_player.play()
        self.ready_set_player.stop()
        self.jack_count = 0
        # # Hide targets
        for target in self.targets:
            target.hide()
            target.reset()
    
    def _end_tooslow_penalty(self):
        self.sync_event('TRIAL_END')
    
    def pseudo_success(self): #function to measure almost success
        if self.target_index == 1: #if peripheral target is displayed 
            target_buffer_dist = self.target_radius + self.shadow_periph_radius #combined radius 
            dist_from_targ = np.linalg.norm(self.plant.get_endpoint_pos() - self.targs[self.target_index]) #vector difference
            if dist_from_targ <= target_buffer_dist:
                self.pseudo_reward += 1 #increment if cursor position is less than the shadow radius plus radius 
                
       