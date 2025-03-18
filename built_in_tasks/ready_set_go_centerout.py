from .target_capture_task import ScreenTargetCapture
from riglib.experiment import traits
import os
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
        leave_center = dict(leave_target="targ_transition", mustmv_complete="hold_penalty"),
        targ_transition = dict(trial_complete="reward", trial_abort="wait", trial_incomplete="target"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition", end_state=True),
        hold_penalty = dict(hold_penalty_end="targ_transition", end_state=True),
        delay_penalty = dict(delay_penalty_end="targ_transition", end_state=True),
        reward = dict(reward_end="wait", stoppable=False, end_state=True)
    )

    #exclude_parent_traits = ['delay_time']

    prepbuff_time = traits.Float(.2, desc="How long after acquiring center target before peripheral target appears")
    mustmv_time = traits.Float(.2, desc="Must leave center target within this time after auditory go cue.")
    
    files = [f for f in os.listdir(audio_path) if '.wav' in f]
    ready_set_sound = traits.OptionsList(files, desc="File in riglib/audio to play on each reward")


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ready_set_player = AudioPlayer(self.ready_set_sound)
        # sound_time = self.ready_set_player.get_length()
        # self.delay_time = 
        # self.prepbuff_time = (sound_time + mustmv_time) - self.delay_time

    ###Test Functions ###
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
        return self.target_index == 1 and time_in_state > self.hold_time
    
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

    ### State Functions ###
    def _start_prepbuff(self):

        #self.sync_event('CURSOR_LEAVE_TARGET', self.gen_indices[self.target_index])
        self.ready_set_player.play()

    def _start_leave_center(self):

        if self.target_index == 0:
            #self.targets[0].hide()
            self.sync_event('CENTER_TARGET_OFF', self.gen_indices[self.target_index])

    def _start_hold_penalty(self):
        if hasattr(super(), '_start_hold_penalty'):
            super()._start_hold_penalty()
        self.ready_set_player.stop()
    
    def _start_delay_penalty(self):
        if hasattr(super(), '_start_delay_penalty'):
            super()._start_delay_penalty()
        self.ready_set_player.stop()