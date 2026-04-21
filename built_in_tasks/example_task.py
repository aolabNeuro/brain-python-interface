'''
Example blank tasks for creating new tasks
'''
import numpy as np

from riglib.experiment import traits, Sequence
from riglib.stereo_opengl.window import Window
from .target_graphics import *



class ExampleSequenceTask(Window, Sequence):
    """
    Show one target after another
    """

    status = dict(
        wait = dict(start_trial="target", start_pause="pause"),
        target = dict(target_done="wait", start_pause="pause", end_state=True), # end_state is just for counting trials
        pause = dict(end_pause="wait", end_state=True),
    )

    # initial state
    state = "wait"

    sequence_generators = ['example_generator']

    example_param = traits.Float(0.0, desc="Example parameter (units)")
    target_on_time = traits.Float(1.0, desc="Time to display target")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = VirtualCircularTarget(target_radius=1, target_color=[1,1,1,1])
        for model in self.target.graphics_models:
            self.add_model(model)

    def init(self):
        self.add_dtype('example_cycle_data_to_save', 'f8', (1,))
        self.trial_dtype = np.dtype([('trial', 'u4'), ('index', 'u4'), ('target', 'f8', (3,))])
        super().init()

    def _cycle(self):
        '''
        Calls any update functions necessary and redraws screen
        '''
        super()._cycle()

    #### TEST FUNCTIONS ####
    def _test_start_trial(self, ts):
        return not self.pause

    def _test_start_pause(self, ts):
        return self.pause
    
    def _test_target_done(self, ts):
        return ts > self.target_on_time
    
    def _test_end_pause(self, ts):
        return not self.pause

    def _parse_next_trial(self):
        '''Check that the generator has the required data'''
        self.target_idx, self.target_location = self.next_trial
        
        # Update the data sinks with trial information
        self.trial_record['trial'] = self.calc_trial_num()
        self.trial_record['index'] = self.target_idx
        self.trial_record['target'] =  self.target_location
        self.sinks.send("trials", self.trial_record)

    ### STATE FUNCTIONS ###
    def _start_wait(self):
        self.sync_event('TARGET_OFF')
        self.target.hide()
        super()._start_wait()

    def _start_target(self):
        self.sync_event('TARGET_ON', 0xd)
        
        self.task_data['example_cycle_data_to_save'] = 2 # maybe you want to save some specific data here
        
        print("driving target to", self.target_location)
        self.target.move_to_position(self.target_location)
        self.target.show()

    def _start_pause(self):
        self.sync_event('PAUSE_START')
        self.target.hide()

    def _end_pause(self):
        self.sync_event('PAUSE_END')

    @classmethod
    def get_desc(cls, params, log_summary):
        duration = round(log_summary['runtime'] / 60, 1)
        return "{}/{} succesful trials in {} min".format(
            log_summary['n_success_trials'], log_summary['n_trials'], duration)

    @staticmethod
    def example_generator():
        '''
        Generates a sequence of targets

        Returns
        -------
        [10 x 1] array of tuples containing trial index and [1 x 3] target coordinates

        '''
        
        for idx in range(10):
            pos = np.array([idx,0,1]).T
            yield idx, pos

