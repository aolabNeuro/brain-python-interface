import numpy as np
import random
import os

from .target_graphics import *
from .target_capture_task import ScreenTargetCapture
from riglib.experiment import traits

class TwoChoiceTargetCapture(ScreenTargetCapture):
    '''
    Add a penalty state when subjects looks away.
    '''

    status = dict(
        wait = dict(start_trial="target"),
        target = dict(timeout="timeout_penalty",
                      enter_target="hold"),
        hold = dict(leave_target="target"),
        delay = dict(leave_target="delay_penalty", 
                     delay_complete="targ_transition"),
        targ_transition = dict(trial_complete="reward", 
                               trial_abort="wait", 
                               trial_incomplete="target"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition", 
                               end_state=True),
        hold_penalty = dict(hold_penalty_end="targ_transition",
                            end_state=True),
        delay_penalty = dict(delay_penalty_end="targ_transition",
                             end_state=True),
        reward = dict(reward_end="wait",
                      stoppable=False, 
                      end_state=True),
    )

    def _start_target(self):
        super()._start_target()

        #show two targets at trial start

        