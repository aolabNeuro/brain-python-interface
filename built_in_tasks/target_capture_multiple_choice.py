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
        hold = dict(leave_target="hold_penalty", hold_complete="delay"),
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

    sequence_generators = ['dual_peripheral']

    def __init__(self, *args, **kwargs):
        kwargs['instantiate_targets'] = False
        super().__init__(*args, **kwargs)
        
        # Create three targets: center + two peripherals
        target_center = VirtualCircularTarget(target_radius=self.target_radius, 
                                            target_color=target_colors[self.target_color])
        target_periph1 = VirtualCircularTarget(target_radius=self.target_radius,
                                            target_color=target_colors[self.target_color])
        target_periph2 = VirtualCircularTarget(target_radius=self.target_radius,
                                            target_color=target_colors[self.target_color])
        
        self.targets = [target_center, target_periph1, target_periph2]
        self.chosen_target = None

    def _start_target(self):
        super()._start_target()
        
        if self.target_index == 0:
            # Show only the center target initially
            self.targets[0].move_to_position(self.targs[0])
            self.targets[0].show()
            self.sync_event('TARGET_ON', 0)

    def _start_hold(self):
        super()._start_hold()
        
        if self.target_index == 0:
            # Just entered center target
            self.sync_event('CURSOR_ENTER_TARGET', 0)
        else:
            # Entered one of the peripheral targets - record which one
            cursor_pos = self.plant.get_endpoint_pos()
            d1 = np.linalg.norm(cursor_pos - self.targs[1])
            d2 = np.linalg.norm(cursor_pos - self.targs[2])
            
            if d1 < d2:
                self.chosen_target = 1
                self.targets[2].hide()  # Hide unchosen target
            else:
                self.chosen_target = 2
                self.targets[1].hide()  # Hide unchosen target
            
            self.sync_event('CURSOR_ENTER_TARGET', self.chosen_target)
            
    def _start_delay(self):
        super()._start_delay()
        
        # After holding center, show BOTH peripheral targets
        if self.target_index == 0:  # Just finished holding center
            self.targets[1].move_to_position(self.targs[1])
            self.targets[1].show()
            
            self.targets[2].move_to_position(self.targs[2])
            self.targets[2].show()
            
            self.sync_event('TARGET_ON', [1, 2])

    def _start_reward(self):
        super()._start_reward()
        # Hide all targets
        for target in self.targets:
            target.hide()
            target.reset()

    @staticmethod
    def dual_peripheral(nblocks=100, distance=10, origin=(0,0,0)):
        '''
        Generates center target + two peripheral targets
        '''
        rng = np.random.default_rng()
        for _ in range(nblocks):
            # Generate two random angles for peripheral targets
            angles = rng.uniform(0, 2*np.pi, size=2)
            
            # Target 0: center
            center = np.array(origin)
            
            # Target 1: first peripheral
            pos1 = np.array([
                distance*np.cos(angles[0]),
                0,
                distance*np.sin(angles[0])
            ]) + origin
            
            # Target 2: second peripheral
            pos2 = np.array([
                distance*np.cos(angles[1]),
                0,
                distance*np.sin(angles[1])
            ]) + origin
            
            # Yield indices and positions for all three targets
            yield [0, 1, 2], [center, pos1, pos2]

            
    def _test_enter_target(self, ts):
        '''
        Check if cursor is in the appropriate target(s)
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        
        if self.target_index == 0:
            # First target: check only center
            d = np.linalg.norm(cursor_pos - self.targs[0])
            return d <= (self.target_radius - self.cursor_radius)
        
        else:
            # After center hold: check if in EITHER peripheral target
            d1 = np.linalg.norm(cursor_pos - self.targs[1])
            in_target1 = d1 <= (self.target_radius - self.cursor_radius)
            
            d2 = np.linalg.norm(cursor_pos - self.targs[2])
            in_target2 = d2 <= (self.target_radius - self.cursor_radius)
            
            return in_target1 or in_target2

    def _test_leave_target(self, ts):
        '''
        Check if cursor leaves current target
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        
        if self.target_index == 0:
            # Leaving center target
            d = np.linalg.norm(cursor_pos - self.targs[0])
        else:
            # Leaving whichever peripheral was chosen
            d = np.linalg.norm(cursor_pos - self.targs[self.chosen_target])
        
        rad = self.target_radius - self.cursor_radius
        return d > rad or super()._test_leave_target(ts)

    def _test_hold_complete(self, time_in_state):
        '''
        Hold complete after holding center OR after holding chosen peripheral
        '''
        return time_in_state > self.hold_time
    
    def _test_trial_complete(self, time_in_state):
        '''
        Trial complete after acquiring either peripheral target (index 1)
        '''
        return self.target_index == 1

    def _test_leave_target(self, ts):
        cursor_pos = self.plant.get_endpoint_pos()
        
        if self.target_index == 0:
            d = np.linalg.norm(cursor_pos - self.targs[0])
        elif self.chosen_target is not None:  # Add safety check
            d = np.linalg.norm(cursor_pos - self.targs[self.chosen_target])
        else:
            return False  # No target chosen yet, can't have left it
        
        rad = self.target_radius - self.cursor_radius
        return d > rad or super()._test_leave_target(ts)