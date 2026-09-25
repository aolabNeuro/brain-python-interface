import numpy as np
from riglib.experiment import traits
from riglib.bmi import clda

import aopy
import glob
import os


class SimpleEndpointIntentionLearner(clda.Learner):
    """
    Minimal intention estimator for endpoint velocity decoders.
   
    The intended state keeps position unchanged and sets velocity to point from
    the current cursor position to the current target position.
    """
    def __init__(self, batch_size, *args, **kwargs):
        self.default_speed = kwargs.pop('default_speed', 1.0)
        super(SimpleEndpointIntentionLearner, self).__init__(batch_size, *args, **kwargs)

    def calc_int_kin(self, current_state, target_state, decoder_output, task_state, state_order=None):
        curr = np.asarray(current_state).reshape(-1)
        targ = np.asarray(target_state).reshape(-1)

        # This learner is intended for [pos(3), vel(3), offset] endpoint states.
        if curr.size < 6 or targ.size < 3:
            return None

        cursor_pos = curr[:3]
        target_pos = targ[:3]
        diff = target_pos - cursor_pos
        dist = np.linalg.norm(diff)

        if not np.isfinite(dist):
            return None

        dir_to_target = diff / (np.spacing(1) + dist)

        curr_speed = np.linalg.norm(curr[3:6])
        speed = curr_speed if curr_speed > 0 else self.default_speed

        intended = curr.copy()
        intended[:3] = cursor_pos
        intended[3:6] = speed * dir_to_target

        if intended.size >= 7:
            intended[6] = 1

        return np.asmatrix(intended).reshape(-1, 1)



class CLDA_KFRML_IntendedVelocity(traits.HasTraits):
    clda_batch_time = traits.Float(1, desc="How frequently to update weights [s]")
    clda_update_half_life = traits.Float(60, desc="Half-life for exponential decay [s] to combine with previous weights.") #[s]
    # clda_update_batch_time = traits.Float(60, desc="How frequently to update weights [s]")
    # clda_learner_batch_time = traits.Float(60, desc="How much data to update the learner with [s]") # Samples to update intended kinematics with
    def create_learner(self):
        '''
        The "learner" uses knowledge of the task goals to determine the "intended"
        action of the BMI subject and pairs this intention estimation with actual observations.
        '''
        self.learn_flag = False
        fmatrix = np.array(self.decoder.filt.B.T/np.max(self.decoder.filt.B))
        self.decoder.filt.F_dict = {
            'target': fmatrix,
            'hold': np.zeros(fmatrix.shape),
            'timeout_penalty': np.zeros(fmatrix.shape),
            'wait': np.zeros(fmatrix.shape),
            'delay': np.zeros(fmatrix.shape),
            'targ_transition': np.zeros(fmatrix.shape),
            'hold_penalty': np.zeros(fmatrix.shape),
            'delay_penalty': np.zeros(fmatrix.shape),
            'reward': np.zeros(fmatrix.shape),
        }

        learner_batch_size = int(self.clda_batch_time/self.decoder.binlen)
        self.learner = clda.OFCLearnerRotateIntendedVelocity(learner_batch_size, self.decoder.filt.A, self.decoder.filt.B, self.decoder.filt.F_dict)

    def create_updater(self):
        '''
        The "updater" uses the output batches of data from the learner and an update rule to
        alter the decoder parameters to better match the intention estimates.
        '''
        self.updater = clda.KFRML(self.clda_batch_time, self.clda_update_half_life)
        self.updater.init(self.decoder)
    

class CLDA_Smoothbatch_IntendedVelocity(traits.HasTraits):
    clda_batch_time = traits.Float(60, desc="How frequently to update weights [s]")
    clda_half_life = traits.Float(60, desc="Half-life for exponential decay [s] to combine with previous weights.") #[s]

    def create_learner(self):
        '''
        The "learner" uses knowledge of the task goals to determine the "intended"
        action of the BMI subject and pairs this intention estimation with actual observations.
        '''
        self.learn_flag = False
        learner_batch_size = int(self.clda_batch_time/self.decoder.binlen)
        self.learner = SimpleEndpointIntentionLearner(learner_batch_size)

    def create_updater(self):
        '''
        The "updater" uses the output batches of data from the learner and an update rule to
        alter the decoder parameters to better match the intention estimates.
        '''
        self.updater = clda.KFSmoothbatch(self.clda_batch_time, self.clda_half_life)
        self.updater.init(self.decoder)