import numpy as np
from riglib.experiment import traits
from riglib.bmi import clda

import aopy
import glob
import os



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