import numpy as np
from riglib.experiment import traits
from riglib.bmi import clda

import aopy
import glob
import os


class CLDA_KFRML_IntendedVelocity(traits.HasTraits):
    """
    Enable online CLDA updates using KFRML and an intended-velocity learner.

    This feature wires together:

    - ``clda.OFCLearnerRotateIntendedVelocity`` to estimate intended kinematics
      from task target state during task ``target`` epochs.
    - ``clda.KFRML`` to recursively update the decoder observation model from
      batches of ``(intended_kin, spike_counts)`` pairs.

    Behavior summary
    ----------------
    - Intention learning is disabled by default (``self.learn_flag = False``)
      and must be toggled on by task-level CLDA controls.
    - The learner forms batched samples at ``clda_batch_time`` cadence.
    - A position-error feedback matrix is used in ``target`` state and zero
      feedback is used in non-target states.
    - Intended velocity magnitude is controlled by ``clda_intended_speed``.
    - The updater applies exponential forgetting using
      ``clda_update_half_life``.

    Notes
    -----
    This mixin assumes a decoder with state-space metadata
    (``decoder.ssm.state_order``) and Kalman-style filter matrices
    (``decoder.filt.A``, ``decoder.filt.B``).
    """

    clda_batch_time = traits.Float(1, desc="How frequently to update weights [s]")
    clda_update_half_life = traits.Float(60, desc="Half-life for exponential decay [s] to combine with previous weights.")
    clda_intended_speed = traits.Float(5.0, desc="Nominal intended cursor speed [decoder velocity units]")

    # clda_update_batch_time = traits.Float(60, desc="How frequently to update weights [s]")
    # clda_learner_batch_time = traits.Float(60, desc="How much data to update the learner with [s]")
    # Samples to update intended kinematics with
    def create_learner(self):
        '''
        The "learner" uses knowledge of the task goals to determine the "intended"
        action of the BMI subject and pairs this intention estimation with actual observations.
        '''
        self.learn_flag = False
        n_ctrl = self.decoder.filt.B.shape[1]
        n_states = self.decoder.filt.B.shape[0]

        fmatrix = np.zeros((n_ctrl, n_states))
        pos_inds = np.where(self.decoder.ssm.state_order == 0)[0]
        n_pos = min(len(pos_inds), n_ctrl)
        if n_pos > 0:
            fmatrix[np.arange(n_pos), pos_inds[:n_pos]] = 1.0

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

        learner_batch_size = int(self.clda_batch_time / self.decoder.binlen)
        self.learner = clda.OFCLearnerRotateIntendedVelocity(
            learner_batch_size,
            self.decoder.filt.A,
            self.decoder.filt.B,
            self.decoder.filt.F_dict,
            intended_speed=self.clda_intended_speed,
        )

    def create_updater(self):
        '''
        The "updater" uses the output batches of data from the learner and an update rule to
        alter the decoder parameters to better match the intention estimates.
        '''
        self.updater = clda.KFRML(self.clda_batch_time, self.clda_update_half_life)
        self.updater.init(self.decoder)