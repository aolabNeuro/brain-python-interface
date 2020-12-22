from bmimultitasks import SimBMIControlMulti, SimBMICosEncKFDec
from features import SaveHDF
from features.simulation_features import get_enc_setup, SimKFDecoderRandom, SimCosineTunedEnc,SimIntentionLQRController
from riglib import experiment

import time
import numpy as np


"""
this task uses 
"""


# build a sequence generator
if __name__ == "__main__":

    #generate task params
    N_TARGETS = 8
    N_TRIALS = 6
    DECODER_MODE = 'random' # in this case we load simulation_features.SimKFDecoderRandom
    ENCODER_TYPE = 'cosine_tuned_encoder'

    seq = SimBMIControlMulti.sim_target_seq_generator_multi(
        N_TARGETS, N_TRIALS)

    #neuron set up : 'std (20 neurons)' or 'toy (4 neurons)' 
    N_NEURONS, N_STATES, sim_C = get_enc_setup(sim_mode = 'toy')

    # set up assist level
    assist_level = (0, 0)

    feats = []

    #set up intention feedbackcontroller
    #this ideally set before the encoder
    feats.append(SimIntentionLQRController)


    #set up the encoder
    if ENCODER_TYPE == 'cosine_tuned_encoder':
        feats.append(SimCosineTunedEnc)
        print(f'{__name__}: selected SimCosineTunedEnc\n')
        

    #take care the decoder setup
    if DECODER_MODE == 'random':
        base_class = SimBMIControlMulti
        feats.append(SimKFDecoderRandom)
        print(f'{__name__}: set base class ')
        print(f'{__name__}: selected SimKFDecoderRandom \n')
    else: #defaul to a cosEnc and a pre-traind KF DEC
        base_class = SimBMICosEncKFDec
        print(f'{__name__}: set baseclass to SimBMICosEncKFDec\n')


    #sav everthing in a kw
    kwargs = dict()
    kwargs['sim_C'] = sim_C

    #spawn the task
    Exp = experiment.make(base_class, feats=feats)
    print(Exp)
    exp = Exp(seq, **kwargs)
    exp.init()
    exp.run()  # start the task

