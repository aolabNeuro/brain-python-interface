import time
import os
import numpy as np
from riglib.experiment import traits
from riglib import quattrocento
from features.neural_sys_features import CorticalBMI
import traceback

class QuattBMI(CorticalBMI):
    '''
    BMI using quattrocento as the datasource.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._neural_src_system_type = quattrocento.EMG

    def init(self):
        self.neural_src_kwargs = dict()
        super().init()

    @property 
    def sys_module(self):
        return quattrocento   

