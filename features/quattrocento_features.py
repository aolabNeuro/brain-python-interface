import time
import os
import numpy as np
from riglib.experiment import traits
from riglib import quattrocento, source
from features.neural_sys_features import CorticalBMI
import traceback

class QuattBMI(CorticalBMI):
    '''
    BMI using quattrocento as the datasource.
    '''

    def __init__(self, *args, **kwargs):
        '''
        This is a bit weird because normally only the readouts are streamed to BMI3D, but in this case
        we're streaming all the data so we can save it locally. So we need to tell the sink manager about
        all the channels (cortical_channels) and then hope the extractor will only use the channels that are
        actually readouts.
        '''
        super().__init__(*args, **kwargs)
        self.cortical_channels = np.arange(1,64+16+8+1) # 64 EMG + 16 AUX + 8 samples

        # These get read by CorticalData when initializing the extractor
        self._neural_src_type = source.MultiChanDataSource
        self._neural_src_kwargs = dict(
            send_data_to_sink_manager=self.send_data_to_sink_manager, 
            channels=self.cortical_channels)
        self._neural_src_system_type = quattrocento.EMG

    @property 
    def sys_module(self):
        return quattrocento   

