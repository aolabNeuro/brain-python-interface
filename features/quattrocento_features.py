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
        quattrocento.EMG.subj = self.subject_name
        quattrocento.EMG.saveid = self.saveid

        # These get read by CorticalData when initializing the extractor
        self._neural_src_type = source.MultiChanDataSource
        self._neural_src_kwargs = dict(
            send_data_to_sink_manager=True, 
            channels=self.cortical_channels)
        self._neural_src_system_type = quattrocento.EMG

    def cleanup(self, database, saveid, **kwargs):
        '''
        Function to run at 'cleanup' time, after the FSM has finished executing. See riglib.experiment.Experiment.cleanup
        This 'cleanup' method links the file created to the database ID for the current TaskEntry
        '''
        super_result = super().cleanup(database, saveid, **kwargs)
        # Sleep time so that the plx file has time to save cleanly
        time.sleep(2)
        dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
        filename = f'/var/tmp/tmp_{str(quattrocento.EMG)}_{self.subject_name}_{saveid}.hdf'
        print(f"Saving {filename} to database {dbname}")
        if saveid is not None:    
            if dbname == 'default':
                database.save_data(filename, "emg", saveid, True, False)
            else:
                database.save_data(filename, "emg", saveid, True, False, dbname=dbname)
        else:
            print('\n\nPlexon file not found properly! It will have to be manually linked!\n\n')
        return super_result

    @property 
    def sys_module(self):
        return quattrocento   

