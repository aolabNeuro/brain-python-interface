import tables
import numpy as np
import tempfile
import datetime
import os

compfilt = tables.Filters(complevel=5, complib="zlib", shuffle=True)

class SupplementaryHDF(object):

    def __init__(self, channels, sink_dtype, source, data_dir='/var/tmp/'):

        dt = datetime.datetime.now()
        tm = dt.time()
        subject = 'test' if not hasattr(source, 'subj') else source.subj
        saveid = 'None' if not hasattr(source, 'saveid') else source.saveid
        self.filename = f'tmp_{str(source)}_{subject}_{saveid}.hdf'
        self.h5_file = tables.open_file(os.path.join(data_dir, self.filename), "w")

        #If sink datatype is not specified: 
        if sink_dtype is None:
            self.dtype = np.dtype([('data',       np.float64),
                              ('ts_arrival', np.float64)])

            self.send_to_sinks_dtype = np.dtype([('chan'+str(chan), self.dtype) for chan in channels])

        else:
            self.send_to_sinks_dtype = sink_dtype
        self.supp_data = self.h5_file.create_table("/", "data", self.send_to_sinks_dtype, filters=compfilt)
        self.supp_data.attrs['samplerate'] = source.update_freq
        self.supp_data.attrs['channels'] = channels
        self.supp_data.attrs['dtype'] = source.dtype
        
        if hasattr(source, 'n_arrays'):
            self.supp_data.attrs['n_arrays'] = source.n_arrays

    def add_data(self, data):
        self.supp_data.append(data)

    def close_data(self):
        self.h5_file.close()
        print("Closed supplementary hdf file")
