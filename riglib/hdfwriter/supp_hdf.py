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
        self.filename = 'tmp_'+str(source)+str(dt.year)+str(dt.month)+str(dt.day)+'_'+tm.isoformat()+'.hdf'
        self.h5_file = tables.open_file(os.path.join(data_dir, self.filename), "w")

        #If sink datatype is not specified: 
        if sink_dtype is None:
            self.dtype = np.dtype([('data',       np.float64),
                              ('ts_arrival', np.float64)])

            self.send_to_sinks_dtype = np.dtype([('chan'+str(chan), self.dtype) for chan in channels])

        else:
            self.send_to_sinks_dtype = sink_dtype
        self.supp_data = self.h5_file.create_table("/", "data", self.send_to_sinks_dtype, filters=compfilt)

    def add_data(self, data):
        self.supp_data.append(data)

    def close_data(self):
        self.h5_file.close()
        print("Closed supplementary hdf file")
