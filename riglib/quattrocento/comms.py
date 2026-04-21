import socket
import time
import numpy as np
import pandas as pd

class IncompleteSampleError(Exception):
    def __init__(self, actual_sample, expected_sample):
        super().__init__()
        self.error_message = \
            'Incomplete sample. Received: {}. Expected: {}.'.format(
                actual_sample, expected_sample)


class Quattrocento():
    """

    Reference EMG device for high-throughput recordings.
    The user manual for the device can be found [here](
    https://www.otbioelettronica.it/attachments/article/70/
    Quattrocento%20User%20Manual%20v1.4.pdf)
    The user manual for the OTBioLab software can be found [here](https://
    www.otbiolab.it/UserManuals/User%20Manual%20OT%20BioLab+%20v1.0.pdf)
    This class will connect to the Quattrocento using a raw TCP network socket
    (see host and port defaults below), then send a configuration buffer using
    the protocol described in https://www.otbioelettronica.it/en/?preview=1&op
    tion=com_dropfiles&format&task=frontfile.download&catid=41&id=70&Itemid=10
    00000000000.
    Among other things, configuration includes the sampling rate and the number
    of electrode arrays (at 96 channels per array) to use.
    When streaming is activated, the Quattrocento is expected to send a
    continuous series of data buffers, each representing a single multi-
    channel sample and structured as contiguous 16-bit little-endian integers.
        EMG: main EMG array channels as contiguous 16-bit little-endian ints
        Aux: 16 auxiliary channels as 16-bit little-endian ints
        Acc: 8 accessory channels as 16-bit
    For example, if this reader is configured with num_arrays=2 (acquisition
    with two out of the possible four electrode arrays active) it will expect
    to receive data buffers exactly 432 bytes in length:
        EMG: 2 arrays * 96 channels per array * 2B per channel = 384 bytes
        Aux: 16 auxiliary channels * 2B per channel = 32 bytes
        Acc: 8 accessory channels * 2B per channel = 16 bytes
    With all four arrays active, each sample buffer will be 816 bytes.
    At present, the auxiliary channels are unused, and only the first
    accessory channel (sample number) is used.
    Other than the expectation that sample_num is monotonic, there are no
    documented framing or checksum structures to assist with finding buffer
    boundaries in the TCP stream. We must know the message size exactly split
    the stream at the appropriate points.
    Note that electrode numbering within the EMG segment is determined by the
    geometry of the surface electrodes used for the recording.
    """

    TIMEOUT = 5.0  # warn if no data received after this many seconds

    RECEIVE_LOOP_PERIOD = 0.05# 100 ms of data

    # communication with Quattrocento device directly
    DIRECT_PORT = 23456
    DIRECT_HOST = '169.254.1.10'

    CHANNELS_PER_ARRAY = 64
    #CHANNELS_PER_ARRAY = (384 + 16 + 8)
    ACCESSORY_CHANNELS = 16
    EXTENDED_CHANNELS = 8

    valid_stream_names = ["quattrocento", "main"]

    df_data_output = pd.DataFrame()

    def __init__(self,
                 f_samp: int = 2048,
                 num_arrays: int = 1,
                 name: str = 'Quattrocento',
                 host: str = DIRECT_HOST,
                 port: int = DIRECT_PORT,
                 emg_output: str = 'temp.npy',
                 data_output: str = 'tmp.bufer'
                 ) -> None:
        self.port = port
        self.host = host
        self.timer = name
        self._last_data_timestamp = time.time()
        self.f_samp = f_samp
        self.name = name
        self.num_arrays = num_arrays
        self.num_emg_channels = self.num_arrays * self.CHANNELS_PER_ARRAY
        self.num_sample_fields = self.num_emg_channels + \
            self.ACCESSORY_CHANNELS + self.EXTENDED_CHANNELS
        self.sample_message_size = self.num_sample_fields * 2  # 2B per field

        self.socket: socket.socket
        self.ready = False
        self.receiving = False
        self.quatt_driver = QuattrocentoDriver()

        self.emg_output = emg_output
        self.data_output = data_output

    def teardown(self):
        stop_acq_str = self.quatt_driver.create_config(
            False, False, self.f_samp, self.num_arrays)
        self.socket.send(stop_acq_str)

        self.socket.close()

    def setup(self):
        server_addr = (self.host, self.port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.settimeout(None)  # setting this to None is blocking
            self.socket.connect(server_addr)
        except Exception as e:
            print("Quattrocento: error: {}".format(e))
            raise
        print(
            "Quattrocento: {} receiver subscribing to socket {}...".format(
                self.name, server_addr))

        self.ready = True

    def _receive_data(self):
        # print("inside _receive_data..")
        # for i in range(2):

        sample_nums = []
        for i in range(int(self.f_samp * self.RECEIVE_LOOP_PERIOD)):
            try:
                # recv waits until buffer of size self.num_channels*2 is full
                data = self.socket.recv(self.sample_message_size,
                                        socket.MSG_WAITALL)  # bytes to read
                # data = self.socket.recv(self.sample_message_size)  # bytes to read

                if len(data) < (self.sample_message_size):
                    raise IncompleteSampleError(
                        len(data), self.num_sample_fields * 2)

            except IncompleteSampleError as e:
                print(f'Quattrocento: {e.error_message}')
                return

            except OSError as e:
                print("Quattrocento: error: {}".format(e))
                data_lag = time.time() - self._last_data_timestamp
                if data_lag >= self.TIMEOUT:
                    print(
                        f"Quattrocento: No data received from {self.name} in"
                        " the last {self.TIMEOUT} seconds.")
                    self._last_data_timestamp = time.time()
                return

            self._last_data_timestamp = time.time()
            emg, sample_num = self._decode_message(data)
            # print("emg: ", type(emg))

        # ### SAVING FUNCTIONS ####
        # with open(self.emg_output, "ab") as f:
        #     np.savetxt(f, [emg], delimiter=',')

        # # https://stackoverflow.com/questions/30376581/save-numpy-array-in-append-mode
        # print(f'Saving to {self.data_output}')
        # f = open(self.data_output, 'ab') # 'tmp.bufer'
        # f.write(data)
        # f.close()

        return emg, sample_nums


    # written by Gus for labgraph
    def get_state(self):
        print("inside get_state..")
        # for i in range(2):
        for i in range(int(self.f_samp * self.RECEIVE_LOOP_PERIOD)):
            print("int i: ", i)
            try:
                # recv waits until buffer of size self.num_channels*2 is full
                data = self.socket.recv(self.sample_message_size,
                                        socket.MSG_WAITALL)  # bytes to read
                # data = self.socket.recv(self.sample_message_size)  # bytes to read

                if len(data) < (self.sample_message_size):
                    raise IncompleteSampleError(
                        len(data), self.num_channels * 2)

            except IncompleteSampleError as e:
                print(f'Quattrocento: {e.error_message}')
                return

            except OSError as e:
                print("Quattrocento: error: {}".format(e))
                data_lag = time.time() - self._last_data_timestamp
                if data_lag >= self.TIMEOUT:
                    print(
                        f"Quattrocento: No data received from {self.name} in"
                        " the last {self.TIMEOUT} seconds.")
                    self._last_data_timestamp = time.time()
                return

            self._last_data_timestamp = time.time()
            emg, sample_num = self._decode_message(data)
            print("returning emg")
            return emg, sample_num


    def start_acq(self):
        start_acq_str = self.quatt_driver.create_config(
            True, False, self.f_samp, self.num_arrays)
        self.socket.send(start_acq_str)

    def stream_transform(self):
        if self.ready and not self.receiving:
            self.start_acq()
            self.receiving = True
        # print("starting receive data")
        emg, sample_num = self._receive_data()
        return emg,sample_num

    # for use in calling from labgraph
    def start_stream(self):
        if self.ready and not self.receiving:
            self.start_acq()
            self.receiving = True
       

    def _decode_message(self, message):
        # Convert the EMG section directly into a Numpy array using frombuffer
        # with int16, little endian arrangement
        dt = np.dtype('int16')
        dt = dt.newbyteorder('<')
        emg = np.frombuffer(message, dtype=dt)
        # Pick out the sample number, which is the first item in the extended
        # channel section, and must be interpreted as an unsigned short.
        #sample_num = struct.unpack('<H', message[-16:-14])[0]
        print(emg.shape)
        sample_num = emg[-self.EXTENDED_CHANNELS]

        return emg[:self.num_emg_channels], sample_num

    def _output(self, emg, sample_num, timestamp):
        sample = {
            "TIME": timestamp,
            "MACHINE_TIME": time.time(),
            "EMG": emg,
            "SAMPLE": sample_num
        }


class QuattrocentoDriver():
    """Driver for interfacing with Quattrocento device. The driver was
    designed based on the configuration protocol from the manufacturer,
    which can be found [here](https://www.otbioelettronica.it/en/
    ?preview=1&option=com_dropfiles&format=&task=frontfile.download
    &catid=41&id=70&Itemid=1000000000000)
    """

    # BYTE 1: ACQ_SETT
    ACQ_OFF = np.uint8(0b00000000)
    ACQ_ON = np.uint8(0b00000001)

    ALL_INPUTS_ACTIVE = np.uint8(0b00000110)
    THREE_FOURTHS_INPUTS_ACTIVE = np.uint8(0b00000100)
    HALF_INPUTS_ACTIVE = np.uint8(0b00000010)
    ONE_FOURTH_INPUTS_ACTIVE = np.uint8(0b00000000)

    SAMPLING_FREQ_512 = np.uint8(0b00000000)
    SAMPLING_FREQ_2048 = np.uint8(0b00001000)
    SAMPLING_FREQ_5120 = np.uint8(0b00010000)
    SAMPLING_FREQ_10240 = np.uint8(0b00011000)

    RECORDING_ON = np.uint8(0b00100000)
    RECORDING_OFF = np.uint8(0b00000000)

    DECIMATOR_ON = np.uint8(0b01000000)
    DECIMATOR_OFF = np.uint8(0b00000000)

    # BYTES 2 and 3: analog outputs
    AN_OUT_IN_SEL = np.uint8(0b00000000)
    AN_OUT_CH_SEL = np.uint8(0b00000000)

    # BYTES 4 - 39: configure 8, 16-chnl IN & 4, 64-chnl MULTIPLE_IN ports
    IN_CONFIG_0 = np.uint8(0b00000000)

    CONFIG_1_ADAPTER_64_CHANNEL = np.uint8(0b00000100)

    CONFIG_2_SIDE_NOT_DEF = np.uint8(0b00000000)
    CONFIG_2_HPF_03HZ = np.uint8(0b00000000)
    CONFIG_2_HPF_10HZ = np.uint8(0b00010000)
    CONFIG_2_LPF_4400HZ = np.uint8(0b00001100)
    CONFIG_2_LPF_900HZ = np.uint8(0b00001000)
    CONFIG_2_MODE_MONO = np.uint8(0b00000000)
    CONFIG_2_MODE_BI = np.uint8(0b00000010)
    CONFIG_2_MODE_DIFF = np.uint8(0b00000001)

    # CRC CHECKSUM
    def crc8(self, vals):
        crc = np.uint8(0)
        for v in vals:
            for j in range(8):
                s = np.not_equal(np.mod(crc, 2), np.mod(v, 2))
                crc = np.uint8(np.floor_divide(crc, 2))
                if s:
                    crc = np.uint8(np.bitwise_xor(crc, 140))
                v = np.uint8(np.floor_divide(v, 2))
        return crc

    def create_config(self, start_acq, start_rec, f_samp, num_arrays):
        if start_acq:
            acq = self.ACQ_ON
        else:
            acq = self.ACQ_OFF

        if start_rec:
            rec = self.RECORDING_ON
        else:
            rec = self.RECORDING_OFF

        num_arrays_to_inputs = {
            1: self.ONE_FOURTH_INPUTS_ACTIVE,
            2: self.HALF_INPUTS_ACTIVE,
            3: self.THREE_FOURTHS_INPUTS_ACTIVE,
            4: self.ALL_INPUTS_ACTIVE
        }

        inputs = num_arrays_to_inputs[num_arrays]

        sampling_rate_selector = {
            512: self.SAMPLING_FREQ_512,
            2048: self.SAMPLING_FREQ_2048,
            5120: self.SAMPLING_FREQ_5120,
            10240: self.SAMPLING_FREQ_10240
        }

        sampling_rate = sampling_rate_selector[f_samp]

        acq_sett_list = [
            np.uint8(0b10000000), acq, inputs, sampling_rate, rec,
            self.DECIMATOR_OFF
        ]
        ACQ_SETT = np.bitwise_or.reduce(acq_sett_list)
        config_list = np.array(ACQ_SETT, dtype=np.uint8)

        config_list = np.append(config_list, self.AN_OUT_IN_SEL)
        config_list = np.append(config_list, self.AN_OUT_CH_SEL)

        # Set up input ports
        IN_CONFIG_1_ON = self.CONFIG_1_ADAPTER_64_CHANNEL
        IN_CONFIG_1_OFF = np.uint8(0b00000000)

        config_2_list = [
            self.CONFIG_2_SIDE_NOT_DEF, self.CONFIG_2_HPF_10HZ,
            self.CONFIG_2_LPF_900HZ, self.CONFIG_2_MODE_MONO
        ]

        IN_CONFIG_2_ON = np.bitwise_or.reduce(config_2_list)
        IN_CONFIG_2_OFF = np.uint8(0b00000000)

        # Set up 8 IN ports
        for i in range(num_arrays * 2):
            config_list = np.append(config_list, self.IN_CONFIG_0)
            config_list = np.append(config_list, IN_CONFIG_1_ON)
            config_list = np.append(config_list, IN_CONFIG_2_ON)

        for i in range(8 - (num_arrays * 2)):
            config_list = np.append(config_list, self.IN_CONFIG_0)
            config_list = np.append(config_list, IN_CONFIG_1_OFF)
            config_list = np.append(config_list, IN_CONFIG_2_OFF)

        # Set up 4 MULTIPLE IN ports
        for i in range(num_arrays):
            config_list = np.append(config_list, self.IN_CONFIG_0)
            config_list = np.append(config_list, IN_CONFIG_1_ON)
            config_list = np.append(config_list, IN_CONFIG_2_ON)

        for i in range(4 - num_arrays):
            config_list = np.append(config_list, self.IN_CONFIG_0)
            config_list = np.append(config_list, IN_CONFIG_1_OFF)
            config_list = np.append(config_list, IN_CONFIG_2_OFF)

        crc = self.crc8(config_list)
        config_list = np.append(config_list, crc)
        config_list = config_list.tobytes()

        return config_list


class QuattroOtlight:    

    emg_channels = 384
    aux_channels = 16
    accessory_channels = 8

    def __init__(self, host='127.0.0.1', port=31000, emg_ch_range=[16*8,16*8+64], refresh_freq=1, sample_freq=2048):
        '''
        Choose the same refresh frequency and sampling frequency as the settings in OTBioLabLight. The EMG channel range
        describes the location of the electrodes in use. Default is MULTI IN 1 channels.

        Parameters
        ----------
        Host should be set to the computer running the OTBioLab Light
        Refresh rate should be set as high as possible (32) and match what is set in OTBioLab
        Sample rate should be set to match what is set in OTBioLab
        '''
        self.host = host
        self.port = port
        self.emg_ch_range = emg_ch_range
        self.refresh_freq = refresh_freq
        self.sample_freq = sample_freq

        self.sample_message_size = self.emg_channels + self.aux_channels + self.accessory_channels
        self.sample_message_size_in_bytes = int(self.sample_message_size * 2) # two bytes per number
        self.batch_size = int(self.sample_freq // self.refresh_freq)

    def setup(self):
        server_addr = (self.host, self.port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.settimeout(None)  # setting this to None is blocking
            self.socket.connect(server_addr)
        except Exception as e:
            print("Quattrocento: error: {}".format(e))
            raise
        print(
            "Quattrocento: {} receiver subscribing to socket {}...".format(
                'quattro', server_addr))

        self.socket.send('startTX'.encode())
        data = self.socket.recv(8).decode() # read in 8 bytes to confirm connection
        if data != 'OTBioLab':
            raise Exception(f'cannot connect to the device at {server_addr}')
        
        print(f'connected to the device ')
        time.sleep(0.1)

    def read_emg(self):
        """
        read in a batch of emg  and decode into a tuple of 
        (emg_channels, aux_channels, sample_counter)
        """
        data = self.socket.recv(self.sample_message_size_in_bytes *self.batch_size,
                                socket.MSG_WAITALL )  # bytes to read
        dt = np.dtype('int16')
        dt = dt.newbyteorder('<')
        # data.shape here is (64, 408) - 408 = 384 (emg channels) + 16 (aux) + 8 (accessory)
        data = np.frombuffer(data, dtype=dt).reshape((-1, self.sample_message_size,))
        # data is (64, 408) - so datapoints x num channels
        
        return self._decode_emg(data)


    def _decode_emg(self, data:np.ndarray):
        """
        decompose data into actual emg, aux and accessory channels
        data is num_data_points by num_channels
        """
        emg_signals  = data[:, :self.emg_channels]
        aux_signals = data[:, self.emg_channels:self.emg_channels+self.aux_channels]
        sample_counter = data[:, -self.accessory_channels:]

        # select only the output channels
        emg_signals = emg_signals[:, self.emg_ch_range[0]:self.emg_ch_range[1]]

        return (emg_signals, aux_signals, sample_counter)

    def tear_down(self):
        self.socket.send('stopTX'.encode())

        self.socket.close()

class QuattroDummy(QuattroOtlight):
    
    supported_dummy_data_types = ['zeros','random']
    
    def __init__(self, chan_range = None, dummy_data_type = 'zeros') -> None:
        
        super().__init__()
        
        if dummy_data_type in self.supported_dummy_data_types: 
            self.dumm_data_type = dummy_data_type
        else:
            raise Exception(f'{dummy_data_type} is not in supported dummy data types {self.supported_dummy_data_types}')
        

    def setup(self):
        pass

    def read_emg(self):
        """
        read in a batch of emg  and decode into a tuple of 
        (emg_channels, aux_channels, sample_counter)
        """
        start_time = time.time()
        
        if self.dumm_data_type == "zeros":
            data = np.zeros((self.batch_size, self.sample_message_size))
        elif self.dumm_data_type == "random":
            data = np.random.rand(self.batch_size, self.sample_message_size)
        else:
            raise NotImplementedError(f'has not implemented for dummy_data_type {self.dumm_data_type}')
        

        decoded_data = self._decode_emg(data)
        end_time = time.time()
        loop_time = end_time - start_time
        
        # empirically subtract 0.01 to ensure the actual refresh_freq = 32 Hz
        time.sleep(1/self.refresh_freq - loop_time - 0.01) 
        return decoded_data
    
    def tear_down(self):
        pass
    

def test_quattrolight():

    qt = QuattroOtlight(host='128.95.215.191', refresh_freq=32)
    qt.setup()
    prev_time = time.perf_counter()

    N_reads = 100
    for _  in range(N_reads):
        qt.read_emg()

    post_time = time.perf_counter()
    print(f'it takes {post_time - prev_time} to do {N_reads} reads')
    print(f'mean read time is {(post_time - prev_time) / N_reads}s')
    print(f'sampling freq is then {N_reads/(post_time - prev_time)}')
    qt.tear_down()

def test_quattodummy():
    qt = QuattroDummy()
    qt.setup()
    prev_time = time.perf_counter()

    N_reads = 10
    for _  in range(N_reads):
        qt.read_emg()

    post_time = time.perf_counter()
    print(f'it takes {post_time - prev_time} to do {N_reads} reads')
    print(f'mean read time is {(post_time - prev_time) / N_reads}s')
    print(f'sampling freq is then {N_reads/(post_time - prev_time)}')
    qt.tear_down()

def test_quottro_functions():
    path = 'C:\\Users\\aolab\\'
    name = 'my_test'
    version = '_v1'
    data_output = path + name + version + '.buffer'
    emg_output = path + name + version + '.csv'
    qt = Quattrocento(emg_output=emg_output, data_output=data_output)
    qt.setup()
    qt.teardown()
    qt.setup()
    
    timestamps = []
    prior_time = time.perf_counter()
    for _ in range(1):
        current_time  = time.perf_counter()
        emg = qt.stream_transform()
        timestamps.append(current_time - prior_time)
        prior_time = current_time

    qt.teardown()
    print(emg)



if __name__ == "__main__":

    test_quattrolight()

    #test_quottro_functions()