import datetime
import time
import json
import socket
import select
import os
import multiprocessing as mp
import threading
import queue
import traceback
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.widgets import Button, Slider, Cursor
import aopy
from riglib.ecube import MultiSource, map_channels_for_multisource
from riglib.source import MultiChanDataSource

class OnlineDataWorker(threading.Thread):
    '''
    Worker process to receive data from BMI3D.
    '''

    def __init__(self, socket, result_queue):
        '''
        Start the server on the specified IP address and port number.
        '''
        self.sock = socket
        self._stop_event = threading.Event()
        self.result_queue = result_queue
        super().__init__()

    def run(self):
        while not self._stop_event.is_set():
            ready = select.select([self.sock], [], [], 0.1) # 100ms timeout to check _stop_event
            if ready[0]:
                data = self.sock.recv(4096)
                key, value = data.decode('utf-8').split('%')
                self.result_queue.put((key, [json.loads(v) for v in value.split('#')]))

    def stop(self):
        self._stop_event.set()

class AnalysisWorker(mp.Process):
    '''
    Plots eye, cursor, and target data from experiments that have them. Performs automatic
    calibration of eye data to target locations when the cursor enters the target.
    '''

    def __init__(self, task_params, data_queue, figsize=(8,8), update_rate=60, fps=60):
        self.task_params = task_params
        self._stop_event = mp.Event()
        self.data_queue = data_queue
        self.figsize = figsize
        self.update_rate = update_rate
        self.fps = fps
        super().__init__()

    def init(self):
        '''
        Initialize the worker. 
        '''
        self.cycle_count = 0

    def handle_data(self, key, values):
        '''
        Do something with incoming data. By default just keeps track of the time
        '''
        if key == 'cycle_count':
            self.cycle_count = values[0]        

    def draw(self):
        '''
        Update the figure.
        '''
        if 'fps' in self.task_params:
            self.time_text.set_text(f"t={int(self.cycle_count/self.task_params['fps'])} s")
        else:
            self.time_text.set_text(f"t={int(self.cycle_count)} cycles")

    def cleanup(self):
        '''
        Cleanup tasks after the experiment ends, e.g. saving the figure.
        '''
        pass

    def update(self):
        while True:
            try:
                key, values = self.data_queue.get(timeout=0.) # continue if no data
                self.handle_data(key, values)
            except queue.Empty:
                break

    def run(self):
        print('Starting analysis worker:', self.__class__.__name__)
        
        # Initialize figure
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_axes([0, 0.2, 1, 0.7])
        experiment_name = self.task_params.get('experiment_name', 'None')
        te_id = self.task_params.get('te_id', 'None')
        self.fig.canvas.manager.set_window_title(f"{experiment_name} ({te_id}) - {self.__class__.__name__}")
        self.time_text = self.ax.text(1., 1.05, '', ha='right', va='center', fontsize=12, transform=self.ax.transAxes)
        self.time_text.set_text('Waiting for data...')
        self.init()
                
        # Pop up the figure
        plt.show(block=False)
        plt.pause(0.1)

        t_update = time.perf_counter()
        while not self._stop_event.is_set():
            if time.perf_counter() - t_update > 1./self.update_rate:
                try:
                    self.update()
                except:
                    traceback.print_exc()
                t_update = time.perf_counter()
            self.draw()
            self.time_text.set_text(f"t={int(self.cycle_count/self.task_params['fps'])}s")
            plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.start_event_loop(1./self.fps) 

        self.cleanup()
        plt.close(self.fig)
        print(self.__class__.__name__, 'stopped')

    def stop(self):
        self._stop_event.set()


class BehaviorAnalysisWorker(AnalysisWorker):
    '''
    Plots eye, cursor, and target data from experiments that have them. Performs automatic
    calibration of eye data to target locations when the cursor enters the target if no
    calibration coefficients are available. 
    '''
   
    def __init__(self, task_params, data_queue, calibration_dir='/var/tmp', buffer_time=10, ylim=1, px_per_cm=51.67, **kwargs):
        super().__init__(task_params, data_queue, **kwargs)
        self.calibration_dir = calibration_dir
        self.buffer_time = buffer_time
        self.ylim = ylim
        self.px_per_cm = px_per_cm

    def init(self):
        super().init()
        self.cursor_pos = np.zeros(2)
        self.eye_pos = np.zeros(2)
        self.target_pos = {}
        self.targets = {}
        self.calibration_data = []
        self.calibration_flag = True
        self.eye_coeff = np.array([[1,0],[1,0]])
        self.eye_coeff_corr = 0.5 # Don't accept anything lower than 0.5 by default
        
        self.eye_diam = np.zeros((int(self.buffer_time*self.task_params['fps']), int(2)))

        # Load previous calibration if it exists
        subject = self.task_params.get('subject_name', 'None')
        self.calibration_filename = f'calib_{subject}_{datetime.date.today()}.pkl'
        filepath = os.path.join(self.calibration_dir, self.calibration_filename)
        if os.path.exists(filepath):
            self.eye_coeff, self.eye_coeff_corr = aopy.data.pkl_read(self.calibration_filename, self.calibration_dir)
            self.calibration_flag = False

        # Turn off automatic calibration if this isn't a MC task
        if 'target_radius' not in self.task_params or self.task_params['target_radius'] >= 10:
            self.calibration_flag = False
        if 'rotation' not in self.task_params or self.task_params['rotation'] != 'yzx':
            self.calibration_flag = False

        # Set up figure
        bounds = self.task_params.get('cursor_bounds', (-10,10,0,0,-10,10))
        self.ax.set_xlim(bounds[0], bounds[1])
        self.ax.set_ylim(bounds[-2], bounds[-1])
        self.ax.set_aspect('equal')
        self.circles = PatchCollection([])
        self.ax.add_collection(self.circles)

        # Set up eye diameter figure  
        self.diam_ax = self.fig.add_axes([0.1, 0.06, 0.8, 0.11])
        self.diam_ax.set_ylim(0, self.ylim)
        self.diam_ax.set_xlim(-self.buffer_time, 0)
        self.diam_ax.set_xlabel('Time (s)')
        self.diam_ax.set_ylabel('Eye Diameter (cm)')
        self.diam_plot = self.diam_ax.plot([], [], 'green')[0]


    def update_eye_calibration(self):
        '''
        Update the eye calibration coefficients using the collected data
        '''
        if self.calibration_flag and len(self.calibration_data) > 3:
            eye_data, cursor_data = zip(*self.calibration_data)
            eye_data = np.array(eye_data)
            cursor_data = np.array(cursor_data)
            slopes, intercepts, correlation_coeff = aopy.analysis.fit_linear_regression(eye_data, cursor_data)
            if np.all(abs(correlation_coeff) > self.eye_coeff_corr):
                self.eye_coeff_corr = np.min(abs(correlation_coeff))
                self.eye_coeff = np.vstack((slopes, intercepts)).T
                print('updated calibration:', self.eye_coeff, self.eye_coeff_corr)
            
    def get_current_pos(self):
        '''
        Get the current cursor, eye, and target positions

        Returns:
            cursor_pos ((2,) tuple): Current cursor position
            eye_pos ((2,) tuple): Current eye position and diameters
            targets (list): List of active targets in (position, radius, color) format
        '''
        calibrated_eye_pos = aopy.data.get_calibrated_eye_data(self.eye_pos[0:2], self.eye_coeff)
        try:
            radius = self.task_params['target_radius']
            color = self.task_params['target_color']
            targets = [(self.target_pos[k], radius, color if v == 1 else 'green') for k, v in self.targets.items() if v]
        except:
            targets = []
        return self.cursor_pos, calibrated_eye_pos, targets

    def handle_data(self, key, values):
        super().handle_data(key, values)
        if key == 'sync_event':
            event_name, event_data = values
            if event_name == 'TARGET_ON':
                self.targets[event_data] = 1
            elif event_name == 'TARGET_OFF':
                self.targets[event_data] = 0
            elif event_name in ['PAUSE', 'TRIAL_END', 'HOLD_PENALTY', 'DELAY_PENALTY', 'TIMEOUT_PENALTY']:
                # Clear targets at the end of the trial
                self.targets = {}
            elif event_name == 'REWARD':
                # Set all active targets to reward
                for target_idx in self.targets.keys():
                    self.targets[target_idx] = 2 if self.targets[target_idx] else 0
            elif event_name == 'CURSOR_ENTER_TARGET' and event_data > 0:
                self.calibration_data.append((self.eye_pos, self.cursor_pos))
                self.update_eye_calibration()
        elif key == 'cursor':
            self.cursor_pos = np.array(values[0])[[0,2]]
        elif key == 'eye_pos':
            self.eye_pos = np.array(values[0])[:2]

            # Update eye diameter
            self.temp = np.array(values[0])[4:]
            self.eye_diam = np.roll(self.eye_diam, -1, axis=0)
            self.eye_diam[-1] = self.temp

        elif key == 'target_location':
            target_idx, target_location = values
            self.target_pos[int(target_idx)] = np.array(target_location)[[0,2]]

    def draw(self):
        super().draw()
        cursor_pos, eye_pos, targets = self.get_current_pos()
        cursor_radius = self.task_params.get('cursor_radius', 0.25)
        print(self.eye_diam[0])

        patches = [
            plt.Circle(cursor_pos, cursor_radius), 
            plt.Circle(eye_pos, self.eye_diam[-1, 0]/self.px_per_cm)
        ] + [plt.Circle(pos, radius) for pos, radius, _ in targets]
        self.circles.set_paths(patches)
        colors = ['b', 'g'] + [c for _, _, c in targets]
        self.circles.set_facecolor(colors)
        self.circles.set_alpha(0.5)

        # Update eye diameter plot
        self.diam_plot.set_data(np.arange(len(self.eye_diam)) * 1/(int(self.task_params['fps'])) - 10, 
                                self.eye_diam[:, 0]/self.px_per_cm)

    def cleanup(self):

        # Save the calibration if it was performed
        filepath = os.path.join(self.calibration_dir, self.calibration_filename)
        if self.calibration_flag and not np.array_equal(self.eye_coeff, np.array([[1,0],[1,0]])):
            aopy.data.pkl_write(self.calibration_filename, (self.eye_coeff, self.eye_coeff_corr), self.calibration_dir)

class ERPAnalysisWorker(AnalysisWorker):
    '''
    Plots ERP data from experiments with an ECoG244 array. Automatically calculates 
    ERPs for flash, movement, or laser events depending on the task.
    '''
    bufferlen = 5 # seconds of data to keep in the buffer

    def __init__(self, task_params, data_queue, update_rate=1, time_before=0.05, 
                 time_after=0.1, figure_dir='/home/aolab/figures', **kwargs):
        super().__init__(task_params, data_queue, update_rate=update_rate, **kwargs)
        self.time_before = time_before
        self.time_after = time_after
        self.figure_dir = figure_dir

    def init(self):
        super().init()

        # Initialize the data source
        self.elec_pos, self.acq_ch, _ = aopy.data.load_chmap('ECoG244')
        self.clock_dch = self.task_params.get('screen_sync_dch', 40)
        self.clock_elapsed = 0
        self.clock_times = np.array([])
        self.trigger_dch = None
        self.trigger_events = []
        self.trigger_cycles = []
        self.trigger_times = []
        self.lfp_downsample = 25

        if 'qwalor_trigger_dch' in self.task_params:
            self.trigger_dch = self.task_params['qwalor_trigger_dch']
            channels = map_channels_for_multisource(headstage_channels=self.acq_ch, 
                                                         digital_channels=[self.clock_dch, self.trigger_dch])
        else:
            self.trigger_events.append('TARGET_ON') # For flash
            channels = map_channels_for_multisource(headstage_channels=self.acq_ch, digital_channels=[self.clock_dch])
        self.ds = MultiChanDataSource(MultiSource, channels=channels, bufferlen=self.bufferlen)
        self.ds.start()
        print('datasource started')

        # Initialize the ERP data
        self.erp = np.zeros((int((self.time_before + self.time_after) * self.ds.source.update_freq/self.lfp_downsample), 
                             len(self.elec_pos), 0), dtype=self.ds.source.dtype)
        
        # Initialize the figure
        self.data_map = np.zeros((16,16))
        self.erp_im = aopy.visualization.plot_spatial_map(self.data_map, self.elec_pos[:,0], self.elec_pos[:,1], 
                                                 cmap='bwr', ax=self.ax)
        self.erp_im.set_clim(-100, 100)
        self.erp_text = self.ax.text(0.75, 1.05, '.', ha='center', va='center', fontsize=12, transform=self.ax.transAxes)

        # Add a slider to control the range
        ax_slider = self.fig.add_axes([0.225, 0.075, 0.6, 0.03])
        self.slider = Slider(
            ax=ax_slider,
            label='Range (uV)',
            valmin=0.1,
            valmax=1000,
            valinit=100
        )
        self.slider.on_changed(lambda val: self.erp_im.set_clim(-val, val))

        # And a button to show/hide channel numbers
        self.labels = []
        elec_pos, acq_ch, _ = aopy.data.load_chmap('ECoG244')
        stim_pos, stim_ch, _ = aopy.data.load_chmap('Opto32')
        for pos, ch in zip(elec_pos, acq_ch):
            self.labels.append(aopy.visualization.annotate_spatial_map(pos, ch, 'c', 12, self.ax))
        for pos, ch in zip(stim_pos, stim_ch):
            self.labels.append(aopy.visualization.annotate_spatial_map(pos, ch, 'm', 12, self.ax))
        def toggle_labels(event):
            for label in self.labels:
                label.set_visible(not label.get_visible())
        toggle_labels(None)
        ax_toggle = self.fig.add_axes([0.1, 0.025, 0.1, 0.025])
        self.toggle = Button(ax_toggle, 'Labels')
        self.toggle.on_clicked(toggle_labels)

    def update(self):
        super().update()
        # Keep track of clock cycles
        clock_data = self.ds.get_new(map_channels_for_multisource(digital_channels=[self.clock_dch]))[0]
        if len(clock_data) == 0:
            return # no new data
        timestamps, edges = aopy.utils.detect_edges(clock_data, self.ds.source.update_freq, rising=True, falling=False)
        self.clock_times = np.concatenate((self.clock_times, timestamps + self.clock_elapsed))
        clock_elapsed_prev = self.clock_elapsed
        clock_elapsed_new = self.clock_elapsed + len(clock_data) / self.ds.source.update_freq

        # Check the trigger if it exists
        if self.trigger_dch:
            trigger_data = self.ds.get_new(map_channels_for_multisource(digital_channels=[self.trigger_dch]))[0]
            timestamps, edges = aopy.utils.detect_edges(trigger_data, self.ds.source.update_freq, rising=True, falling=False)   
            self.trigger_times = np.concatenate((self.trigger_times, timestamps + clock_elapsed_prev)).tolist()

        # Append new ERPs from trigger events
        fs = self.ds.source.update_freq / self.lfp_downsample
        nt = 2./self.update_rate # seconds
        npts = int(nt * self.ds.source.update_freq)
        lfp = self.ds.get(npts, map_channels_for_multisource(headstage_channels=self.acq_ch))
        lfp = np.array(lfp)[:,::self.lfp_downsample].T # reshape and downsample (nt, nch)
        ignored_trigger_cycles = []
        while len(self.trigger_cycles) > 0:
            cycle = self.trigger_cycles.pop()
            if cycle >= len(self.clock_times):
                ignored_trigger_cycles.append(cycle)
                continue
            time = self.clock_times[cycle]
            if time + self.time_after > clock_elapsed_new:
                ignored_trigger_cycles.append(cycle)
                continue
            elif time - self.time_before < clock_elapsed_new - nt:
                print('missed cycle', cycle, 'at time', time)
                continue
            erp = aopy.analysis.calc_erp(lfp, [time - clock_elapsed_new + nt], self.time_before, self.time_after, fs)
            self.erp = np.concatenate((self.erp, erp), axis=2)
        self.trigger_cycles = ignored_trigger_cycles

        # Check for new digital triggers
        ignored_trigger_times = []
        while len(self.trigger_times) > 0:
            time = self.trigger_times.pop()
            if time + self.time_after > clock_elapsed_new:
                ignored_trigger_times.append(time)
                continue
            elif time - self.time_before < clock_elapsed_new - nt:
                print('missed trigger', time)
                continue
            erp = aopy.analysis.calc_erp(lfp, [time - clock_elapsed_new + nt], self.time_before, self.time_after, fs)
            self.erp = np.concatenate((self.erp, erp), axis=2)
        self.trigger_times = ignored_trigger_times

        # Update the total elapsed time
        self.clock_elapsed = clock_elapsed_new

        # Update the ERP
        fs = self.ds.source.update_freq / self.lfp_downsample
        max_erp = aopy.analysis.get_max_erp(self.erp, self.time_before, self.time_after, fs, trial_average=True)
        self.data_map = aopy.visualization.get_data_map(max_erp*1.907348633e-7*1e6, self.elec_pos[:,0], self.elec_pos[:,1])

    def handle_data(self, key, values):
        super().handle_data(key, values)
        if key == 'sync_event':
            event_name, event_data = values
            if event_name in self.trigger_events:
                self.trigger_cycles.append(self.cycle_count)

    def draw(self):
        super().draw()
        self.erp_im.set_data(self.data_map)
        self.erp_text.set_text(f"{self.erp.shape[2]} trials")

    def cleanup(self):
        '''
        Cleanup tasks after the experiment ends, e.g. saving the figure.
        '''
        self.ds.stop()      
        # TO-DO: implenent saving figures
        subject = self.task_params.get('subject_name', 'None')
        te_id = self.task_params.get('te_id', 'None')
        if te_id == 'None':
            return
        date = datetime.date.today()
        filename = f'online_erp_{subject}_{te_id}_{date}.png'
        plt.figure(self.fig)
        aopy.visualization.savefig(self.figure_dir, filename, transparent=False)

class OnlineDataServer(threading.Thread):
    '''
    Interface for accumulating and analyzing BMI3D data in real-time.
    '''

    def __init__(self, host_ip, port=5000):
        '''
        Initialize the server on the specified IP address and port number.

        Args:
            host_ip (str): IP address of the machine running the online analysis
            port (int): Port number for the online analysis server
        '''

        # Initialize socket
        self.sock = socket.socket(
            socket.AF_INET, # Internet
              socket.SOCK_DGRAM) # UDP 
        self.sock.bind((host_ip, port))
        self.sock.setblocking(0)
                
        # Initialize workers
        self.data_worker = None
        self.analysis_workers = []
        try:
            MultiSource.pre_init()
            print('eCube streaming initialized')
        except:
            pass

        # Initialize the server
        self._stop_event = threading.Event()
        self.reset()
        self.is_completed = False
        super().__init__()

    def _stop(self):
        # Stop all the workers
        for worker, _ in self.analysis_workers:
            if worker.is_alive():
                worker.stop()
                worker.join()
        self.analysis_workers = []
        if self.data_worker and self.data_worker.is_alive():
            self.data_worker.stop()
            self.data_worker.join()
        self.data_worker = None

    def reset(self):
        self._stop()

        # Start new workers
        self.result_queue = queue.Queue()
        self.data_worker = OnlineDataWorker(self.sock, self.result_queue)
        self.data_worker.start()
        self.is_running = False
        self.is_completed = True
        self.task_params = {}
        self.state = None

    def init(self):
        '''
        Once the experiment is initialized but before it starts, we spin up the analysis processes
        based on what kind of experiment is running.
        '''
        # Always start with the behavior analysis worker
        print('init in state', self.state)
        data_queue = mp.Queue()
        self.analysis_workers.append((BehaviorAnalysisWorker(self.task_params, data_queue), data_queue))

        # Is there an ECoG array?
        if 'record_headstage' in self.task_params and self.task_params['record_headstage']:
            data_queue = mp.Queue()
            self.analysis_workers.append((ERPAnalysisWorker(self.task_params, data_queue), data_queue))

        # Start all the workers
        for worker, _ in self.analysis_workers:
            worker.start()
        
    def update(self):
        '''
        Get the latest data from the online server
        '''
        try:
            key, values = self.result_queue.get(timeout=0.1)
        except queue.Empty:
            time.sleep(0.1)
            return False
        if key == 'state':
            self.state = values[0]
            if values[0] is None:
                print('Experiment finished')
                self.reset()
                self.is_running = False
                self.is_completed = True
            for _, data_queue in self.analysis_workers:
                data_queue.put((key, values))
        elif key == 'init' and values[0]:
            self.is_running = True
            self.init()
        elif key == 'param':
            name, value = values
            self.task_params[name] = value
        else:
            # Send everything else back onto the queues for the analysis workers
            for _, data_queue in self.analysis_workers:
                data_queue.put((key, values))
        return True

    def run(self):
        '''
        Main loop to run the server
        '''
        while True:
            self.update()
            if self._stop_event.is_set():
                break

        self._stop()
        self.sock.close()
        print('OnlineDataServer shut down.')

    def stop(self):
        self._stop_event.set()


if __name__ == '__main__':
    import signal
    import sys

    # Parse arguments
    # online_analysis [hostname] [port] [display]
    if len(sys.argv) >= 2:
        hostname = sys.argv[1]
    else:
        hostname = 'localhost'
    
    if len(sys.argv) >= 3:
        port = sys.argv[2]
    else:
        port = 5000

    if len(sys.argv) >= 4:
        display = sys.argv[3]
    else:
        display = ':0'

    # Spin up servernode
    # if hostname == '0.0.0.0':
    #     import subprocess
    #     subprocess.Popen('/home/aolab/code/bmi3d/riglib/ecube/servernode-control')

    # Start server
    print(hostname, port, display)
    os.environ['DISPLAY'] = display
    analysis = OnlineDataServer(hostname, port)
    analysis.start()

    # Catch interrupts
    def signal_handler(*args, **kwargs):
        analysis.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()

