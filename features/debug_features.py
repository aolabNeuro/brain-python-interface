import cProfile
import pstats
import socket
import traceback
from riglib.experiment import traits
import numpy as np
import json

class Profiler():
    
    def run(self):
        pr = cProfile.Profile()
        pr.enable()
        super().run()
        pr.disable()
        with open('profile.csv', 'w') as f:
            ps = pstats.Stats(pr, stream=f).sort_stats('time')
            ps.print_stats()

class OnlineAnalysis(traits.HasTraits):
    '''
    Feature to send task data to an online analysis server.

    In the future this could be expanded to make use of the riglib.sinks interface.
    For now it is a simple UDP socket interface that sends messages about the 
    task params, state transitions, and sync events to a server for online analysis.
    '''

    online_analysis_ip = traits.String("localhost", desc="IP address of the machine running the online analysis")
    online_analysis_port = traits.Int(5000, desc="Port number for the online analysis server")
        
    def _send_online_analysis_msg(self, key, *values):
        '''
        Helper function to send messages to the online analysis server
        '''
        strings = []
        for v in values:
            if isinstance(v, np.ndarray) and v.size > 1:
                strings.append(json.dumps(v.tolist()))
            elif isinstance(v, np.ndarray):
                strings.append(json.dumps(v[0]))
            else:
                strings.append(json.dumps(v))
        payload = '#'.join(strings)
        self.online_analysis_sock.sendto(f'{key}%{payload}'.encode('utf-8'), (self.online_analysis_ip, self.online_analysis_port))

    def init(self):
        '''
        Send basic experiment info to the online analysis server
        '''
        super().init()
        print('Starting online analysis socket...')
        self.online_analysis_sock = socket.socket(
            socket.AF_INET, # Internet
            socket.SOCK_DGRAM) # UDP
        
        # Send entry metadata
        self._send_online_analysis_msg('param', 'experiment_name', self.__class__.__name__)
        if hasattr(self, 'saveid'):
            self._send_online_analysis_msg('param', 'te_id', self.saveid)
        else:
            self._send_online_analysis_msg('param', 'te_id', 'None')
        if hasattr(self, 'subject_name'):
            self._send_online_analysis_msg('param', 'subject_name', self.subject_name)

        # Send task metadata
        for key, value in self.get_trait_values().items():
            if key in self.object_trait_names:
                if key == 'decoder':
                    self._send_online_analysis_msg('param', 'decoder_channels', value.channels)
            else:
                self._send_online_analysis_msg('param', key, value)
        if hasattr(self, 'sync_params'):
            for key, value in self.sync_params.items():
                self._send_online_analysis_msg('param', key, value)
        
        # Send init message to trigger analysis workers
        self._send_online_analysis_msg('init', 'None')
        print('...done!')

    def _start_wait(self):
        if hasattr(super(), '_start_wait'):
            super()._start_wait()
        if hasattr(self, 'targs') and hasattr(self, 'gen_indices'):
            for i in range(len(self.targs)):
                self._send_online_analysis_msg('target_location', self.gen_indices[i], self.targs[i])

    def _cycle(self):
        '''
        Send cursor and eye position data to the online analysis server
        '''
        super()._cycle()
        self._send_online_analysis_msg('cycle_count', self.cycle_count)
        if hasattr(self, 'plant'):
            self._send_online_analysis_msg('cursor', self.plant.get_endpoint_pos())
        if hasattr(self, 'eye_pos'):
            self._send_online_analysis_msg('eye_pos', self.eye_pos[0,:])

    def set_state(self, condition, **kwargs):
        '''
        Send task state transitions to the online analysis server
        '''
        self._send_online_analysis_msg('state', condition)
        super().set_state(condition, **kwargs)

    def sync_event(self, event_name, event_data=0, immediate=False):
        '''
        Send sync events to the online analysis server
        '''
        self._send_online_analysis_msg('sync_event', event_name, event_data)
        super().sync_event(event_name, event_data=event_data, immediate=immediate)