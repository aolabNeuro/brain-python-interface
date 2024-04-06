import cProfile
import pstats
import traits
import socket

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

    online_analysis_ip = traits.String("localhost", desc="IP address of the machine running the online analysis")
    online_analysis_port = traits.Int(5000, desc="Port number for the online analysis server")

    def __init__(self, *args, **kwargs):
        self.online_analysis_sock = socket.socket(
            socket.AF_INET, # Internet
            socket.SOCK_DGRAM) # UDP
        
    def _send_online_analysis_msg(self, key, value):
        '''
        Helper function to send messages to the online analysis server
        '''
        self.online_analysis_sock.sendto(f'{key}:{value};', (self.online_analysis_ip, self.online_analysis_port))

    def init(self):
        '''
        Send basic experiment info to the online analysis server
        '''
        super().init()
        self._send_online_analysis_msg('experiment_name', self.__class__.__name__)
        self._send_online_analysis_msg('subject', self.subject)
        self._send_online_analysis_msg('save_id', self.save_id)
        for key, value in self.get_trait_values:
            self._send_online_analysis_msg(key, value)
        self._send_online_analysis_msg('init', 'None')

    def _cycle(self):
        '''
        Send ticks to the online analysis server
        '''
        self._send_online_analysis_msg('cycle', self.cycle_count)
        super()._cycle()

    def set_state(self, condition, **kwargs):
        '''
        Send task state transitions to the online analysis server
        '''
        self._send_online_analysis_msg('state', condition)
        super().set_state(condition, **kwargs)

    def sync_event(self, event_name, event_data=None, immediate=False):
        '''
        Send sync events to the online analysis server
        '''
        self._send_online_analysis_msg('sync_event', f'{event_name}/{event_data}')
        super().sync_event(event_name, event_data=event_data, immediate=immediate)