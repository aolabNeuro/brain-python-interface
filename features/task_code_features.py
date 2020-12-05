from riglib.dio.NIUSB6501 import write_to_comedi
    
class TaskCodeStreamer(object):

    '''
    TaskCodeDict = {
        'wait': 1,
        'target':2, #target appears
        'hold': 15,
        'targ_transition': 6,
        'reward': 0
    }
    '''
    #binary state, reward or not
    TaskCodeDict = {
        'wait': 0,
        'target':0, #target appears
        'hold': 0,
        'targ_transition': 0,
        'reward': 1
    }


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #clear the output
        write_to_comedi(0)

    def set_state(self, condition, **kwargs):
        '''
        Extension of riglib.experiment.Experiment.set_state. 

        Parameters
        ----------
        condition : string
            Name of new state.
        **kwargs : dict 
            Passed to 'super' set_state function

        Returns
        -------
        None
        '''
        if condition in self.TaskCodeDict.keys():
            print(f'transition to {condition} with task code {self.TaskCodeDict[condition]}')
        else:
            print(f'transition to {condition}')

        write_to_comedi(self.TaskCodeDict[condition])

        super().set_state(condition, **kwargs)