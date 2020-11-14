
    
    
class TaskCodeStreamer(object):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        print(f'transition to {condition}')
        
        super().set_state(condition, **kwargs)