from riglib.dio.NIUSB6501 import write_to_comedi
import time

'''
TO-DO 
this needs abstracton and encapsulation
'''
    
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
        'target':1, #target appears
        'hold': 2,
        'targ_transition': 3,
        'reward': 4,
        'None':255
    }
    NONE_CODE = 255


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
            write_to_comedi(self.TaskCodeDict[condition])
        elif condition is None:
            print(f'transition to {condition}')
            write_to_comedi(self.NONE_CODE)
        else:
            print(f'transition to {condition}')

        

        super().set_state(condition, **kwargs)



if __name__ == "__main__":
    #testing script
    #that flashes every 0.4 second

    while 1:
        write_to_comedi(0)
        time.sleep(0.2)

        write_to_comedi(255)
        time.sleep(0.2)
    