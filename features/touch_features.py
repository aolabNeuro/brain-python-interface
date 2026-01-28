
'''
Features for a touch sensor on the neurosync arduino
'''
from riglib.experiment import traits
from riglib import touch_data
import numpy as np
import pygame
from riglib.experiment import traits
from riglib.touch_data import TabletTouchData

########################################################################################################
# Touch sensor datasources
########################################################################################################

class TabletTouch(traits.HasTraits):
    
    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        
        # Create a source to buffer the touch data
        from riglib import source
        self.touch = source.DataSource(TabletTouchData)

        # Save to the sink
        from riglib import sink
        sink_manager = sink.SinkManager.get_instance()
        sink_manager.register(self.touch)
        super().init()


    def _get_manual_position(self):
        ''' Overridden method to get input coordinates based on motion data'''

        # Get data from optitrack datasource
        data = self.touch.get() # List of (list of features)
        if len(data) == 0:
            return [np.nan, np.nan, np.nan]
        
        pos = data[-1] # get the most recent event
        pos[0] = (pos[0] / self.window_size[0] - 0.5) * self.screen_cm[0]
        pos[1] = -(pos[1] / self.window_size[1] - 0.5) * self.screen_cm[1] # pygame counts (0,0) as the top left

        return [pos[0], 0, pos[1]]



class MouseEmulateTouch(traits.HasTraits):
    '''
    Emulate a touch screen by detecting when mouse position doesn't update
    '''

    mouse_repeat_delay = traits.Float(1., desc="Time in seconds before the same mouse position causes the cursor to disappear")

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        n_repeat_delay = int(self.mouse_repeat_delay * self.fps)
        self.joystick = MouseHistory(self.window_size, self.screen_cm, 
                                     np.array(self.starting_pos[::2]), n_repeat_delay=n_repeat_delay)

class MouseHistory():
    '''
    Pretend to be a data source
    '''

    def __init__(self, window_size, screen_cm, start_pos, n_repeat_delay=3, init_frames=3):
        self.window_size = window_size
        self.screen_cm = screen_cm
        self.history = np.zeros((n_repeat_delay, 2))
        self.pos = [0., 0.]
        self.pos[0] = start_pos[0]
        self.pos[1] = start_pos[1]
        self.init_frames = init_frames

    def get(self):
        pos = pygame.mouse.get_pos()
        self.pos[0] = (pos[0] / self.window_size[0] - 0.5) * self.screen_cm[0]
        self.pos[1] = -(pos[1] / self.window_size[1] - 0.5) * self.screen_cm[1] # pygame counts (0,0) as the top left
        
        # Have to ignore the first few positions because they can change when the screen is initializing
        if self.init_frames > 0:
            self.history[:] = self.pos
            self.init_frames -= 1

        # Save a buffer of previous positions and if they are all the same, then set pos to NaN
        self.history[:-1, :] = self.history[1:, :]
        self.history[-1, :] = self.pos
        if np.all(np.diff(self.history, axis=0) == 0):
            self.pos[0] = np.nan
            self.pos[1] = np.nan
        return [self.pos]

class TouchDataFeature(traits.HasTraits):
    '''
    Enable reading of data from touch sensor
    '''

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up the DataSource for interacting with the 
        kinarm system and registers the source with the SinkRegister so that the data gets saved 
        to file as it is collected.
        '''
        from riglib import source
        System  = touch_data.TouchData
        self.touch_data = source.DataSource(System)
        from riglib import sink
        sink_manager = sink.SinkManager.get_instance()
        sink_manager.register(self.touch_data)
        super(TouchDataFeature, self).init()
    
    @property
    def source_class(self):
        '''
        Specify the source class as a function
        '''
        from riglib import touch_data
        return touch_data.TouchData()

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the motiontracker source prior to starting the experiment's 
        main thread/process, and handle any errors by stopping the source
        '''
        self.touch_data.start()
        try:
            super(TouchDataFeature, self).run()
        finally:
            self.touch_data.stop()
    
    def join(self):
        '''
        See riglib.experiment.Experiment.join(). Re-join the 'motiondata' source process before cleaning up the experiment thread
        '''
        #self.touch_data.join()
        super(TouchDataFeature, self).join()
    
    def _start_None(self):
        '''
        Code to run before the 'None' state starts (i.e., the task stops)
        '''
        self.touch_data.stop()
        super(TouchDataFeature, self)._start_None()
