'''
Peripheral interface device features
'''

import types
import time
import numpy as np
import pygame
from riglib import gpio
from config.rig_defaults import force_sensor_address
from riglib.experiment import traits

###### CONSTANTS
sec_per_min = 60

class Joystick(traits.HasTraits):
    '''
    Grab x,y position data from a usb joystick using pygame
    '''
    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.joystick = JoystickInput(self.screen_cm)

class JoystickInput():
    '''
    Pretend to be a data source for a joystick. Scales the (-1,1) axis outputs to the screen dimensions
    '''

    def __init__(self, screen_cm):
        self.screen_cm = screen_cm

        # Enable polling the controller when the mouse isn't on the screen
        import os
        os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"

        # Check if joystick is connected
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise Exception("No joystick connected!")
        self.joystick = pygame.joystick.Joystick(0)

    def get(self):
        pos = np.array([
            self.joystick.get_axis(0) * self.screen_cm[0],
            -self.joystick.get_axis(1) * self.screen_cm[1]
        ])
        return [pos]

class PhidgetsJoystick(object):
    '''
    Code to use an analog joystick with signals digitized by the phidgets board
    '''
    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' instantiates a DataSource with 2 channels for the two analog 
        inputs from the phidgets joystick. 
        '''
        from riglib import source, phidgets, sink
        sink_manager = sink.SinkManager.get_instance()
        #System = phidgets.make(2, 1)
        #self.joystick = source.DataSource(System)

        self.register_num_channels()
        super(Joystick, self).init()
        sink_manager.register(self.joystick)

    def register_num_channels(self):
        from riglib import source, phidgets, sink
        System = phidgets.make(2, 1)
        self.joystick = source.DataSource(System)

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the joystick source and stops it after the FSM has finished running
        '''
        self.joystick.start()
        try:
            super(Joystick, self).run()
        finally:
            self.joystick.stop()

    def join(self):
        '''
        See riglib.experiment.Experiment.join(). Re-join the joystick source process before cleaning up the experiment thread
        '''
        self.joystick.join()
        super(Joystick, self).join()

class ArduinoJoystick(Joystick):
    def init(self):
        '''
        Same as above, w/o Phidgets import
        '''
        from riglib import source, sink
        self.sinks = sink.sinks # sink_manager = sink.SinkManager.get_instance()
        self.register_num_channels()
        super(Joystick, self).init()
        self.sinks.register(self.joystick)

    def register_num_channels(self):
        from riglib import arduino_joystick, source, sink
        System = arduino_joystick.make(2, 1)
        self.joystick = source.DataSource(System)

class ArduinoIMU(object):
    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' instantiates a DataSource with 2 channels for the
        inputs from the IMU
        '''
        from riglib import sink
        self.register_num_channels()
        super(ArduinoIMU, self).init()

        sink_manager = sink.SinkManager.get_instance()
        sink_manager.register(self.arduino_imu)

    def register_num_channels(self):
        from riglib import source, arduino_imu
        System = arduino_imu.make(6, 1)
        self.arduino_imu = source.DataSource(System)

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the joystick source and stops it after the FSM has finished running
        '''
        self.arduino_imu.start()
        try:
            super(ArduinoIMU, self).run()
        finally:
            self.arduino_imu.stop()

    def join(self):
        '''
        See riglib.experiment.Experiment.join(). Re-join the joystick source process before cleaning up the experiment thread
        '''
        self.arduino_imu.join()
        super(ArduinoIMU, self).join()    

class Joystick_plus_TouchSensor(Joystick):
    '''
    code to use touch sensor (attached to joystick in exorig) plus joystick
    '''
    def register_num_channels(self):
        from riglib import source, phidgets, sink
        System = phidgets.make(3, 1)
        self.joystick = source.DataSource(System)

class DualJoystick(object):
    '''
    A two-joystick interface, similar to Joystick
    '''
    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' creates a 4-channel DataSource, two channels for each joystick
        -------
        '''
        from riglib import source, phidgets
        System = phidgets.make(4, 1)
        self.dualjoystick = source.DataSource(System)
        super(DualJoystick, self).init()

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the dual_joystick source and stops it after the FSM has finished running
        '''
        self.dualjoystick.start()
        try:
            super(DualJoystick, self).run()
        finally:
            self.dualjoystick.stop()

    def join(self):
        '''
        See riglib.experiment.Experiment.join(). Re-join the joystick source process before cleaning up the experiment thread
        '''
        self.dualjoystick.join()
        super(DualJoystick, self).join()

class Button(object):
    '''
    Deprecated!

    Adds the ability to respond to the button, as well as to keyboard responses
    The "button" was a switch connected to a modified mouse so that the digital input went through 
    the mouse interface (hence the calls to pygame's mouse interface)
    '''
    def screen_init(self):
        super(Button, self).screen_init()
        import pygame
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

    def _get_event(self):
        import pygame
        btnmap = {1:1, 3:4}
        for btn in pygame.event.get(pygame.MOUSEBUTTONDOWN):
            if btn.button in btnmap:
                return btnmap[btn.button]

        return super(Button, self)._get_event()
    
    def _while_reward(self):
        super(Button, self)._while_reward()
        import pygame
        pygame.event.clear()
    
    def _while_penalty(self):
        #Clear out the button buffers
        super(Button, self)._while_penalty()
        import pygame
        pygame.event.clear()
    
    def _while_wait(self):
        super(Button, self)._while_wait()
        import pygame
        pygame.event.clear()

class EyeControl(object):
    '''
    this class implements a python cursor control task. This is just for testing eye related task. 
    '''

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.joystick = Eye(np.array(self.starting_pos[::2]))

class Eye(object):
    '''
    Pretend to be a data source. This is just for testing eye related task. 
    '''
    
    def __init__(self, start_pos):
        self.pos = [0., 0.]
        self.pos[0] = start_pos[0]
        self.pos[1] = start_pos[1]
        self.move_step = 1 # cm, before scaling
        self.calibration = np.array([2,2]) # TODO load calibration data

    def get(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                if event.type == pygame.K_q:
                    pygame.quit()
                    quit()
                if event.key == pygame.K_LEFT:
                    self.pos[0] -= self.move_step
                if event.key == pygame.K_RIGHT:
                    self.pos[0] += self.move_step
                if event.key == pygame.K_UP:
                    self.pos[1] += self.move_step
                if event.key == pygame.K_DOWN:
                    self.pos[1] -= self.move_step

        calibrated_pos = self.calibration*self.pos
        return [calibrated_pos]

class KeyboardControl(object):
    '''
    this class implements a python cursor control task for human
    '''

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.joystick = Keyboard(np.array(self.starting_pos[::2]))

class Keyboard():
    '''
    Pretend to be a data source
    '''

    def __init__(self, start_pos):
        self.pos = [0., 0.]
        self.pos[0] = start_pos[0]
        self.pos[1] = start_pos[1]
        self.move_step = 1 # cm, before scaling

    def get(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                if event.type == pygame.K_q:
                    pygame.quit()
                    quit()
                if event.key == pygame.K_LEFT:
                    self.pos[0] -= self.move_step
                if event.key == pygame.K_RIGHT:
                    self.pos[0] += self.move_step
                if event.key == pygame.K_UP:
                    self.pos[1] += self.move_step
                if event.key == pygame.K_DOWN:
                    self.pos[1] -= self.move_step
        return [self.pos]

class MouseControl(KeyboardControl):
    '''
    this class implements a python cursor control task for human
    '''

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.joystick = Mouse(self.window_size, self.screen_cm, np.array(self.starting_pos[::2]))

class Mouse():
    '''
    Pretend to be a data source
    '''

    def __init__(self, window_size, screen_cm, start_pos):
        self.window_size = window_size
        self.screen_cm = screen_cm
        self.pos = [0., 0.]
        self.pos[0] = start_pos[0]
        self.pos[1] = start_pos[1]

    def get(self):
        pos = pygame.mouse.get_pos()
        self.pos[0] = (pos[0] / self.window_size[0] - 0.5) * self.screen_cm[0]
        self.pos[1] = -(pos[1] / self.window_size[1] - 0.5) * self.screen_cm[1] # pygame counts (0,0) as the top left
        return [self.pos]
    
class ForceControl():
    '''
    Read the voltage from pin 0 on connected arduino at '/dev/forcesensor'
    '''
    def init(self, *args, **kwargs):
        if hasattr(super(), 'init'):
            super().init(*args, **kwargs)
        self.joystick = gpio.ArduinoGPIO(force_sensor_address, enable_analog=True)
        def get(self):
            return self.analog_read(0)
        self.joystick.get = types.MethodType(get, self.joystick)


class CthulhuTDUFeedback(traits.HasTraits):
    '''
    Stream a low-resolution world image (target + cursor) to the Cthulhu TDU
    Arduino sketch in riglib/cthulhu_display/cthulhu_display.ino.
    '''
    cthulhu_display_port = traits.String('/dev/cthulhu_display_tdu', desc="Serial port for Cthulhu TDU Arduino. Use 'auto' to auto-detect.")
    cthulhu_display_baudrate = traits.Int(115200, desc='Serial baudrate for Cthulhu TDU Arduino')
    cthulhu_display_refresh_hz = traits.Float(25.0, desc='Frame update rate to TDU')
    cthulhu_display_grid_shape = traits.Tuple((8, 8), desc='Grid rows, cols sent over serial')
    cthulhu_display_world_axes = traits.Tuple((0, 2), desc='Position vector indices mapped to TDU x/y')
    cthulhu_display_cursor_sigma = traits.Float(1.2, desc='Gaussian width (cm) for cursor blob')
    cthulhu_display_target_sigma = traits.Float(1.8, desc='Gaussian width (cm) for target blob')
    cthulhu_display_cursor_weight = traits.Float(1.0, desc='Cursor contribution weight')
    cthulhu_display_target_weight = traits.Float(0.8, desc='Target contribution weight')
    cthulhu_display_smooth_alpha = traits.Float(0.35, desc='Arduino exponential smoothing alpha')
    cthulhu_display_activation_threshold = traits.Float(0.04, desc='Arduino activation threshold (0..1)')
    cthulhu_display_intensity = traits.Float(0.45, desc='Global TDU intensity gain (0..1)')

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)

        self._cthulhu = None
        self._cthulhu_last_send = 0.0
        self._cthulhu_bounds = self._cthulhu_get_world_bounds()

        try:
            from riglib.cthulhu_display.cthulhu_display_tdu import CthulhuTDU
            port = self.cthulhu_display_port
            if str(port).strip().lower() in ['', 'auto', 'none']:
                port = None
            self._cthulhu = CthulhuTDU(port=port, baudrate=self.cthulhu_display_baudrate)
            self._cthulhu.set_smoothing(self.cthulhu_display_smooth_alpha, self.cthulhu_display_activation_threshold)
            self._cthulhu.set_intensity(self.cthulhu_display_intensity)
            print('CthulhuTDUFeedback connected on %s' % self._cthulhu.port_name)
        except Exception as err:
            print('CthulhuTDUFeedback disabled (%s)' % err)
            self._cthulhu = None

    def _cthulhu_get_world_bounds(self):
        if hasattr(self, 'cursor_bounds'):
            x_idx, y_idx = self.cthulhu_display_world_axes
            x_min = float(self.cursor_bounds[2 * x_idx])
            x_max = float(self.cursor_bounds[2 * x_idx + 1])
            y_min = float(self.cursor_bounds[2 * y_idx])
            y_max = float(self.cursor_bounds[2 * y_idx + 1])
            return (x_min, x_max, y_min, y_max)
        return (-10.0, 10.0, -10.0, 10.0)

    def _cthulhu_get_cursor_pos(self):
        if hasattr(self, 'plant') and hasattr(self.plant, 'get_endpoint_pos'):
            return self.plant.get_endpoint_pos()
        if hasattr(self, 'decoder') and hasattr(self.decoder, 'state'):
            return np.asarray(self.decoder.state).ravel()
        return None

    def _cthulhu_get_target_pos(self):
        if hasattr(self, 'target_location'):
            return np.asarray(self.target_location).ravel()
        return None

    def _cthulhu_send_world_frame(self):
        if self._cthulhu is None:
            return

        refresh_hz = max(1.0, float(self.cthulhu_display_refresh_hz))
        now = time.time()
        if now - self._cthulhu_last_send < (1.0 / refresh_hz):
            return

        cursor_pos = self._cthulhu_get_cursor_pos()
        target_pos = self._cthulhu_get_target_pos()

        try:
            self._cthulhu.send_world(
                cursor_pos=cursor_pos,
                target_pos=target_pos,
                bounds=self._cthulhu_bounds,
                grid_shape=self.cthulhu_display_grid_shape,
                axes=self.cthulhu_display_world_axes,
                cursor_sigma_cm=self.cthulhu_display_cursor_sigma,
                target_sigma_cm=self.cthulhu_display_target_sigma,
                cursor_weight=self.cthulhu_display_cursor_weight,
                target_weight=self.cthulhu_display_target_weight,
            )
            self._cthulhu_last_send = now
        except Exception as err:
            print('CthulhuTDUFeedback send error (%s)' % err)
            self._cthulhu = None

    def _cycle(self):
        self._cthulhu_send_world_frame()
        super()._cycle()

    def cleanup(self, database, saveid, **kwargs):
        if self._cthulhu is not None:
            try:
                self._cthulhu.clear()
                self._cthulhu.close()
            except Exception:
                pass
        return super().cleanup(database, saveid, **kwargs)
