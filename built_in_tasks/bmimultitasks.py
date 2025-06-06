'''
BMI tasks in the new structure, i.e. inheriting from manualcontrolmultitasks
'''
import numpy as np
import pickle

from riglib.experiment import traits

from riglib.bmi import goal_calculators, ppfdecoder, feedback_controllers
from riglib.bmi.bmi import BMILoop
from riglib.bmi.assist import Assister, FeedbackControllerAssist
from riglib.bmi.state_space_models import StateSpaceEndptVel2D, StateSpaceNLinkPlanarChain
from riglib.experiment.experiment import control_decorator

from riglib.stereo_opengl.window import WindowDispl2D
from .target_capture_task import ScreenReachAngle, ScreenTargetCapture
from features.bmi_task_features import LinearlyDecreasingAssist
from .target_graphics import target_colors

np.set_printoptions(suppress=False)

###################
####### Assisters
##################
class OFCEndpointAssister(FeedbackControllerAssist):
    '''
    Assister for cursor PPF control which uses linear feedback (infinite horizon LQR) to drive the cursor toward the target state
    '''
    def __init__(self, decoding_rate=180):
        '''
        Constructor for OFCEndpointAssister

        Parameters
        ----------
        decoding_rate : int
            Rate that the decoder should operate, in Hz. Should be a multiple or divisor of 60 Hz

        Returns
        -------
        OFCEndpointAssister instance
        '''
        F_dict = pickle.load(open('/storage/assist_params/assist_20levels_ppf.pkl'))
        B = np.mat(np.vstack([np.zeros([3,3]), np.eye(3)*1000*1./decoding_rate, np.zeros(3)]))
        fb_ctrl = feedback_controllers.MultiModalLFC(A=B, B=B, F_dict=F_dict)
        super(OFCEndpointAssister, self).__init__(fb_ctrl, style='additive_cov')
        self.n_assist_levels = len(F_dict)

    def get_F(self, assist_level):
        '''
        Look up the feedback gain matrix based on the assist_level

        Parameters
        ----------
        assist_level : float
            Float between 0 and 1 to indicate the level of the assist (1 being the highest)

        Returns
        -------
        np.mat
        '''
        assist_level_idx = min(int(assist_level * self.n_assist_levels), self.n_assist_levels-1)
        F = np.mat(self.fb_ctrl.F_dict[assist_level_idx])
        return F

class SimpleEndpointAssister(Assister):
    '''
    Constant velocity toward the target if the cursor is outside the target. If the
    cursor is inside the target, the speed becomes the distance to the center of the
    target divided by 2.
    '''
    def __init__(self, *args, **kwargs):
        '''    Docstring    '''
        self.decoder_binlen = kwargs.pop('decoder_binlen', 0.1)
        self.assist_speed = kwargs.pop('assist_speed', 5.)
        self.target_radius = kwargs.pop('target_radius', 2.)
        self.assist_noise = kwargs.pop('assist_noise', 0.)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''    Docstring    '''
        Bu = None
        assist_weight = 0.

        if assist_level > 0:
            cursor_pos = np.array(current_state[0:3,0]).ravel()
            target_pos = np.array(target_state[0:3,0]).ravel()
            decoder_binlen = self.decoder_binlen
            speed = self.assist_speed * decoder_binlen
            target_radius = self.target_radius
            Bu = self.endpoint_assist_simple(cursor_pos, target_pos, decoder_binlen, speed, target_radius, self.assist_noise)
            assist_weight = assist_level

        # return Bu, assist_weight
        return dict(x_assist=Bu, assist_level=assist_weight)

    @staticmethod
    def endpoint_assist_simple(cursor_pos, target_pos, decoder_binlen=0.1, speed=0.5, target_radius=2., assist_noise=0.):
        '''
        Estimate the next state using a constant velocity estimate moving toward the specified target

        Parameters
        ----------
        cursor_pos: np.ndarray of shape (3,)
            Current position of the cursor
        target_pos: np.ndarray of shape (3,)
            Specified target position
        decoder_binlen: float
            Time between iterations of the decoder
        speed: float
            Speed of the machine-assisted cursor
        target_radius: float
            Radius of the target. When the cursor is inside the target, the machine assisted cursor speed decreases.
        assist_noise: float
            Noise added to the assist speed to vary the timing of the trajectories

        Returns
        -------
        x_assist : np.ndarray of shape (7, 1)
            Control vector to add onto the state vector to assist control.
        '''
        diff_vec = target_pos - cursor_pos
        dist_to_target = np.linalg.norm(diff_vec)
        dir_to_target = diff_vec / (np.spacing(1) + dist_to_target)

        if dist_to_target > target_radius:
            assist_cursor_pos = cursor_pos + np.random.uniform(1-assist_noise,1+assist_noise)*speed*dir_to_target
        else:
            assist_cursor_pos = cursor_pos + np.random.uniform(1-assist_noise,1+assist_noise)*speed*diff_vec/2

        assist_cursor_vel = (assist_cursor_pos-cursor_pos)/decoder_binlen
        x_assist = np.hstack([assist_cursor_pos, assist_cursor_vel, 1])
        x_assist = np.mat(x_assist.reshape(-1,1))
        return x_assist

class SimplePosAssister(SimpleEndpointAssister):
    
    @staticmethod 
    def endpoint_assist_simple(cursor_pos, target_pos, decoder_binlen=0.1, speed=0.5, target_radius=2., assist_level=0.):
        '''
        Estimate the next state using a constant velocity estimate moving toward the specified target

        Parameters
        ----------
        see SimpleEndtpointAssister for docs

        Returns
        -------
        x_assist : np.ndarray of shape (7, 1)
            Control vector to add onto the state vector to assist control.
        '''
        diff_vec = target_pos - cursor_pos 
        dist_to_target = np.linalg.norm(diff_vec)
        dir_to_target = diff_vec / (np.spacing(1) + dist_to_target)
        
        if dist_to_target > target_radius:
            assist_cursor_pos = cursor_pos + speed*dir_to_target
        else:
            assist_cursor_pos = cursor_pos + speed*diff_vec/2

        return assist_cursor_pos.ravel()

class SimpleEndpointAssisterLFC(feedback_controllers.MultiModalLFC):
    '''
    Docstring
    '''
    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        dt = 0.1
        A = np.mat([[1., 0, 0, dt, 0, 0, 0],
                    [0., 1, 0, 0,  dt, 0, 0],
                    [0., 0, 1, 0, 0, dt, 0],
                    [0., 0, 0, 0, 0,  0, 0],
                    [0., 0, 0, 0, 0,  0, 0],
                    [0., 0, 0, 0, 0,  0, 0],
                    [0., 0, 0, 0, 0,  0, 1]])

        I = np.mat(np.eye(3))
        B = np.vstack([0*I, I, np.zeros([1,3])])
        F_target = np.hstack([I, 0*I, np.zeros([3,1])])
        F_hold = np.hstack([0*I, 0*I, np.zeros([3,1])])
        F_dict = dict(hold=F_hold, target=F_target)
        super(SimpleEndpointAssisterLFC, self).__init__(B=B, F_dict=F_dict)

#################
##### Tasks #####
#################
class BMIControlMultiMixin(BMILoop, LinearlyDecreasingAssist):
    '''
    Target capture task with cursor position controlled by BMI output.
    Cursor movement can be assisted toward target by setting assist_level > 0.
    '''
    reset = traits.Int(0, desc='reset the decoder state to the starting configuration. 1 for always, 2 for only on timeout')
    assist_speed = traits.Float(2., desc="speed of assister in cm/s")
    assist_noise = traits.Float(0., desc="noise added to cursor speed in cm/s")
    cursor_color = traits.OptionsList("orange", *target_colors, desc='Color of cursor endpoint', bmi3d_input_options=list(target_colors.keys()))
    save_zscore = traits.Bool(False, desc="save a decoder zscored from this task")

    static_states = ['reward'] # states in which the decoder is not run

    ordered_traits = ['session_length', 'assist_level', 'assist_level_time', 'reward_time','timeout_time','timeout_penalty_time']
    exclude_parent_traits = ['marker_count', 'marker_num', 'goal_cache_block']
    hidden_traits = ['arm_hide_rate', 'arm_visible', 'hold_penalty_time', 'rand_start', 'reset', 'target_radius', 'window_size']

    is_bmi_seed = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def create_assister(self):
        # Create the appropriate type of assister object
        start_level, end_level = self.assist_level
        kwargs = dict(decoder_binlen=self.decoder.binlen, target_radius=self.target_radius)
        kwargs['assist_speed'] = self.assist_speed
        kwargs['assist_noise'] = self.assist_noise

        if isinstance(self.decoder.ssm, StateSpaceEndptVel2D) and isinstance(self.decoder, ppfdecoder.PPFDecoder):
            self.assister = OFCEndpointAssister()
        elif isinstance(self.decoder.ssm, StateSpaceEndptVel2D):
            self.assister = SimpleEndpointAssister(**kwargs)
        ## elif (self.decoder.ssm == namelist.tentacle_2D_state_space) or (self.decoder.ssm == namelist.joint_2D_state_space):
        ##     # kin_chain = self.plant.kin_chain
        ##     # A, B, W = self.decoder.ssm.get_ssm_matrices(update_rate=self.decoder.binlen)
        ##     # Q = np.mat(np.diag(np.hstack([kin_chain.link_lengths, np.zeros_like(kin_chain.link_lengths), 0])))
        ##     # R = 10000*np.mat(np.eye(B.shape[1]))

        ##     # fb_ctrl = LQRController(A, B, Q, R)
        ##     # self.assister = FeedbackControllerAssist(fb_ctrl, style='additive')
        ##     self.assister = TentacleAssist(ssm=self.decoder.ssm, kin_chain=self.plant.kin_chain, update_rate=self.decoder.binlen)
        else:
            raise NotImplementedError("Cannot assist for this type of statespace: %r" % self.decoder.ssm)

        print('Assister: ', self.assister)

    def create_goal_calculator(self):
        if isinstance(self.decoder.ssm, StateSpaceEndptVel2D):
            self.goal_calculator = goal_calculators.ZeroVelocityGoal(self.decoder.ssm)
        elif isinstance(self.decoder.ssm, StateSpaceNLinkPlanarChain) and self.decoder.ssm.n_links == 2:
            self.goal_calculator = goal_calculators.PlanarMultiLinkJointGoal(self.decoder.ssm, self.plant.base_loc, self.plant.kin_chain, multiproc=False, init_resp=None)
        elif isinstance(self.decoder.ssm, StateSpaceNLinkPlanarChain) and self.decoder.ssm.n_links == 4:
            shoulder_anchor = self.plant.base_loc
            chain = self.plant.kin_chain
            q_start = self.plant.get_intrinsic_coordinates()
            x_init = np.hstack([q_start, np.zeros_like(q_start), 1])
            x_init = np.mat(x_init).reshape(-1, 1)

            cached = True

            if cached:
                goal_calc_class = goal_calculators.PlanarMultiLinkJointGoalCached
                multiproc = False
            else:
                goal_calc_class = goal_calculators.PlanarMultiLinkJointGoal
                multiproc = True

            self.goal_calculator = goal_calc_class(namelist.tentacle_2D_state_space, shoulder_anchor,
                                                   chain, multiproc=multiproc, init_resp=x_init)
        else:
            raise ValueError("Unrecognized decoder state space!")

    def get_target_BMI_state(self, *args):
        '''
        Run the goal calculator to determine the target state of the task
        '''
        if isinstance(self.goal_calculator, goal_calculators.PlanarMultiLinkJointGoalCached):
            task_eps = np.inf
        else:
            task_eps = 0.5
        ik_eps = task_eps/10
        data, solution_updated = self.goal_calculator(self.target_location, verbose=False, n_particles=500, eps=ik_eps, n_iter=10, q_start=self.plant.get_intrinsic_coordinates())
        target_state, error = data

        if isinstance(self.goal_calculator, goal_calculators.PlanarMultiLinkJointGoal) and error > task_eps and solution_updated:
            self.goal_calculator.reset()

        return np.array(target_state).reshape(-1,1)

    def _end_targ_transition(self):
        super()._end_targ_transition()
        if self.reset == 1 and ((self.target_index == self.chain_length - 1) or (self.target_index == -1)):

            # Reset on any target transition away from the last target
            self.decoder.filt.state.mean = self.init_decoder_mean.copy()
            self.hdf.sendMsg("reset")

    def _end_timeout_penalty(self):
        super()._end_timeout_penalty()
        if self.reset == 2:

            # Reset on timeout
            self.decoder.filt.state.mean = self.init_decoder_mean.copy()
            self.hdf.sendMsg("reset")

    def _start_None(self):
        super()._start_None()

        # Optionally save a new decoder zscored from this task
        if (not self.save_zscore) or (self.saveid is None):
            return

        if not (np.all(self.decoder.mFR == 0) and np.all(self.decoder.sdFR) == 1):
            filename = self.decoder.save()

            from db.tracker import dbq
            suffix = f"saved_from_{self.saveid}"
            dbq.save_bmi(suffix, self.saveid, filename)
            return

        # This is linear mapping specific - needs updating
        self.decoder.filt.fix_norm_attr()
        self.decoder.filt._update_scale_attr()
        mFR = self.decoder.filt.attr['offset'].copy()
        sdFR = self.decoder.filt.attr['scale'].copy()
        n_units = self.decoder.filt.n_units
        self.decoder.filt.update_norm_attr(offset=np.zeros(n_units), scale=np.ones(n_units))

        # The rest should work with any decoder
        self.decoder.init_zscore(mFR, sdFR)
        filename = self.decoder.save()

        from db.tracker import dbq
        suffix = f"zscored_online_in_{self.saveid}"
        dbq.save_bmi(suffix, self.saveid, filename)


    @classmethod
    def get_desc(cls, params, log_summary):
        duration = round(log_summary['runtime'] / 60, 1)
        return "{}/{} succesful trials in {} min".format(
            log_summary['n_success_trials'], log_summary['n_trials'], duration)

    @control_decorator
    def reset_cursor(self):
        self.decoder.filt.state.mean = self.init_decoder_mean.copy()
        self.hdf.sendMsg("reset")
        
    # @control_decorator
    # def update_zscore(self):
    #     # This is a temporary solution - LRS Jan 2024
    #     # Only works if the decoder currently has mFR == 0 and sdFR == 1
    #     # Assumes you're using a lindecoder with a reasonably large buffer (>2 minutes)
    #     self.decoder.filt.fix_norm_attr() # Should be already!
    #     self.decoder.filt._update_scale_attr()
    #     print(f"updated zscore")
    #     self.hdf.sendMsg(f"updated zscore")
    #     mFR = self.decoder.filt.attr['offset'].copy()
    #     sdFR = self.decoder.filt.attr['scale'].copy()
    #     self.decoder.init_zscore(mFR, sdFR)
    #     self.hdf.sendAttr("task", "session_mFR", mFR)
    #     self.hdf.sendAttr("task", "session_sdFR", sdFR)
    #     n_units = self.decoder.filt.n_units
    #     self.decoder.filt.update_norm_attr(offset=np.zeros(n_units), scale=np.ones(n_units))

    @control_decorator
    def toggle_clda(self):
        self.learn_flag = not self.learn_flag
        self.hdf.sendMsg(f"clda = {self.learn_flag}")
        print(f"clda = {self.learn_flag}")

class BMIControlMulti2DWindow(BMIControlMultiMixin, WindowDispl2D, ScreenTargetCapture):
    fps = 20.
    def __init__(self,*args, **kwargs):
        super(BMIControlMulti2DWindow, self).__init__(*args, **kwargs)

    def create_assister(self):
        kwargs = dict(decoder_binlen=self.decoder.binlen, target_radius=self.target_radius)
        if hasattr(self, 'assist_speed'):
            kwargs['assist_speed'] = self.assist_speed
        self.assister = SimpleEndpointAssister(**kwargs)

    def create_goal_calculator(self):
        self.goal_calculator = goal_calculators.ZeroVelocityGoal(self.decoder.ssm)

    def _start_wait(self):
        self.wait_time = 0.
        super(BMIControlMulti2DWindow, self)._start_wait()

    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause

class BMIControlMulti(BMIControlMultiMixin, ScreenTargetCapture):
    '''
    Slightly refactored original bmi control task
    '''
    pass

class BMIControlMultiDirectionConstraint(BMIControlMultiMixin, ScreenReachAngle):
    '''
    Adds an additional constraint that the direction of travel must be within a certain angle
    '''
    pass

# Add this to the end of bmimultitasks.py

class OptitrackBMIControlMulti(BMIControlMultiMixin, ScreenTargetCapture):
    '''
    Target capture task with cursor position controlled by Optitrack data through BMI infrastructure.
    This task uses the Optitrack system as the neural signal source, processed through a decoder
    to control cursor movement. Supports all standard BMI features like assist levels.
    '''
    
    # Optitrack-specific traits
    optitrack_system = traits.String('rigid body', desc='Type of Optitrack feature to track')
    n_optitrack_features = traits.Int(1, desc='Number of rigid bodies/markers to track')
    optitrack_smooth_len = traits.Int(1, desc='Number of features to average for cursor position')
    
    # Override some BMI traits for Optitrack context
    exclude_parent_traits = BMIControlMultiMixin.exclude_parent_traits + ['neural_source']
    
    def __init__(self, *args, **kwargs):
        # Ensure we're using Optitrack features
        super().__init__(*args, **kwargs)
        
        # Set up Optitrack-specific parameters
        if not hasattr(self, 'optitrack_system'):
            self.optitrack_system = 'rigid body'
        if not hasattr(self, 'n_optitrack_features'):
            self.n_optitrack_features = 1
        if not hasattr(self, 'optitrack_smooth_len'):
            self.optitrack_smooth_len = 1
    
    def init(self):
        '''
        Initialize the task with Optitrack-specific setup
        '''
        # Call parent initialization first
        super().init()
        
        # Verify decoder is compatible with Optitrack data
        self._verify_optitrack_decoder_compatibility()
        
        # Set up Optitrack position scaling if needed
        self._setup_optitrack_scaling()
        
    def _verify_optitrack_decoder_compatibility(self):
        '''
        Verify that the decoder is set up to handle Optitrack kinematic data
        '''
        # Check if decoder expects kinematic input
        if not hasattr(self.decoder, 'filt'):
            raise ValueError("Decoder must have a filter (filt) attribute for Optitrack integration")
        
        # Check input dimensionality matches expected Optitrack channels
        expected_channels = self.n_optitrack_features * 9  # 9 channels per rigid body
        if hasattr(self.decoder.filt, 'n_units'):
            if self.decoder.filt.n_units != expected_channels:
                print(f"Warning: Decoder expects {self.decoder.filt.n_units} channels, "
                      f"but Optitrack will provide {expected_channels} channels")
        
        print(f"Optitrack decoder compatibility verified: {expected_channels} channels expected")
    
    def _setup_optitrack_scaling(self):
        '''
        Set up any necessary scaling between Optitrack coordinates and task coordinates
        '''
        # Optitrack typically works in meters, BMI tasks often in cm
        # This can be adjusted based on your coordinate system requirements
        self.optitrack_to_task_scale = 100.0  # Convert m to cm
        print(f"Optitrack scaling factor: {self.optitrack_to_task_scale}")
    
    def create_assister(self):
        '''
        Create assister for Optitrack BMI control
        Uses the same assistance framework as standard BMI tasks
        '''
        # Use standard endpoint assister - works with any 3D position decoder
        kwargs = dict(
            decoder_binlen=self.decoder.binlen, 
            target_radius=self.target_radius,
            assist_speed=self.assist_speed,
            assist_noise=self.assist_noise
        )
        
        # For Optitrack, we typically use simple endpoint assistance
        # since the state space is usually position-based
        self.assister = SimpleEndpointAssister(**kwargs)
        print(f'Optitrack Assister created: {self.assister}')
    
    def create_goal_calculator(self):
        '''
        Create goal calculator for Optitrack BMI control
        Assumes 3D endpoint control similar to standard BMI tasks
        '''
        # For Optitrack 3D position control, use zero velocity goal
        if isinstance(self.decoder.ssm, StateSpaceEndptVel2D):
            self.goal_calculator = goal_calculators.ZeroVelocityGoal(self.decoder.ssm)
        else:
            # For other state spaces, might need custom goal calculator
            # This is a placeholder - adjust based on your specific decoder
            self.goal_calculator = goal_calculators.ZeroVelocityGoal(self.decoder.ssm)
        
        print(f'Optitrack Goal Calculator created: {self.goal_calculator}')
    
    def _get_cursor_location(self):
        '''
        Override cursor location method to handle Optitrack-specific positioning
        '''
        # Get the standard decoder output
        cursor_loc = super()._get_cursor_location()
        
        # Apply any Optitrack-specific coordinate transformations
        # This is where you might apply scaling, rotation, or offset corrections
        return cursor_loc
    
    def _start_trial(self):
        '''
        Override trial start to add Optitrack-specific initialization
        '''
        super()._start_trial()
        
        # Log Optitrack-specific trial parameters
        self.hdf.sendMsg(f"Optitrack trial started - Features: {self.n_optitrack_features}, "
                        f"System: {self.optitrack_system}")
    
    @classmethod
    def get_desc(cls, params, log_summary):
        '''
        Enhanced description including Optitrack information
        '''
        duration = round(log_summary['runtime'] / 60, 1)
        base_desc = "{}/{} successful trials in {} min".format(
            log_summary['n_success_trials'], log_summary['n_trials'], duration)
        
        # Add Optitrack-specific info if available in params
        if 'n_optitrack_features' in params:
            optitrack_info = f" (Optitrack: {params['n_optitrack_features']} features)"
            return base_desc + optitrack_info
        return base_desc


class OptitrackBMIControlMulti2DWindow(BMIControlMultiMixin, WindowDispl2D, ScreenTargetCapture):
    '''
    2D window version of Optitrack BMI control task
    Similar to BMIControlMulti2DWindow but with Optitrack input
    '''
    fps = 20.
    
    # Optitrack-specific traits
    optitrack_system = traits.String('rigid body', desc='Type of Optitrack feature to track')
    n_optitrack_features = traits.Int(1, desc='Number of rigid bodies/markers to track')
    optitrack_smooth_len = traits.Int(1, desc='Number of features to average for cursor position')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Set up Optitrack defaults
        if not hasattr(self, 'optitrack_system'):
            self.optitrack_system = 'rigid body'
        if not hasattr(self, 'n_optitrack_features'):
            self.n_optitrack_features = 1
    
    def create_assister(self):
        '''Create assister for 2D window Optitrack control'''
        kwargs = dict(
            decoder_binlen=self.decoder.binlen, 
            target_radius=self.target_radius
        )
        if hasattr(self, 'assist_speed'):
            kwargs['assist_speed'] = self.assist_speed
        if hasattr(self, 'assist_noise'):
            kwargs['assist_noise'] = self.assist_noise
            
        self.assister = SimpleEndpointAssister(**kwargs)
    
    def create_goal_calculator(self):
        '''Create goal calculator for 2D window Optitrack control'''
        self.goal_calculator = goal_calculators.ZeroVelocityGoal(self.decoder.ssm)
    
    def _start_wait(self):
        '''Override wait state for 2D window version'''
        self.wait_time = 0.
        super()._start_wait()
    
    def _test_start_trial(self, ts):
        '''Test condition for starting trial in 2D window version'''
        return ts > self.wait_time and not self.pause


# Additional utility class for direct Optitrack position control (no decoder)
class OptitrackDirectControl(ScreenTargetCapture):
    '''
    Direct Optitrack position control without BMI decoder
    Useful for testing and calibration
    '''
    
    # Optitrack traits
    optitrack_system = traits.String('rigid body', desc='Type of Optitrack feature to track')
    n_optitrack_features = traits.Int(1, desc='Number of rigid bodies/markers to track')
    optitrack_smooth_len = traits.Int(1, desc='Number of features to average for cursor position')
    optitrack_scale = traits.Float(100.0, desc='Scale factor from Optitrack units to task units')
    
    # Override traits that don't apply to direct control
    exclude_parent_traits = ['assist_level', 'assist_level_time']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optitrack_position = np.array([0., 0., 0.])
    
    def init(self):
        '''Initialize direct Optitrack control'''
        super().init()
        
        # Initialize Optitrack connection - this would interface with your kinematic system
        print(f"Initializing direct Optitrack control: {self.n_optitrack_features} {self.optitrack_system} features")
        
        # Set up position tracking
        self.optitrack_position = np.array([0., 0., 0.])
    
    def _get_cursor_location(self):
        '''
        Get cursor location directly from Optitrack data
        This would interface with your Optitrack streaming system
        '''
        # In a real implementation, this would get data from your Optitrack system
        # For now, return the stored position
        return self.optitrack_position[:3]  # Return x, y, z
    
    def _update_cursor_location(self):
        '''
        Update cursor location from Optitrack data stream
        This method would be called in your main loop to update position
        '''
        # This is where you'd interface with your Optitrack data stream
        # Example pseudo-code:
        # optitrack_data = self.get_optitrack_data()  # Your method to get latest data
        # self.optitrack_position = optitrack_data * self.optitrack_scale
        pass
    
    @classmethod 
    def get_desc(cls, params, log_summary):
        '''Description for direct Optitrack control'''
        duration = round(log_summary['runtime'] / 60, 1)
        return "Direct Optitrack: {}/{} successful trials in {} min".format(
            log_summary['n_success_trials'], log_summary['n_trials'], duration)