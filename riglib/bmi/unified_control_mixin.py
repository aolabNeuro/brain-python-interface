"""
Unified control mixin for integrating coordinate transforms into BMI and manual control tasks.

This module provides the UnifiedControlMixin class that tasks can inherit from to gain access
to unified coordinate transformation capabilities.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from riglib.experiment import traits
from .unified_control import CoordinateMapper, _axis_angle_rotation


class UnifiedControlMixin(traits.HasTraits):
    """
    Mixin providing unified input and coordinate handling for both BMI and manual control tasks.
    
    Subclasses can optionally override create_control_input() to specify how to handle input.
    By default (for BMI tasks), no ControlInput wrapper is created as the decoder handles it.
    """
    
    # Coordinate transform traits
    scale = traits.Float(1.0, desc="Scale factor for input")
    offset = traits.Array(value=np.array([0, 0, 0]), desc="Offset in 3D space")
    
    # The traits system will populate these from imported dicts
    rotation = traits.String('identity', desc="Base rotation")
    baseline_rotation = traits.String('identity', desc="Baseline workspace rotation")
    exp_rotation = traits.String('identity', desc="Experimental rotation")
    
    perturbation_rotation_x = traits.Float(0.0, desc="X-axis perturbation [deg]")
    perturbation_rotation_y = traits.Float(0.0, desc="Y-axis perturbation [deg]")
    perturbation_rotation_z = traits.Float(0.0, desc="Z-axis perturbation [deg]")
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinate_mapper = None
    
    def create_control_input(self):
        """
        Create and configure control input wrapper.
        
        Called during task init. Subclasses should override to create appropriate
        ControlInput instance (NeuralControlInput for BMI, ManualControlInput for MC).
        
        For BMI tasks using BMILoop, this is often not needed (decoder handles it).
        For manual control tasks, this should wrap joystick input.
        """
        # Default: no-op. Subclasses override.
        pass
    
    def create_coordinate_mapper(self, rotation_dict=None, baseline_rotation_dict=None,
                                 exp_rotation_dict=None):
        """
        Build and store coordinate mapper from traits.
        
        Parameters
        ----------
        rotation_dict : dict, optional
            Dict mapping rotation names to 3x3 matrices for base rotation
        baseline_rotation_dict : dict, optional
            Dict mapping rotation names to 3x3 matrices for baseline rotation
        exp_rotation_dict : dict, optional
            Dict mapping rotation names to 3x3 matrices for experimental rotation
        """
        rotation_dict = rotation_dict or {}
        baseline_rotation_dict = baseline_rotation_dict or {}
        exp_rotation_dict = exp_rotation_dict or {}
        
        # Build perturbation rotations dict, excluding zeros for efficiency
        perturb_rots = {}
        if self.perturbation_rotation_x != 0:
            perturb_rots['x'] = self.perturbation_rotation_x
        if self.perturbation_rotation_y != 0:
            perturb_rots['y'] = self.perturbation_rotation_y
        if self.perturbation_rotation_z != 0:
            perturb_rots['z'] = self.perturbation_rotation_z
        
        # Get rotation matrices from dicts, defaulting to identity
        rot_matrix = rotation_dict.get(self.rotation, np.eye(3))
        baseline_rot_matrix = baseline_rotation_dict.get(self.baseline_rotation, np.eye(3))
        exp_rot_matrix = exp_rotation_dict.get(self.exp_rotation, np.eye(3))
        
        self.coordinate_mapper = CoordinateMapper(
            scale=self.scale,
            offset=self.offset,
            rotation_matrix=rot_matrix,
            baseline_rotation=baseline_rot_matrix,
            exp_rotation=exp_rot_matrix,
            perturbation_rotations=perturb_rots
        )
    
    def init(self, *args, **kwargs):
        """
        Secondary initialization called after primary __init__.
        
        Creates coordinate mapper and control input wrapper.
        """
        # Create coordinate mapper first if not already created
        if self.coordinate_mapper is None:
            self.create_coordinate_mapper()
        
        # Create control input wrapper (subclass-specific)
        self.create_control_input()
        
        super().init(*args, **kwargs)
    
    def get_world_control(self, control_vector, apply_limit2d=False, apply_limit1d=False):
        """
        Apply coordinate transforms to control vector to get world space coordinates.
        
        Parameters
        ----------
        control_vector : np.ndarray of shape (3,), (7,), or None
            Control vector from input source
        apply_limit2d : bool, default=False
            If True, constrain to XZ plane (zero Y)
        apply_limit1d : bool, default=False
            If True, constrain to Z axis only (zero X and Y)
        
        Returns
        -------
        np.ndarray of shape (3,) or None
            Transformed world coordinates
        """
        if self.coordinate_mapper is None:
            # Fallback if mapper not initialized
            if control_vector is None:
                return None
            control_vector = np.asarray(control_vector).ravel()
            pos = control_vector[:3].copy()
            if apply_limit1d:
                pos[0] = pos[1] = 0
            elif apply_limit2d:
                pos[1] = 0
            return pos
        
        return self.coordinate_mapper(
            control_vector,
            apply_limit2d=apply_limit2d,
            apply_limit1d=apply_limit1d
        )
    
    def update_rotation(self, rotation_name, rotation_dict):
        """Update base rotation."""
        self.rotation = rotation_name
        if self.coordinate_mapper:
            self.coordinate_mapper.set_rotation(rotation_dict.get(rotation_name, np.eye(3)))
    
    def update_baseline_rotation(self, rotation_name, baseline_rotation_dict):
        """Update baseline rotation."""
        self.baseline_rotation = rotation_name
        if self.coordinate_mapper:
            self.coordinate_mapper.set_baseline_rotation(
                baseline_rotation_dict.get(rotation_name, np.eye(3))
            )
    
    def update_exp_rotation(self, rotation_name, exp_rotation_dict):
        """Update experimental rotation."""
        self.exp_rotation = rotation_name
        if self.coordinate_mapper:
            self.coordinate_mapper.set_exp_rotation(exp_rotation_dict.get(rotation_name, np.eye(3)))
    
    def update_perturbation_rotation(self, axis, angle_deg):
        """
        Update a perturbation rotation.
        
        Parameters
        ----------
        axis : str
            'x', 'y', or 'z'
        angle_deg : float
            Rotation angle in degrees
        """
        setattr(self, f'perturbation_rotation_{axis}', angle_deg)
        if self.coordinate_mapper:
            self.coordinate_mapper.set_perturbation(axis, angle_deg)
