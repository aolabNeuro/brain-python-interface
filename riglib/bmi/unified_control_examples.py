"""
Example integrations of the Unified Control Architecture with existing tasks.

This module demonstrates how to:
1. Create unified manual control tasks
2. Create unified BMI tasks  
3. Blend neural and manual control

These examples show the patterns for migrating existing tasks to the unified architecture.
"""

import numpy as np

from riglib.experiment import traits
from riglib.bmi.bmi import BMILoop
from riglib.bmi.unified_control import NeuralControlInput, ManualControlInput, CoordinateMapper
from riglib.bmi.unified_control_mixin import UnifiedControlMixin

# Import example target capture task (adjust path as needed)
from built_in_tasks.target_capture_task import ScreenTargetCapture
from built_in_tasks.rotation_matrices import rotations, baseline_rotations, exp_rotations


class UnifiedManualControl(UnifiedControlMixin, ScreenTargetCapture):
    """
    Example: Manual control task using unified coordinate transforms.
    
    This task demonstrates how to migrate an existing manual control task
    to use the unified control architecture, gaining access to shared
    coordinate transform infrastructure.
    """
    
    # Inherit coordinate transform traits from UnifiedControlMixin
    # (scale, offset, rotation, baseline_rotation, exp_rotation, perturbation_rotation_*)
    
    velocity_control = traits.Bool(False, desc="Use velocity or position control")
    
    def create_control_input(self):
        """Create manual control input wrapper."""
        self.control_input = ManualControlInput(
            scale=self.scale,  # Note: scale can also be in UnifiedControlMixin
            velocity_control=self.velocity_control
        )
    
    def create_coordinate_mapper(self):
        """Override to provide rotation dicts to parent method."""
        super().create_coordinate_mapper(
            rotation_dict=rotations,
            baseline_rotation_dict=baseline_rotations,
            exp_rotation_dict=exp_rotations
        )
    
    def move_effector(self):
        """
        Sets cursor position using unified pipeline.
        
        Pipeline:
        1. Get raw joystick input via _get_manual_position()
        2. Map through ManualControlInput (scale, padding)
        3. Apply coordinate transforms via get_world_control()
        4. Drive plant to world coordinates
        """
        # Get raw input
        raw_coords = self._get_manual_position()
        
        if raw_coords is None or len(raw_coords) < 1:
            self.task_data['manual_input'] = np.ones((3,)) * np.nan
            self.task_data['user_screen'] = np.ones((3,)) * np.nan
            return
        
        # Record raw input
        self.task_data['manual_input'] = raw_coords.copy()
        
        # Map through control input (scale + padding)
        control_vector = self.control_input(raw_coords, dt=1./self.fps)
        
        # Apply coordinate transforms
        world_coords = self.get_world_control(
            control_vector,
            apply_limit2d=getattr(self, 'limit2d', False),
            apply_limit1d=getattr(self, 'limit1d', False)
        )
        
        # Record transformed output
        self.task_data['user_screen'] = world_coords
        
        # Drive plant
        if not self.velocity_control:
            self.current_pt = world_coords
        else:
            # Velocity control
            epsilon = 2 * (10 ** -2)
            if np.sum(world_coords ** 2) > epsilon:
                self.current_pt = world_coords / self.fps + self.last_pt
            else:
                self.current_pt = self.last_pt
        
        self.plant.set_endpoint_pos(self.current_pt)
        self.last_pt = self.plant.get_endpoint_pos()
    
    def _get_manual_position(self):
        """Get raw joystick position (implementation depends on hardware)."""
        if not hasattr(self, 'joystick'):
            return None
        
        pt = self.joystick.get()
        if len(pt) == 0:
            return None
        
        pt = pt[-1]  # Use latest coordinate
        if len(pt) == 2:
            pt = np.concatenate((np.reshape(pt, -1), [0]))
        
        return pt


class UnifiedBMIControl(UnifiedControlMixin, BMILoop, ScreenTargetCapture):
    """
    Example: BMI control task using unified coordinate transforms.
    
    This task shows how BMI tasks can leverage the unified coordinate
    transform pipeline while maintaining existing decoder infrastructure.
    """
    
    # Inherit coordinate transform traits from UnifiedControlMixin
    
    def create_control_input(self):
        """
        Create neural control input wrapper around BMILoop's decoder.
        
        Note: BMILoop already has decoder, extractor, and feature_accumulator
        set up during its initialization. We just wrap them.
        """
        if hasattr(self, 'decoder') and hasattr(self, 'extractor') and \
           hasattr(self, 'feature_accumulator'):
            self.control_input = NeuralControlInput(
                self.decoder,
                self.extractor,
                self.feature_accumulator
            )
    
    def create_coordinate_mapper(self):
        """Override to provide rotation dicts."""
        super().create_coordinate_mapper(
            rotation_dict=rotations,
            baseline_rotation_dict=baseline_rotations,
            exp_rotation_dict=exp_rotations
        )
    
    def move_plant(self):
        """
        Move plant using decoded neural signal with unified transforms.
        
        Pipeline:
        1. Get spike data / features from BMI system
        2. Decode via control_input (NeuralControlInput wraps decoder)
        3. Apply coordinate transforms
        4. Drive plant
        """
        # Get neural features (BMILoop's job)
        features = self.get_features()
        
        if features is None:
            self.task_data['plant_pos'] = np.nan
            return
        
        # Decode features to control vector via neural input wrapper
        if hasattr(self, 'control_input'):
            control_vector = self.control_input(features)
        else:
            # Fallback: use decoder directly (for backward compat)
            self.decoder(features)
            control_vector = self.decoder.get_state()
        
        # Apply coordinate transforms
        world_pos = self.get_world_control(
            control_vector,
            apply_limit2d=getattr(self, 'limit2d', False),
            apply_limit1d=getattr(self, 'limit1d', False)
        )
        
        # Drive plant
        if world_pos is not None:
            self.plant.set_endpoint_pos(world_pos)
            self.task_data['plant_pos'] = world_pos
        else:
            self.task_data['plant_pos'] = np.nan


class HybridControlTask(UnifiedControlMixin, BMILoop, ScreenTargetCapture):
    """
    Example: Blended neural and manual control task.
    
    Demonstrates how the unified architecture enables easy blending
    of multiple input modalities.
    """
    
    # Blending parameter
    neural_weight = traits.Float(
        0.5, 
        desc="Weight for neural control [0=manual, 1=neural]"
    )
    
    # Manual control parameter
    velocity_control = traits.Bool(False, desc="Manual velocity or position control")
    
    def create_control_input(self):
        """Create both neural and manual control inputs."""
        # Neural input (wraps BMILoop decoder)
        if hasattr(self, 'decoder') and hasattr(self, 'extractor') and \
           hasattr(self, 'feature_accumulator'):
            self.neural_input = NeuralControlInput(
                self.decoder,
                self.extractor,
                self.feature_accumulator
            )
        
        # Manual input (wraps joystick)
        self.manual_input = ManualControlInput(
            scale=1.0,
            velocity_control=self.velocity_control
        )
    
    def create_coordinate_mapper(self):
        """Override to provide rotation dicts."""
        super().create_coordinate_mapper(
            rotation_dict=rotations,
            baseline_rotation_dict=baseline_rotations,
            exp_rotation_dict=exp_rotations
        )
    
    def move_plant(self):
        """
        Move plant using blended neural and manual control.
        
        Pipeline:
        1. Get neural features and manual input
        2. Decode to control vectors via respective inputs
        3. Blend based on neural_weight
        4. Apply unified coordinate transforms
        5. Drive plant
        """
        # Get neural signal
        features = self.get_features()
        neural_control = None
        if features is not None and hasattr(self, 'neural_input'):
            neural_control = self.neural_input(features)
        
        # Get manual input
        manual_raw = self._get_manual_position()
        manual_control = None
        if manual_raw is not None and hasattr(self, 'manual_input'):
            manual_control = self.manual_input(manual_raw, dt=1./self.fps)
        
        # Blend controls
        blended_control = self._blend_controls(
            neural_control, 
            manual_control,
            self.neural_weight
        )
        
        # Apply transforms
        world_pos = self.get_world_control(
            blended_control,
            apply_limit2d=getattr(self, 'limit2d', False),
            apply_limit1d=getattr(self, 'limit1d', False)
        )
        
        # Drive plant
        if world_pos is not None:
            self.plant.set_endpoint_pos(world_pos)
    
    def _blend_controls(self, neural, manual, neural_weight):
        """
        Blend neural and manual control signals.
        
        Parameters
        ----------
        neural : np.ndarray or None
            Neural-decoded control (3D or 7D)
        manual : np.ndarray or None
            Manual control (3D)
        neural_weight : float
            Weight for neural [0=manual, 1=neural]
        
        Returns
        -------
        np.ndarray or None
            Blended control (3D)
        """
        if neural is None and manual is None:
            return None
        
        if neural is None:
            return manual[:3] if manual is not None else None
        
        if manual is None:
            return neural[:3]
        
        # Blend: take only position components (first 3)
        neural_pos = neural[:3]
        manual_pos = manual[:3]
        
        blended = neural_weight * neural_pos + (1 - neural_weight) * manual_pos
        return blended
    
    def _get_manual_position(self):
        """Get raw manual input."""
        if not hasattr(self, 'joystick'):
            return None
        
        pt = self.joystick.get()
        if len(pt) == 0:
            return None
        
        pt = pt[-1]
        if len(pt) == 2:
            pt = np.concatenate((np.reshape(pt, -1), [0]))
        
        return pt


class UnifiedBMIWithPerturbation(UnifiedBMIControl):
    """
    Example: BMI task with easy perturbation application.
    
    Demonstrates the power of the unified architecture: perturbations
    are trivial to add to any task by just setting a trait.
    """
    
    # Perturbation settings (inherited from UnifiedControlMixin)
    # - perturbation_rotation_x
    # - perturbation_rotation_y  
    # - perturbation_rotation_z
    
    # These can be set at runtime and the coordinate mapper automatically
    # applies them to all control vectors.
    
    def apply_visuomotor_rotation(self, degrees):
        """Apply a visuomotor rotation perturbation about Z axis."""
        self.update_perturbation_rotation('z', degrees)
    
    def apply_force_field(self, direction_deg, magnitude=0.5):
        """
        Apply a force field perturbation.
        
        For example, a velocity-dependent force field can be approximated
        by rotating the perceived motion direction.
        
        Parameters
        ----------
        direction_deg : float
            Direction of force field perturbation in degrees
        magnitude : float
            Magnitude factor (currently unused, but could scale rotations)
        """
        # In a real implementation, you could apply different perturbations
        # based on magnitude or other parameters
        self.apply_visuomotor_rotation(magnitude * direction_deg)


# ============================================================================
# Usage Examples
# ============================================================================

def example_usage():
    """
    Example of how to use the unified control tasks.
    
    In practice, these would be instantiated through the BMI3D GUI
    or experiment runner. This shows the general pattern.
    """
    
    # Example 1: Unified manual control
    # mc_task = UnifiedManualControl(
    #     scale=1.5,
    #     offset=np.array([0, 0, 0]),
    #     rotation='identity',
    #     baseline_rotation='identity',
    #     exp_rotation='identity',
    #     perturbation_rotation_z=0,
    # )
    # mc_task.init()
    
    # Example 2: Unified BMI control
    # bmi_task = UnifiedBMIControl(
    #     scale=1.0,
    #     exp_rotation='90_deg_z',  # Apply 90-degree rotation
    # )
    # bmi_task.init()
    
    # Example 3: Hybrid task
    # hybrid_task = HybridControlTask(
    #     neural_weight=0.5,  # 50/50 blend
    #     perturbation_rotation_y=15.0,  # 15-degree perturbation
    # )
    # hybrid_task.init()
    
    # Example 4: Apply perturbation at runtime
    # bmi_with_perturb = UnifiedBMIWithPerturbation()
    # bmi_with_perturb.init()
    # bmi_with_perturb.apply_visuomotor_rotation(30)  # 30-degree rotation
    
    pass
