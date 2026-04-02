"""
Unified Control Architecture for BMI and Manual Control Tasks

This module provides a decoupled input pipeline architecture that allows both
neural (BMI) and manual control tasks to share a common coordinate transform
system (CoordinateMapper). This enables uniform experimental manipulations
(rotations, offsets, scales) across both input modalities.

Components:
- ControlInput: Base abstraction for input sources
- NeuralControlInput: Wraps neural decoder for BMI control
- ManualControlInput: Maps joystick/motion tracker to 3D control
- CoordinateMapper: Applies all coordinate transforms to control output
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def _axis_angle_rotation(axis, angle_deg):
    """
    Create a rotation matrix from axis and angle in degrees.
    
    Parameters
    ----------
    axis : str
        'x', 'y', or 'z'
    angle_deg : float
        Rotation angle in degrees
    
    Returns
    -------
    np.ndarray of shape (3, 3)
        Rotation matrix
    """
    rot = R.from_euler(axis, angle_deg, degrees=True)
    return rot.as_matrix()


def _coerce_rotation_matrix(matrix, use_legacy_row_rotations=False):
    """
    Normalize rotation matrix input to a 3x3 numpy array.

    Supports both legacy 4x4 homogeneous rotation matrices and 3x3 matrices.
    For legacy row-vector semantics, returns the transposed 3x3 block so
    column-vector multiplication preserves historical behavior.
    """
    if matrix is None:
        return np.eye(3)

    matrix = np.asarray(matrix)

    if matrix.shape == (4, 4):
        matrix3 = matrix[:3, :3]
        if use_legacy_row_rotations:
            return matrix3.T
        return matrix3

    if matrix.shape == (3, 3):
        return matrix

    raise ValueError("Rotation matrix must be 3x3 or 4x4, got shape %r" % (matrix.shape,))


class ControlInput:
    """
    Base class for input sources that map raw input to 3D control vectors.
    
    Subclasses implement different input modalities (neural, manual, etc.)
    while providing a unified interface.
    """
    
    def __call__(self, raw_input, **kwargs):
        """
        Convert raw input to 3D control vector.
        
        Parameters
        ----------
        raw_input : np.ndarray or similar
            Raw input from the device/decoder
        **kwargs : dict
            Additional keyword arguments (dt, mode, etc.)
        
        Returns
        -------
        np.ndarray of shape (3,) or (7,)
            3D control vector representing position or velocity
            7D vector if includes velocity components (Px, Py, Pz, Vx, Vy, Vz, mode)
        """
        raise NotImplementedError
    
    def get_state(self):
        """
        Optionally return internal state (e.g., decoder state estimate).
        
        Returns
        -------
        np.ndarray or None
            Internal state if available, None otherwise
        """
        return None


class NeuralControlInput(ControlInput):
    """
    Wraps existing neural decoder to provide unified ControlInput interface.
    
    Integrates with:
    - Feature extractor
    - Feature accumulator
    - Kalman filter decoder
    """
    
    def __init__(self, decoder, extractor, feature_accumulator):
        """
        Initialize neural control input.
        
        Parameters
        ----------
        decoder : riglib.bmi.kfdecoder.KalmanFilterDecoder or similar
            Decoder that produces state estimates from features
        extractor : callable
            Feature extractor that converts spike data to features
        feature_accumulator : callable
            Feature accumulator that buffers features and determines decode timing
        """
        self.decoder = decoder
        self.extractor = extractor
        self.feature_accumulator = feature_accumulator
    
    def __call__(self, spike_data, **kwargs):
        """
        Decode spike data to 3D control vector.
        
        Parameters
        ----------
        spike_data : np.ndarray
            Spike counts or sorted unit data
        **kwargs : dict
            Additional arguments passed to decoder (e.g., assist_level, mode)
        
        Returns
        -------
        np.ndarray of shape (7,)
            Decoder state estimate [Px, Py, Pz, Vx, Vy, Vz, mode]
        """
        if spike_data is None or len(spike_data) == 0:
            return None
        
        # Extract features
        features = self.extractor(spike_data)
        
        # Accumulate features and check if should decode
        decodable_obs, should_decode = self.feature_accumulator(features)
        
        # Decode if accumulator indicates
        if should_decode:
            self.decoder(decodable_obs, **kwargs)
        
        # Return current decoder state estimate
        return self.decoder.get_state()
    
    def get_state(self):
        """Return decoder state estimate."""
        return self.decoder.get_state()


class ManualControlInput(ControlInput):
    """
    Maps joystick or motion tracker input to 3D control vector.
    
    Supports both position and velocity control modes.
    """
    
    def __init__(self, scale=1.0, velocity_control=False):
        """
        Initialize manual control input.
        
        Parameters
        ----------
        scale : float, default=1.0
            Scaling factor applied to raw input
        velocity_control : bool, default=False
            If True, output is velocity (input derivative)
            If False, output is position
        """
        self.scale = scale
        self.velocity_control = velocity_control
        self.last_pos = np.zeros(3)
    
    def __call__(self, raw_input, dt=1./60., **kwargs):
        """
        Convert raw joystick/motion tracker input to 3D control vector.
        
        Parameters
        ----------
        raw_input : np.ndarray of shape (2,) or (3,)
            Raw 2D or 3D input from joystick/motion tracker
        dt : float, default=1./60.
            Time step, used for velocity calculation
        **kwargs : dict
            Additional arguments (unused for manual control)
        
        Returns
        -------
        np.ndarray of shape (3,) or None
            3D position or velocity vector, or None if no input
        """
        if raw_input is None or len(raw_input) == 0:
            return None
        
        raw_input = np.asarray(raw_input).ravel()
        
        # Pad 2D input to 3D
        if len(raw_input) == 2:
            raw_input = np.concatenate([raw_input, [0]])
        elif len(raw_input) != 3:
            # Take first 3 components if longer
            raw_input = raw_input[:3]
        
        # Apply scale
        pos = raw_input * self.scale
        
        if self.velocity_control:
            # Compute velocity as derivative
            vel = (pos - self.last_pos) / dt
            self.last_pos = pos.copy()
            return vel
        else:
            # Return position
            self.last_pos = pos.copy()
            return pos
    
    def reset(self):
        """Reset internal state (last position)."""
        self.last_pos = np.zeros(3)


class CoordinateMapper:
    """
    Applies coordinate transforms to map control space to world space.
    
    Transforms are applied in sequence:
    1. Scale factor
    2. Offset
    3. Base rotation
    4. Baseline rotation (motion direction)
    5. Experimental rotation/perturbation
    6. Fine perturbations (x, y, z axis rotations)
    
    Optionally applies 1D or 2D constraints.
    """
    
    def __init__(self, scale=1.0, offset=None, rotation_matrix=None,
                 baseline_rotation=None, exp_rotation=None,
                 perturbation_rotations=None, use_legacy_row_rotations=False):
        """
        Initialize coordinate mapper.
        
        Parameters
        ----------
        scale : float, default=1.0
            Scale factor applied to control vector
        offset : np.ndarray of shape (3,), optional
            Offset in 3D space to add after scaling
        rotation_matrix : np.ndarray of shape (3, 3), optional
            Base rotation matrix
        baseline_rotation : np.ndarray of shape (3, 3), optional
            Baseline workspace rotation (defines motion direction)
        exp_rotation : np.ndarray of shape (3, 3), optional
            Experimental/perturbation rotation
        perturbation_rotations : dict, optional
            Dict mapping 'x', 'y', 'z' to rotation angles in degrees
            Example: {'y': 15.0, 'z': -10.0}
        use_legacy_row_rotations : bool, default=False
            If True, preserves legacy row-vector rotation behavior used by
            historical task rotation dictionaries.
        """
        self.scale = scale
        self.offset = offset if offset is not None else np.zeros(3)
        self.use_legacy_row_rotations = use_legacy_row_rotations
        self.rotation_matrix = _coerce_rotation_matrix(rotation_matrix, use_legacy_row_rotations=self.use_legacy_row_rotations)
        self.baseline_rotation = _coerce_rotation_matrix(baseline_rotation, use_legacy_row_rotations=self.use_legacy_row_rotations)
        self.exp_rotation = _coerce_rotation_matrix(exp_rotation, use_legacy_row_rotations=self.use_legacy_row_rotations)
        self.perturbation_rotations = perturbation_rotations or {}
    
    def __call__(self, control_vector, apply_limit2d=False, apply_limit1d=False):
        """
        Apply all coordinate transforms in sequence.
        
        Parameters
        ----------
        control_vector : np.ndarray of shape (3,), (7,), or None
            3D control vector or 7D state estimate
        apply_limit2d : bool, default=False
            If True, zero Y component (constrain to XZ plane)
        apply_limit1d : bool, default=False
            If True, zero X and Y (constrain to Z axis)
        
        Returns
        -------
        np.ndarray of shape (3,) or None
            Transformed 3D world coordinates, or None if input is None
        """
        if control_vector is None:
            return None
        
        control_vector = np.asarray(control_vector).ravel()
        
        # Extract position (first 3 components)
        # Handle both 3D and 7D (with velocity) vectors
        pos = control_vector[:3].copy()
        
        # Apply endpoint limits
        if apply_limit1d:
            pos[0] = 0
            pos[1] = 0
        elif apply_limit2d:
            pos[1] = 0
        
        # Apply scale
        pos = pos * self.scale
        
        # Apply offset
        pos = pos + self.offset
        
        # Apply rotation sequence
        pos = self.rotation_matrix @ pos
        pos = self.baseline_rotation @ pos
        pos = self.exp_rotation @ pos
        
        # Apply fine perturbations (axis-specific rotations)
        for axis, angle_deg in self.perturbation_rotations.items():
            if angle_deg != 0:
                rot = _axis_angle_rotation(axis, angle_deg)
                if self.use_legacy_row_rotations:
                    pos = rot.T @ pos
                else:
                    pos = rot @ pos
        
        return pos
    
    def set_rotation(self, rotation_matrix):
        """Set base rotation matrix."""
        self.rotation_matrix = _coerce_rotation_matrix(rotation_matrix, use_legacy_row_rotations=self.use_legacy_row_rotations)
    
    def set_baseline_rotation(self, rotation_matrix):
        """Set baseline rotation matrix."""
        self.baseline_rotation = _coerce_rotation_matrix(rotation_matrix, use_legacy_row_rotations=self.use_legacy_row_rotations)
    
    def set_exp_rotation(self, rotation_matrix):
        """Set experimental rotation matrix."""
        self.exp_rotation = _coerce_rotation_matrix(rotation_matrix, use_legacy_row_rotations=self.use_legacy_row_rotations)
    
    def set_perturbation(self, axis, angle_deg):
        """
        Set perturbation rotation for a single axis.
        
        Parameters
        ----------
        axis : str
            'x', 'y', or 'z'
        angle_deg : float
            Rotation angle in degrees
        """
        self.perturbation_rotations[axis] = angle_deg
    
    def clear_perturbations(self):
        """Clear all perturbation rotations."""
        self.perturbation_rotations.clear()
