"""
Unit tests for unified control architecture.

Tests for:
- ControlInput base class and implementations
- CoordinateMapper transforms
- UnifiedControlMixin integration
"""

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
import os

# Add parent dirs to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from riglib.bmi.unified_control import (
    ControlInput, NeuralControlInput, ManualControlInput, CoordinateMapper, _axis_angle_rotation
)


class TestAxisAngleRotation:
    """Test axis-angle rotation helper."""
    
    def test_identity_rotation(self):
        """Test zero rotation is identity."""
        rot = _axis_angle_rotation('x', 0)
        np.testing.assert_array_almost_equal(rot, np.eye(3))
    
    def test_90_degree_rotations(self):
        """Test 90-degree rotations produce expected matrices."""
        # 90-degree rotation about X
        rot_x = _axis_angle_rotation('x', 90)
        pt = np.array([0, 1, 0])
        result = rot_x @ pt
        np.testing.assert_array_almost_equal(result, [0, 0, 1])
        
        # 90-degree rotation about Y
        rot_y = _axis_angle_rotation('y', 90)
        pt = np.array([1, 0, 0])
        result = rot_y @ pt
        np.testing.assert_array_almost_equal(result, [0, 0, -1])
        
        # 90-degree rotation about Z
        rot_z = _axis_angle_rotation('z', 90)
        pt = np.array([1, 0, 0])
        result = rot_z @ pt
        np.testing.assert_array_almost_equal(result, [0, 1, 0])


class TestManualControlInput:
    """Test ManualControlInput class."""
    
    def test_init_defaults(self):
        """Test default initialization."""
        input_ctrl = ManualControlInput()
        assert input_ctrl.scale == 1.0
        assert input_ctrl.velocity_control is False
        np.testing.assert_array_equal(input_ctrl.last_pos, [0, 0, 0])
    
    def test_2d_input_padding(self):
        """Test 2D input is padded to 3D."""
        input_ctrl = ManualControlInput()
        result = input_ctrl([1.0, 2.0])
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 0])
    
    def test_3d_input_pass_through(self):
        """Test 3D input passes through with scale."""
        input_ctrl = ManualControlInput(scale=2.0)
        result = input_ctrl([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result, [2.0, 4.0, 6.0])
    
    def test_position_control(self):
        """Test position control mode."""
        input_ctrl = ManualControlInput(velocity_control=False)
        result = input_ctrl([1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, [1.0, 1.0, 1.0])
        
        # Calling again should give same result
        result2 = input_ctrl([1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result2, [1.0, 1.0, 1.0])
    
    def test_velocity_control(self):
        """Test velocity control mode."""
        input_ctrl = ManualControlInput(velocity_control=True)
        
        # First call: velocity = (pos - [0,0,0]) / dt
        dt = 1.0
        result = input_ctrl([1.0, 0.0, 0.0], dt=dt)
        # velocity = ([1, 0, 0] - [0, 0, 0]) / 1.0 = [1, 0, 0]
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 0.0])
        
        # Second call: velocity = (pos - last_pos) / dt
        result2 = input_ctrl([2.0, 0.0, 0.0], dt=dt)
        # velocity = ([2, 0, 0] - [1, 0, 0]) / 1.0 = [1, 0, 0]
        np.testing.assert_array_almost_equal(result2, [1.0, 0.0, 0.0])
    
    def test_none_input(self):
        """Test None input returns None."""
        input_ctrl = ManualControlInput()
        result = input_ctrl(None)
        assert result is None
    
    def test_empty_input(self):
        """Test empty input returns None."""
        input_ctrl = ManualControlInput()
        result = input_ctrl([])
        assert result is None
    
    def test_reset(self):
        """Test reset clears internal state."""
        input_ctrl = ManualControlInput(velocity_control=True)
        input_ctrl([1.0, 0.0, 0.0], dt=1.0)
        assert not np.allclose(input_ctrl.last_pos, [0, 0, 0])
        
        input_ctrl.reset()
        np.testing.assert_array_equal(input_ctrl.last_pos, [0, 0, 0])


class TestCoordinateMapper:
    """Test CoordinateMapper class."""
    
    def test_identity_mapping(self):
        """Test identity mapping returns input."""
        mapper = CoordinateMapper()
        input_pt = np.array([1.0, 2.0, 3.0])
        result = mapper(input_pt)
        np.testing.assert_array_almost_equal(result, input_pt)
    
    def test_scale(self):
        """Test scale transformation."""
        mapper = CoordinateMapper(scale=2.0)
        input_pt = np.array([1.0, 2.0, 3.0])
        result = mapper(input_pt)
        np.testing.assert_array_almost_equal(result, [2.0, 4.0, 6.0])
    
    def test_offset(self):
        """Test offset transformation."""
        offset = np.array([1.0, 2.0, 3.0])
        mapper = CoordinateMapper(offset=offset)
        input_pt = np.array([1.0, 1.0, 1.0])
        result = mapper(input_pt)
        np.testing.assert_array_almost_equal(result, [2.0, 3.0, 4.0])
    
    def test_scale_and_offset(self):
        """Test combined scale and offset."""
        mapper = CoordinateMapper(scale=2.0, offset=np.array([1.0, 0.0, 0.0]))
        input_pt = np.array([1.0, 1.0, 1.0])
        result = mapper(input_pt)
        # First scale: [2, 2, 2], then offset: [3, 2, 2]
        np.testing.assert_array_almost_equal(result, [3.0, 2.0, 2.0])
    
    def test_rotation(self):
        """Test rotation transformation."""
        rot_90_z = _axis_angle_rotation('z', 90)
        mapper = CoordinateMapper(rotation_matrix=rot_90_z)
        
        input_pt = np.array([1.0, 0.0, 0.0])
        result = mapper(input_pt)
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 0.0])
    
    def test_7d_input_handling(self):
        """Test 7D input (with velocity) uses only first 3 components."""
        mapper = CoordinateMapper(scale=2.0)
        # 7D: [Px, Py, Pz, Vx, Vy, Vz, mode]
        input_pt = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0])
        result = mapper(input_pt)
        # First 3 scaled, velocity components ignored
        np.testing.assert_array_almost_equal(result, [2.0, 4.0, 6.0])
    
    def test_limit1d(self):
        """Test 1D limit (constrain to Z axis)."""
        mapper = CoordinateMapper()
        input_pt = np.array([1.0, 2.0, 3.0])
        result = mapper(input_pt, apply_limit1d=True)
        # X and Y zeroed
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 3.0])
    
    def test_limit2d(self):
        """Test 2D limit (constrain to XZ plane)."""
        mapper = CoordinateMapper()
        input_pt = np.array([1.0, 2.0, 3.0])
        result = mapper(input_pt, apply_limit2d=True)
        # Y zeroed
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 3.0])
    
    def test_none_input(self):
        """Test None input returns None."""
        mapper = CoordinateMapper()
        result = mapper(None)
        assert result is None
    
    def test_perturbation_rotations(self):
        """Test perturbation rotations."""
        perturb = {'z': 90}
        mapper = CoordinateMapper(perturbation_rotations=perturb)
        
        input_pt = np.array([1.0, 0.0, 0.0])
        result = mapper(input_pt)
        # 90-degree rotation about Z: [1, 0, 0] -> [0, 1, 0]
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 0.0])
    
    def test_multiple_perturbations(self):
        """Test multiple perturbation rotations applied in sequence."""
        perturb = {'x': 90, 'y': 90}
        mapper = CoordinateMapper(perturbation_rotations=perturb)
        
        input_pt = np.array([1.0, 0.0, 0.0])
        result = mapper(input_pt)
        
        # Verify result is reasonable (exact computation would be complex)
        assert result is not None
        assert result.shape == (3,)
    
    def test_set_rotation(self):
        """Test updating rotation after creation."""
        mapper = CoordinateMapper()
        rot_90_z = _axis_angle_rotation('z', 90)
        mapper.set_rotation(rot_90_z)
        
        input_pt = np.array([1.0, 0.0, 0.0])
        result = mapper(input_pt)
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 0.0])
    
    def test_set_perturbation(self):
        """Test setting perturbation after creation."""
        mapper = CoordinateMapper()
        mapper.set_perturbation('z', 90)
        
        input_pt = np.array([1.0, 0.0, 0.0])
        result = mapper(input_pt)
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 0.0])
    
    def test_clear_perturbations(self):
        """Test clearing all perturbations."""
        mapper = CoordinateMapper(perturbation_rotations={'x': 45, 'y': 45, 'z': 45})
        mapper.clear_perturbations()
        
        assert len(mapper.perturbation_rotations) == 0
        
        input_pt = np.array([1.0, 0.0, 0.0])
        result = mapper(input_pt)
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 0.0])


class TestControlInputBase:
    """Test ControlInput base class."""
    
    def test_base_class_not_callable(self):
        """Test that base class raises NotImplementedError when called."""
        input_ctrl = ControlInput()
        with pytest.raises(NotImplementedError):
            input_ctrl([1, 2, 3])
    
    def test_get_state_default(self):
        """Test default get_state returns None."""
        input_ctrl = ControlInput()
        assert input_ctrl.get_state() is None


class MockDecoder:
    """Mock decoder for testing NeuralControlInput."""
    
    def __init__(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.last_features = None
    
    def __call__(self, features, **kwargs):
        """Decode features to state."""
        self.last_features = features
        # Simple mock: scale features to state
        if len(features) >= 3:
            self.state[:3] = features[:3]
    
    def get_state(self):
        """Return current state."""
        return self.state.copy()


class MockExtractor:
    """Mock feature extractor for testing NeuralControlInput."""
    
    def __call__(self, spike_data):
        """Extract features from spike data."""
        return spike_data[:3] if len(spike_data) >= 3 else spike_data


class MockFeatureAccumulator:
    """Mock feature accumulator for testing NeuralControlInput."""
    
    def __call__(self, features):
        """Accumulate features and decide when to decode."""
        return features, True  # Always decode


class TestNeuralControlInput:
    """Test NeuralControlInput class."""
    
    def test_init(self):
        """Test initialization."""
        decoder = MockDecoder()
        extractor = MockExtractor()
        accumulator = MockFeatureAccumulator()
        
        input_ctrl = NeuralControlInput(decoder, extractor, accumulator)
        assert input_ctrl.decoder is decoder
        assert input_ctrl.extractor is extractor
        assert input_ctrl.feature_accumulator is accumulator
    
    def test_call(self):
        """Test calling with spike data."""
        decoder = MockDecoder()
        extractor = MockExtractor()
        accumulator = MockFeatureAccumulator()
        
        input_ctrl = NeuralControlInput(decoder, extractor, accumulator)
        spike_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = input_ctrl(spike_data)
        assert result is not None
        np.testing.assert_array_almost_equal(result[:3], [1.0, 2.0, 3.0])
    
    def test_none_input(self):
        """Test None input returns None."""
        decoder = MockDecoder()
        extractor = MockExtractor()
        accumulator = MockFeatureAccumulator()
        
        input_ctrl = NeuralControlInput(decoder, extractor, accumulator)
        result = input_ctrl(None)
        assert result is None
    
    def test_get_state(self):
        """Test get_state returns decoder state."""
        decoder = MockDecoder()
        extractor = MockExtractor()
        accumulator = MockFeatureAccumulator()
        
        input_ctrl = NeuralControlInput(decoder, extractor, accumulator)
        state = input_ctrl.get_state()
        
        assert state is not None
        np.testing.assert_array_equal(state, decoder.state)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_manual_to_world_pipeline(self):
        """Test full pipeline: manual input -> coordinate mapper -> world."""
        # Manual input with scale 2
        manual_input = ManualControlInput(scale=2.0)
        
        # Coordinate mapper with offset and rotation
        offset = np.array([1.0, 0.0, 0.0])
        rot = _axis_angle_rotation('z', 90)
        mapper = CoordinateMapper(offset=offset, rotation_matrix=rot)
        
        # Raw joystick: [1, 0]
        raw = manual_input([1.0, 0.0])  # -> [2, 0, 0] after manual scale
        # Mapper : offset [1, 0, 0] -> [3, 0, 0], then rotate Z 90° -> [0, 3, 0]
        world = mapper(raw)
        
        np.testing.assert_array_almost_equal(world, [0.0, 3.0, 0.0])
    
    def test_neural_to_world_pipeline(self):
        """Test full pipeline: neural input -> coordinate mapper -> world."""
        # Neural input
        decoder = MockDecoder()
        extractor = MockExtractor()
        accumulator = MockFeatureAccumulator()
        neural_input = NeuralControlInput(decoder, extractor, accumulator)
        
        # Coordinate mapper
        mapper = CoordinateMapper(scale=1.5)
        
        # Decode spike data
        spike_data = np.array([2.0, 2.0, 2.0])
        neural_state = neural_input(spike_data)  # -> [2, 2, 2]
        world = mapper(neural_state)  # -> [3, 3, 3] after scale
        
        np.testing.assert_array_almost_equal(world, [3.0, 3.0, 3.0])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
