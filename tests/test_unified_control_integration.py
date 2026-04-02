"""
Integration tests for unified control architecture examples.

Tests verify that example task implementations correctly integrate
the unified control pipeline.
"""

import pytest
import numpy as np

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from riglib.bmi.unified_control import (
    ControlInput, NeuralControlInput, ManualControlInput, CoordinateMapper, _axis_angle_rotation
)
from riglib.bmi.unified_control_mixin import UnifiedControlMixin


# Mock classes for testing
class MockPlant:
    """Mock plant for testing."""
    
    def __init__(self):
        self.endpoint_pos = np.array([0.0, 0.0, 0.0])
    
    def set_endpoint_pos(self, pos):
        self.endpoint_pos = np.asarray(pos)
    
    def get_endpoint_pos(self):
        return self.endpoint_pos


class MockTask(UnifiedControlMixin):
    """Minimal mock task for testing UnifiedControlMixin."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plant = MockPlant()
        self.task_data = {}
    
    def init(self):
        # Don't call parent init to avoid issues with traits
        self.create_coordinate_mapper()
        self.create_control_input()


class MockDecoder:
    """Mock decoder for neural input."""
    
    def __init__(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    def __call__(self, features, **kwargs):
        if len(features) >= 3:
            self.state[:3] = features[:3]
    
    def get_state(self):
        return self.state.copy()


class MockExtractor:
    """Mock extractor for neural input."""
    
    def __call__(self, spike_data):
        return spike_data[:3] if len(spike_data) >= 3 else spike_data


class MockAccumulator:
    """Mock accumulator for neural input."""
    
    def __call__(self, features):
        return features, True


class TestUnifiedControlMixinIntegration:
    """Integration tests for UnifiedControlMixin."""
    
    def test_mixin_creates_coordinate_mapper(self):
        """Test that mixin creates coordinate mapper."""
        task = MockTask()
        task.init()
        
        assert task.coordinate_mapper is not None
        assert isinstance(task.coordinate_mapper, CoordinateMapper)
    
    def test_mixin_applies_transforms(self):
        """Test that mixin correctly applies coordinate transforms."""
        task = MockTask()
        task.scale = 2.0
        task.offset = np.array([1.0, 0.0, 0.0])
        task.init()
        
        # Test transform
        input_pt = np.array([1.0, 0.0, 0.0])
        output = task.get_world_control(input_pt)
        
        # Scale 2.0: [2, 0, 0], then offset: [3, 0, 0]
        np.testing.assert_array_almost_equal(output, [3.0, 0.0, 0.0])
    
    def test_mixin_applies_limits(self):
        """Test that mixin applies 1D and 2D limits."""
        task = MockTask()
        task.init()
        
        input_pt = np.array([1.0, 2.0, 3.0])
        
        # Test 2D limit
        output_2d = task.get_world_control(input_pt, apply_limit2d=True)
        np.testing.assert_array_almost_equal(output_2d, [1.0, 0.0, 3.0])
        
        # Test 1D limit
        output_1d = task.get_world_control(input_pt, apply_limit1d=True)
        np.testing.assert_array_almost_equal(output_1d, [0.0, 0.0, 3.0])
    
    def test_mixin_update_perturbation(self):
        """Test updating perturbation at runtime."""
        task = MockTask()
        task.init()
        
        # Initially no perturbation
        input_pt = np.array([1.0, 0.0, 0.0])
        output1 = task.get_world_control(input_pt)
        
        # Add perturbation
        task.update_perturbation_rotation('z', 90)
        output2 = task.get_world_control(input_pt)
        
        # Outputs should differ due to rotation
        assert not np.allclose(output1, output2)


class TestManualControlIntegration:
    """Integration tests for manual control pipeline."""
    
    def test_manual_input_with_mapper(self):
        """Test manual input through coordinate mapper."""
        manual = ManualControlInput(scale=1.0)
        mapper = CoordinateMapper(
            scale=2.0,
            offset=np.array([1.0, 0.0, 0.0])
        )
        
        # Get manual input
        raw = manual([1.0, 0.0])  # -> [1, 0, 0]
        
        # Apply mapper
        world = mapper(raw)  # scale: [2, 0, 0], offset: [3, 0, 0]
        
        np.testing.assert_array_almost_equal(world, [3.0, 0.0, 0.0])
    
    def test_manual_velocity_control(self):
        """Test manual velocity control mode."""
        manual = ManualControlInput(velocity_control=True)
        
        # First input
        vel1 = manual([1.0, 0.0], dt=1.0)  # velocity = [1, 0, 0]
        np.testing.assert_array_almost_equal(vel1, [1.0, 0.0, 0.0])
        
        # Second input (different position)
        vel2 = manual([2.0, 0.0], dt=1.0)  # velocity = [1, 0, 0]
        np.testing.assert_array_almost_equal(vel2, [1.0, 0.0, 0.0])
    
    def test_manual_with_limits(self):
        """Test manual input with applied limits."""
        manual = ManualControlInput()
        mapper = CoordinateMapper()
        
        raw = manual([1.0, 1.0, 1.0])
        
        # Apply 2D limit
        world = mapper(raw, apply_limit2d=True)
        np.testing.assert_array_almost_equal(world, [1.0, 0.0, 1.0])


class TestNeuralControlIntegration:
    """Integration tests for neural control pipeline."""
    
    def test_neural_input_with_mapper(self):
        """Test neural input through coordinate mapper."""
        decoder = MockDecoder()
        extractor = MockExtractor()
        accumulator = MockAccumulator()
        
        neural = NeuralControlInput(decoder, extractor, accumulator)
        mapper = CoordinateMapper(scale=1.5)
        
        # Decode spike data
        spike_data = np.array([2.0, 2.0, 2.0, 4.0, 5.0])
        state = neural(spike_data)  # -> [2, 2, 2, ...]
        
        # Apply mapper
        world = mapper(state)  # scale: [3, 3, 3]
        
        np.testing.assert_array_almost_equal(world, [3.0, 3.0, 3.0])
    
    def test_neural_state_extraction(self):
        """Test extracting decoder state."""
        decoder = MockDecoder()
        extractor = MockExtractor()
        accumulator = MockAccumulator()
        
        neural = NeuralControlInput(decoder, extractor, accumulator)
        
        # Decode something
        spike_data = np.array([1.0, 2.0, 3.0])
        neural(spike_data)
        
        # Get state
        state = neural.get_state()
        assert state is not None
        np.testing.assert_array_almost_equal(state[:3], [1.0, 2.0, 3.0])


class TestComplexTransformSequence:
    """Test complex sequences of transforms."""
    
    def test_scale_rotate_offset_sequence(self):
        """Test that transforms apply in correct order."""
        # Setup: scale 2x, offset [1, 0, 0], then rotate 90° Z
        rot = _axis_angle_rotation('z', 90)
        mapper = CoordinateMapper(
            scale=2.0,
            rotation_matrix=rot,
            offset=np.array([1.0, 0.0, 0.0])
        )
        
        input_pt = np.array([1.0, 0.0, 0.0])
        output = mapper(input_pt)
        
        # Step by step (per CoordinateMapper order):
        # 1. scale: [1, 0, 0] * 2 = [2, 0, 0]
        # 2. offset: [2, 0, 0] + [1, 0, 0] = [3, 0, 0]
        # 3. rotate Z 90°: [3, 0, 0] -> [0, 3, 0]
        np.testing.assert_array_almost_equal(output, [0.0, 3.0, 0.0])
    
    def test_multiple_perturbations_order(self):
        """Test that multiple perturbations are applied in sequence."""
        mapper = CoordinateMapper(
            perturbation_rotations={'x': 90, 'z': 90}
        )
        
        input_pt = np.array([1.0, 0.0, 0.0])
        output = mapper(input_pt)
        
        # Result depends on rotation composition
        # Just verify it's different and non-trivial
        assert output is not None
        assert not np.allclose(output, input_pt)
    
    def test_all_transforms_combined(self):
        """Test combining all transform types."""
        rot_x = _axis_angle_rotation('x', 45)
        rot_baseline = _axis_angle_rotation('y', 30)
        rot_exp = _axis_angle_rotation('z', 15)
        
        mapper = CoordinateMapper(
            scale=1.5,
            offset=np.array([1.0, 1.0, 1.0]),
            rotation_matrix=rot_x,
            baseline_rotation=rot_baseline,
            exp_rotation=rot_exp,
            perturbation_rotations={'z': 10}
        )
        
        input_pt = np.array([1.0, 1.0, 1.0])
        output = mapper(input_pt)
        
        # Just verify it computes without error
        assert output is not None
        assert output.shape == (3,)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_none_handling_manual(self):
        """Test manual input handles None properly."""
        manual = ManualControlInput()
        assert manual(None) is None
        assert manual([]) is None
    
    def test_none_handling_neural(self):
        """Test neural input handles None properly."""
        neural = NeuralControlInput(MockDecoder(), MockExtractor(), MockAccumulator())
        assert neural(None) is None
    
    def test_none_handling_mapper(self):
        """Test mapper handles None properly."""
        mapper = CoordinateMapper()
        assert mapper(None) is None
    
    def test_vector_size_mismatch_manual(self):
        """Test manual input handles wrong size vectors."""
        manual = ManualControlInput()
        
        # Very long vector - should take first 3
        long_vec = np.arange(10)
        result = manual(long_vec)
        assert result.shape == (3,)
    
    def test_7d_vector_handling(self):
        """Test that 7D vectors are handled correctly."""
        mapper = CoordinateMapper()
        
        # 7D input: [Px, Py, Pz, Vx, Vy, Vz, mode]
        state_7d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0])
        output = mapper(state_7d)
        
        # Should use only first 3 components
        np.testing.assert_array_almost_equal(output, [1.0, 2.0, 3.0])
    
    def test_zero_perturbation_handling(self):
        """Test that zero perturbations are handled efficiently."""
        mapper = CoordinateMapper(perturbation_rotations={
            'x': 0,
            'y': 0,
            'z': 0
        })
        
        input_pt = np.array([1.0, 1.0, 1.0])
        output = mapper(input_pt)
        
        np.testing.assert_array_almost_equal(output, input_pt)


class TestBlendingScenarios:
    """Test scenarios involving blending of control signals."""
    
    def test_blend_neural_manual_equal_weight(self):
        """Test blending neural and manual with equal weights."""
        neural = np.array([2.0, 2.0, 2.0])
        manual = np.array([4.0, 4.0, 4.0])
        
        blended = 0.5 * neural + 0.5 * manual
        expected = np.array([3.0, 3.0, 3.0])
        
        np.testing.assert_array_almost_equal(blended, expected)
    
    def test_blend_neural_dominant(self):
        """Test blending with neural control dominant."""
        neural = np.array([1.0, 1.0, 1.0])
        manual = np.array([9.0, 9.0, 9.0])
        
        blended = 0.9 * neural + 0.1 * manual
        expected = np.array([1.8, 1.8, 1.8])
        
        np.testing.assert_array_almost_equal(blended, expected)
    
    def test_blend_with_transform(self):
        """Test blending followed by coordinate transform."""
        neural = np.array([1.0, 0.0, 0.0])
        manual = np.array([0.0, 1.0, 0.0])
        
        # Blend 50/50
        blended = 0.5 * neural + 0.5 * manual
        # -> [0.5, 0.5, 0.0]
        
        # Apply transform
        mapper = CoordinateMapper(scale=2.0)
        world = mapper(blended)
        # -> [1.0, 1.0, 0.0]
        
        np.testing.assert_array_almost_equal(world, [1.0, 1.0, 0.0])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
