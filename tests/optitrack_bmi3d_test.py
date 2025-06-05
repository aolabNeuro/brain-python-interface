#!/usr/bin/env python3
"""
Test suite for BMI3D Optitrack integration with NatNet SDK 4.3
Tests the updated optitrack features and kinematic system compatibility
"""
from riglib.optitrack_client_update_natnet.updated_kinematic_sdk43 import KinematicSystem, SimulatedClient
from features.optitrack_bmi_features_updated import OptitrackBMI, Optitrack, OptitrackSimulate
import unittest
import numpy as np
import time
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import threading

# Mock the BMI3D modules that might not be available
sys.modules['riglib'] = Mock()
sys.modules['riglib.experiment'] = Mock()
sys.modules['riglib.experiment.traits'] = Mock()
sys.modules['riglib.optitrack_client_update_natnet'] = Mock()
sys.modules['riglib.stereo_opengl'] = Mock()
sys.modules['riglib.stereo_opengl.primitives'] = Mock()
sys.modules['riglib.source'] = Mock()
sys.modules['config'] = Mock()
sys.modules['config.rig_defaults'] = Mock()
sys.modules['features'] = Mock()
sys.modules['features.neural_sys_features'] = Mock()
sys.modules['riglib.sink'] = Mock()

# Mock traits classes
class MockTrait:
    def __init__(self, default=None, desc=""):
        self.default = default
        self.desc = desc

class MockTraits:
    OptionsList = lambda x: MockTrait()
    Int = lambda x, desc="": MockTrait(x, desc)
    Float = lambda x, desc="": MockTrait(x, desc)
    Array = lambda value=None, desc="": MockTrait(value, desc)
    String = lambda x, desc="": MockTrait(x, desc)
    Bool = lambda x, desc="": MockTrait(x, desc)

class MockHasTraits:
    def __init__(self, *args, **kwargs):
        pass
    def init(self):
        pass

# Set up mock modules
sys.modules['riglib.experiment'].traits = MockTraits()
sys.modules['riglib.experiment.traits'].HasTraits = MockHasTraits
sys.modules['config.rig_defaults'].optitrack = {
    'scale': 1.0,
    'offset': np.array([0, 0, 0]),
    'save_path': '/tmp/',
    'sync_dch': 1,
    'server_address': '127.0.0.1',
    'client_address': '127.0.0.1',
    'use_multicast': True
}

# Mock neural features
class MockCorticalBMI:
    def __init__(self, *args, **kwargs):
        self.cortical_channels = np.array([])
        
    def init(self):
        pass
        
    def cleanup(self, database, saveid, **kwargs):
        return True

class MockCorticalData:
    pass

sys.modules['features.neural_sys_features'].CorticalBMI = MockCorticalBMI
sys.modules['features.neural_sys_features'].CorticalData = MockCorticalData

# Mock source classes
class MockDataSourceSystem:
    def __init__(self, *args, **kwargs):
        pass
        
class MockMultiChanDataSource:
    def __init__(self, *args, **kwargs):
        pass
        
class MockDataSource:
    def __init__(self, system):
        self.system = system
        self.running = False
        
    def start(self):
        self.running = True
        
    def stop(self):
        self.running = False
        
    def join(self):
        pass
        
    def get(self):
        # Return mock kinematic data
        return np.array([(np.random.random(9), time.time())], 
                       dtype=[('data', np.float64, (9,)), ('timestamp', np.float64)])

sys.modules['riglib.source'].DataSourceSystem = MockDataSourceSystem
sys.modules['riglib.source'].MultiChanDataSource = MockMultiChanDataSource
sys.modules['riglib.source'].DataSource = MockDataSource

# Mock sink manager
class MockSinkManager:
    @staticmethod
    def get_instance():
        return MockSinkManager()
        
    def register(self, source):
        pass

sys.modules['riglib.sink'].SinkManager = MockSinkManager

# Now import the modules to test
try:
    # Import the kinematic system
    exec(open('updated_kinematic_sdk43.py').read())
    from riglib.optitrack_client_update_natnet.updated_kinematic_sdk43 import KinematicSystem, SimulatedClient
    
    # Import the optitrack features  
    exec(open('optitrack_bmi_features_updated.py').read())
    from features.optitrack_bmi_features_updated import OptitrackBMI, Optitrack, OptitrackSimulate
    
except Exception as e:
    print(f"Warning: Could not import modules directly: {e}")
    print("Proceeding with embedded code testing...")


class TestKinematicSystem(unittest.TestCase):
    """Test the KinematicSystem class"""
    
    def setUp(self):
        """Set up test environment"""
        self.system = KinematicSystem(
            n_features=2,
            feature_type="rigid body",
            server_address="127.0.0.1",
            client_address="127.0.0.1"
        )
        
    def test_initialization(self):
        """Test KinematicSystem initialization"""
        self.assertEqual(self.system.n_features, 2)
        self.assertEqual(self.system.feature_type, "rigid body")
        self.assertEqual(self.system.n_kinematic_channels, 18)  # 2 features * 9 channels each
        
    def test_data_dimensions(self):
        """Test data array dimensions are correct"""
        self.assertEqual(self.system.get_data_len(), 18)
        channels = self.system.get_channels()
        self.assertEqual(len(channels), 18)
        self.assertEqual(channels[0], 1)
        self.assertEqual(channels[-1], 18)
        
    def test_dtype_structure(self):
        """Test the numpy dtype structure matches EMG format"""
        expected_dtype = np.dtype([
            ('data', np.float64, (18,)),
            ('timestamp', np.float64)
        ])
        self.assertEqual(self.system.dtype, expected_dtype)
        
    def test_simulated_client(self):
        """Test simulated client functionality"""
        sim_client = SimulatedClient()
        self.assertTrue(sim_client.connected())
        self.assertTrue(sim_client.run('d'))
        
        # Test frame simulation
        self.assertTrue(sim_client.simulate_frame())
        
    def test_nan_handling(self):
        """Test NaN data handling"""
        # Create test data with NaN values
        test_data = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] * 2)
        
        # Test NaN replacement
        cleaned_data = self.system._handle_nan_data(test_data)
        self.assertFalse(np.isnan(cleaned_data).any())
        
    def test_position_extraction(self):
        """Test position data extraction from frame data"""
        # Create mock frame data
        mock_rb = Mock()
        mock_rb.tracking_valid = True
        mock_rb.pos_x = 1.0
        mock_rb.pos_y = 2.0
        mock_rb.pos_z = 3.0
        
        mock_rb_data = Mock()
        mock_rb_data.rigid_body_list = [mock_rb]
        
        mock_frame = Mock()
        mock_frame.rigid_body_data = mock_rb_data
        
        self.system.frame_data = mock_frame
        positions = self.system._extract_positions()
        
        self.assertIsNotNone(positions)
        self.assertEqual(positions.shape, (2, 3))  # 2 features, 3 coordinates
        

class TestOptitrackBMI(unittest.TestCase):
    """Test the OptitrackBMI class"""
    
    def setUp(self):
        """Set up test environment"""
        with patch('features.optitrack_bmi_features_updated.optitrack'):
            self.bmi = OptitrackBMI()
            self.bmi.n_features = 1
            self.bmi.optitrack_feature = "rigid body"
            self.bmi.kinematic_channels = np.array([])
            
    def test_initialization(self):
        """Test OptitrackBMI initialization"""
        self.assertEqual(self.bmi.n_features, 1)
        self.assertEqual(len(self.bmi.cortical_channels), 9)  # 1 feature * 9 channels
        
    def test_channel_assignment(self):
        """Test channel assignment for decoder compatibility"""
        # Should automatically assign position channels if none specified
        expected_pos_channels = np.array([1, 2, 3])  # x, y, z of first feature
        
        # Reset and reinitialize to test auto-assignment
        self.bmi.kinematic_channels = np.array([])
        self.bmi.__init__()
        
        np.testing.assert_array_equal(self.bmi.kinematic_channels, expected_pos_channels)
        
    def test_multi_feature_channels(self):
        """Test channel assignment for multiple features"""
        self.bmi.n_features = 2
        self.bmi.kinematic_channels = np.array([])
        self.bmi.__init__()
        
        # Should have 18 total channels (2 features * 9 each)
        self.assertEqual(len(self.bmi.cortical_channels), 18)
        
        # Position channels should be [1,2,3,10,11,12] for 2 features
        expected_pos_channels = np.array([1, 2, 3, 10, 11, 12])
        np.testing.assert_array_equal(self.bmi.kinematic_channels, expected_pos_channels)


class TestOptitrack(unittest.TestCase):
    """Test the base Optitrack class"""
    
    def setUp(self):
        """Set up test environment"""
        self.optitrack = Optitrack()
        self.optitrack.n_features = 1
        self.optitrack.optitrack_feature = "rigid body"
        self.optitrack.smooth_features = 1
        
    def test_manual_position_extraction(self):
        """Test manual position extraction for cursor control"""
        # Mock the data source
        mock_data = np.array([(np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03]), time.time())],
                           dtype=[('data', np.float64, (9,)), ('timestamp', np.float64)])
        
        self.optitrack.motiondata = Mock()
        self.optitrack.motiondata.get.return_value = mock_data
        
        position = self.optitrack._get_manual_position()
        
        # Should return position in centimeters (scaled by 100)
        expected_position = np.array([100.0, 200.0, 300.0])  # meters to cm
        np.testing.assert_array_almost_equal(position, expected_position)
        
    def test_nan_position_handling(self):
        """Test NaN handling in position extraction"""
        # Mock data with NaN values
        mock_data = np.array([(np.array([np.nan, 2.0, 3.0, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03]), time.time())],
                           dtype=[('data', np.float64, (9,)), ('timestamp', np.float64)])
        
        self.optitrack.motiondata = Mock()
        self.optitrack.motiondata.get.return_value = mock_data
        
        position = self.optitrack._get_manual_position()
        self.assertIsNone(position)  # Should return None for invalid data
        
    def test_multi_feature_averaging(self):
        """Test averaging multiple features for cursor position"""
        self.optitrack.n_features = 2
        self.optitrack.smooth_features = 2
        
        # Mock data for 2 features
        kinematic_data = np.array([
            1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03,  # Feature 1
            2.0, 3.0, 4.0, 0.2, 0.3, 0.4, 0.02, 0.03, 0.04   # Feature 2
        ])
        mock_data = np.array([(kinematic_data, time.time())],
                           dtype=[('data', np.float64, (18,)), ('timestamp', np.float64)])
        
        self.optitrack.motiondata = Mock()
        self.optitrack.motiondata.get.return_value = mock_data
        
        position = self.optitrack._get_manual_position()
        
        # Should average the two feature positions and convert to cm
        expected_position = np.array([150.0, 250.0, 350.0])  # Average of [1,2,3] and [2,3,4] * 100
        np.testing.assert_array_almost_equal(position, expected_position)


class TestDataFormatCompatibility(unittest.TestCase):
    """Test data format compatibility with BMI infrastructure"""
    
    def test_emg_data_format_compatibility(self):
        """Test that kinematic data matches EMG data format"""
        system = KinematicSystem(n_features=1)
        
        # Simulate getting data
        system.frame_data = self._create_mock_frame_data()
        data = system.get()
        
        # Check structure matches expected EMG format
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(len(data), 1)  # Single sample
        self.assertIn('data', data.dtype.names)
        self.assertIn('timestamp', data.dtype.names)
        
        # Check data dimensions
        kinematic_data = data[0]['data']
        self.assertEqual(len(kinematic_data), 9)  # 1 feature * 9 channels
        self.assertIsInstance(data[0]['timestamp'], float)
        
    def test_decoder_channel_mapping(self):
        """Test that channels map correctly for decoder input"""
        bmi = OptitrackBMI()
        bmi.n_features = 2
        bmi.kinematic_channels = np.array([])
        bmi.__init__()
        
        # Check that position channels are correctly identified
        pos_channels = bmi.kinematic_channels
        self.assertEqual(len(pos_channels), 6)  # 2 features * 3 position channels each
        
        # Verify channel indices are correct
        self.assertIn(1, pos_channels)  # x of feature 1
        self.assertIn(2, pos_channels)  # y of feature 1
        self.assertIn(3, pos_channels)  # z of feature 1
        self.assertIn(10, pos_channels) # x of feature 2
        self.assertIn(11, pos_channels) # y of feature 2
        self.assertIn(12, pos_channels) # z of feature 2
        
    def _create_mock_frame_data(self):
        """Create mock frame data for testing"""
        mock_rb = Mock()
        mock_rb.tracking_valid = True
        mock_rb.pos_x = 1.0
        mock_rb.pos_y = 2.0
        mock_rb.pos_z = 3.0
        
        mock_rb_data = Mock()
        mock_rb_data.rigid_body_list = [mock_rb]
        
        mock_frame = Mock()
        mock_frame.rigid_body_data = mock_rb_data
        
        return mock_frame


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete integration scenarios"""
    
    def test_bmi_experiment_workflow(self):
        """Test complete BMI experiment workflow"""
        # Initialize BMI system
        bmi = OptitrackBMI()
        bmi.n_features = 1
        bmi.optitrack_feature = "rigid body"
        
        # Mock the neural system
        mock_neural_sys = Mock()
        mock_neural_sys.optitrack_status = 'simulated'
        bmi.neural_sys = mock_neural_sys
        
        # Test initialization
        with patch('features.optitrack_bmi_features_updated.optitrack'):
            bmi.init()
            
        self.assertEqual(bmi.optitrack_status, 'simulated')
        
    def test_cursor_control_pipeline(self):
        """Test the cursor control data pipeline"""
        optitrack = Optitrack()
        optitrack.n_features = 1
        optitrack.optitrack_feature = "rigid body"
        
        # Mock kinematic system with simulated status
        mock_system = Mock()
        mock_system.optitrack_status = 'simulated'
        optitrack.kinematic_system = mock_system
        optitrack.optitrack_status = 'simulated'
        
        # Mock data source
        mock_data = np.array([(np.array([0.5, 0.6, 0.7, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03]), time.time())],
                           dtype=[('data', np.float64, (9,)), ('timestamp', np.float64)])
        
        optitrack.motiondata = Mock()
        optitrack.motiondata.get.return_value = mock_data
        
        # Test position extraction
        position = optitrack._get_manual_position()
        self.assertIsNotNone(position)
        self.assertEqual(len(position), 3)
        
        # Position should be in centimeters
        self.assertTrue(all(abs(p) > 10 for p in position))  # Should be scaled to cm
        
    def test_error_handling(self):
        """Test error handling and fallback mechanisms"""
        # Test connection failure fallback
        system = KinematicSystem()
        system._init_client()  # Should fall back to simulated
        
        self.assertIn(system.optitrack_status, ['simulated', 'disconnected', 'NatNet SDK 4.3 not found'])
        
    def test_data_continuity(self):
        """Test data continuity and buffering"""
        system = KinematicSystem(n_features=1)
        
        # Test multiple data retrievals
        for i in range(5):
            system.frame_data = self._create_mock_frame_data()
            data = system.get()
            self.assertIsNotNone(data)
            self.assertEqual(len(data[0]['data']), 9)
            
    def _create_mock_frame_data(self):
        """Create mock frame data for testing"""
        mock_rb = Mock()
        mock_rb.tracking_valid = True
        mock_rb.pos_x = np.random.uniform(-1, 1)
        mock_rb.pos_y = np.random.uniform(-1, 1)
        mock_rb.pos_z = np.random.uniform(-1, 1)
        
        mock_rb_data = Mock()
        mock_rb_data.rigid_body_list = [mock_rb]
        
        mock_frame = Mock()
        mock_frame.rigid_body_data = mock_rb_data
        
        return mock_frame


class TestPerformance(unittest.TestCase):
    """Test performance and timing characteristics"""
    
    def test_data_rate_performance(self):
        """Test that data can be retrieved at expected rates"""
        system = KinematicSystem(n_features=1)
        system.update_freq = 240  # 240 Hz
        
        # Test data retrieval timing
        start_time = time.time()
        for i in range(10):
            system.frame_data = self._create_mock_frame_data()
            data = system.get()
            self.assertIsNotNone(data)
            
        elapsed_time = time.time() - start_time
        
        # Should complete 10 retrievals quickly
        self.assertLess(elapsed_time, 1.0)  # Less than 1 second for 10 samples
        
    def test_memory_usage(self):
        """Test memory usage with data buffering"""
        system = KinematicSystem(n_features=2)
        
        # Fill position history
        for i in range(10):
            positions = np.random.random((2, 3))
            system.position_history.append(positions)
            if len(system.position_history) > system.max_history:
                system.position_history.pop(0)
                
        # History should be limited
        self.assertLessEqual(len(system.position_history), system.max_history)
        
    def _create_mock_frame_data(self):
        """Create mock frame data for testing"""
        mock_rb = Mock()
        mock_rb.tracking_valid = True
        mock_rb.pos_x = np.random.uniform(-1, 1)
        mock_rb.pos_y = np.random.uniform(-1, 1) 
        mock_rb.pos_z = np.random.uniform(-1, 1)
        
        mock_rb_data = Mock()
        mock_rb_data.rigid_body_list = [mock_rb]
        
        mock_frame = Mock()
        mock_frame.rigid_body_data = mock_rb_data
        
        return mock_frame


def run_integration_tests():
    """Run all integration tests"""
    print("="*60)
    print("BMI3D Optitrack Integration Test Suite")
    print("="*60)
    
    # Create test suite
    test_classes = [
        TestKinematicSystem,
        TestOptitrackBMI, 
        TestOptitrack,
        TestDataFormatCompatibility,
        TestIntegrationScenarios,
        TestPerformance
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
            
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
            
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)