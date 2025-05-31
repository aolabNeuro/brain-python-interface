'''
Unit tests for OptiTrack Quattro Stream compatibility
Tests both simulation and real client integration
'''
import unittest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.append('/home/aolab/NatNet_SDK_4.2_ubuntu/samples/PythonClient')
import os

# Add the parent directory to the path for imports  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the classes we're testing
from features.optitrack_quattro_compatible import (
    OptiTrackQuattroStream, 
    SimulatedOptiTrackClient, 
    create_optitrack_quattro_source
)


class TestOptiTrackQuattroStream(unittest.TestCase):
    """Test cases for OptiTrackQuattroStream class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_client = Mock()
        self.n_rigid_bodies = 2
        self.channels_per_body = 3
        
    def test_initialization(self):
        """Test proper initialization of OptiTrackQuattroStream"""
        stream = OptiTrackQuattroStream(
            self.mock_client, 
            self.n_rigid_bodies, 
            self.channels_per_body
        )
        
        self.assertEqual(stream.n_rigid_bodies, self.n_rigid_bodies)
        self.assertEqual(stream.channels_per_body, self.channels_per_body)
        self.assertEqual(stream.n_channels, self.n_rigid_bodies * self.channels_per_body)
        self.assertEqual(stream.latest_data.shape, (1, 6))  # 2 bodies * 3 channels
        np.testing.assert_array_equal(stream.latest_data, np.zeros((1, 6)))
    
    def test_callback_setup_new_client(self):
        """Test callback setup for new NatNet client"""
        mock_client = Mock()
        mock_client.rigid_body_listener = None
        
        stream = OptiTrackQuattroStream(mock_client, 1, 3)
        
        # Should have set the rigid_body_listener
        self.assertIsNotNone(mock_client.rigid_body_listener)
        self.assertEqual(mock_client.rigid_body_listener, stream._rigid_body_callback)
    
    def test_callback_setup_legacy_client(self):
        """Test callback setup for legacy client"""
        mock_client = Mock()
        mock_client.set_callback = Mock()
        # Remove rigid_body_listener to simulate legacy client
        if hasattr(mock_client, 'rigid_body_listener'):
            delattr(mock_client, 'rigid_body_listener')
        
        stream = OptiTrackQuattroStream(mock_client, 1, 3)
        
        # Should have called set_callback
        mock_client.set_callback.assert_called_once_with(stream._legacy_callback)
    
    def test_rigid_body_callback(self):
        """Test rigid body callback for new NatNet client"""
        stream = OptiTrackQuattroStream(Mock(), 2, 3)
        
        # Test callback with first rigid body
        position = [0.1, 0.2, 0.3]  # meters
        rotation = [0, 0, 0, 1]     # quaternion
        stream._rigid_body_callback(0, position, rotation)
        
        # Should convert to cm and store in correct position
        expected_pos_cm = np.array(position) * 100
        np.testing.assert_array_almost_equal(
            stream.latest_data[0, 0:3], 
            expected_pos_cm
        )
        
        # Test second rigid body
        position2 = [0.4, 0.5, 0.6]
        stream._rigid_body_callback(1, position2, rotation)
        
        expected_pos2_cm = np.array(position2) * 100
        np.testing.assert_array_almost_equal(
            stream.latest_data[0, 3:6], 
            expected_pos2_cm
        )
    
    def test_legacy_callback(self):
        """Test legacy callback for old-style clients"""
        stream = OptiTrackQuattroStream(Mock(), 2, 3)
        
        # Create mock rigid bodies
        rb1 = Mock()
        rb1.position = np.array([10, 20, 30])  # cm
        rb1._already_converted_to_cm = True
        
        rb2 = Mock()
        rb2.position = np.array([0.04, 0.05, 0.06])  # meters
        rb2._already_converted_to_cm = False
        
        rigid_bodies = [rb1, rb2]
        stream._legacy_callback(rigid_bodies, [], [], [])
        
        # First body should be stored as-is (already in cm)
        np.testing.assert_array_almost_equal(
            stream.latest_data[0, 0:3], 
            [10, 20, 30]
        )
        
        # Second body should be converted to cm
        np.testing.assert_array_almost_equal(
            stream.latest_data[0, 3:6], 
            [4, 5, 6]
        )
    
    def test_get_method(self):
        """Test get method returns correct format"""
        mock_client = Mock()
        mock_client.run_once = Mock()
        
        stream = OptiTrackQuattroStream(mock_client, 1, 3)
        
        # Set some test data
        test_data = np.array([[1, 2, 3]])
        stream.latest_data = test_data
        
        result = stream.get()
        
        # Should call run_once for legacy clients
        mock_client.run_once.assert_called_once_with(timeout=0.01)
        
        # Should return copy of data
        np.testing.assert_array_equal(result, test_data)
        self.assertIsNot(result, stream.latest_data)  # Should be a copy
        
        # Should have correct shape
        self.assertEqual(result.shape, (1, 3))
    
    def test_start_stop_methods(self):
        """Test start and stop methods"""
        # Test with new client
        mock_client = Mock()
        mock_client.run = Mock()
        
        stream = OptiTrackQuattroStream(mock_client, 1, 3)
        
        with patch('threading.Thread') as mock_thread:
            stream.start()
            
            # Should create and start thread
            mock_thread.assert_called_once()
            mock_thread.return_value.start.assert_called_once()
        
        # Test stop
        mock_client.stop_threads = False
        stream.stop()
        self.assertTrue(mock_client.stop_threads)
    
    def test_thread_safety(self):
        """Test thread safety of data access"""
        stream = OptiTrackQuattroStream(Mock(), 1, 3)
        
        # Function to modify data from another thread
        def modify_data():
            for i in range(100):
                stream._rigid_body_callback(0, [i/100, i/100, i/100], [0, 0, 0, 1])
                time.sleep(0.001)
        
        # Start modifying data in background
        modifier_thread = threading.Thread(target=modify_data)
        modifier_thread.start()
        
        # Read data from main thread
        for _ in range(50):
            data = stream.get()
            # Should always get valid data shape
            self.assertEqual(data.shape, (1, 3))
            time.sleep(0.002)
        
        modifier_thread.join()


class TestSimulatedOptiTrackClient(unittest.TestCase):
    """Test cases for SimulatedOptiTrackClient"""
    
    def test_initialization(self):
        """Test client initialization"""
        client = SimulatedOptiTrackClient(n_bodies=3)
        self.assertEqual(client.n_bodies, 3)
        self.assertIsNone(client.callback)
    
    def test_callback_setting(self):
        """Test callback setting"""
        client = SimulatedOptiTrackClient()
        callback_func = Mock()
        
        client.set_callback(callback_func)
        self.assertEqual(client.callback, callback_func)
    
    def test_run_once_generates_data(self):
        """Test that run_once generates motion data"""
        client = SimulatedOptiTrackClient(n_bodies=2)
        callback_mock = Mock()
        client.set_callback(callback_mock)
        
        client.run_once()
        
        # Should have called callback with rigid body data
        callback_mock.assert_called_once()
        args = callback_mock.call_args[0]
        rigid_bodies, skeletons, markers, timing = args
        
        self.assertEqual(len(rigid_bodies), 2)
        self.assertEqual(len(skeletons), 0)
        self.assertEqual(len(markers), 0)
        
        # Check rigid body data
        for rb in rigid_bodies:
            self.assertTrue(hasattr(rb, 'position'))
            self.assertTrue(hasattr(rb, '_already_converted_to_cm'))
            self.assertTrue(rb._already_converted_to_cm)
            self.assertEqual(len(rb.position), 3)
    
    def test_motion_generation(self):
        """Test that motion data changes over time"""
        client = SimulatedOptiTrackClient(n_bodies=1)
        positions = []
        
        def capture_position(rigid_bodies, *args):
            positions.append(rigid_bodies[0].position.copy())
        
        client.set_callback(capture_position)
        
        # Collect several positions
        for _ in range(5):
            client.run_once()
            time.sleep(0.01)  # Small delay to advance time
        
        # Positions should be different (motion simulation)
        self.assertGreater(len(positions), 1)
        
        # Check that positions actually change
        first_pos = positions[0]
        last_pos = positions[-1]
        self.assertFalse(np.allclose(first_pos, last_pos, atol=0.001))


class TestFactoryFunction(unittest.TestCase):
    """Test cases for create_optitrack_quattro_source factory function"""
    
    @patch('optitrack_quattro_compatible.create_client')
    def test_factory_with_natnet_client(self, mock_create_client):
        """Test factory function with real NatNet client"""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        source = create_optitrack_quattro_source(
            n_rigid_bodies=2,
            client_type="natnet",
            server_ip="192.168.1.100"
        )
        
        # Should have called create_client with correct parameters
        mock_create_client.assert_called_once_with(
            "natnet", 
            server_ip="192.168.1.100"
        )
        
        # Should return OptiTrackQuattroStream instance
        self.assertIsInstance(source, OptiTrackQuattroStream)
        self.assertEqual(source.n_rigid_bodies, 2)
    
    @patch('optitrack_quattro_stream.create_client')
    def test_factory_with_simulation_client(self, mock_create_client):
        """Test factory function with simulation client"""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        source = create_optitrack_quattro_source(
            n_rigid_bodies=1,
            client_type="simulation"
        )
        
        mock_create_client.assert_called_once_with("simulation")
        self.assertIsInstance(source, OptiTrackQuattroStream)
    
    def test_factory_fallback_to_simulation(self):
        """Test factory fallback when import fails"""
        with patch('optitrack_quattro_stream.create_client', side_effect=ImportError):
            source = create_optitrack_quattro_source(n_rigid_bodies=1)
            
            # Should fallback and still return valid source
            self.assertIsInstance(source, OptiTrackQuattroStream)


class TestIntegration(unittest.TestCase):
    """Integration tests with simulated client"""
    
    def test_end_to_end_simulation(self):
        """Test complete workflow with simulated client"""
        # Create source with simulation
        source = create_optitrack_quattro_source(
            n_rigid_bodies=2,
            client_type="simulation" if hasattr(create_optitrack_quattro_source, '__globals__') else None
        )
        
        # This will use fallback simulation if import fails
        if source.client.__class__.__name__ == 'SimulatedOptiTrackClient':
            # Test the workflow
            source.start()
            
            try:
                # Collect some data
                data_samples = []
                for _ in range(5):
                    data = source.get()
                    data_samples.append(data)
                    time.sleep(0.05)
                
                # Verify data format
                for data in data_samples:
                    self.assertEqual(data.shape, (1, 6))  # 2 bodies * 3 coords
                    self.assertFalse(np.all(np.isnan(data)))  # Should have valid data
                
                # Verify data changes over time (motion)
                first_sample = data_samples[0]
                last_sample = data_samples[-1]
                self.assertFalse(np.allclose(first_sample, last_sample, atol=0.1))
                
            finally:
                source.stop()
    
    def test_quattrocento_format_compatibility(self):
        """Test that output format matches Quattrocento expectations"""
        source = create_optitrack_quattro_source(n_rigid_bodies=3)
        
        # Test with fallback simulation
        data = source.get()
        
        # Should match Quattrocento format: (1, n_channels)
        expected_channels = 3 * 3  # 3 bodies * 3 coordinates
        self.assertEqual(data.shape, (1, expected_channels))
        self.assertEqual(data.dtype, np.float64)
        
        # Should be suitable for decoder input
        self.assertEqual(len(data.shape), 2)
        self.assertEqual(data.shape[0], 1)  # Single time sample


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_invalid_rigid_body_id(self):
        """Test handling of invalid rigid body IDs"""
        stream = OptiTrackQuattroStream(Mock(), n_rigid_bodies=2)
        
        # Test with ID beyond range
        stream._rigid_body_callback(5, [0.1, 0.2, 0.3], [0, 0, 0, 1])
        
        # Data should remain zeros (not crash)
        np.testing.assert_array_equal(stream.latest_data, np.zeros((1, 6)))
    
    def test_none_position_handling(self):
        """Test handling of None positions"""
        stream = OptiTrackQuattroStream(Mock(), n_rigid_bodies=1)
        
        # Create rigid body with None position
        rb = Mock()
        rb.position = None
        
        # Should not crash
        stream._legacy_callback([rb], [], [], [])
        
        # Data should remain unchanged
        np.testing.assert_array_equal(stream.latest_data, np.zeros((1, 3)))
    
    def test_missing_attributes(self):
        """Test handling of missing attributes"""
        stream = OptiTrackQuattroStream(Mock(), n_rigid_bodies=1)
        
        # Test rigid body without position attribute
        rb = Mock(spec=[])  # Empty spec means no attributes
        
        # Should not crash
        try:
            stream._legacy_callback([rb], [], [], [])
        except AttributeError:
            self.fail("Should handle missing position attribute gracefully")


def create_test_suite():
    """Create and return the test suite"""
    test_classes = [
        TestOptiTrackQuattroStream,
        TestSimulatedOptiTrackClient, 
        TestFactoryFunction,
        TestIntegration,
        TestErrorHandling
    ]
    
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    
    for test_class in test_classes:
        # Fixed: use correct method name (lowercase 's')
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


def run_tests():
    """Run all tests with detailed output"""
    suite = create_test_suite()
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0] if 'AssertionError' in traceback else 'See details above'
            print(f"- {test}: {error_msg}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            error_msg = traceback.split('\n')[-2] if '\n' in traceback else traceback
            print(f"- {test}: {error_msg}")
    
    return result


if __name__ == '__main__':
    # You can run tests in different ways:
    
    # Option 1: Run with our custom runner (recommended)
    print("Running OptiTrack Quattro Stream Tests")
    print("="*50)
    result = run_tests()
    
    # Option 2: Standard unittest discovery (alternative)
    # unittest.main(verbosity=2)
    
    # Option 3: Run specific test class
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestOptiTrackQuattroStream)
    # unittest.TextTestRunner(verbosity=2).run(suite)