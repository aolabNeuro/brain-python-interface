#!/usr/bin/env python3
"""
Comprehensive test suite for OptiTrack client and feature implementations
Tests both the updated client and the feature classes
"""

import sys
import os
import time
import numpy as np
import threading
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add the parent directory to path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules to test
try:
    from optitrack_client_updated import (
        System, RigidBody, Marker, NatNetClientWrapper, 
        SimulatedClient, PlaybackClient, create_client, make
    )
    CLIENT_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import optitrack_client_updated: {e}")
    CLIENT_MODULE_AVAILABLE = False

try:
    from optitrack_updated import (
        OptitrackNatNetClient, OptitrackUpdated, 
        OptitrackSimulateUpdated, HidePlantOnPause, SpheresToCylinders
    )
    FEATURE_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import optitrack_updated: {e}")
    FEATURE_MODULE_AVAILABLE = False


class TestOptiTrackClient(unittest.TestCase):
    """Test cases for the OptiTrack client module"""
    
    def setUp(self):
        if not CLIENT_MODULE_AVAILABLE:
            self.skipTest("Client module not available")
    
    def test_rigid_body_creation(self):
        """Test RigidBody class creation and properties"""
        position = [1.0, 2.0, 3.0]
        rotation = [0.0, 0.0, 0.0, 1.0]
        rb_id = 1
        
        rb = RigidBody(position, rotation, rb_id)
        
        self.assertTrue(np.array_equal(rb.position, np.array(position)))
        self.assertTrue(np.array_equal(rb.rotation, np.array(rotation)))
        self.assertEqual(rb.id, rb_id)
        self.assertTrue(rb.tracking_valid)
        self.assertEqual(rb.error, 0.0)
    
    def test_marker_creation(self):
        """Test Marker class creation"""
        position = [1.0, 2.0, 3.0]
        marker_id = 1
        
        marker = Marker(position, marker_id)
        
        self.assertTrue(np.array_equal(marker.position, np.array(position)))
        self.assertEqual(marker.id, marker_id)
    
    def test_simulated_client(self):
        """Test the simulated client functionality"""
        client = SimulatedClient(n=1)
        
        # Test callback setting
        callback_called = False
        received_data = None
        
        def test_callback(rigid_bodies, skeletons, markers, timing):
            nonlocal callback_called, received_data
            callback_called = True
            received_data = rigid_bodies
        
        client.set_callback(test_callback)
        client.run_once()
        
        self.assertTrue(callback_called)
        self.assertIsNotNone(received_data)
        self.assertEqual(len(received_data), 1)
        self.assertIsInstance(received_data[0], RigidBody)
        
        # Test recording functions
        self.assertTrue(client.start_recording())
        self.assertTrue(client.stop_recording())
    
    def test_system_with_simulated_client(self):
        """Test the System class with simulated client"""
        client = SimulatedClient(n=1)
        system = System(client, feature="rigid body", n_features=1)
        
        # Start the system
        system.start()
        
        # Give it a moment to generate data
        time.sleep(0.1)
        
        # Get data
        data = system.get()
        
        self.assertIsNotNone(data)
        self.assertEqual(data.shape, (1, 1, 3))  # (time, n_features, xyz)
        
        # Stop the system
        system.stop()
    
    def test_system_data_conversion(self):
        """Test coordinate conversion from meters to centimeters"""
        client = SimulatedClient(n=1)
        system = System(client, feature="rigid body", n_features=1)
        
        # Create a rigid body with known position
        test_position = np.array([1.0, 2.0, 3.0])  # meters
        rb = RigidBody(test_position)
        rb._already_converted_to_cm = False  # Mark as needing conversion
        
        # Manually set the data
        with system.data_lock:
            system.rigid_bodies = [rb]
        
        data = system.get()
        expected_cm = test_position * 100  # Convert to cm
        
        # Check if conversion happened correctly
        np.testing.assert_array_almost_equal(data[0, 0, :], expected_cm, decimal=2)
    
    @patch('optitrack_client_updated.SDK_AVAILABLE', False)
    def test_create_client_fallback(self):
        """Test client creation when SDK is not available"""
        client = create_client("natnet")
        self.assertIsInstance(client, SimulatedClient)
    
    def test_make_function(self):
        """Test the dynamic class creation function"""
        client = SimulatedClient()
        CustomSystem = make(System, client, "rigid body", num_features=2)
        
        instance = CustomSystem()
        self.assertEqual(instance.n_features, 2)
        self.assertEqual(instance.feature, "rigid body")


class TestOptiTrackFeatures(unittest.TestCase):
    """Test cases for the OptiTrack feature module"""
    
    def setUp(self):
        if not FEATURE_MODULE_AVAILABLE:
            self.skipTest("Feature module not available")
    
    @patch('optitrack_updated.SDK_AVAILABLE', False)
    def test_optitrack_natnet_client_no_sdk(self):
        """Test OptitrackNatNetClient when SDK is not available"""
        client = OptitrackNatNetClient()
        
        with self.assertRaises(Exception):
            client.connect()
    
    def test_optitrack_simulate_updated(self):
        """Test the simulation version of OptiTrack features"""
        
        # Mock the parent class methods to avoid full BMI3D dependency
        with patch.multiple(OptitrackSimulateUpdated, 
                           init=Mock(), 
                           __init__=lambda self: None):
            
            simulator = OptitrackSimulateUpdated()
            simulator.scale = 1.0
            simulator.offset = [0, 0, 0]
            simulator.optitrack_client = None
            simulator.optitrack_status = 'simulation'
            simulator.position_history = []
            simulator.sim_position = np.array([0.0, 0.0, 0.0])
            simulator.sim_time = time.time()
            
            # Test position generation
            position = simulator._get_manual_position()
            
            self.assertIsNotNone(position)
            self.assertEqual(len(position), 3)
            self.assertIsInstance(position, np.ndarray)


class TestIntegration(unittest.TestCase):
    """Integration tests combining both modules"""
    
    def setUp(self):
        if not (CLIENT_MODULE_AVAILABLE and FEATURE_MODULE_AVAILABLE):
            self.skipTest("Both modules required for integration tests")
    
    def test_end_to_end_simulation(self):
        """Test complete simulation pipeline"""
        
        # Create a simulated client
        client = SimulatedClient(n=1)
        
        # Create system
        system = System(client, feature="rigid body", n_features=1)
        system.start()
        
        # Let it run for a bit
        time.sleep(0.2)
        
        # Get some data points
        data_points = []
        for _ in range(5):
            data = system.get()
            if data is not None and not np.isnan(data).all():
                data_points.append(data)
            time.sleep(0.05)
        
        system.stop()
        
        # Check that we got some valid data
        self.assertGreater(len(data_points), 0)
        for data in data_points:
            self.assertEqual(data.shape, (1, 1, 3))


class TestErrorConditions(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def setUp(self):
        if not CLIENT_MODULE_AVAILABLE:
            self.skipTest("Client module not available")
    
    def test_system_with_invalid_feature(self):
        """Test system with invalid feature type"""
        client = SimulatedClient()
        system = System(client, feature="invalid_feature", n_features=1)
        
        with self.assertRaises(AttributeError):
            system.get()
    
    def test_rigid_body_with_none_position(self):
        """Test RigidBody creation with None position"""
        rb = RigidBody(None)
        expected_position = np.array([0, 0, 0])
        np.testing.assert_array_equal(rb.position, expected_position)
    
    def test_system_thread_safety(self):
        """Test thread safety of system data access"""
        client = SimulatedClient(n=1)
        system = System(client, feature="rigid body", n_features=1)
        system.start()
        
        # Access data from multiple threads
        results = []
        errors = []
        
        def access_data():
            try:
                for _ in range(10):
                    data = system.get()
                    results.append(data)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=access_data) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        system.stop()
        
        # Should not have any thread-related errors
        self.assertEqual(len(errors), 0)
        self.assertGreater(len(results), 0)


def run_interactive_test():
    """Run an interactive test to see data in real-time"""
    print("\n" + "="*60)
    print("INTERACTIVE OPTITRACK TEST")
    print("="*60)
    
    if not CLIENT_MODULE_AVAILABLE:
        print("❌ Client module not available")
        return
    
    print("🚀 Starting simulated OptiTrack client...")
    
    # Create and start system
    client = SimulatedClient(n=1, radius=(5.0, 2.0, 3.0), speed=(1.0, 2.0, 0.5))
    system = System(client, feature="rigid body", n_features=1)
    system.start()
    
    print("📊 Collecting data for 5 seconds...")
    print("Position data (X, Y, Z in cm):")
    print("-" * 40)
    
    try:
        start_time = time.time()
        while time.time() - start_time < 5.0:
            data = system.get()
            if data is not None and not np.isnan(data).all():
                pos = data[0, 0, :]  # First feature position
                print(f"  X: {pos[0]:8.2f}, Y: {pos[1]:8.2f}, Z: {pos[2]:8.2f}")
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    
    finally:
        system.stop()
        print("\n✅ Test completed")


def run_validation_tests():
    """Run validation tests and report results"""
    print("\n" + "="*60)
    print("OPTITRACK VALIDATION TESTS")
    print("="*60)
    
    # Module availability check
    print(f"📦 Client module available: {'✅' if CLIENT_MODULE_AVAILABLE else '❌'}")
    print(f"📦 Feature module available: {'✅' if FEATURE_MODULE_AVAILABLE else '❌'}")
    
    if not CLIENT_MODULE_AVAILABLE and not FEATURE_MODULE_AVAILABLE:
        print("❌ No modules available for testing")
        return
    
    # Run unit tests
    suite = unittest.TestSuite()
    
    if CLIENT_MODULE_AVAILABLE:
        suite.addTest(unittest.makeSuite(TestOptiTrackClient))
        suite.addTest(unittest.makeSuite(TestErrorConditions))
    
    if FEATURE_MODULE_AVAILABLE:
        suite.addTest(unittest.makeSuite(TestOptiTrackFeatures))
    
    if CLIENT_MODULE_AVAILABLE and FEATURE_MODULE_AVAILABLE:
        suite.addTest(unittest.makeSuite(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n📈 Test Results:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, trace in result.failures:
            print(f"   {test}: {trace.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\n⚠️  Errors:")
        for test, trace in result.errors:
            print(f"   {test}: {trace.split('\\n')[-2]}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n🎯 Success rate: {success_rate:.1f}%")


if __name__ == "__main__":
    print("OptiTrack Test Suite")
    print("Choose test mode:")
    print("1. Validation tests (unit tests)")
    print("2. Interactive test (see data in real-time)")
    print("3. Both")
    
    try:
        choice = input("Enter choice (1/2/3): ").strip()
        
        if choice in ['1', '3']:
            run_validation_tests()
        
        if choice in ['2', '3']:
            run_interactive_test()
        
        if choice not in ['1', '2', '3']:
            print("Invalid choice, running validation tests...")
            run_validation_tests()
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
