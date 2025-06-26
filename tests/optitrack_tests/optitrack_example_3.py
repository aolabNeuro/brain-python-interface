"""
Example usage of OptiTrack features with BMI3D

This script demonstrates how to use the OptiTrack features for:
1. Basic data streaming (OptiTrackFeature)
2. BMI integration with decoders (OptiTrackBMI)
"""

from riglib.experiment import Experiment
from features.optitrack_features_5 import OptiTrackFeature, OptiTrackBMI, OptiTrackData  # OptiTrackData is alias
from features.optitrack_features_5 import DEFAULT_OPTITRACK_SERVER_IP, DEFAULT_OPTITRACK_CLIENT_IP
from riglib.bmi.bmi import Decoder
import time


# Centralized network configuration - modify these for your setup
OPTITRACK_SERVER_IP = DEFAULT_OPTITRACK_SERVER_IP  # Change to your Motive server IP
OPTITRACK_CLIENT_IP = DEFAULT_OPTITRACK_CLIENT_IP  # Change to your Ubuntu machine IP


# Example 1: Basic OptiTrack data streaming
class OptiTrackStreamingTask(Experiment, OptiTrackFeature):
    """
    Simple task that streams OptiTrack data without BMI integration
    """
    
    # Configure OptiTrack settings
    optitrack_server_ip = OPTITRACK_SERVER_IP
    optitrack_client_ip = OPTITRACK_CLIENT_IP
    optitrack_use_multicast = False
    optitrack_update_freq = 120.0
    
    # Enable data saving
    register_with_sink_manager = True
    send_data_to_sink_manager = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial_duration = 1.0  # seconds
    
    def _start_wait(self):
        """Wait state at beginning of trial"""
        print("Starting OptiTrack data collection...")
        return True
    
    def _test_start_trial(self, ts):
        """Transition to trial state"""
        return ts > 1.0  # Start trial after 1 second
    
    def _start_trial(self):
        """Main trial state"""
        print("Trial started - collecting OptiTrack data")
        return True
    
    def _test_end_trial(self, ts):
        """Check if trial should end"""
        return ts > self.trial_duration
    
    def _end_trial(self):
        """End trial state"""
        print("Trial ended")
        return True
    
    def _test_end_wait(self, ts):
        """Transition to end state"""
        return ts > 1.0
    
    def _end_wait(self):
        """Final state"""
        print("Task completed")
        return True


# Alternative using the alias (backward compatibility)
class OptiTrackStreamingTaskAlias(Experiment, OptiTrackData):
    """
    Same as above but using the OptiTrackData alias for backward compatibility
    """
    
    # Configure OptiTrack settings
    optitrack_server_ip = OPTITRACK_SERVER_IP
    optitrack_client_ip = OPTITRACK_CLIENT_IP
    optitrack_use_multicast = False
    optitrack_update_freq = 120.0
    
    # Enable data saving
    register_with_sink_manager = True
    send_data_to_sink_manager = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial_duration = 1.0
    
    def _start_wait(self):
        print("Starting OptiTrack data collection (using alias)...")
        return True
    
    def _test_start_trial(self, ts):
        return ts > 1.0
    
    def _start_trial(self):
        print("Trial started - collecting OptiTrack data")
        return True
    
    def _test_end_trial(self, ts):
        return ts > self.trial_duration
    
    def _end_trial(self):
        print("Trial ended")
        return True
    
    def _test_end_wait(self, ts):
        return ts > 1.0
    
    def _end_wait(self):
        print("Task completed")
        return True


# Example 2: OptiTrack BMI integration
class OptiTrackBMITask(Experiment, OptiTrackBMI):
    """
    Task that uses OptiTrack data with BMI decoder
    """
    
    # Configure OptiTrack settings
    optitrack_server_ip = OPTITRACK_SERVER_IP
    optitrack_client_ip = OPTITRACK_CLIENT_IP
    optitrack_use_multicast = False
    optitrack_update_freq = 120.0
    
    # Enable data saving
    register_with_sink_manager = True
    send_data_to_sink_manager = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial_duration = 1.0
        self.decoded_output = None
    
    def _start_wait(self):
        """Wait state at beginning of trial"""
        print("Starting OptiTrack BMI task...")
        print(f"Decoder channels: {len(self.cortical_channels)}")
        return True
    
    def _test_start_trial(self, ts):
        """Transition to trial state"""
        return ts > 1.0
    
    def _start_trial(self):
        """Main trial state with BMI decoding"""
        print("Trial started - running BMI decoder on OptiTrack data")
        
        # Get latest neural data (OptiTrack positions)
        if hasattr(self, 'neurondata') and self.neurondata:
            try:
                # Get new data from OptiTrack
                data, timestamps = self.neurondata.get_new_data()
                
                if len(data) > 0:
                    print(f"Received {len(data)} samples with {data.shape[1]} channels")
                    
                    # Run decoder if available
                    if hasattr(self, 'decoder') and self.decoder:
                        try:
                            # Use the most recent sample for decoding
                            latest_sample = data[-1, :]
                            self.decoded_output = self.decoder(latest_sample.reshape(1, -1))
                            print(f"Decoder output: {self.decoded_output}")
                        except Exception as e:
                            print(f"Decoder error: {e}")
                
            except Exception as e:
                print(f"Error getting OptiTrack data: {e}")
        
        return True
    
    def _test_end_trial(self, ts):
        """Check if trial should end"""
        return ts > self.trial_duration
    
    def _end_trial(self):
        """End trial state"""
        print("Trial ended")
        if self.decoded_output is not None:
            print(f"Final decoder output: {self.decoded_output}")
        return True
    
    def _test_end_wait(self, ts):
        """Transition to end state"""
        return ts > 1.0
    
    def _end_wait(self):
        """Final state"""
        print("BMI task completed")
        return True


# Example 3: Testing OptiTrack connection
def test_optitrack_connection(server_ip=OPTITRACK_SERVER_IP, client_ip=OPTITRACK_CLIENT_IP):
    """
    Test OptiTrack connection without full BMI3D integration
    """
    # Import the system class directly to avoid confusion
    from riglib.optitrack_client_update.optitrack_system_5 import OptiTrackData as OptiTrackSystemData
    
    print(f"Testing OptiTrack connection...")
    print(f"Server IP: {server_ip}")
    print(f"Client IP: {client_ip}")
    
    # Create OptiTrack data source using system class
    optitrack = OptiTrackSystemData(
        server_ip=server_ip,
        client_ip=client_ip,
        use_multicast=False,
        update_freq=120.0
    )
    
    try:
        # Start streaming
        print("Starting OptiTrack streaming...")
        optitrack.start()
        
        # Collect data for 10 seconds
        print("Collecting data for 1 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 1.0:
            data, timestamps = optitrack.get_new_data()
            
            if len(data) > 0:
                print(f"Frame {optitrack.frame_number}: {len(data)} samples, "
                      f"{data.shape[1]} channels, "
                      f"{len(optitrack.rigid_bodies)} rigid bodies, "
                      f"{len(optitrack.skeletons)} skeletons")
                
                # Print sample data
                if len(data) > 0:
                    latest_sample = data[-1, :]
                    print(f"  Latest sample: {latest_sample[:min(6, len(latest_sample))]}")
            
            time.sleep(0.5)  # Check every 100ms
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop streaming
        print("Stopping OptiTrack...")
        optitrack.stop()


if __name__ == "__main__":
    # Test connection first
    print("=" * 50)
    print("Testing OptiTrack Connection")
    print("=" * 50)
    
    # Replace these IPs with your actual network configuration
    test_optitrack_connection(
        server_ip=OPTITRACK_SERVER_IP,  # Your Motive server IP
        client_ip=OPTITRACK_CLIENT_IP   # Your Ubuntu machine IP
    )
    
    print("\n" + "=" * 50)
    print("Example task classes are defined above.")
    print("To run them, create instances and call .run() method:")
    print("task = OptiTrackStreamingTask()")
    print("task.run()")
    print("=" * 50)
