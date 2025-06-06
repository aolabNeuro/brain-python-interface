"""
Example usage of OptiTrack features with BMI3D

This script demonstrates how to use the OptiTrack features for:
1. Basic data streaming (OptiTrackData)
2. BMI integration with decoders (OptiTrackBMI)
"""

from riglib.experiment import Experiment
from features.optitrack_features_update import OptiTrackData, OptiTrackBMI
from riglib.bmi.bmi import Decoder
import time


# Example 1: Basic OptiTrack data streaming
class OptiTrackStreamingTask(Experiment, OptiTrackData):
    """
    Simple task that streams OptiTrack data without BMI integration
    """
    
    # Configure OptiTrack settings
    optitrack_server_ip = "128.95.215.191"  # Replace with your Motive server IP
    optitrack_client_ip = "128.95.215.213"  # Replace with your Ubuntu machine IP
    optitrack_use_multicast = False
    optitrack_update_freq = 120.0
    
    # Enable data saving
    register_with_sink_manager = True
    send_data_to_sink_manager = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial_duration = 10.0  # seconds
    
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


# Example 2: OptiTrack BMI integration
class OptiTrackBMITask(Experiment, OptiTrackBMI):
    """
    Task that uses OptiTrack data with BMI decoder
    """
    
    # Configure OptiTrack settings
    optitrack_server_ip = "192.168.1.100"  # Replace with your Motive server IP
    optitrack_client_ip = "192.168.1.101"  # Replace with your Ubuntu machine IP
    optitrack_use_multicast = False
    optitrack_update_freq = 120.0
    
    # Enable data saving
    register_with_sink_manager = True
    send_data_to_sink_manager = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial_duration = 10.0
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
def test_optitrack_connection(server_ip="127.0.0.1", client_ip="127.0.0.1"):
    """
    Test OptiTrack connection without full BMI3D integration
    """
    from riglib.optitrack_client_update.optitrack_system import OptiTrackData
    
    print(f"Testing OptiTrack connection...")
    print(f"Server IP: {server_ip}")
    print(f"Client IP: {client_ip}")
    
    # Create OptiTrack data source
    optitrack = OptiTrackData(
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
        print("Collecting data for 10 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 10.0:
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
            
            time.sleep(0.1)  # Check every 100ms
        
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
        server_ip="127.0.0.1",  # Your Motive server IP
        client_ip="127.0.0.1"   # Your Ubuntu machine IP
    )
    
    print("\n" + "=" * 50)
    print("Example task classes are defined above.")
    print("To run them, create instances and call .run() method:")
    print("task = OptiTrackStreamingTask()")
    print("task.run()")
    print("=" * 50)
