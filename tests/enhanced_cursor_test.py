#!/usr/bin/env python3
'''
Enhanced test script for OptiTrack cursor control functionality
Tests new rigid body selection modes, hand tracking, and BMI3D integration
'''

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import threading

# Add paths for importing modules (adjust as needed)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from optitrack_cursor_control import (
        OptitrackCursorControl, OptitrackCursorControlUpdated, 
        OptitrackCursorSimulated, create_optitrack_cursor_system,
        OptitrackCursorControlEnhanced, OptitrackManualControlTask,
        add_optitrack_to_existing_task
    )
    CURSOR_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import cursor control module: {e}")
    CURSOR_MODULE_AVAILABLE = False

class EnhancedOptitrackCursorSimulated(OptitrackCursorControlEnhanced):
    '''
    Enhanced simulated version for testing new features
    '''
    
    def init(self):
        '''Initialize enhanced simulation'''
        # Create multiple simulated rigid bodies
        self.sim_rigid_bodies = {
            1: {'id': 1, 'position': np.array([0., 0., 0.]), 'stable_factor': 1.0},
            2: {'id': 2, 'position': np.array([5., 5., 2.]), 'stable_factor': 0.3},
            3: {'id': 3, 'position': np.array([-3., 8., -1.]), 'stable_factor': 0.8},
            4: {'id': 4, 'position': np.array([10., -5., 3.]), 'stable_factor': 0.1},
        }
        
        # Simulated hand skeleton data
        self.hand_skeleton_data = {
            'wrist': {'position': np.array([0., 0., 0.])},
            'thumb_tip': {'position': np.array([2., 1., 1.])},
            'index_tip': {'position': np.array([4., 0., 0.])},
            'middle_tip': {'position': np.array([4.5., 0., 0.])},
            'ring_tip': {'position': np.array([4., 0., -1.])},
            'pinky_tip': {'position': np.array([3., 0., -2.])},
        }
        
        super().init()
    
    def _get_all_rigid_bodies(self):
        '''Return simulated rigid body data with realistic movement'''
        current_time = time.time()
        t = current_time * 0.5
        
        rigid_bodies = []
        
        for rb_id, rb_data in self.sim_rigid_bodies.items():
            # Add realistic movement with different stability levels
            base_pos = rb_data['position'].copy()
            stability = rb_data['stable_factor']
            
            # More stable = less random movement
            noise_scale = (1.0 - stability) * 2.0
            
            position = base_pos + np.array([
                2 * np.cos(t + rb_id) + np.random.normal(0, noise_scale),
                2 * np.sin(t * 1.5 + rb_id) + np.random.normal(0, noise_scale),
                1 * np.sin(t * 2 + rb_id) + np.random.normal(0, noise_scale * 0.5)
            ])
            
            rigid_bodies.append({
                'id': rb_id,
                'name': f'RigidBody_{rb_id}',
                'position': position,
                'rotation': np.array([0., 0., 0., 1.]),
                'timestamp': current_time,
                'tracking_valid': True
            })
        
        return rigid_bodies

def test_enhanced_cursor_control():
    '''Test enhanced cursor control features'''
    print("\n" + "="*60)
    print("TESTING ENHANCED OPTITRACK CURSOR CONTROL")
    print("="*60)
    
    if not CURSOR_MODULE_AVAILABLE:
        print("❌ Enhanced cursor control module not available")
        return
    
    cursor_system = EnhancedOptitrackCursorSimulated()
    cursor_system.rigid_body_selection_mode = "manual_id"
    cursor_system.target_rigid_body_id = 1
    cursor_system.init()
    
    print("✅ Enhanced cursor control system initialized")
    
    # Test basic functionality
    print("\n📊 Testing enhanced cursor tracking...")
    positions = []
    timestamps = []
    
    start_time = time.time()
    while time.time() - start_time < 5.0:
        cursor_pos = cursor_system._get_manual_position()
        if cursor_pos is not None:
            positions.append(cursor_pos.copy())
            timestamps.append(time.time() - start_time)
        time.sleep(0.1)
    
    print(f"✅ Collected {len(positions)} enhanced cursor samples")
    return cursor_system, positions, timestamps

def test_rigid_body_detection():
    '''Test enhanced rigid body detection and selection'''
    print("\n" + "="*60)
    print("TESTING RIGID BODY DETECTION")
    print("="*60)
    
    cursor_system = EnhancedOptitrackCursorSimulated()
    cursor_system.init()
    
    # Let system run for a bit to detect rigid bodies
    print("🔍 Running detection for 3 seconds...")
    start_time = time.time()
    while time.time() - start_time < 3.0:
        cursor_system._get_manual_position()  # This triggers detection
        time.sleep(0.1)
    
    # Check detected rigid bodies
    available_rbs = cursor_system.get_available_rigid_bodies()
    print(f"📊 Detected {len(available_rbs)} rigid bodies:")
    for rb in available_rbs:
        print(f"   ID {rb['id']}: stable={rb['stable']}, detected {rb['detection_time']:.1f}s ago")
    
    return available_rbs

def test_auto_detection_mode():
    '''Test automatic rigid body detection'''
    print("\n" + "="*60)
    print("TESTING AUTOMATIC RIGID BODY DETECTION")
    print("="*60)
    
    cursor_system = EnhancedOptitrackCursorSimulated()
    cursor_system.rigid_body_selection_mode = "auto_detect"
    cursor_system.auto_detect_timeout = 8.0
    cursor_system.init()
    
    print("🤖 Running auto-detection...")
    start_time = time.time()
    detected_rb = None
    
    while time.time() - start_time < 10.0:
        cursor_pos = cursor_system._get_manual_position()
        
        if cursor_system.rigid_body_found and detected_rb is None:
            detected_rb = cursor_system.target_rigid_body_id
            print(f"✅ Auto-detected rigid body ID: {detected_rb}")
            break
        
        if int(time.time() - start_time) % 2 == 0:
            available = cursor_system.get_available_rigid_bodies()
            print(f"   Evaluating {len(available)} rigid bodies...")
        
        time.sleep(0.2)
    
    if detected_rb:
        print(f"🎯 Successfully auto-detected RB ID: {detected_rb}")
    else:
        print("❌ Auto-detection failed or timed out")
    
    return detected_rb

def test_selection_modes():
    '''Test different rigid body selection modes'''
    print("\n" + "="*60)
    print("TESTING RIGID BODY SELECTION MODES")
    print("="*60)
    
    modes = ["manual_id", "auto_detect", "closest_to_origin"]
    
    for mode in modes:
        print(f"\n🎮 Testing {mode} selection mode:")
        
        cursor_system = EnhancedOptitrackCursorSimulated()
        cursor_system.rigid_body_selection_mode = mode
        if mode == "manual_id":
            cursor_system.target_rigid_body_id = 3
        cursor_system.init()
        
        # Run for a few seconds
        start_time = time.time()
        selected_rb = None
        
        while time.time() - start_time < 5.0:
            cursor_pos = cursor_system._get_manual_position()
            if cursor_system.rigid_body_found:
                selected_rb = cursor_system.target_rigid_body_id
                break
            time.sleep(0.1)
        
        print(f"   Selected rigid body ID: {selected_rb}")

def test_hand_skeleton_mode():
    '''Test hand skeleton cursor control'''
    print("\n" + "="*60)
    print("TESTING HAND SKELETON CURSOR CONTROL")
    print("="*60)
    
    cursor_system = EnhancedOptitrackCursorSimulated()
    cursor_system.use_hand_skeleton = True
    cursor_system.hand_control_mode = "fingertip"
    cursor_system.target_finger = "index"
    cursor_system.init()
    
    print("🖐️ Testing fingertip cursor control...")
    
    positions = []
    for i in range(10):
        cursor_pos = cursor_system._get_manual_position()
        if cursor_pos is not None:
            positions.append(cursor_pos.copy())
            print(f"   Fingertip cursor: [{cursor_pos[0]:.2f}, {cursor_pos[1]:.2f}, {cursor_pos[2]:.2f}]")
        time.sleep(0.2)
    
    # Test different fingers
    fingers = ["thumb", "middle", "ring", "pinky"]
    for finger in fingers:
        cursor_system.target_finger = finger
        cursor_pos = cursor_system._get_manual_position()
        if cursor_pos is not None:
            print(f"   {finger} tip: [{cursor_pos[0]:.2f}, {cursor_pos[1]:.2f}, {cursor_pos[2]:.2f}]")

def test_coordinate_transformations_enhanced():
    '''Test enhanced coordinate transformations'''
    print("\n" + "="*60)
    print("TESTING ENHANCED COORDINATE TRANSFORMATIONS")
    print("="*60)
    
    # Test different coordinate system configurations
    configs = [
        {
            "name": "Standard BMI3D (X-right, Y-forward, Z-up)",
            "axes": [0, 1, 2], 
            "flips": [1, 1, 1],
            "offset": [0, 0, 0]
        },
        {
            "name": "OptiTrack to BMI3D (swap Y-Z, flip Z)",
            "axes": [0, 2, 1],
            "flips": [1, 1, -1], 
            "offset": [0, 0, 10]
        },
        {
            "name": "Custom workspace mapping",
            "axes": [1, 0, 2],
            "flips": [-1, 1, 1],
            "offset": [5, -5, 0]
        }
    ]
    
    for config in configs:
        print(f"\n🔄 Testing: {config['name']}")
        
        cursor_system = EnhancedOptitrackCursorSimulated()
        cursor_system.optitrack_to_cursor_axes = np.array(config['axes'])
        cursor_system.flip_axes = np.array(config['flips'])
        cursor_system.position_offset = np.array(config['offset'])
        cursor_system.init()
        
        # Test a few positions
        for i in range(3):
            cursor_pos = cursor_system._get_manual_position()
            if cursor_pos is not None:
                print(f"   Sample {i+1}: [{cursor_pos[0]:6.2f}, {cursor_pos[1]:6.2f}, {cursor_pos[2]:6.2f}]")
            time.sleep(0.1)

def test_bmi3d_integration():
    '''Test BMI3D task integration'''
    print("\n" + "="*60)
    print("TESTING BMI3D TASK INTEGRATION")
    print("="*60)
    
    try:
        # Test manual control task
        print("🧠 Testing OptitrackManualControlTask...")
        
        # Create a mock BMI3D-style task
        class MockBMITask:
            def __init__(self):
                self.cursor_pos = np.array([0., 0., 0.])
                self.target_pos = np.array([10., 5., 0.])
                self.target_radius = 3.0
            
            def init(self):
                print("   Mock BMI task initialized")
            
            def _cycle(self):
                pass
        
        # Create enhanced task
        class MockOptitrackTask(MockBMITask, EnhancedOptitrackCursorSimulated):
            def init(self):
                MockBMITask.init(self)
                EnhancedOptitrackCursorSimulated.init(self)
        
        task = MockOptitrackTask()
        task.target_rigid_body_id = 1
        task.init()
        
        print("✅ BMI3D integration task created successfully")
        
        # Test cursor control in task context
        print("🎯 Testing cursor control in BMI3D task...")
        for i in range(5):
            cursor_pos = task._get_manual_position()
            if cursor_pos is not None:
                distance = np.linalg.norm(cursor_pos - task.target_pos)
                print(f"   Cycle {i+1}: cursor [{cursor_pos[0]:6.1f}, {cursor_pos[1]:6.1f}, {cursor_pos[2]:6.1f}], distance to target: {distance:.1f} cm")
            time.sleep(0.2)
        
        return True
        
    except Exception as e:
        print(f"❌ BMI3D integration test failed: {e}")
        return False

def test_performance_monitoring():
    '''Test performance and monitoring features'''
    print("\n" + "="*60)
    print("TESTING PERFORMANCE MONITORING")
    print("="*60)
    
    cursor_system = EnhancedOptitrackCursorSimulated()
    cursor_system.init()
    
    # Test cursor info monitoring
    print("📊 Testing cursor info monitoring...")
    
    for i in range(10):
        cursor_pos = cursor_system._get_manual_position()
        info = cursor_system.get_cursor_info()
        
        if i % 3 == 0:  # Print every 3rd sample
            print(f"   Sample {i+1}:")
            print(f"      Position: [{info['position'][0]:6.2f}, {info['position'][1]:6.2f}, {info['position'][2]:6.2f}]")
            print(f"      RB found: {info['rigid_body_found']}")
            print(f"      Target ID: {info['target_rb_id']}")
            print(f"      History samples: {info['num_history_samples']}")
        
        time.sleep(0.1)
    
    # Test available rigid bodies monitoring
    available_rbs = cursor_system.get_available_rigid_bodies()
    print(f"\n📋 Available rigid bodies: {len(available_rbs)}")
    for rb in available_rbs:
        print(f"   ID {rb['id']}: stable={rb['stable']}, detection_time={rb['detection_time']:.1f}s")

def visualize_enhanced_tracking(cursor_system, positions, timestamps):
    '''Enhanced visualization with multiple rigid bodies'''
    if positions is None or len(positions) == 0:
        print("❌ No position data to visualize")
        return
    
    print("\n📈 Creating enhanced cursor tracking visualization...")
    
    try:
        positions = np.array(positions)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Enhanced OptiTrack Cursor Control Analysis')
        
        # Original trajectory plot
        ax1 = axes[0, 0]
        ax1.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.7, linewidth=2)
        ax1.scatter(positions[0, 0], positions[0, 1], c='green', s=100, label='Start', marker='o')
        ax1.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, label='End', marker='s')
        ax1.set_xlabel('X (cm)')
        ax1.set_ylabel('Y (cm)')
        ax1.set_title('X-Y Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 3D trajectory
        ax2 = axes[0, 1]
        ax2.scatter(positions[:, 0], positions[:, 1], c=timestamps, cmap='viridis', s=20)
        ax2.set_xlabel('X (cm)')
        ax2.set_ylabel('Y (cm)')
        ax2.set_title('Trajectory (colored by time)')
        ax2.grid(True, alpha=0.3)
        
        # Velocity analysis
        ax3 = axes[0, 2]
        if len(positions) > 1:
            velocities = np.diff(positions, axis=0) / np.diff(timestamps).reshape(-1, 1)
            speeds = np.linalg.norm(velocities, axis=1)
            ax3.plot(timestamps[1:], speeds, 'r-', linewidth=2)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Speed (cm/s)')
            ax3.set_title('Cursor Speed')
            ax3.grid(True, alpha=0.3)
        
        # Position components over time
        ax4 = axes[1, 0]
        ax4.plot(timestamps, positions[:, 0], 'r-', label='X', linewidth=2)
        ax4.plot(timestamps, positions[:, 1], 'g-', label='Y', linewidth=2)
        ax4.plot(timestamps, positions[:, 2], 'b-', label='Z', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Position (cm)')
        ax4.set_title('Position Components vs Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Statistics
        ax5 = axes[1, 1]
        stats_text = f"""
Position Statistics:
X: {positions[:, 0].mean():.1f} ± {positions[:, 0].std():.1f} cm
Y: {positions[:, 1].mean():.1f} ± {positions[:, 1].std():.1f} cm  
Z: {positions[:, 2].mean():.1f} ± {positions[:, 2].std():.1f} cm

Range:
X: [{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}] cm
Y: [{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}] cm
Z: [{positions[:, 2].min():.1f}, {positions[:, 2].max():.1f}] cm

Samples: {len(positions)}
Duration: {timestamps[-1]:.1f} s
        """
        ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=9)
        ax5.set_title('Tracking Statistics')
        ax5.axis('off')
        
        # Rigid body info
        ax6 = axes[1, 2]
        if hasattr(cursor_system, 'get_available_rigid_bodies'):
            rbs = cursor_system.get_available_rigid_bodies()
            rb_text = "Available Rigid Bodies:\n\n"
            for i, rb in enumerate(rbs):
                rb_text += f"ID {rb['id']}: {'Stable' if rb['stable'] else 'Unstable'}\n"
                rb_text += f"  Detected: {rb['detection_time']:.1f}s ago\n\n"
            
            if cursor_system.rigid_body_found:
                rb_text += f"\nActive RB ID: {cursor_system.target_rigid_body_id}\n"
                rb_text += f"Selection Mode: {cursor_system.rigid_body_selection_mode}"
            
            ax6.text(0.05, 0.95, rb_text, transform=ax6.transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=9)
        ax6.set_title('Rigid Body Status')
        ax6.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Enhanced visualization created successfully")
        
    except Exception as e:
        print(f"❌ Error creating enhanced visualization: {e}")

def main():
    '''Run enhanced cursor control test suite'''
    print("Enhanced OptiTrack Cursor Control Test Suite")
    print("Testing new rigid body selection, hand tracking, and BMI3D integration")
    
    # Run enhanced tests
    cursor_system, positions, timestamps = test_enhanced_cursor_control()
    available_rbs = test_rigid_body_detection()
    detected_rb = test_auto_detection_mode()
    test_selection_modes()
    test_hand_skeleton_mode()
    test_coordinate_transformations_enhanced()
    bmi3d_success = test_bmi3d_integration()
    test_performance_monitoring()
    
    # Enhanced visualization
    try:
        visualize_enhanced_tracking(cursor_system, positions, timestamps)
    except Exception as e:
        print(f"📊 Enhanced visualization failed: {e}")
    
    # Summary report
    print("\n" + "="*60)
    print("ENHANCED TEST SUITE SUMMARY")
    print("="*60)
    
    print(f"✅ Basic cursor control: {'PASS' if positions else 'FAIL'}")
    print(f"✅ Rigid body detection: {'PASS' if available_rbs else 'FAIL'}")
    print(f"✅ Auto-detection: {'PASS' if detected_rb else 'FAIL'}")
    print(f"✅ BMI3D integration: {'PASS' if bmi3d_success else 'FAIL'}")
    
    print("\n🎯 Enhanced Integration Points for BMI3D:")
    print("   1. Use OptitrackCursorControlEnhanced for advanced rigid body selection")
    print("   2. Set rigid_body_selection_mode to 'auto_detect' for automatic setup")
    print("   3. Enable use_hand_skeleton=True for hand/glove control")
    print("   4. Use add_optitrack_to_existing_task() decorator for existing tasks")
    print("   5. Monitor performance with get_available_rigid_bodies() and get_cursor_info()")
    print("   6. The enhanced system handles coordinate transformations and safety limits")

if __name__ == "__main__":
    main()
