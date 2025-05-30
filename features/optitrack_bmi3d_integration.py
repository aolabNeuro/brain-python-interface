"""
Enhanced OptiTrack Cursor Control with proper BMI3D integration
Adds rigid body selection, hand processing, and task integration
"""

import numpy as np
import time
from riglib.experiment import traits
from collections import deque
import threading
from bmimultitasks import BMIControlMulti

class OptitrackCursorControlEnhanced(OptitrackCursorControl):
    """
    Enhanced cursor control with rigid body selection and hand processing
    """
    
    # Enhanced rigid body selection
    available_rigid_body_ids = traits.List([1,2,3,4,5], desc="Available rigid body IDs")
    rigid_body_selection_mode = traits.OptionsList(("manual_id", "auto_detect", "closest_to_origin"))
    auto_detect_timeout = traits.Float(5.0, desc="Timeout for auto-detection in seconds")
    
    # Hand/glove specific settings
    use_hand_skeleton = traits.Bool(False, desc="Use hand skeleton instead of rigid body")
    hand_control_mode = traits.OptionsList(("fingertip", "palm_center", "gesture", "joint_angles"))
    target_finger = traits.OptionsList(("thumb", "index", "middle", "ring", "pinky"))
    
    # Advanced mapping options
    enable_gesture_control = traits.Bool(False, desc="Enable gesture-based control")
    gesture_threshold = traits.Float(0.7, desc="Confidence threshold for gestures")
    
    def init(self):
        super().init()
        
        # Enhanced tracking variables
        self.detected_rigid_bodies = {}
        self.rigid_body_detection_start = None
        self.hand_skeleton_data = None
        self.last_gesture = None
        
        # Start rigid body detection if in auto mode
        if self.rigid_body_selection_mode == "auto_detect":
            self._start_rigid_body_detection()
    
    def _get_all_rigid_bodies(self):
        """
        Enhanced rigid body data retrieval with proper error handling
        """
        rigid_bodies = []
        
        try:
            # Method 1: Direct client interface (for single RB systems)
            if hasattr(self, 'optitrack_client') and self.optitrack_client:
                if hasattr(self.optitrack_client, 'get_latest_data'):
                    data = self.optitrack_client.get_latest_data()
                    if data and self._validate_rigid_body_data(data):
                        rigid_bodies.append(self._standardize_rigid_body_data(data))
            
            # Method 2: System interface (for multi-RB systems)
            if hasattr(self, 'optitrack_system') and self.optitrack_system:
                with getattr(self.optitrack_system, 'data_lock', threading.Lock()):
                    for rb in getattr(self.optitrack_system, 'rigid_bodies', []):
                        if self._is_rigid_body_valid(rb):
                            rb_data = self._extract_rigid_body_data(rb)
                            if self._validate_rigid_body_data(rb_data):
                                rigid_bodies.append(self._standardize_rigid_body_data(rb_data))
            
            # Method 3: Direct streaming data access
            if hasattr(self, 'latest_optitrack_data') and self.latest_optitrack_data:
                for rb_id, rb_data in self.latest_optitrack_data.items():
                    if self._validate_rigid_body_data(rb_data):
                        rigid_bodies.append(self._standardize_rigid_body_data(rb_data))
        
        except Exception as e:
            print(f"Error retrieving rigid body data: {e}")
        
        # Update detection registry
        self._update_detection_registry(rigid_bodies)
        
        return rigid_bodies
    
    def _validate_rigid_body_data(self, data):
        """Validate that rigid body data is complete and reasonable"""
        if not data:
            return False
        
        required_keys = ['position']
        if not all(key in data for key in required_keys):
            return False
        
        position = np.array(data['position'])
        if len(position) != 3:
            return False
        
        # Check for reasonable position values (not NaN, not too large)
        if np.any(np.isnan(position)) or np.any(np.abs(position) > 1000):
            return False
        
        return True
    
    def _standardize_rigid_body_data(self, data):
        """Standardize rigid body data format"""
        return {
            'id': data.get('id', 0),
            'name': data.get('name', ''),
            'position': np.array(data['position']),
            'rotation': np.array(data.get('rotation', [0, 0, 0, 1])),
            'timestamp': data.get('timestamp', time.time()),
            'tracking_valid': data.get('tracking_valid', True)
        }
    
    def _is_rigid_body_valid(self, rb):
        """Check if rigid body object is valid for tracking"""
        return (hasattr(rb, 'position') and 
                hasattr(rb, 'tracking_valid') and 
                getattr(rb, 'tracking_valid', False))
    
    def _extract_rigid_body_data(self, rb):
        """Extract data from rigid body object"""
        return {
            'id': getattr(rb, 'id', 0),
            'name': getattr(rb, 'name', ''),
            'position': getattr(rb, 'position', [0, 0, 0]),
            'rotation': getattr(rb, 'rotation', [0, 0, 0, 1]),
            'tracking_valid': getattr(rb, 'tracking_valid', False)
        }
    
    def _update_detection_registry(self, rigid_bodies):
        """Update registry of detected rigid bodies"""
        current_time = time.time()
        
        for rb in rigid_bodies:
            rb_id = rb['id']
            if rb_id not in self.detected_rigid_bodies:
                self.detected_rigid_bodies[rb_id] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'positions': deque(maxlen=10),
                    'stable': False
                }
            
            self.detected_rigid_bodies[rb_id]['last_seen'] = current_time
            self.detected_rigid_bodies[rb_id]['positions'].append(rb['position'])
            
            # Check if rigid body is stable (low variance in position)
            if len(self.detected_rigid_bodies[rb_id]['positions']) >= 5:
                positions = np.array(list(self.detected_rigid_bodies[rb_id]['positions']))
                variance = np.var(positions, axis=0)
                self.detected_rigid_bodies[rb_id]['stable'] = np.all(variance < 1.0)  # cm
    
    def _find_target_rigid_body(self, rigid_body_data):
        """Enhanced rigid body selection with multiple modes"""
        if not rigid_body_data:
            return None
        
        if self.rigid_body_selection_mode == "manual_id":
            return super()._find_target_rigid_body(rigid_body_data)
        
        elif self.rigid_body_selection_mode == "auto_detect":
            return self._auto_detect_rigid_body(rigid_body_data)
        
        elif self.rigid_body_selection_mode == "closest_to_origin":
            return self._find_closest_rigid_body(rigid_body_data)
        
        return None
    
    def _auto_detect_rigid_body(self, rigid_body_data):
        """Automatically detect the best rigid body for cursor control"""
        if self.rigid_body_detection_start is None:
            self.rigid_body_detection_start = time.time()
        
        # Find most stable rigid body
        best_rb = None
        best_stability = -1
        
        for rb in rigid_body_data:
            rb_id = rb['id']
            if rb_id in self.detected_rigid_bodies:
                detection_info = self.detected_rigid_bodies[rb_id]
                
                # Calculate stability score
                stability_score = 0
                if detection_info['stable']:
                    stability_score += 2
                
                detection_time = time.time() - detection_info['first_seen']
                stability_score += min(detection_time / 2.0, 3.0)  # Max 3 points for time
                
                if stability_score > best_stability:
                    best_stability = stability_score
                    best_rb = rb
        
        if best_rb and best_stability > 3.0:
            self.target_rigid_body_id = best_rb['id']
            self.rigid_body_found = True
            print(f"Auto-detected stable rigid body ID: {best_rb['id']}")
            return best_rb
        
        # Timeout check
        if time.time() - self.rigid_body_detection_start > self.auto_detect_timeout:
            if rigid_body_data:
                print("Auto-detection timeout, using first available rigid body")
                return rigid_body_data[0]
        
        return None
    
    def _find_closest_rigid_body(self, rigid_body_data):
        """Find rigid body closest to origin"""
        closest_rb = None
        min_distance = float('inf')
        
        for rb in rigid_body_data:
            distance = np.linalg.norm(rb['position'])
            if distance < min_distance:
                min_distance = distance
                closest_rb = rb
        
        if closest_rb:
            self.target_rigid_body_id = closest_rb['id']
            self.rigid_body_found = True
        
        return closest_rb
    
    def _get_manual_position(self):
        """
        Enhanced position getter with hand skeleton support
        """
        if self.use_hand_skeleton:
            return self._get_hand_cursor_position()
        else:
            return super()._get_manual_position()
    
    def _get_hand_cursor_position(self):
        """
        Get cursor position from hand skeleton data
        """
        if not hasattr(self, 'hand_skeleton_data') or not self.hand_skeleton_data:
            return self._get_fallback_position()
        
        if self.hand_control_mode == "fingertip":
            return self._get_fingertip_position()
        elif self.hand_control_mode == "palm_center":
            return self._get_palm_center_position()
        elif self.hand_control_mode == "gesture":
            return self._get_gesture_cursor_position()
        elif self.hand_control_mode == "joint_angles":
            return self._get_joint_angle_cursor_position()
        
        return self._get_fallback_position()
    
    def _get_fingertip_position(self):
        """Get position of specified fingertip"""
        # This would interface with your hand skeleton data
        # Implementation depends on your hand tracking data format
        finger_joints = {
            'thumb': 'thumb_tip',
            'index': 'index_tip', 
            'middle': 'middle_tip',
            'ring': 'ring_tip',
            'pinky': 'pinky_tip'
        }
        
        target_joint = finger_joints[self.target_finger]
        if target_joint in self.hand_skeleton_data:
            position = np.array(self.hand_skeleton_data[target_joint]['position'])
            return self._transform_coordinates(position)
        
        return self._get_fallback_position()
    
    def _get_palm_center_position(self):
        """Get center of palm position"""
        # Calculate centroid of palm joints
        palm_joints = ['wrist', 'thumb_base', 'index_base', 'middle_base', 'ring_base', 'pinky_base']
        positions = []
        
        for joint in palm_joints:
            if joint in self.hand_skeleton_data:
                positions.append(self.hand_skeleton_data[joint]['position'])
        
        if positions:
            centroid = np.mean(positions, axis=0)
            return self._transform_coordinates(centroid)
        
        return self._get_fallback_position()
    
    def get_available_rigid_bodies(self):
        """Get list of currently detected rigid bodies"""
        current_time = time.time()
        active_rbs = []
        
        for rb_id, info in self.detected_rigid_bodies.items():
            if current_time - info['last_seen'] < 2.0:  # Active within last 2 seconds
                active_rbs.append({
                    'id': rb_id,
                    'stable': info['stable'],
                    'detection_time': current_time - info['first_seen']
                })
        
        return active_rbs
    
    def set_rigid_body_selection_mode(self, mode):
        """Change rigid body selection mode"""
        self.rigid_body_selection_mode = mode
        if mode == "auto_detect":
            self._start_rigid_body_detection()
    
    def _start_rigid_body_detection(self):
        """Start/restart rigid body detection process"""
        self.rigid_body_detection_start = time.time()
        self.detected_rigid_bodies.clear()
        print("Started automatic rigid body detection...")


class OptitrackManualControlTask(BMIControlMulti, OptitrackCursorControlEnhanced):
    """
    BMI3D task that uses OptiTrack for manual cursor control
    Can be used with existing center-out and other BMI tasks
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def init(self):
        """Initialize both BMI task and OptiTrack cursor control"""
        super().init()
        print("OptiTrack Manual Control Task initialized")
        print(f"Target rigid body ID: {self.target_rigid_body_id}")
        print(f"Cursor mapping mode: {self.cursor_mapping_mode}")
    
    def _cycle(self):
        """Override cycle to add OptiTrack-specific monitoring"""
        super()._cycle()
        
        # Optional: Add debugging/monitoring here
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
        
        # Print cursor info every 100 cycles (adjust as needed)
        if self._debug_counter % 100 == 0:
            info = self.get_cursor_info()
            if info['rigid_body_found']:
                pos = info['position']
                print(f"Cursor: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] cm")


# Integration with existing BMI3D tasks
def add_optitrack_to_existing_task(TaskClass):
    """
    Decorator to add OptiTrack cursor control to any existing BMI3D task
    
    Usage:
        @add_optitrack_to_existing_task
        class MyCenterOutTask(ExistingCenterOutTask):
            pass
    """
    class OptitrackEnabledTask(TaskClass, OptitrackCursorControlEnhanced):
        def init(self):
            super().init()
            print(f"Added OptiTrack cursor control to {TaskClass.__name__}")
    
    return OptitrackEnabledTask

# Example integrations with common BMI3D tasks
try:
    from riglib.experiment.experiments.centerout_2D_discrete import CenterOutReach
    OptitrackCenterOut2D = add_optitrack_to_existing_task(CenterOutReach)
except ImportError:
    pass

try:
    from riglib.experiment.experiments.cursor_tasks import TargetCaptureVFB2DWindow
    OptitrackTargetCapture = add_optitrack_to_existing_task(TargetCaptureVFB2DWindow)
except ImportError:
    pass

# Factory functions for easy task creation
def create_optitrack_centerout_task(**kwargs):
    """Create OptiTrack-based center-out task"""
    return OptitrackManualControlTask(**kwargs)

def create_optitrack_manual_task(**kwargs):
    """Create generic OptiTrack manual control task"""
    return OptitrackManualControlTask(**kwargs)
