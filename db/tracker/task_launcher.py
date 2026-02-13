'''
Simplified task launcher using Django Channels for WebSocket communication.

Replaces tasktrack.py and websocket.py with a cleaner architecture:
- Launches experiments in background processes
- Uses simple queue-based RPC instead of custom pipe logic
- WebSocket communication via Django Channels instead of Tornado
- No custom synchronization logic (simpler, less error-prone)
'''

import multiprocessing as mp
import threading
import queue
import time
import sys
import io
import traceback
import json
from datetime import datetime

from riglib import experiment
from . import models
from .json_param import Parameters

# Use fork method on Unix/macOS to avoid pickle issues with dynamic classes
if sys.platform in ['linux', 'darwin']:
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        # Already set, ignore
        pass

MAX_TASK_QUEUE_WAIT = 10  # seconds to wait for task response


class RPCMixin:
    """
    Mixin to add RPC command processing to an Experiment class.
    
    This allows the task's _cycle() method to check for and process RPC commands
    while the FSM is running. Subclasses should call super()._cycle() to ensure
    RPC processing happens alongside normal cycle operations.
    """
    
    def _cycle_with_rpc(self):
        """Check for RPC commands and run parent _cycle()"""
        # Process any pending RPC commands
        if hasattr(self, '_task_process') and self._task_process is not None:
            try:
                cmd = self._task_process.rpc_queue.get_nowait()
                self._task_process._handle_rpc_command(cmd)
            except queue.Empty:
                pass
        
        # Call the parent _cycle() method
        super()._cycle()


class TaskProcess(mp.Process):
    """
    Background process that runs an Experiment instance.
    
    Communicates with the main process via:
    - rpc_queue: receives method calls and command data
    - status_queue: sends task state updates and responses  
    """
    
    def __init__(self, target_class, params, subject_name, saveid=None, log_filename=None):
        """
        Initialize a task process.
        
        Parameters
        ----------
        target_class : class
            Experiment class (with features already mixed in if needed)
        params : dict
            Parameters to pass to the experiment __init__
        subject_name : str
            Name of the subject running the task
        saveid : int, optional
            Database ID of the TaskEntry record
        log_filename : str, optional
            Path to log file for errors
        """
        super().__init__()
        self.target_class = target_class
        self.params = params
        self.subject_name = subject_name
        self.saveid = saveid
        self.log_filename = log_filename
        
        # Queues for IPC
        self.rpc_queue = mp.Queue()  # Commands FROM main process
        self.status_queue = mp.Queue()  # Status updates TO main process
        
        # Task instance (created in subprocess)
        self.task = None
        
    def log_error(self, err):
        """Log an error to file"""
        if self.log_filename:
            with open(self.log_filename, 'a') as fp:
                fp.write(str(err) + '\n')
                traceback.print_exc(file=fp)
    
    def run(self):
        """Main process loop - called by mp.Process.start()"""
        try:
            self._run_task()
        except Exception as e:
            self.log_error(e)
            self.status_queue.put({
                'type': 'error',
                'message': traceback.format_exc()
            })
    
    def _run_task(self):
        """Initialize and run the task"""
        # Set up parameters
        params = self.params.copy()
        params['subject_name'] = self.subject_name
        # Avoid duplicate saveid in params
        if 'saveid' not in params:
            params['saveid'] = self.saveid
        
        # Pre-initialization hook (start recording devices, etc)
        self.target_class.pre_init(subject_name=self.subject_name, saveid=self.saveid)
        
        # Create task instance
        self.task = self.target_class(**params)
        
        # Store reference to this process in the task so _cycle can access RPC queue
        self.task._task_process = self
        
        # Mix in RPC handling to the task's _cycle method
        # We'll replace the _cycle method with one that checks for RPC commands and sends status
        original_cycle = self.task._cycle
        
        cycle_counter = [0]  # Use list to allow modification in nested function
        
        def _cycle_with_rpc_and_status():
            # Check for RPC commands
            try:
                cmd = self.rpc_queue.get_nowait()
                self._handle_rpc_command(cmd)
            except queue.Empty:
                pass
            
            # Call the original _cycle method
            original_cycle()
            
            # Send status update periodically (every ~1 second, assuming 60 FPS)
            cycle_counter[0] += 1
            if cycle_counter[0] >= self.task.fps:
                cycle_counter[0] = 0
                try:
                    self.status_queue.put({
                        'type': 'status',
                        'state': self.task.state,
                        'reportstats': dict(self.task.reportstats)
                    })
                except Exception as e:
                    print(f"Error sending status: {e}")
        
        # Replace the task's _cycle method with our wrapped version
        self.task._cycle = _cycle_with_rpc_and_status
        
        try:
            # Run the task's FSM loop (which includes its own main loop)
            self.task.run()
        finally:
            # Cleanup after task completion
            self._cleanup()
    
    def _handle_rpc_command(self, cmd):
        """
        Handle an RPC method call from the main process.
        
        Format of cmd:
        {
            'method': str (method name),
            'args': tuple,
            'kwargs': dict,
            'id': str (optional, for request/response matching)
        }
        """
        try:
            method_name = cmd.get('method')
            args = cmd.get('args', ())
            kwargs = cmd.get('kwargs', {})
            cmd_id = cmd.get('id')
            
            # Call the method
            method = getattr(self.task, method_name)
            result = method(*args, **kwargs)
            
            # Send response back to main process
            response = {'type': 'rpc_response', 'result': result}
            if cmd_id:
                response['id'] = cmd_id
            self.status_queue.put(response)
            
        except Exception as e:
            error_msg = traceback.format_exc()
            self.log_error(e)
            
            response = {
                'type': 'rpc_error',
                'error': error_msg
            }
            if cmd_id:
                response['id'] = cmd_id
            self.status_queue.put(response)
    
    def _cleanup(self):
        """Cleanup after task completion"""
        if self.task is None:
            return
        
        try:
            self.task.join()
        except:
            pass
        
        # Database cleanup if needed
        if self.saveid is not None:
            try:
                self._db_cleanup()
            except Exception as e:
                self.log_error(e)
        
        try:
            self.task.terminate()
        except:
            pass
    
    def _db_cleanup(self):
        """Save task results to database"""
        import os
        import xmlrpc.client
        
        port = int(os.environ.get('BMI3D_PORT', 8000))
        try:
            database = xmlrpc.client.ServerProxy(
                f"http://localhost:{port}/RPC2/",
                allow_none=True
            )
            self.task.cleanup(database, self.saveid, subject=None)
            database.cleanup(self.saveid)
        except Exception as e:
            self.log_error(f"Database cleanup failed: {e}")


class TaskTracker:
    """
    Manages task lifecycle and communication.
    
    This replaces the singleton Track class with a simpler, non-singleton approach.
    """
    
    def __init__(self, log_filename=None):
        """
        Initialize task tracker.
        
        Parameters
        ----------
        log_filename : str, optional
            Path to log errors
        """
        self.log_filename = log_filename
        self.proc = None
        self.status = ''  # 'running', 'testing', 'stopped', 'error'
        self._rpc_response_map = {}  # Map command IDs to response queues
        self._response_reader_thread = None
        self.reportstats = {}  # Cache the last reportstats from the task
    
    def start_task(self, target_class, params, subject_name, saveid=None):
        """
        Start an experiment task in a background process.
        
        Parameters
        ----------
        target_class : class
            Experiment class (with features already mixed in if needed)
        params : dict
            Task parameters
        subject_name : str
            Subject name
        saveid : int, optional
            Database TaskEntry ID
        """
        if self.proc is not None and self.proc.is_alive():
            raise RuntimeError("Task already running")
        
        self.proc = TaskProcess(
            target_class=target_class,
            params=params,
            subject_name=subject_name,
            saveid=saveid,
            log_filename=self.log_filename
        )
        self.proc.start()
        self.status = 'running' if saveid else 'testing'
        
        # Start thread to read status updates from task
        self._start_response_reader()
    
    def stop_task(self):
        """Stop the running task gracefully"""
        if self.proc is None or not self.proc.is_alive():
            self.status = 'stopped'
            return
        
        # Try to call end_task on the experiment, but don't fail if it doesn't work
        try:
            self.call_task_method('end_task')
        except Exception as e:
            print(f"Error calling end_task: {e}")
        
        # Wait for process to finish
        self.proc.join(timeout=5)
        
        if self.proc.is_alive():
            # Force terminate if needed
            self.proc.terminate()
            self.proc.join(timeout=1)
        
        self.status = 'stopped'
    
    def call_task_method(self, method_name, *args, **kwargs):
        """
        Call a method on the running task.
        
        Parameters
        ----------
        method_name : str
            Name of method to call
        *args, **kwargs
            Arguments to pass to method
            
        Returns
        -------
        Result from the method call
        """
        if self.proc is None or not self.proc.is_alive():
            raise RuntimeError("No task running")
        
        cmd_id = str(time.time())  # Simple unique ID
        response_queue = queue.Queue()
        self._rpc_response_map[cmd_id] = response_queue
        
        try:
            # Send command
            cmd = {
                'method': method_name,
                'args': args,
                'kwargs': kwargs,
                'id': cmd_id
            }
            self.proc.rpc_queue.put(cmd)
            
            # Wait for response
            response = response_queue.get(timeout=MAX_TASK_QUEUE_WAIT)
            
            if response.get('type') == 'rpc_error':
                raise RuntimeError(f"Task method error: {response['error']}")
            
            return response.get('result')
            
        finally:
            del self._rpc_response_map[cmd_id]
    
    def _start_response_reader(self):
        """Start thread to read status updates from task process"""
        if self._response_reader_thread is not None and self._response_reader_thread.is_alive():
            return
        
        self._response_reader_thread = threading.Thread(
            target=self._read_responses,
            daemon=True
        )
        self._response_reader_thread.start()
    
    def _read_responses(self):
        """Thread function: read messages from task status queue"""
        while self.proc and self.proc.is_alive():
            try:
                msg = self.proc.status_queue.get(timeout=0.5)
                self._handle_status_message(msg)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error reading task status: {e}")
    
    def _handle_status_message(self, msg):
        """Handle a message from the task process"""
        msg_type = msg.get('type')
        
        if msg_type == 'rpc_response' or msg_type == 'rpc_error':
            # Route to appropriate response queue
            cmd_id = msg.get('id')
            if cmd_id in self._rpc_response_map:
                self._rpc_response_map[cmd_id].put(msg)
        
        elif msg_type == 'status':
            # Status update - broadcast to WebSocket clients and cache reportstats
            state = msg.get('state')
            reported_stats = msg.get('reportstats', {})
            self.reportstats = reported_stats  # Cache for AJAX access
            try:
                from .consumers import sync_broadcast_status
                sync_broadcast_status(state, reported_stats)
            except Exception as e:
                print(f"Failed to broadcast status: {e}")
        
        elif msg_type == 'error':
            self.status = 'error'
            print(f"Task error: {msg.get('message')}")
        
        elif msg_type == 'complete':
            self.status = 'complete'
    
    def update_alive(self):
        """Check if the remote process is still alive, and if dead, reset"""
        if self.proc is not None and not self.proc.is_alive():
            self.reset()
    
    def get_status(self):
        """Get the current status string"""
        return self.status
    
    def reset(self):
        """Reset tracker state, clearing process and status"""
        self.proc = None
        self.status = ''
        self._rpc_response_map = {}


# Global instance shared across all modules
_task_tracker = TaskTracker()
