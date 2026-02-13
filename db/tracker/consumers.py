'''
Django Channels WebSocket consumer for task communication.

Replaces the Tornado websocket.py with Django's native WebSocket support.
Simpler, more maintainable, and fully integrated with Django.
'''

import json
import asyncio
import threading
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.layers import get_channel_layer
from asgiref.sync import sync_to_async, async_to_sync


class TaskConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time task status updates and RPC calls.
    
    Handles:
    - Client connection/disconnection
    - Broadcasting task status to all connected clients
    - Receiving RPC commands from clients and forwarding to task
    """
    
    async def connect(self):
        """Called when a WebSocket connects"""
        self.task_group = 'tasks'
        
        # Add to a group so we can broadcast to all clients
        await self.channel_layer.group_add(
            self.task_group,
            self.channel_name
        )
        await self.accept()
    
    async def disconnect(self, close_code):
        """Called when a WebSocket disconnects"""
        await self.channel_layer.group_discard(
            self.task_group,
            self.channel_name
        )
    
    async def receive(self, text_data):
        """Receive a message from the WebSocket client"""
        try:
            data = json.loads(text_data)
            await self.handle_message(data)
        except json.JSONDecodeError:
            await self.send_error("Invalid JSON")
        except Exception as e:
            await self.send_error(f"Error: {str(e)}")
    
    async def handle_message(self, data):
        """
        Route incoming messages to appropriate handlers.
        
        Message types:
        - 'rpc': Remote procedure call on running task
        - 'ping': Keep-alive message
        - Custom types can be added as needed
        """
        msg_type = data.get('type')
        
        if msg_type == 'rpc':
            await self.handle_rpc(data)
        elif msg_type == 'ping':
            await self.send_json({'type': 'pong'})
        else:
            await self.send_error(f"Unknown message type: {msg_type}")
    
    async def handle_rpc(self, data):
        """
        Handle a request to call a method on the running task.
        
        Data format:
        {
            'type': 'rpc',
            'method': str (method name),
            'args': list,
            'kwargs': dict,
            'id': str (optional, for response matching)
        }
        """
        # This would interface with the TaskTracker singleton
        # Implementation depends on how we want to structure this
        
        # For now, send an echo response
        response = {
            'type': 'rpc_response',
            'id': data.get('id'),
            'result': 'RPC handler not yet implemented'
        }
        await self.send_json(response)
    
    async def send_error(self, message):
        """Send an error message to the client"""
        await self.send_json({
            'type': 'error',
            'message': message
        })
    
    async def send_json(self, content):
        """Send JSON content to client"""
        await self.send(text_data=json.dumps(content))
    
    # Handlers for group messages (from background tasks)
    
    async def task_status(self, event):
        """
        Receive a task status update from the task process and forward to client.
        
        Called when a message with type='task_status' is sent to the group.
        """
        await self.send_json(event['data'])
    
    async def task_error(self, event):
        """
        Receive a task error message and forward to client.
        """
        await self.send_json({
            'type': 'error',
            'message': event.get('message'),
            'traceback': event.get('traceback')
        })


class TaskStatusBroadcaster:
    """
    Utility class to broadcast task status updates to all connected WebSocket clients.
    
    This is used by the TaskTracker/TaskProcess to send updates to the frontend.
    """
    
    @staticmethod
    async def broadcast_status(state, reportstats):
        """
        Broadcast task status to all connected clients.
        
        Parameters
        ----------
        state : str
            Current FSM state of the task
        reportstats : dict
            Report statistics from the task
        """
        channel_layer = get_channel_layer()
        await channel_layer.group_send(
            'tasks',
            {
                'type': 'task_status',
                'data': {
                    'type': 'task_status',
                    'state': state,
                    'reportstats': reportstats
                }
            }
        )
    
    @staticmethod
    async def broadcast_error(message, traceback_str=''):
        """
        Broadcast an error message to all connected clients.
        
        Parameters
        ----------
        message : str
            Error message
        traceback_str : str, optional
            Full traceback if available
        """
        channel_layer = get_channel_layer()
        await channel_layer.group_send(
            'tasks',
            {
                'type': 'task_error',
                'message': message,
                'traceback': traceback_str
            }
        )


# Synchronous wrappers for use in non-async code
def sync_broadcast_status(state, reportstats):
    """Synchronous wrapper to broadcast task status"""
    # Use async_to_sync to call the async broadcast function from sync code
    async_to_sync(TaskStatusBroadcaster.broadcast_status)(state, reportstats)


def sync_broadcast_error(message, traceback_str=''):
    """Synchronous wrapper to broadcast error"""
    # Use async_to_sync to call the async broadcast function from sync code
    async_to_sync(TaskStatusBroadcaster.broadcast_error)(message, traceback_str)
