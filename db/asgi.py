'''
ASGI config for Django project with Channels support.

This is used instead of the normal WSGI application when using Django Channels.
'''

import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'db.db_settings')

# Initialize Django ASGI application early to ensure the AppRegistry
# is populated before importing code that may import ORM models.
django_asgi_app = get_asgi_application()

from db.tracker import consumers
from django.urls import re_path

application = ProtocolTypeRouter({
    # Django's ASGI application to handle traditional HTTP requests
    "http": django_asgi_app,
    
    # WebSocket chat handler with auth middleware
    "websocket": AuthMiddlewareStack(
        URLRouter([
            re_path(r"ws/tasks/$", consumers.TaskConsumer.as_asgi()),
        ])
    ),
})
