
import os
from django.apps import AppConfig


class TrackerConfig(AppConfig):
    name = 'db.tracker'
    verbose_name = "exp_log_tracker"

    def ready(self):
        """Auto-register video files from the visual_stimuli system on startup."""
        self._register_visual_stimuli_files()

    def _register_visual_stimuli_files(self):
        """Scan visual_stimuli directory and register any missing video files."""
        try:
            from . import models
            
            # Get the visual_stimuli system
            system = models.System.objects.get(name='visual_stimuli')
            
            if not os.path.isdir(system.path):
                return
            
            # Find all video files
            video_extensions = ('.mp4', '.mov', '.mkv', '.avi', '.m4v', '.webm', '.flv', '.wmv')
            for root, dirs, files in os.walk(system.path):
                for filename in files:
                    if filename.lower().endswith(video_extensions):
                        filepath = os.path.abspath(os.path.join(root, filename))
                        
                        # Check if already registered
                        if models.DataFile.objects.filter(system=system, path=filepath).exists():
                            continue
                        
                        # Register new video
                        try:
                            models.DataFile.objects.create(
                                system=system,
                                path=filepath,
                                local=True,
                                archived=False
                            )
                        except Exception:
                            pass  # Silently skip errors during startup
        except Exception:
            pass  # Silently fail if system doesn't exist yet