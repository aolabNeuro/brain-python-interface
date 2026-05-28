
import os
from django.apps import AppConfig


class TrackerConfig(AppConfig):
    name = 'db.tracker'
    verbose_name = "exp_log_tracker"

    def ready(self):
        """Auto-register video files from the visual_stimuli system on startup."""
        self._register_visual_stimuli_files()

    def _register_visual_stimuli_files(self):
        """Scan visual_stimuli directory and keep DataFile rows in sync."""
        try:
            from . import models
            
            # Get the visual_stimuli system
            system = models.System.objects.get(name='visual_stimuli')
            
            if not os.path.isdir(system.path):
                return
            
            # Find all video files currently present on disk.
            video_extensions = ('.mp4', '.mov', '.mkv', '.avi', '.m4v', '.webm', '.flv', '.wmv')
            discovered_paths = set()
            for root, dirs, files in os.walk(system.path):
                for filename in files:
                    if filename.lower().endswith(video_extensions):
                        filepath = os.path.abspath(os.path.join(root, filename))
                        discovered_paths.add(os.path.realpath(filepath))
                        
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

            # Remove stale DB entries for files no longer present on disk.
            existing_entries = models.DataFile.objects.filter(system=system)
            for datafile in existing_entries:
                if os.path.isabs(datafile.path):
                    full_path = datafile.path
                else:
                    full_path = os.path.join(system.path, datafile.path)

                full_path = os.path.realpath(os.path.abspath(full_path))
                if full_path not in discovered_paths:
                    try:
                        datafile.delete()
                    except Exception:
                        pass
        except Exception:
            pass  # Silently fail if system doesn't exist yet