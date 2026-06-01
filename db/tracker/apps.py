
import os
from django.apps import AppConfig
from django.db import connections


class TrackerConfig(AppConfig):
    name = 'db.tracker'
    verbose_name = "exp_log_tracker"

    def ready(self):
        """Auto-register video files from the visual_stimuli system on startup."""
        for dbname in connections.databases.keys():
            self._register_visual_stimuli_files(dbname=dbname)

    def _register_visual_stimuli_files(self, dbname='default'):
        """Scan visual_stimuli directory and keep DataFile rows in sync."""
        try:
            from . import models
            
            # Get the visual_stimuli system
            system = models.System.objects.using(dbname).get(name='visual_stimuli')
            
            if not os.path.isdir(system.path):
                return
            
            # Find all video files currently present on disk.
            video_extensions = ('.mp4', '.mov', '.mkv', '.avi', '.m4v', '.webm', '.flv', '.wmv')
            discovered_paths = set()
            for root, dirs, files in os.walk(system.path):
                for filename in files:
                    if filename.startswith('._'):
                        continue
                    if filename.lower().endswith(video_extensions):
                        filepath = os.path.abspath(os.path.join(root, filename))
                        discovered_paths.add(os.path.realpath(filepath))
                        
                        # Check if already registered
                        if models.DataFile.objects.using(dbname).filter(system=system, path=filepath).exists():
                            continue
                        
                        # Register new video
                        try:
                            models.DataFile.objects.using(dbname).create(
                                system=system,
                                path=filepath,
                                local=True,
                                archived=False
                            )
                        except Exception:
                            pass  # Silently skip errors during startup

            # Remove stale DB entries for files no longer present on disk.
            existing_entries = models.DataFile.objects.using(dbname).filter(system=system)
            for datafile in existing_entries:
                if os.path.isabs(datafile.path):
                    full_path = datafile.path
                else:
                    full_path = os.path.join(system.path, datafile.path)

                full_path = os.path.realpath(os.path.abspath(full_path))
                if full_path in discovered_paths:
                    continue

                # Remove only the DB row. QuerySet delete bypasses DataFile.delete().
                try:
                    models.DataFile.objects.using(dbname).filter(pk=datafile.pk).delete()
                except Exception:
                    pass
        except Exception:
            pass  # Silently fail if system doesn't exist yet