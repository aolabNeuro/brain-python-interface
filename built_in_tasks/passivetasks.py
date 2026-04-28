'''
Tasks which control a plant under pure machine control. Used typically for initializing BMI decoder parameters.
'''
import numpy as np
import os
import tables
import time
import subprocess
import tempfile
from OpenGL.GL import GL_RGB, GL_RGB8, GL_UNSIGNED_BYTE

from riglib.experiment import traits, Experiment
from riglib.bmi.state_space_models import StateSpaceEndptVel2D
from riglib.bmi.bmi import Decoder, BMILoop, MachineOnlyFilter
from riglib.bmi.extractor import DummyExtractor
from riglib.stereo_opengl.primitives import TexPlane
from riglib.stereo_opengl.textures import Texture
from riglib.stereo_opengl.window import Window, WindowDispl2D, Window2D
from config.rig_defaults import window as window_defaults

from built_in_tasks.manualcontrolmultitasks import ScreenTargetCapture
from built_in_tasks.bmimultitasks import BMIControlMulti

from .target_graphics import *

bmi_ssm_options = ['Endpt2D', 'Tentacle', 'Joint2L']

class EndPostureFeedbackController(BMILoop, traits.HasTraits):
    ssm_type_options = bmi_ssm_options
    ssm_type = traits.OptionsList(*bmi_ssm_options, bmi3d_input_options=bmi_ssm_options)
    decoder_update_rate = traits.Float(60, desc="Assist feedback rate (Hz)")

    def load_decoder(self):
        self.ssm = StateSpaceEndptVel2D()
        A, B, W = self.ssm.get_ssm_matrices()
        filt = MachineOnlyFilter(A, W)
        units = []
        self.decoder = Decoder(filt, units, self.ssm, binlen=1./self.decoder_update_rate)
        self.decoder.n_features = 1

    def create_feature_extractor(self):
        self.extractor = DummyExtractor()
        self._add_feature_extractor_dtype()


class TargetCaptureVisualFeedback(EndPostureFeedbackController, BMIControlMulti):
    assist_level = (1, 1)
    is_bmi_seed = True

    def move_effector(self):
        pass

class TargetCaptureVFB2DWindow(TargetCaptureVisualFeedback, WindowDispl2D):
    fps = 20.
    def __init__(self,*args, **kwargs):
        super(TargetCaptureVFB2DWindow, self).__init__(*args, **kwargs)
        self.assist_level = (1, 1)

    def _start_wait(self):
        self.wait_time = 0.
        super(TargetCaptureVFB2DWindow, self)._start_wait()

    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause

    @classmethod
    def get_desc(cls, params, report):
        if isinstance(report, list) and len(report) > 0:
            duration = report[-1][-1] - report[0][-1]
            reward_count = 0
            for item in report:
                if item[0] == "reward":
                    reward_count += 1
            return "{} rewarded trials in {} min".format(reward_count, int(np.ceil(duration / 60)))
        elif isinstance(report, dict):
            duration = report['runtime'] / 60
            reward_count = report['n_success_trials']
            return "{} rewarded trials in {} min".format(reward_count, int(np.ceil(duration / 60)))
        else:
            return "No trials"

from .target_graphics import target_colors

class TargetCaptureReplay(ScreenTargetCapture):
    '''
    Reads the frame-by-frame cursor and trial-by-trial target positions from a saved
    HDF file to display an exact copy of a previous experiment. 
    Doesn't really work, do not recommend using this.
    '''

    hdf_filepath = traits.String("", desc="Filepath of hdf file to replay")

    exclude_parent_traits = list(set(ScreenTargetCapture.class_traits().keys()) - \
        set(['window_size', 'fullscreen']))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t0 = time.perf_counter()
        with tables.open_file(self.hdf_filepath, 'r') as f:
            task = f.root.task.read()
            state = f.root.task_msgs.read()
            trial = f.root.trials.read()
            params = f.root.task.attrs._f_list("user")
            self.task_meta = {k : getattr(f.root.task.attrs, k) for k in params}
        self.replay_state = state
        self.replay_task = task
        self.replay_trial = trial
        for k, v in self.task_meta.items():
            if k in self.exclude_parent_traits:
                print("setting {} to {}".format(k, v))
                setattr(self, k, v)

        # Have to additionally reset the targets since they are created in super().__init__()
        target1 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])
        target2 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])
        self.targets = [target1, target2]

    def _test_start_trial(self, time_in_state):
        '''Wait for the state change in the HDF file in case there is autostart enabled'''
        trials = self.replay_state[self.replay_state['msg'] == b'target']
        upcoming_trials = [t['time']-1 for t in trials if self.replay_task[t['time']]['trial'] >= self.calc_trial_num()]
        return (np.array(upcoming_trials) <= self.cycle_count).any()

    def _parse_next_trial(self):
        '''Ignore the generator'''
        self.targs = []
        self.gen_indices = []
        trial_num = self.calc_trial_num()
        for trial in self.replay_trial:
            if trial['trial'] == trial_num:
                self.targs.append(trial['target'])
                self.gen_indices.append(trial['index'])

    def _cycle(self):
        '''Have to fudge the cycle_count a bit in case the fps isn't exactly the same'''
        super()._cycle()
        t1 = time.perf_counter() - self.t0
        self.cycle_count = int(t1*self.fps)

    def move_effector(self):
        current_pt = self.replay_task['cursor'][self.cycle_count]
        self.plant.set_endpoint_pos(current_pt)

    def _test_stop(self, ts):
        return super()._test_stop(ts) or self.cycle_count == len(self.replay_task)


class VideoPlayer(Window2D, Window, Experiment):
    status = dict(wait=dict(stop=None))
    state = "wait"

    media_file = traits.String("", desc="Path to local video file (mp4, mov, mkv, etc.)")
    audio_volume = traits.Float(1.0, desc="Playback volume from 0.0 to 1.0")

    def __init__(self, *args, **kwargs):
        import cv2

        self.event = None
        self._video_capture = None
        self._video_fps = 0.0
        self._next_frame_idx = 0
        self._video_finished = False
        self._audio_path = None
        self._audio_started = False
        self._playback_t0 = None
        super().__init__(*args, **kwargs)

        # Extract video metadata and prepare for playback
        media_file = os.path.expanduser(self.media_file)
        if not media_file:
            raise ValueError("media_file must be set to a valid local video path")
        if not os.path.isfile(media_file):
            raise IOError("media_file does not exist: %s" % media_file)

        self._video_capture = cv2.VideoCapture(media_file)
        if not self._video_capture.isOpened():
            raise RuntimeError("Could not open video file: %s" % media_file)
        self._video_fps = float(self._video_capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if self._video_fps <= 0:
            self._video_fps = float(self.fps)
        # Aspect ratio
        width = int(self._video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width > 0 and height > 0:
            self.video_aspect = width / height
        else:
            self.video_aspect = self.screen_cm[0] / self.screen_cm[1]

        # Extract audio
        self._audio_path = self._extract_audio_track(media_file)



    def init(self):
        super().init()
        
        # Create a fullscreen plane on which to display the video frames
        video_size = (int(self.screen_cm[0]), int(self.screen_cm[0] / self.video_aspect))
        self.tex = Texture(
            np.zeros((video_size[1], video_size[0], 3), dtype=np.uint8),
            size=(video_size[0], video_size[1]),
            iformat=GL_RGB8,
            exformat=GL_RGB,
            dtype=GL_UNSIGNED_BYTE,
        )
        self.video_surface = TexPlane(self.screen_cm[0], self.screen_cm[1], tex=self.tex, specular_color=(0,0,0,0))
        self.video_surface.rotate_x(90).translate(-self.screen_cm[0]/2, 0, -self.screen_cm[1]/2)
        self.add_model(self.video_surface)

    def _extract_audio_track(self, media_file):
        audio_fd, audio_path = tempfile.mkstemp(prefix="bmi3d_video_audio_", suffix=".wav")
        os.close(audio_fd)

        command = [
            "ffmpeg",
            "-y",
            "-i", media_file,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            audio_path,
        ]
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0 or not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            try:
                os.remove(audio_path)
            except OSError:
                pass
            return None
        return audio_path

    def _set_video_frame(self, frame):
        import cv2

        frame_rgb = np.flipud(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        # if (frame_rgb.shape[1], frame_rgb.shape[0]) != self.window_size:
        #     frame_rgb = cv2.resize(frame_rgb, self.window_size)

        # Create a new texture and assign it to the video surface
        self.tex = Texture(
            frame_rgb,
            size=(frame_rgb.shape[1], frame_rgb.shape[0]),
            iformat=GL_RGB8,
            exformat=GL_RGB,
            dtype=GL_UNSIGNED_BYTE,
        )
        self.video_surface.replace_texture(self.tex)

    def _start_wait(self):
        import cv2
        import pygame

        ok, first_frame = self._video_capture.read()
        if not ok:
            raise RuntimeError("Could not decode first frame from video")

        self._next_frame_idx = 1
        self._set_video_frame(first_frame)

        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(44100, -16, 2, 2048)
            pygame.mixer.init()

        if self._audio_path is not None:
            pygame.mixer.music.load(self._audio_path)
            pygame.mixer.music.set_volume(np.clip(self.audio_volume, 0.0, 1.0))
            pygame.mixer.music.play(loops=0)
            self._audio_started = True
        else:
            self._audio_started = False
            self._playback_t0 = self.get_time()

    def stop_video(self):
        import pygame

        if pygame.mixer.get_init():
            try:
                pygame.mixer.music.stop()
                if hasattr(pygame.mixer.music, "unload"):
                    pygame.mixer.music.unload()
            except Exception:
                pass

        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None

        if self._audio_path is not None and os.path.exists(self._audio_path):
            try:
                os.remove(self._audio_path)
            except OSError:
                pass
            self._audio_path = None

    def _test_stop(self, ts):
        return self._video_finished or super()._test_stop(ts)

    def _start_None(self):
        super()._start_None()
        self.stop_video()

    def _target_frame_idx(self):
        import pygame

        if self._audio_started:
            pos_ms = pygame.mixer.music.get_pos()
            if pos_ms < 0:
                self._video_finished = True
                return self._next_frame_idx
            return int((pos_ms / 1000.0) * self._video_fps)

        if self._playback_t0 is None:
            self._playback_t0 = self.get_time()
        elapsed = self.get_time() - self._playback_t0
        return int(elapsed * self._video_fps)

    def _cycle(self):
        super()._cycle()

        if self._video_finished:
            return

        target_idx = self._target_frame_idx()

        while self._next_frame_idx <= target_idx and not self._video_finished:
            ok, frame = self._video_capture.read()
            if not ok:
                self._video_finished = True
                break
            self._set_video_frame(frame)
            self._next_frame_idx += 1

        self.draw_world()
