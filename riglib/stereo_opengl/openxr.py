import time
import os
import numpy as np
from ctypes import byref, cast, POINTER
from OpenGL.GL import *
import pygame
try:
    import xr
except:
    print("No OpenXR support")
    
from ..experiment import traits
from .render import stereo, render, ssao, shadow_map
from .models import Group
from .xfm import Quaternion, Transform
from .window import Window
from .environment import Grid

class Clock():

    def __init__(self, n_ticks=10):
        self.start_time = self.get_time()
        self.prev_ticks = np.zeros((n_ticks,))

    def tick(self, fps):
        self.prev_ticks = np.roll(self.prev_ticks, 1)
        self.prev_ticks[0] = self.get_time()

    def get_time(self):
        return time.perf_counter()
    
    def get_fps(self):
        return -1/np.mean(np.diff(self.prev_ticks))
    
class WindowVR(Window):
    '''
    An OpenXR window for rendering in VR to an HMD
    '''
    
    show_grid = traits.Bool(True, desc="Show a textured grid on the floor")
    grid_size = traits.Float(130, desc="Size of the grid in cm")
    grid_position = traits.Tuple((0, 0, 0), desc="Position of the grid in cm. If you want the floor of the grid to be on the floor of the world, set the z component to (grid_size - camera_offset[2])")
    camera_offset = traits.Tuple((0, -130, 40), desc="Offset virtual screen to the camera in cm")
    camera_position = traits.Tuple((0.0, 0.0, -40.0), desc="Absolute position of the camera (x,y,z) in cm world coordinates. Only used if fixed_camera_position is True")
    camera_orientation = traits.Tuple((1.0, 0.0, 0.0, 0.0), desc="Orientation of the camera (w, x, y, z) as a quaternion. Only used if fixed_camera_orientation is True")
    fixed_camera_position = traits.Bool(False, desc="Fixed position of the camera")
    fixed_camera_orientation = traits.Bool(False, desc="Fixed orientation of the camera")
    xr_runtime_json = traits.String("", desc="Optional path to OpenXR runtime JSON. If empty, uses XR_RUNTIME_JSON from the environment")
    swapchain_color_format = traits.OptionsList("auto", "srgb8a8", "rgba8", desc="Preferred OpenXR swapchain color format")
    xr_ignore_window_close = traits.Bool(True, desc="Ignore hidden OpenXR helper-window close signal to keep frame loop running")
    setup_steamvr_env = traits.Bool(True, desc="Set SteamVR helper environment paths for stable runtime startup")

    hidden_traits = ['fps', 'window_size', 'screen_dist']

    def init(self):
        self.add_dtype('view_pose_position', 'f8', (2,3))
        self.add_dtype('view_pose_rotation', 'f8', (2,4))
        self.add_dtype('modelview', 'f8', (2,4,4))
        self.add_dtype('camera_position', 'f8', (3,))
        self.add_dtype('camera_orientation', 'f8', (4,))
        super().init()

    def screen_init(self):
        from ctypes import byref, c_int32, c_void_p, cast, POINTER, pointer, Structure

        if self.setup_steamvr_env:
            self._setup_steamvr_env()
        if self.xr_runtime_json:
            os.environ['XR_RUNTIME_JSON'] = self.xr_runtime_json
        pygame.init()
        self.clock = Clock()
        self.fps = 90

        context = xr.ContextObject(
            instance_create_info=xr.InstanceCreateInfo(
                enabled_extension_names=[
                    xr.KHR_OPENGL_ENABLE_EXTENSION_NAME,
                ],
            ),
            reference_space_create_info=xr.ReferenceSpaceCreateInfo(
                reference_space_type=xr.ReferenceSpaceType.STAGE,
                pose_in_reference_space=xr.Posef((0,0,0,1), (0,0,0)),
            ),
            
        )
        # context.__enter__()

        '''
        Ideally we would use the context manager here, but it uses the default
        swapchain image format, which is not guaranteed to be an SRGB format.
        There might be a better way to handle this but for now this works.
        '''
        context.instance = xr.create_instance(
            create_info=context._instance_create_info,
        )
        context.system_id = xr.get_system(
            instance=context.instance,
            get_info=xr.SystemGetInfo(
                form_factor=context.form_factor,
            ),
        )

        if context._session_create_info.next is None:
            context.graphics = xr.OpenGLGraphics(
                instance=context.instance,
                system=context.system_id,
                title=context._instance_create_info.application_info.application_name.decode()
            )
            context.graphics_binding_pointer = cast(pointer(context.graphics.graphics_binding), c_void_p)
            context._session_create_info.next = context.graphics_binding_pointer
        else:
            context.graphics_binding_pointer = context._session_create_info.next

        context._session_create_info.system_id = context.system_id
        context.session = xr.create_session(
            instance=context.instance,
            create_info=context._session_create_info,
        )
        context.space = xr.create_reference_space(
            session=context.session,
            create_info=context._reference_space_create_info
        )
        context.default_action_set = xr.create_action_set(
            instance=context.instance,
            create_info=xr.ActionSetCreateInfo(
                action_set_name="default_action_set",
                localized_action_set_name="Default Action Set",
                priority=0,
            ),
        )
        context.action_sets.append(context.default_action_set)

        # Create swapchains
        config_views = xr.enumerate_view_configuration_views(
            instance=context.instance,
            system_id=context.system_id,
            view_configuration_type=context.view_configuration_type,
        )
        context.graphics.initialize_resources()
        swapchain_formats = xr.enumerate_swapchain_formats(context.session)
        runtime_default_format = context.graphics.select_color_swapchain_format(swapchain_formats)

        if self.swapchain_color_format == "srgb8a8":
            preferred_formats = [GL_SRGB8_ALPHA8]
        elif self.swapchain_color_format == "rgba8":
            preferred_formats = [GL_RGBA8]
        else:
            # Prefer sRGB when supported, otherwise use the runtime's preferred format.
            preferred_formats = [GL_SRGB8_ALPHA8, runtime_default_format, GL_RGBA8]

        color_swapchain_format = None
        for fmt in preferred_formats:
            if fmt in swapchain_formats:
                color_swapchain_format = fmt
                break
        if color_swapchain_format is None:
            color_swapchain_format = runtime_default_format

        self._xr_swapchain_srgb = color_swapchain_format == GL_SRGB8_ALPHA8
        # Create a swapchain for each view.
        context.swapchains.clear()
        context.swapchain_image_buffers.clear()
        context.swapchain_image_ptr_buffers.clear()
        for vp in config_views:
            # Create the swapchain.
            swapchain_create_info = xr.SwapchainCreateInfo(
                array_size=1,
                format=color_swapchain_format,
                width=vp.recommended_image_rect_width,
                height=vp.recommended_image_rect_height,
                mip_count=1,
                face_count=1,
                sample_count=vp.recommended_swapchain_sample_count,
                usage_flags=xr.SwapchainUsageFlags.SAMPLED_BIT | xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT,
            )
            swapchain = xr.context_object.SwapchainStruct(
                xr.create_swapchain(
                    session=context.session,
                    create_info=swapchain_create_info,
                ),
                swapchain_create_info.width,
                swapchain_create_info.height,
            )
            context.swapchains.append(swapchain)
            swapchain_image_buffer = xr.enumerate_swapchain_images(
                swapchain=swapchain.handle,
                element_type=context.graphics.swapchain_image_type,
            )
            # Keep the buffer alive by moving it into the list of buffers.
            context.swapchain_image_buffers.append(swapchain_image_buffer)
            capacity = len(swapchain_image_buffer)
            swapchain_image_ptr_buffer = (POINTER(xr.SwapchainImageBaseHeader) * capacity)()
            for ix in range(capacity):
                swapchain_image_ptr_buffer[ix] = cast(
                    byref(swapchain_image_buffer[ix]),
                    POINTER(xr.SwapchainImageBaseHeader))
            context.swapchain_image_ptr_buffers.append(swapchain_image_ptr_buffer)
        context.graphics.make_current()
        '''
        End of context initialization
        '''

        # Query the swapchain size
        config_views = xr.enumerate_view_configuration_views(
            instance=context.instance,
            system_id=context.system_id,
            view_configuration_type=context.view_configuration_type,
        )
        self.window_size = (
            config_views[0].recommended_image_rect_width * 2,
            config_views[0].recommended_image_rect_height)

        glDisable(GL_FRAMEBUFFER_SRGB)
        glEnable(GL_BLEND)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(*self.background)
        glClearDepth(1.0)
        glDepthMask(GL_TRUE)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        self.renderer = self._get_renderer()

        if self.show_grid:
            self.add_model(Grid(self.grid_size*2).translate(self.grid_position[0], self.grid_position[1], self.grid_position[2]))
        self.world = Group(self.models)
        self.world.init()
        self.set_eye((0,0,0), (0,0))
        self.xr_frame_generator = context.frame_loop()
        self.xr_context = context
        if self.xr_ignore_window_close:
            context.graphics.poll_events = lambda: False
        print("Initialized OpenXR window")
        print(f"OpenXR swapchain format: {color_swapchain_format} (sRGB={self._xr_swapchain_srgb})")

    def _get_renderer(self):
        near = 1
        far = 1024
        if self.stereo_mode == 'mirror':
            glFrontFace(GL_CW);  # Switch to clockwise winding for mirrored objects
        return shadow_map.ShadowMapper(self.window_size, self.fov, near, far)

    def _setup_steamvr_env(self):
        steamvr_root = os.path.expanduser('~/.local/share/Steam/steamapps/common/SteamVR')
        vrenv = os.path.join(steamvr_root, 'bin', 'vrenv.sh')
        if os.path.exists(vrenv):
            os.environ.setdefault('STEAMVR_VRENV', vrenv)

        # Reduce hard process termination when auxiliary SteamVR helpers fail.
        os.environ.setdefault('STEAMVR_DISABLE_THREAD_WATCHDOGS', '1')
        os.environ.setdefault('STEAMVR_DISABLE_ASSERT_MINIDUMP', '1')
        os.environ.setdefault('STEAMVR_DISABLE_CRASH_REPORTING', '1')

        candidates = [
            os.path.join(steamvr_root, 'bin', 'linux64'),
            os.path.join(steamvr_root, 'bin', 'linux64', 'qt', 'lib'),
            os.path.join(steamvr_root, 'bin', 'vrwebhelper', 'linux64'),
            os.path.join(steamvr_root, 'tools', 'lighthouse', 'bin', 'linux64'),
        ]
        existing = [p for p in os.environ.get('LD_LIBRARY_PATH', '').split(':') if p]

        # Prevent conda/OpenCV libs from shadowing SteamVR's bundled runtime libs.
        filtered = []
        for p in existing:
            pl = p.lower()
            if 'miniconda' in pl or 'anaconda' in pl or 'site-packages/cv2' in pl:
                continue
            filtered.append(p)

        for p in reversed(candidates):
            if os.path.isdir(p) and p not in filtered:
                filtered.insert(0, p)

        os.environ['LD_LIBRARY_PATH'] = ':'.join(filtered)

    def _view_loop_retry(self, frame_state):
        if not frame_state.should_render:
            return

        view_state, views = xr.locate_views(
            session=self.xr_context.session,
            view_locate_info=xr.ViewLocateInfo(
                view_configuration_type=self.xr_context.view_configuration_type,
                display_time=frame_state.predicted_display_time,
                space=self.xr_context.space,
            )
        )
        num_views = len(views)
        projection_layer_views = tuple(xr.CompositionLayerProjectionView() for _ in range(num_views))

        vsf = view_state.view_state_flags
        if ((vsf & xr.VIEW_STATE_POSITION_VALID_BIT == 0)
                or (vsf & xr.VIEW_STATE_ORIENTATION_VALID_BIT == 0)):
            if not hasattr(self, "_warned_invalid_view_state"):
                self._warned_invalid_view_state = False
            if not self._warned_invalid_view_state:
                print(f"OpenXR warning: view state flags invalid ({int(vsf)}). Rendering anyway.")
                self._warned_invalid_view_state = True

        for view_index, view in enumerate(views):
            view_swapchain = self.xr_context.swapchains[view_index]
            swapchain_image_index = xr.acquire_swapchain_image(
                swapchain=view_swapchain.handle,
                acquire_info=xr.SwapchainImageAcquireInfo(),
            )
            xr.wait_swapchain_image(
                swapchain=view_swapchain.handle,
                wait_info=xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION),
            )

            layer_view = projection_layer_views[view_index]
            layer_view.pose = view.pose
            layer_view.fov = view.fov
            layer_view.sub_image.swapchain = view_swapchain.handle
            layer_view.sub_image.image_rect.offset[:] = [0, 0]
            layer_view.sub_image.image_rect.extent[:] = [
                view_swapchain.width, view_swapchain.height,
            ]

            swapchain_image_ptr = self.xr_context.swapchain_image_ptr_buffers[view_index][swapchain_image_index]
            swapchain_image = cast(swapchain_image_ptr, POINTER(xr.SwapchainImageOpenGLKHR)).contents
            color_texture = swapchain_image.image

            try:
                self.xr_context.graphics.begin_frame(layer_view, color_texture)
            except Exception:
                # Some runtimes can invalidate/recreate GL objects; rebuild and retry once.
                self.xr_context.graphics.make_current()
                self.xr_context.graphics.swapchain_framebuffer = glGenFramebuffers(1)
                while glGetError() != GL_NO_ERROR:
                    pass
                self.xr_context.graphics.begin_frame(layer_view, color_texture)

            yield view_index, view

            self.xr_context.graphics.end_frame()
            xr.release_swapchain_image(
                swapchain=view_swapchain.handle,
                release_info=xr.SwapchainImageReleaseInfo(),
            )

        layer = xr.CompositionLayerProjection(space=self.xr_context.space)
        layer.views = projection_layer_views
        self.xr_context.render_layers.append(byref(layer))
    
    def draw_world(self):
        graphics = self.xr_context.graphics
        graphics.make_current()
        swapchain_fbo = getattr(graphics, "swapchain_framebuffer", None)
        if not swapchain_fbo or not glIsFramebuffer(swapchain_fbo):
            graphics.initialize_resources()
        # Clear any stale GL errors before OpenXR's per-view begin_frame calls.
        while glGetError() != GL_NO_ERROR:
            pass

        # Get the OpenXR views
        try:
            frame_state = next(self.xr_frame_generator)
        except StopIteration:
            return

        for view_index, view in self._view_loop_retry(frame_state):
            projection = xr.Matrix4x4f.create_projection_fov(
                graphics_api=xr.GraphicsAPI.OPENGL,
                fov=view.fov,
                near_z=0.05,
                far_z=1024,
            ).as_numpy().reshape(4,4).T

            if self.fixed_camera_position:
                position = np.array(self.camera_position, dtype=float) - np.array([1,0,0])*self.iod*(view_index-0.5)
            else:
                position = -np.array([
                    view.pose.position[0]*100 + self.camera_offset[0],
                    view.pose.position[1]*100 + self.camera_offset[1],
                    view.pose.position[2]*100 + self.camera_offset[2],
                ])
                self.camera_position = tuple(position + np.array([1,0,0])*self.iod*(view_index-0.5))
            if self.fixed_camera_orientation:
                rotation = self.camera_orientation
            else:
                rotation = np.array([
                    view.pose.orientation.w,
                    -view.pose.orientation.x,
                    -view.pose.orientation.y,
                    -view.pose.orientation.z,
                ])
                self.camera_orientation = tuple(rotation)
            xfm = Transform(move=position, rotate=Quaternion(*rotation)) 
            self.modelview = xfm.to_mat(reverse=True)

            # Optionally mirror the view along the y-axis
            if self.stereo_mode == 'mirror':
                self.modelview = np.dot(self.modelview, np.diag([-1,1,1,1]))

            # Draw the world
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.renderer.draw(self.world, p_matrix=projection, modelview=self.modelview)
        
            # Save the per-eye pose data
            if hasattr(self, 'task_data'):
                self.task_data['view_pose_position'][:,view_index,:] = position
                self.task_data['view_pose_rotation'][:,view_index,:] = rotation
                self.task_data['modelview'][:,view_index] = self.modelview

        try:
            graphics.make_current()
            self.renderer.draw_done()
        except Exception as exc:
            if not hasattr(self, '_warned_draw_done_error'):
                self._warned_draw_done_error = False
            if not self._warned_draw_done_error:
                print(f"OpenXR warning: draw_done failed ({exc}). Continuing.")
                self._warned_draw_done_error = True

        # Save the cylopian pose data
        if hasattr(self, 'task_data'):
            self.task_data['camera_position'] = self.camera_position
            self.task_data['camera_orientation'] = self.camera_orientation

    def _test_stop(self, ts):
        super_stop = super(Window, self)._test_stop(ts)
        return super_stop

    def _start_None(self):
        self.xr_context.__exit__(None, None, None)
        super(WindowVR, self)._start_None()
