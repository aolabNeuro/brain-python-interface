from OpenGL.GL import *
from matplotlib import projections
from .fbo import FBO, FBOrender
from ..utils import look_at, orthographic, perspective
import numpy as np

class ShadowMapper(FBOrender):
    def __init__(self, *args, blur=False, fovea_radius=2, gaze_world=(0,0,0), kernel_size=13, **kwargs):
        super(ShadowMapper, self).__init__(*args, **kwargs)
        self.size = (1024, 1024)
        self.blur_enabled = blur
        self.fovea_radius_cm = fovea_radius
        self.gaze_world = gaze_world
        self.kernel_size = kernel_size
        
        # Create shadow map FBO
        self.shadow_map = FBO(["depth"], size=self.size)

        # Add shaders for shadow pass
        self.add_shader("shadow_pass_v", GL_VERTEX_SHADER, "shadow_pass.v.glsl")
        self.add_shader("shadow_pass_f", GL_FRAGMENT_SHADER, "shadow_pass.f.glsl")
        self.add_program("shadow_pass", ("shadow_pass_v", "shadow_pass_f"))

        self.add_shader("fsquad", GL_VERTEX_SHADER, "fsquad.v.glsl")
        self.add_shader("fsquad_frag", GL_FRAGMENT_SHADER, "fsquad.f.glsl")
        self.add_program("none", ("fsquad", "fsquad_frag"))

        # Calculate light space matrix
        light_target = [0, 0, 0]
        light_up = [0, 1, 0]
        near_plane = 25.0
        far_plane = 100.0
        light_projection = perspective(75, 1, near_plane, far_plane)
        light_view = look_at(-50*np.array(self.light_direction[:3])/np.linalg.norm(self.light_direction[:3]), light_target, light_up)
        self.light_space_matrix = np.dot(light_projection, light_view)

        # Blur pass setup
        if self.blur_enabled:
            self.blur_fbo = FBO(["color0"], size=self.size)
            self.add_shader("blur_passthru", GL_VERTEX_SHADER, "boxblur.v.glsl")
            self.add_shader("blur", GL_FRAGMENT_SHADER, "boxblur.f.glsl")
            self.add_program("blur", ("blur_passthru", "blur"))


    def generate_shadow_map(self, root, **kwargs):
        # Save current viewport and framebuffer
        original_viewport = glGetIntegerv(GL_VIEWPORT)
        original_framebuffer = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
        original_winding_order = glGetIntegerv(GL_FRONT_FACE)

        # Set viewport for shadow map
        glViewport(0, 0, self.size[0], self.size[1])
        glFrontFace(GL_CCW)
        glCullFace(GL_FRONT)

        # Render shadow map
        self.draw_to_fbo(self.shadow_map, root, shader="shadow_pass", apply_default=True, 
                         light_space_matrix=self.light_space_matrix, **kwargs)

        # Restore original viewport and framebuffer
        glFrontFace(original_winding_order)
        glViewport(*original_viewport)
        glBindFramebuffer(GL_FRAMEBUFFER, original_framebuffer)
        glCullFace(GL_BACK)

        return self.shadow_map['depth'], self.light_space_matrix
    
    def draw(self, root, **kwargs):
        # Generate shadow map
        shadow_map, light_space_matrix = self.generate_shadow_map(root, **kwargs)

        # Main scene render to FBO
        if self.blur_enabled:
            original_viewport = glGetIntegerv(GL_VIEWPORT)
            glViewport(0, 0, self.size[0], self.size[1])
            self.draw_to_fbo(self.blur_fbo, root, light_space_matrix=light_space_matrix,
                             shadow_map=shadow_map, **kwargs)
            glViewport(*original_viewport)

            # Calculate gaze_uv and fovea_radius_uv
            p_matrix = kwargs.get('p_matrix', self.projection)
            modelview = kwargs.get('modelview', self.modelview)
            gaze_uv = p_matrix @ modelview @ np.array(self.gaze_world + (1,))
            gaze_uv = gaze_uv[:2] / gaze_uv[3]
            u, v = ((gaze_uv + 1) / 2).astype(float)
            gaze_uv = (float(u), float(v))
            fovea_radius_uv = float(self.fovea_radius_cm / original_viewport[2])

            # Blur pass
            self.draw_fsquad("blur", tex=self.blur_fbo['color0'], gaze_uv=gaze_uv,
                             fovea_radius=fovea_radius_uv, max_kernel=self.kernel_size, **kwargs)
        else:
            # Draw the scene with shadows directly
            super().draw(root, light_space_matrix=light_space_matrix,
                         shadow_map=shadow_map, **kwargs)