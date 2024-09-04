import inspect
from OpenGL import GL
from OpenGL.GL.shaders import compileShader, compileProgram
import xr

from riglib.stereo_opengl.utils import offaxis_frusta

def initialize():
    # Create OpenXR context
    context = xr.ContextObject(
        instance_create_info=xr.InstanceCreateInfo(
            enabled_extension_names=[
                xr.KHR_OPENGL_ENABLE_EXTENSION_NAME,
            ],
        ),
    )
    context.__enter__()
    
    # Compile shaders
    vertex_shader = compileShader(
        inspect.cleandoc("""
        #version 430
        
        // Adapted from @jherico's RiftDemo.py in pyovr
        
        /*  Draws a cube:
        
           2________ 3
           /|      /|
         6/_|____7/ |
          | |_____|_| 
          | /0    | /1
          |/______|/
          4       5          

         */

        layout(location = 0) uniform mat4 Projection = mat4(1);
        layout(location = 4) uniform mat4 ModelView = mat4(1);
        layout(location = 8) uniform float Size = 0.3;

        // Minimum Y value is zero, so cube sits on the floor in room scale
        const vec3 UNIT_CUBE[8] = vec3[8](
          vec3(-1.0, -0.0, -1.0), // 0: lower left rear
          vec3(+1.0, -0.0, -1.0), // 1: lower right rear
          vec3(-1.0, +2.0, -1.0), // 2: upper left rear
          vec3(+1.0, +2.0, -1.0), // 3: upper right rear
          vec3(-1.0, -0.0, +1.0), // 4: lower left front
          vec3(+1.0, -0.0, +1.0), // 5: lower right front
          vec3(-1.0, +2.0, +1.0), // 6: upper left front
          vec3(+1.0, +2.0, +1.0)  // 7: upper right front
        );

        const vec3 UNIT_CUBE_NORMALS[6] = vec3[6](
          vec3(0.0, 0.0, -1.0),
          vec3(0.0, 0.0, 1.0),
          vec3(1.0, 0.0, 0.0),
          vec3(-1.0, 0.0, 0.0),
          vec3(0.0, 1.0, 0.0),
          vec3(0.0, -1.0, 0.0)
        );

        const int CUBE_INDICES[36] = int[36](
          0, 1, 2, 2, 1, 3, // rear
          4, 6, 5, 6, 7, 5, // front
          0, 2, 4, 4, 2, 6, // left
          1, 3, 5, 5, 3, 7, // right
          2, 6, 3, 6, 3, 7, // top
          0, 1, 4, 4, 1, 5  // bottom
        );

        out vec3 _color;

        void main() {
          _color = vec3(1.0, 0.0, 0.0);
          int vertexIndex = CUBE_INDICES[gl_VertexID];
          int normalIndex = gl_VertexID / 6;

          _color = UNIT_CUBE_NORMALS[normalIndex];
          if (any(lessThan(_color, vec3(0.0)))) {
              _color = vec3(1.0) + _color;
          }

          gl_Position = Projection * ModelView * vec4(UNIT_CUBE[vertexIndex] * Size, 1.0);
        }
        """), GL.GL_VERTEX_SHADER)
    fragment_shader = compileShader(
        inspect.cleandoc("""
        #version 430
        
        in vec3 _color;
        out vec4 FragColor;

        void main() {
          FragColor = vec4(_color, 1.0);
        }
        """), GL.GL_FRAGMENT_SHADER)
    shader = compileProgram(vertex_shader, fragment_shader)
    vao = GL.glGenVertexArrays(1)
    GL.glEnable(GL.GL_BLEND)
    GL.glDepthFunc(GL.GL_LESS)
    GL.glEnable(GL.GL_DEPTH_TEST)
    # glEnable(GL_TEXTURE_2D)
    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
    GL.glClearColor(0., 0., 0., 1.)
    GL.glClearDepth(1.0)
    GL.glDepthMask(GL.GL_TRUE)
    GL.glEnable(GL.GL_CULL_FACE) # temporary solution to alpha blending issue with spheres. just draw the front half of the sphere
    GL.glCullFace(GL.GL_BACK)
    
    return context, shader, vao

def draw_view(view, shader, vao):
    projection = xr.Matrix4x4f.create_projection_fov(
        graphics_api=xr.GraphicsAPI.OPENGL,
        fov=view.fov,
        near_z=0.05,
        far_z=100.0,
    )
    to_view = xr.Matrix4x4f.create_translation_rotation_scale(
        translation=view.pose.position,
        rotation=view.pose.orientation,
        scale=(1, 1, 1),
    )
    view_matrix = xr.Matrix4x4f.invert_rigid_body(to_view)

    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glUseProgram(shader)
    GL.glUniformMatrix4fv(0, 1, False, projection.as_numpy())
    GL.glUniformMatrix4fv(4, 1, False, view_matrix.as_numpy())
    GL.glBindVertexArray(vao)
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, 36)

def frame_loop(context, shader, vao):
    frame_generator = context.frame_loop()

    while True:
        # Get the next frame state
        try:
            frame_state = next(frame_generator)
        except StopIteration:
            break  # Exit loop if the generator is exhausted

        # Handle non-graphics related computations here if needed
        # e.g., input handling, physics calculations, etc.

        for view_index, view in enumerate(context.view_loop(frame_state)):
            draw_view(view, shader, vao)

        # Early exit condition if needed
        # if some_condition:
        #     break

def main():
    # Initialization
    context, shader, vao = initialize()
    
    # Main loop (frame handling)
    frame_loop(context, shader, vao)
    
    # Cleanup if needed
    # context.destroy()
    # GL.glDeleteVertexArrays(1, vao)
    context.__exit__()


if __name__ == "__main__":
    main()
