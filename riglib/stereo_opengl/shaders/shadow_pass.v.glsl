#version 330 core

uniform mat4 light_space_matrix;
uniform mat4 xfm;

layout (location = 0) in vec3 position;

void main()
{
    vec4 world_pos = xfm * vec4(position, 1.0);
    gl_Position = light_space_matrix * world_pos;
}