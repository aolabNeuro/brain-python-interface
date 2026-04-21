#version 330 core

uniform mat4 p_matrix;
uniform mat4 xfm;
uniform mat4 modelview;
uniform vec4 basecolor;

in vec4 position;
in vec2 texcoord;

out vec2 vtexcoord;

void main(void) {
    vec4 eye_position = modelview * xfm * position;
    vtexcoord = texcoord;

    gl_Position = p_matrix * eye_position;
}
