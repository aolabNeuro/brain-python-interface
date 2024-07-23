#version 110

uniform mat4 p_matrix;
uniform mat4 xfm;
uniform vec4 basecolor;

attribute vec4 position;
attribute vec2 texcoord;

varying vec2 vtexcoord;

void main(void) {
    vec4 eye_position = xfm * position;
    vtexcoord = texcoord;

    gl_Position = p_matrix * eye_position;
}
