#version 330 core

uniform mat4 p_matrix;
uniform mat4 xfm;
uniform mat4 modelview;
uniform vec4 basecolor;
uniform float shininess;

in vec4 position;
in vec2 texcoord;
in vec4 normal;

out vec3 vposition;
out vec3 vnormal;
out vec2 vtexcoord;
out float vshininess;
out mat4 transform;

void main(void) {
    vec4 eye_position = xfm * position;
    gl_Position = p_matrix * eye_position;
    
    vposition = eye_position.xyz;
    vnormal   = (xfm * vec4(normal.xyz, 0.0)).xyz;
    vtexcoord = texcoord;
    vshininess = shininess;
    transform = xfm;
}
