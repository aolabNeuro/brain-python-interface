#version 330 core

uniform mat4 p_matrix;
uniform mat4 xfm;
uniform mat4 modelview;
uniform mat4 light_space_matrix;
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
out vec4 frag_pos_light_space;

void main(void) {
    vec4 eye_position = modelview * xfm * position;
    gl_Position = p_matrix * eye_position;
    
    vposition = eye_position.xyz;
    vnormal   = (xfm * vec4(normal.xyz, 0.0)).xyz;
    vtexcoord = texcoord;
    vshininess = shininess;
    transform = inverse(modelview);
    frag_pos_light_space = light_space_matrix * xfm * position;
}
