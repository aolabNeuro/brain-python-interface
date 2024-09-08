#version 330 core

in vec3 vnormal;

uniform float near_clip;
uniform float far_clip;

out vec4 normal_out;

void main() {
    // Output view-space normal
    normal_out = vec4(normalize(vnormal), 1.0);
}