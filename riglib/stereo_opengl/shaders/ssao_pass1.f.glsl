#version 330 core

in vec3 vnormal;

out vec4 normal_out;

void main() {
    // Output view-space normal
    normal_out = vec4(normalize(vnormal), 1.0);
}