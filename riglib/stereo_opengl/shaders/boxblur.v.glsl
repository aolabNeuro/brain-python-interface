#version 330 core

in vec2 position;
out vec2 uv;

void main(void) {
    uv = position * 0.5 + 0.5; // map [-1,1] to [0,1]
    gl_Position = vec4(position, 0.0, 1.0);
}