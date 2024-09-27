#version 330 core

in vec4 position;

out vec2 uv;

void main(void) {
    gl_Position = position;
    uv = (vec2(position.x, position.y) + vec2(1.0)) * 0.5;
}