#version 330 core

vec4 phong(); // Forward declaration of the phong function

void main() {
    FragColor = phong();
}