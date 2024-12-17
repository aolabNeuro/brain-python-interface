#version 330 core

vec4 phong(); // Forward declaration of the phong function

void main() {
    FragColor = phong();
    // FragColor = vec4(vnormal, 1.0); // useful visualization for debugging triangle meshes
}
