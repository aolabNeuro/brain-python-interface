#version 330 core

uniform sampler2D shadow;
uniform vec4 window;

vec4 phong(); // Forward declaration

void main() {
    vec2 uv = (gl_FragCoord.xy - window.xy) / window.zw;
    float shade = texture(shadow, uv).r;
    vec4 phongColor = phong();
    FragColor = vec4(phongColor.rgb * shade, phongColor.a);
}