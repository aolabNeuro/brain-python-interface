#version 330 core

uniform vec4 basecolor;
uniform sampler2D textureSampler;

in vec2 vtexcoord;

out vec4 FragColor;

void main() {
    vec4 texcolor = texture(textureSampler, vtexcoord);
    vec4 frag_diffuse = vec4(
        texcolor.rgb + basecolor.rgb,
        texcolor.a * basecolor.a
    );
    FragColor = frag_diffuse;
}
