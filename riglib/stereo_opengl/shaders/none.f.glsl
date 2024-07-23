#version 110

uniform vec4 basecolor;
uniform sampler2D texture;

varying vec2 vtexcoord;

void main() {
    vec4 frag_diffuse = vec4(texture2D(texture, vtexcoord).rgb + basecolor.rgb, basecolor.a);
    gl_FragColor = frag_diffuse;
}
