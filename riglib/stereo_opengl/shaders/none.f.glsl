#version 110

uniform vec4 basecolor;
uniform sampler2D texture;

varying vec2 vtexcoord;

void main() {
    vec4 texcolor = texture2D(texture, vtexcoord);
    vec4 frag_diffuse = vec4(
        texcolor.rgb + basecolor.rgb,
        texcolor.a * basecolor.a
    );
    gl_FragColor = frag_diffuse;
}
