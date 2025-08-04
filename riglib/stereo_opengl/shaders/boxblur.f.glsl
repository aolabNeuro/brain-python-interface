#version 330 core

uniform sampler2D tex;
uniform vec2 gaze_uv;
uniform float fovea_radius;
uniform int max_kernel; // e.g. 7 for a 7x7 kernel

in vec2 uv;
out vec4 FragColor;

vec4 variable_blur(sampler2D tex, vec2 uv, int kernel) {
    vec2 texel = 1.0 / textureSize(tex, 0);
    vec4 sum = vec4(0.0);
    int count = 0;
    for (int x = -kernel; x <= kernel; ++x)
        for (int y = -kernel; y <= kernel; ++y) {
            sum += texture(tex, uv + vec2(x, y) * texel);
            count++;
        }
    return sum / float(count);
}

void main() {
    float dist = distance(uv, gaze_uv);
    float t = smoothstep(fovea_radius, 1.0, dist);
    float kernel_f = mix(0.0, float(max_kernel), t);

    int kernel_lo = int(floor(kernel_f));
    int kernel_hi = int(ceil(kernel_f));
    float frac = kernel_f - float(kernel_lo);

    vec4 color_lo = (kernel_lo == 0) ? texture(tex, uv) : variable_blur(tex, uv, kernel_lo);
    vec4 color_hi = (kernel_hi == 0) ? texture(tex, uv) : variable_blur(tex, uv, kernel_hi);

    FragColor = mix(color_lo, color_hi, frac);
}