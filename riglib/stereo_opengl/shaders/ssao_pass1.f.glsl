#version 120

varying vec3 vnormal;
varying vec3 vposition;

uniform float near_clip;
uniform float far_clip;

void main() {
    // Output view-space normal
    gl_FragData[0] = vec4(normalize(vnormal), 1.0);
    
    // Calculate and output linear depth
    float depth = length(vposition);
    float linear_depth = (depth - near_clip) / (far_clip - near_clip);
    gl_FragData[1] = vec4(linear_depth, linear_depth, linear_depth, 1.0);
}