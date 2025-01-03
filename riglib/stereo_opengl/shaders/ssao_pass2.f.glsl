#version 330 core

uniform sampler2D rnm;
uniform sampler2D normalMap;
uniform sampler2D depthMap;

uniform float nearclip;
uniform float farclip;

in vec2 uv;
const float totStrength = 0.5;
const float strength = 1.5;
const float falloff = 0.001;
const float rad = .005;
#define SAMPLES 64
const float invSamples = -totStrength/float(SAMPLES);

out vec4 FragColor;

float lindepth(vec2 uv) {
   float n = nearclip; // camera z near
   float f = farclip; // camera z far
   float z = texture(depthMap, uv).x;
   return (2.0 * n) / (f + n - z * (f - n));
}

// Function to generate a random vector
vec3 randomVec(vec2 uv) {
    vec3 noise = texture(rnm, uv).xyz;
    return normalize(noise * 2.0 - 1.0);
}

void main(void) {
    vec3 randVec = randomVec(uv);
    
    vec3 norm = normalize(texture(normalMap,uv).xyz);
    float depth = lindepth(uv);

    // current fragment coords in screen space
    vec3 ep = vec3(uv.xy, depth);

    float bl = 0.0;
    float radD = rad/depth;

    for(int i = 0; i < SAMPLES; ++i) {
        // Generate sample point in hemisphere
        float theta = 2.0 * 3.14159265 * float(i) / float(SAMPLES);
        float cosTheta = cos(theta);
        float sinTheta = sin(theta);
        float r = float(i) / float(SAMPLES);
        
        vec3 sampleVec = vec3(r * cosTheta, r * sinTheta, sqrt(1.0 - r * r));
        
        // Orient the hemisphere with the normal
        vec3 tangent = normalize(randVec - norm * dot(randVec, norm));
        vec3 bitangent = cross(norm, tangent);
        mat3 TBN = mat3(tangent, bitangent, norm);
        vec3 ray = TBN * sampleVec;

        // get the coordinate of the sample point
        vec2 occUV = ep.xy + sign(dot(ray,norm)) * ray.xy * radD;

        // get the depth of the occluder fragment
        float occluderDepth = lindepth(occUV);
        
        // difference between fragment and occluder depths
        float depthDifference = depth - occluderDepth;

        // calculate occlusion factor
        bl += step(falloff, depthDifference) * (1.0 - smoothstep(falloff, strength, depthDifference));
    }

    // output the result
    FragColor = vec4(clamp(1.0 + bl * invSamples, 0.0, 1.0));
}