#version 330 core

uniform sampler2D textureSampler;
uniform vec4 basecolor;
uniform vec4 spec_color;
uniform vec4 light_direction;
uniform sampler2D shadow_map;

in vec3 vposition;
in vec3 vnormal;
in vec2 vtexcoord;
in float vshininess;
in mat4 transform;
in vec4 frag_pos_light_space;

out vec4 FragColor;

const vec4 light_diffuse = vec4(0.6, 0.6, 0.6, 0.0);
const vec4 light_ambient = vec4(0.2, 0.2, 0.2, 1.);
const vec4 light_specular = vec4(1.0, 1.0, 1.0, 1.0);
const float bias = 0.001;

float shadow_calc()
{
    // Perform perspective divide
    vec3 projCoords = frag_pos_light_space.xyz / frag_pos_light_space.w;
    projCoords = projCoords * 0.5 + 0.5;
    float currentDepth = projCoords.z;

    // Check clips
    if (currentDepth > 1.0 || frag_pos_light_space.z < -0.0) return 0.0;
    if (length(spec_color) == 0.0) return 0.0;

    // PCF
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadow_map, 0);
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(shadow_map, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }    
    }
    shadow /= 9.0;
    return shadow;
}

vec4 phong() {
    vec3 mv_light_direction = (transform * light_direction).xyz,
         normal = normalize(vnormal),
         eye = normalize(-vposition),
         reflection = normalize(-reflect(mv_light_direction, normal));
    normal = normalize((transform * vec4(vnormal, 0)).xyz);

    vec4 texcolor = texture(textureSampler, vtexcoord);
    vec4 frag_diffuse = vec4(
        texcolor.rgb + basecolor.rgb,
        texcolor.a * basecolor.a
    );

    vec4 diffuse_factor
        = max(-dot(normal, mv_light_direction), 0.0) * light_diffuse;
    vec4 ambient_diffuse_factor = diffuse_factor + light_ambient;
    
    vec4 specular_factor
        = pow(max(dot(-reflection, eye), 0.0), vshininess) * light_specular;
    
    float shadow = shadow_calc();

    vec4 ambient = light_ambient * frag_diffuse;
    vec4 diffuse = diffuse_factor * frag_diffuse;
    vec4 specular = specular_factor * spec_color;
    return ambient + (1.0 - shadow) * (diffuse + specular);
}