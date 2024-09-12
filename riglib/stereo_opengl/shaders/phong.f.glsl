#version 330 core

uniform sampler2D textureSampler;
uniform vec4 basecolor;
uniform vec4 spec_color;

in vec3 vposition;
in vec3 vnormal;
in vec2 vtexcoord;
in float vshininess;
in mat4 transform;

out vec4 FragColor;

const vec4 light_direction = vec4(-1, -2, -2, 0.0);
const vec4 light_diffuse = vec4(0.6, 0.6, 0.6, 0.0);
const vec4 light_ambient = vec4(0.2, 0.2, 0.2, 1.);
const vec4 light_specular = vec4(1.0, 1.0, 1.0, 1.0);

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
    
    return ambient_diffuse_factor * frag_diffuse + specular_factor * spec_color;
}
