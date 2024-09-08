#version 330 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec4 position[];
in vec3 normal[];
in vec2 texcoord[];

out vec4 vposition;
out vec3 vnormal;
out vec2 vtexcoord;

void main() {
    vec4 vec1 = gl_in[1].gl_Position - gl_in[0].gl_Position;
    vec4 vec2 = gl_in[2].gl_Position - gl_in[0].gl_Position;
    vec3 face_normal = normalize(cross(vec3(vec1), vec3(vec2)));

    for(int i = 0; i < 3; i++) {
        gl_Position = gl_in[i].gl_Position;
        vposition = position[i];
        vnormal = face_normal;
        vtexcoord = texcoord[i];
        EmitVertex();
    }
    EndPrimitive();
}