#version 330 core
uniform sampler2D tex; // this should hold the texture rendered by the horizontal blur pass
in vec2 uv;
 
uniform float blur;

out vec4 FragColor;
 
void main(void)
{
   vec4 sum = vec4(0.0);
 
   // blur in y (vertical)
   // take nine samples, with the distance blur between them
   sum += texture(tex, vec2(uv.x, uv.y - 4.0*blur)) * 0.05;
   sum += texture(tex, vec2(uv.x, uv.y - 3.0*blur)) * 0.09;
   sum += texture(tex, vec2(uv.x, uv.y - 2.0*blur)) * 0.12;
   sum += texture(tex, vec2(uv.x, uv.y - blur)) * 0.15;
   sum += texture(tex, vec2(uv.x, uv.y)) * 0.16;
   sum += texture(tex, vec2(uv.x, uv.y + blur)) * 0.15;
   sum += texture(tex, vec2(uv.x, uv.y + 2.0*blur)) * 0.12;
   sum += texture(tex, vec2(uv.x, uv.y + 3.0*blur)) * 0.09;
   sum += texture(tex, vec2(uv.x, uv.y + 4.0*blur)) * 0.05;
 
   FragColor = sum;
}