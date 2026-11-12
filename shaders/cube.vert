#version 450 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec3 out_world_position;
layout(location = 1) out vec3 out_world_normal;
layout(location = 2) out vec2 out_uv;

layout(std140, set = 0, binding = 0) uniform GlobalUniforms {
	mat4 view_projection;
	mat4 model;
	vec4 camera_position;
	vec4 ambient;
	vec4 light_direction;
	vec4 light_color;
	vec4 albedo;
	vec4 specular;
} global_uniforms;

void main() {
	out_uv = in_uv;
	vec4 world_position = global_uniforms.model * vec4(in_position, 1);
	gl_Position = global_uniforms.view_projection * world_position;

	out_world_position = world_position.xyz;
	// Normal matrix keeps normals perpendicular to faces even with non-uniform scale
	out_world_normal = transpose(inverse(mat3(global_uniforms.model))) * in_normal;
}