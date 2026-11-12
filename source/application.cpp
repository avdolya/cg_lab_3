#include "application.hpp"
#include "math.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb_image.h"

#include <imgui.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace application {

struct Vertex {
	float position[3];
	float normal[3];
	float uv[2];
};


struct GlobalUniforms {
	float view_projection[4][4];
	float model[4][4];
	float camera_position[4];  // где камера
	float ambient[4];          // rgb = color, w = intensity фоновое освещение 
	float light_direction[4];  // xyz, куда светит свет
	float light_color[4];      // rgb = color, w = intensity цвет и сила света источника 
	float albedo[4];           // rgb цвет материала 
	float specular[4];         // цвет блика + shininess в 4-м числе
};

VkBuffer vk_vertex_buffer;
VmaAllocation vk_vertex_buffer_allocation;
void* vk_vertex_buffer_memory;

VkBuffer vk_index_buffer;
VmaAllocation vk_index_buffer_allocation;
void* vk_index_buffer_memory;

VkBuffer vk_uniform_buffer_global;
VmaAllocation vk_uniform_buffer_global_allocation;
GlobalUniforms* vk_uniform_buffer_global_memory;

VkDescriptorSetLayout vk_descriptor_set_layout;
VkDescriptorPool vk_descriptor_pool;
VkDescriptorSet vk_descriptor_set;

VkPipelineLayout vk_pipeline_layout;
VkPipeline vk_pipeline;
VkImage texture_image;
VmaAllocation texture_allocation;
VkImageView texture_view;
VkSampler texture_sampler;
VkDeviceSize uniform_stride;

// One-time upload commands are completed before staging memory is released.
bool uploadResources(const void* vertices, size_t vertex_size,
                     const void* indices, size_t index_size) {
    auto& c = graphics::internal::context;
    int width, height, channels;
    stbi_uc* pixels = stbi_load("textures/checker.png", &width, &height, &channels, 4);
    if (!pixels) {
        std::cerr << "Cannot load textures/checker.png: " << stbi_failure_reason() << '\n';
        return false;
    }
    const VkDeviceSize pixel_size = VkDeviceSize(width) * height * 4;
    VkBuffer staging;
    VmaAllocation staging_allocation;
    VmaAllocationInfo staging_info{};
    VkBufferCreateInfo staging_desc{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = vertex_size + index_size + pixel_size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    };
    VmaAllocationCreateInfo host{
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };
    if (vmaCreateBuffer(c.allocator, &staging_desc, &host, &staging,
                        &staging_allocation, &staging_info) != VK_SUCCESS) {
        stbi_image_free(pixels);
        return false;
    }
    auto* mapped = static_cast<unsigned char*>(staging_info.pMappedData);
    memcpy(mapped, vertices, vertex_size);
    memcpy(mapped + vertex_size, indices, index_size);
    memcpy(mapped + vertex_size + index_size, pixels, size_t(pixel_size));
    stbi_image_free(pixels);
    vmaFlushAllocation(c.allocator, staging_allocation, 0, VK_WHOLE_SIZE);

    VmaAllocationCreateInfo gpu{
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    };
    VkBufferCreateInfo buffer{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = vertex_size,
        .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };
    bool ok = vmaCreateBuffer(c.allocator, &buffer, &gpu, &vk_vertex_buffer,
                              &vk_vertex_buffer_allocation, nullptr) == VK_SUCCESS;
    buffer.size = index_size;
    buffer.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    ok = ok && vmaCreateBuffer(c.allocator, &buffer, &gpu, &vk_index_buffer,
                               &vk_index_buffer_allocation, nullptr) == VK_SUCCESS;
    VkImageCreateInfo image{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_SRGB,
        .extent = {uint32_t(width), uint32_t(height), 1},
        .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
    };
    ok = ok && vmaCreateImage(c.allocator, &image, &gpu, &texture_image,
                              &texture_allocation, nullptr) == VK_SUCCESS;
    VkCommandPool pool{};
    VkCommandPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = c.graphics_queue_index,
    };
    ok = ok && vkCreateCommandPool(c.device, &pool_info, nullptr, &pool) == VK_SUCCESS;
    VkCommandBuffer cmd{};
    VkCommandBufferAllocateInfo alloc{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    ok = ok && vkAllocateCommandBuffers(c.device, &alloc, &cmd) == VK_SUCCESS;
    if (ok) {
        VkCommandBufferBeginInfo begin{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        ok = vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS;
        if (ok) {
            VkBufferCopy copy{0, 0, vertex_size};
            vkCmdCopyBuffer(cmd, staging, vk_vertex_buffer, 1, &copy);
            copy = {vertex_size, 0, index_size};
            vkCmdCopyBuffer(cmd, staging, vk_index_buffer, 1, &copy);
            VkImageMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = texture_image,
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
            };
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
            VkBufferImageCopy region{
                .bufferOffset = vertex_size + index_size,
                .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .imageExtent = image.extent,
            };
            vkCmdCopyBufferToImage(cmd, staging, texture_image,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkMemoryBarrier geometry{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT,
            };
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                                 0, 1, &geometry, 0, nullptr, 1, &barrier);
            ok = vkEndCommandBuffer(cmd) == VK_SUCCESS;
            if (ok) {
                VkSubmitInfo submit{
                    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .commandBufferCount = 1, .pCommandBuffers = &cmd,
                };
                ok = vkQueueSubmit(c.graphics_queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS;
                if (ok) ok = vkQueueWaitIdle(c.graphics_queue) == VK_SUCCESS;
            }
        }
    }
    if (pool) vkDestroyCommandPool(c.device, pool, nullptr);
    vmaDestroyBuffer(c.allocator, staging, staging_allocation);
    if (!ok) return false;
    VkImageViewCreateInfo view{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = texture_image, .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = image.format,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    if (vkCreateImageView(c.device, &view, nullptr, &texture_view) != VK_SUCCESS) return false;
    VkSamplerCreateInfo sampler{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .maxLod = 0.0f,
    };
    return vkCreateSampler(c.device, &sampler, nullptr, &texture_sampler) == VK_SUCCESS;
}

VkShaderModule loadShaderModule(const char path[]) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	const size_t size = file.tellg();
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));

	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(graphics::internal::context.device,
	                         &info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

	return result;
}

// Теперь создается массив на 24 вершины и 36 индексов
// Вершины углов куба дублируются (по 3 штуки на каждый угол) чтобы у каждой из 6 граней была своя уникальная нормаль
bool initialize() {
	auto& context = graphics::internal::context;

	const math::Vector3 corners[8] = {
		{ -0.5f, -0.5f, -0.5f },
		{ +0.5f, -0.5f, -0.5f },
		{ +0.5f, +0.5f, -0.5f },
		{ -0.5f, +0.5f, -0.5f },
		{ -0.5f, -0.5f, +0.5f },
		{ +0.5f, -0.5f, +0.5f },
		{ +0.5f, +0.5f, +0.5f },
		{ -0.5f, +0.5f, +0.5f },
	};

	const int faces[6][4] = {
		{ 0, 1, 2, 3 },
		{ 5, 4, 7, 6 },
		{ 4, 0, 3, 7 },
		{ 1, 5, 6, 2 },
		{ 1, 0, 4, 5 },
		{ 3, 2, 6, 7 },
	};

	const float uv[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
	Vertex vertices[24];
	uint32_t indices[36];
// Процедурный расчет нормалей в цикле
	for (int face = 0; face < 6; ++face) {
		const math::Vector3& a = corners[faces[face][0]];
		const math::Vector3& b = corners[faces[face][1]];
		const math::Vector3& c = corners[faces[face][2]];
		const math::Vector3 normal = math::Vector3::normalize(math::Vector3::cross(c - a, b - a));

		for (int corner = 0; corner < 4; ++corner) {
			const math::Vector3& position = corners[faces[face][corner]];
			vertices[face * 4 + corner] = {
				{ position.x, position.y, position.z },
				{ normal.x, normal.y, normal.z },
				{ uv[corner][0], uv[corner][1] },
			};
		}

		const uint32_t first = face * 4;
		const uint32_t face_indices[6] = { 0, 1, 2, 0, 2, 3 };
		for (int i = 0; i < 6; ++i) {
			indices[face * 6 + i] = first + face_indices[i];
		}
	}

	if (!uploadResources(vertices, sizeof(vertices), indices, sizeof(indices))) return false;
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(context.physical_device, &properties);
    const VkDeviceSize alignment = properties.limits.minUniformBufferOffsetAlignment;
    uniform_stride = (sizeof(GlobalUniforms) + alignment - 1) & ~(alignment - 1);

	const VkBufferCreateInfo global_uniform_buffer = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = uniform_stride,
		.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo global_uniform_buffer_allocation = {
		.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
		         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	if (vmaCreateBuffer(context.allocator, &global_uniform_buffer,
	                    &global_uniform_buffer_allocation,
	                    &vk_uniform_buffer_global, &vk_uniform_buffer_global_allocation,
	                    nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create and allocate uniform buffer\n";
		return false;
	}

	if (vmaMapMemory(context.allocator, vk_uniform_buffer_global_allocation,
	                 (void **)&vk_uniform_buffer_global_memory) != VK_SUCCESS) {
		std::cerr << "Failed to map uniform buffer memory\n";
		return false;
	}

	const VkDescriptorSetLayoutBinding descriptor_set_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
	};

	const VkDescriptorSetLayoutCreateInfo descriptor_set_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(descriptor_set_bindings) / sizeof(descriptor_set_bindings[0]),
		.pBindings = descriptor_set_bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &descriptor_set_layout, nullptr,
	                                &vk_descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		return false;
	}

	const VkPipelineLayoutCreateInfo pipeline_layout = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &vk_descriptor_set_layout,
	};

	if (vkCreatePipelineLayout(context.device, &pipeline_layout, nullptr,
	                           &vk_pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		return false;
	}

	const VkDescriptorPoolSize descriptor_pool_sizes[] = {
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.descriptorCount = 4,
		},
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
	};

	const VkDescriptorPoolCreateInfo descriptor_pool = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1,
		.poolSizeCount = sizeof(descriptor_pool_sizes) / sizeof(descriptor_pool_sizes[0]),
		.pPoolSizes = descriptor_pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &descriptor_pool, nullptr,
	                           &vk_descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	const VkDescriptorSetAllocateInfo descriptor_set = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &vk_descriptor_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &descriptor_set,
	                             &vk_descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor set\n";
		return false;
	}

	const VkDescriptorBufferInfo global_uniform_buffer_descriptor = {
		.buffer = vk_uniform_buffer_global,
		.offset = 0,
		.range = sizeof(GlobalUniforms),
	};

    const VkDescriptorImageInfo texture_descriptor{
        .sampler = texture_sampler, .imageView = texture_view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
	const VkWriteDescriptorSet descriptor_writes[] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = vk_descriptor_set, .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &texture_descriptor,
        },
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk_descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.pBufferInfo = &global_uniform_buffer_descriptor,
		},
	};

	vkUpdateDescriptorSets(context.device, sizeof(descriptor_writes) / sizeof(descriptor_writes[0]),
	                       descriptor_writes, 0, nullptr);

	VkShaderModule vk_vertex_shader = loadShaderModule("shaders/cube.vert.spv");
	VkShaderModule vk_fragment_shader = loadShaderModule("shaders/cube.frag.spv");

	VkPipelineShaderStageCreateInfo stage_infos[2];
	stage_infos[0] = VkPipelineShaderStageCreateInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.stage = VK_SHADER_STAGE_VERTEX_BIT,
		.module = vk_vertex_shader,
		.pName = "main",
	};
	stage_infos[1] = VkPipelineShaderStageCreateInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
		.module = vk_fragment_shader,
		.pName = "main",
	};

	const VkVertexInputBindingDescription vertex_bindings[] = {
		{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		},
	};

	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, normal),
		},
        {
            .location = 2, .binding = 0,
            .format = VK_FORMAT_R32G32_SFLOAT,
            .offset = offsetof(Vertex, uv),
        },
	};

	VkPipelineVertexInputStateCreateInfo input_state_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = vertex_bindings,
		.vertexAttributeDescriptionCount = 3,
		.pVertexAttributeDescriptions = vertex_attributes,
	};

	VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	VkPipelineViewportStateCreateInfo viewport_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	VkPipelineRasterizationStateCreateInfo raster_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	VkPipelineMultisampleStateCreateInfo sample_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	VkPipelineDepthStencilStateCreateInfo depth_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = true,
		.depthWriteEnable = true,
		.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
	};

	VkPipelineColorBlendAttachmentState attachment_info{
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
		                  VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT |
		                  VK_COLOR_COMPONENT_A_BIT,
	};

	VkPipelineColorBlendStateCreateInfo blend_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &attachment_info
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
		.pDynamicStates = dynamic_states,
	};

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stage_infos,
		.pVertexInputState = &input_state_info,
		.pInputAssemblyState = &assembly_state_info,
		.pViewportState = &viewport_info,
		.pRasterizationState = &raster_info,
		.pMultisampleState = &sample_info,
		.pDepthStencilState = &depth_info,
		.pColorBlendState = &blend_info,
		.pDynamicState = &dynamic_state,
		.layout = vk_pipeline_layout,
		.renderPass = context.render_pass,
	};

	if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info,
	                              nullptr, &vk_pipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline\n";
		return false;
	}

	vkDestroyShaderModule(context.device, vk_vertex_shader, nullptr);
	vkDestroyShaderModule(context.device, vk_fragment_shader, nullptr);

	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, vk_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);
	vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_descriptor_set_layout, nullptr);

	vmaUnmapMemory(context.allocator, vk_uniform_buffer_global_allocation);
	vmaDestroyBuffer(context.allocator, vk_uniform_buffer_global, vk_uniform_buffer_global_allocation);

	vkDestroySampler(context.device, texture_sampler, nullptr);
	vkDestroyImageView(context.device, texture_view, nullptr);
	vmaDestroyImage(context.allocator, texture_image, texture_allocation);
	vmaDestroyBuffer(context.allocator, vk_index_buffer, vk_index_buffer_allocation);


	vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vk_vertex_buffer_allocation);
}

constexpr float degrees_to_radians = 3.14159265f / 180.0f;

enum ProjectionType : int {
	PROJECTION_PERSPECTIVE = 0,
	PROJECTION_ORTHOGRAPHIC = 1,
};

int projection_type = PROJECTION_PERSPECTIVE;
float fov_degrees = 60.0f;
float ortho_height = 1.5f;

float camera_position[3] = { 0.0f, 0.0f, -3.0f };
float camera_yaw_degrees = 0.0f;
float camera_pitch_degrees = 0.0f;

float position[3] = { 0.0f, 0.0f, 0.0f };
float rotation_degrees[3] = { 20.0f, 30.0f, 0.0f };
float scale_xyz[3] = { 1.0f, 1.0f, 1.0f };

bool auto_rotate = true;
float auto_rotate_speed = 30.0f;
float auto_rotate_angle = 0.0f;

bool trajectory_enabled = false;
float trajectory_speed = 1.0f;
float trajectory_radius = 1.0f;
float trajectory_time = 0.0f;

float ambient_color[3] = { 1.0f, 1.0f, 1.0f };
float ambient_intensity = 0.15f;

float light_direction[3] = { 0.4f, 0.6f, 1.0f };
float light_color[3] = { 1.0f, 1.0f, 1.0f };
float light_intensity = 1.0f;

float material_albedo[3] = { 1.0f, 1.0f, 1.0f };
float material_specular[3] = { 1.0f, 1.0f, 1.0f };
float material_shininess = 32.0f;

double previous_time = 0.0;

void drawInterface() {
	ImGui::Begin("Cube");

	ImGui::SeparatorText("Projection");
	ImGui::RadioButton("Perspective", &projection_type, PROJECTION_PERSPECTIVE);
	ImGui::SameLine();
	ImGui::RadioButton("Orthographic", &projection_type, PROJECTION_ORTHOGRAPHIC);

	if (projection_type == PROJECTION_PERSPECTIVE) {
		ImGui::SliderFloat("FOV", &fov_degrees, 20.0f, 120.0f, "%.0f deg");
	} else {
		ImGui::SliderFloat("View height", &ortho_height, 0.5f, 5.0f);
	}

	ImGui::SeparatorText("Camera");
	ImGui::SliderFloat3("Position##camera", camera_position, -10.0f, 10.0f);
	ImGui::SliderFloat("Yaw", &camera_yaw_degrees, -180.0f, 180.0f, "%.0f deg");
	ImGui::SliderFloat("Pitch", &camera_pitch_degrees, -89.0f, 89.0f, "%.0f deg");

	ImGui::SeparatorText("Transform");
	ImGui::SliderFloat3("Position", position, -2.0f, 2.0f);
	ImGui::SliderFloat3("Rotation", rotation_degrees, -180.0f, 180.0f, "%.0f deg");
	ImGui::SliderFloat3("Scale", scale_xyz, 0.1f, 3.0f);

	ImGui::SeparatorText("Auto rotation");
	ImGui::Checkbox("Enabled", &auto_rotate);
	ImGui::SliderFloat("Speed", &auto_rotate_speed, -180.0f, 180.0f, "%.0f deg/s");

	ImGui::SeparatorText("Trajectory");
	ImGui::Checkbox("Play##trajectory", &trajectory_enabled);
	ImGui::SliderFloat("Speed##trajectory", &trajectory_speed, 0.1f, 5.0f);
	ImGui::SliderFloat("Radius", &trajectory_radius, 0.0f, 2.0f);

	ImGui::SeparatorText("Directional light");
	ImGui::SliderFloat3("Direction", light_direction, -1.0f, 1.0f);
	ImGui::ColorEdit3("Color##light", light_color);
	ImGui::SliderFloat("Intensity##light", &light_intensity, 0.0f, 5.0f);

	ImGui::SeparatorText("Ambient light");
	ImGui::ColorEdit3("Color##ambient", ambient_color);
	ImGui::SliderFloat("Intensity##ambient", &ambient_intensity, 0.0f, 1.0f);

	ImGui::SeparatorText("Material");
	ImGui::ColorEdit3("Albedo", material_albedo);
	ImGui::ColorEdit3("Specular color", material_specular);
	ImGui::SliderFloat("Shininess", &material_shininess, 1.0f, 256.0f, "%.0f",
	                   ImGuiSliderFlags_Logarithmic);

	if (ImGui::Button("Reset")) {
		projection_type = PROJECTION_PERSPECTIVE;
		fov_degrees = 60.0f;
		ortho_height = 1.5f;
		camera_position[0] = camera_position[1] = 0.0f;
		camera_position[2] = -3.0f;
		camera_yaw_degrees = 0.0f;
		camera_pitch_degrees = 0.0f;
		position[0] = position[1] = position[2] = 0.0f;
		rotation_degrees[0] = 20.0f;
		rotation_degrees[1] = 30.0f;
		rotation_degrees[2] = 0.0f;
		scale_xyz[0] = scale_xyz[1] = scale_xyz[2] = 1.0f;
		auto_rotate = true;
		auto_rotate_speed = 30.0f;
		auto_rotate_angle = 0.0f;
		trajectory_enabled = false;
		trajectory_speed = 1.0f;
		trajectory_radius = 1.0f;
		trajectory_time = 0.0f;
		light_direction[0] = 0.4f;
		light_direction[1] = 0.6f;
		light_direction[2] = 1.0f;
		light_color[0] = light_color[1] = light_color[2] = 1.0f;
		light_intensity = 1.0f;
		ambient_color[0] = ambient_color[1] = ambient_color[2] = 1.0f;
		ambient_intensity = 0.15f;
		material_albedo[0] = material_albedo[1] = material_albedo[2] = 1.0f;
		material_specular[0] = material_specular[1] = material_specular[2] = 1.0f;
		material_shininess = 32.0f;
	}

	ImGui::End();
}

void update(double time) {
	auto& context = graphics::internal::context;

	const float delta_time = float(time - previous_time);
	previous_time = time;

	drawInterface();

	if (auto_rotate) {
		auto_rotate_angle += auto_rotate_speed * delta_time;
	}

	if (trajectory_enabled) {
		trajectory_time += trajectory_speed * delta_time;
	}

	const float trajectory_x = trajectory_radius * std::sin(trajectory_time);
	const float trajectory_y = trajectory_radius * std::sin(2.0f * trajectory_time) / 2.0f;

	const math::Matrix4 model =
		math::Matrix4::translation(position[0] + trajectory_x,
		                           position[1] + trajectory_y,
		                           position[2]) *
		math::Matrix4::rotationY((rotation_degrees[1] + auto_rotate_angle) * degrees_to_radians) *
		math::Matrix4::rotationX(rotation_degrees[0] * degrees_to_radians) *
		math::Matrix4::rotationZ(rotation_degrees[2] * degrees_to_radians) *
		math::Matrix4::scale(scale_xyz[0], scale_xyz[1], scale_xyz[2]);

	const math::Matrix4 view =
		math::Matrix4::rotationX(-camera_pitch_degrees * degrees_to_radians) *
		math::Matrix4::rotationY(-camera_yaw_degrees * degrees_to_radians) *
		math::Matrix4::translation(-camera_position[0], -camera_position[1], -camera_position[2]);

	const float aspect = float(context.swapchain_extent.width) /
	                     float(context.swapchain_extent.height);
	const math::Matrix4 projection = projection_type == PROJECTION_PERSPECTIVE
		? math::Matrix4::perspective(fov_degrees * degrees_to_radians, aspect, 0.1f, 100.0f)
		: math::Matrix4::orthographic(ortho_height, aspect, 0.1f, 100.0f);

	const math::Matrix4 view_projection = projection * view;

	GlobalUniforms& uniforms = *vk_uniform_buffer_global_memory;

	memcpy(uniforms.view_projection, view_projection.elements, sizeof(view_projection.elements));
	memcpy(uniforms.model, model.elements, sizeof(model.elements));

	const float camera_position_xyzw[4] = { camera_position[0], camera_position[1], camera_position[2], 1.0f };
	const float ambient_rgba[4] = { ambient_color[0], ambient_color[1], ambient_color[2], ambient_intensity };
	const float light_direction_xyzw[4] = { light_direction[0], light_direction[1], light_direction[2], 0.0f };
	const float light_color_rgba[4] = { light_color[0], light_color[1], light_color[2], light_intensity };
	const float albedo_rgba[4] = { material_albedo[0], material_albedo[1], material_albedo[2], 1.0f };
	const float specular_rgba[4] = { material_specular[0], material_specular[1], material_specular[2], material_shininess };

	memcpy(uniforms.camera_position, camera_position_xyzw, sizeof(camera_position_xyzw));
	memcpy(uniforms.ambient, ambient_rgba, sizeof(ambient_rgba));
	memcpy(uniforms.light_direction, light_direction_xyzw, sizeof(light_direction_xyzw));
	memcpy(uniforms.light_color, light_color_rgba, sizeof(light_color_rgba));
	memcpy(uniforms.albedo, albedo_rgba, sizeof(albedo_rgba));
	memcpy(uniforms.specular, specular_rgba, sizeof(specular_rgba));
	vmaFlushAllocation(context.allocator, vk_uniform_buffer_global_allocation, 0, sizeof(GlobalUniforms));
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

	vkResetCommandBuffer(fd.command_buffer, 0);

	const VkCommandBufferBeginInfo command_buffer_begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(fd.command_buffer, &command_buffer_begin);

	const VkClearValue clear_values[] = {
		{ .color = { .float32 = { 0.1f, 0.1f, 0.12f, 1.0f } } },
		{ .depthStencil = { .depth = 1.0f, .stencil = 0 } },
	};

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = sizeof(clear_values) / sizeof(clear_values[0]),
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.x = 0, .y = 0,
		.width = float(context.swapchain_extent.width),
		.height = float(context.swapchain_extent.height),
		.minDepth = 0, .maxDepth = 1,
	};

	const VkRect2D scissor = { .extent = context.swapchain_extent };

	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	const VkDeviceSize vertex_buffer_offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_vertex_buffer, &vertex_buffer_offset);
	vkCmdBindIndexBuffer(fd.command_buffer, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);

	const uint32_t model_offset = 0; // The scene contains one model.
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                        vk_pipeline_layout, 0, 1, &vk_descriptor_set,
	                        1, &model_offset);

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

	vkCmdDrawIndexed(fd.command_buffer, 36, 1, 0, 0, 0);

	vkCmdEndRenderPass(fd.command_buffer);

	vkEndCommandBuffer(fd.command_buffer);
}

}
