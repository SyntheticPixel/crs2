#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define GLM_FORCE_CUDA
#define GLM_RIGHT_HANDED
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>

#include "camera.h"
#include "rand.h"

using namespace std;
using namespace crs;

__host__ __device__ crs::Camera::Camera(){
	resolution_x = 100;
	resolution_y = 100;

	fov = 90.0f;			// default 90° vertical fov
	focus_distance = 100;
	image_plane = 100;
	aperture_radius = 0.0f;
}

__host__ __device__ crs::Camera::~Camera(){

}

__host__ __device__ void crs::Camera::update(){

	// update FOV
	if(fov <= 0.0f) fov = 0.1f;
	if(fov >= 180.0f) fov = 179.9f;

	image_plane = resolution_y * tan( glm::radians((180.0f - fov) * 0.5f) );

	//update the camera matrix
	matrix = glm::lookAt(position, lookat, up);
}

__device__ void cast(HitRecord *r, Camera *camera, unsigned long id, unsigned int seed){
	// Generate a 2D random coordinate with a uniform distribution
	curandState rngState;
	glm::vec2 sample;
	curand_init(crs::WangHash(seed) + id, 0, 0, &rngState);
	sample = crs::RandUniformSquare(&rngState);
	sample -= vec2(0.5f, 0.5f);

	// Calculate pixel indices
	float x_index = fmod( (float)id, camera->resolution_x );
	float y_index = id / camera->resolution_x;

	// Calculate the direction in camera space
	float half_width = camera->resolution_x * 0.5f;
	float half_height = camera->resolution_y * 0.5f;
	float u = (sample.x + x_index) - half_width;
	float v = (sample.y + y_index) - half_height;
	float w = camera->image_plane;

	// Calculate focus point in camera space, invert v and w for right-handed coordinate system
	vec4 local_ray = glm::normalize( vec4(u, -v, -w, 0.0f));
	vec4 local_focus = (local_ray  * camera->focus_distance);

	// Calculate the new origin and direction in camera space
	// TODO: Code a decent RandUniformDisc function for CUDA
	vec3 point_in_disc = crs::RandUniformInSphere(&rngState) * camera->aperture_radius;
	vec4 local_origin = vec4(point_in_disc, 0.0f);
	vec4 local_direction = glm::normalize( local_focus - local_origin );

	// Convert origin and direction to world space
	vec4 world_origin =  vec4(camera->position, 0.0f) + local_origin;
	vec4 world_direction = glm::inverse(camera->matrix) * local_direction;

	// Copy back to buffer
	r->wi.origin.x = world_origin.x;
	r->wi.origin.y = world_origin.y;
	r->wi.origin.z = world_origin.z;
	r->wi.direction.x = world_direction.x;
	r->wi.direction.y = world_direction.y;
	r->wi.direction.z = world_direction.z;
	r->wi.frequency = 0.0f;
	r->wi.length = FLT_MAX;

}

// Generates rays
__global__ void crs::KERNEL_CAST_CAMERA_RAYS(HitRecord *hitrecords, Camera *camera, unsigned int seed){
	unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned long threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	if (threadId >= camera->resolution_x * camera->resolution_y) return;

	// Cast
	cast(&hitrecords[threadId], camera, threadId, seed);
}
