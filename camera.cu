#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define GLM_FORCE_CUDA
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>

#include "camera.h"
#include "rand.h"

using namespace std;
using namespace crs;

__host__ __device__ crs::Camera::Camera(){
	width = 100;
	height = 100;
	resolution = 1.0f;
	focusplane = 1.0f;

	fov = 90.0f;			// default 90° vertical fov
	aperture = 0.0f;
}

__host__ __device__ crs::Camera::~Camera(){

}

__host__ __device__ void crs::Camera::update(){
	// scale to world coordinates
	//width /= resolution;
	//height /= resolution;

	// update FOV and matrix
	updateFOV();
	updateMatrix();
}

__host__ __device__ void crs::Camera::updateFOV(){
	// given the vertical fov, calculate the distance to the focal plane
	//float theta = fov * ((float)M_PI/180.0f);
	//float half_height = height * 0.5f;

	// TODO
	focusplane = (float)width*2.0f;
}

__host__ __device__ void crs::Camera::updateMatrix(){
	matrix = glm::lookAt(position, lookat, up);
}

__device__ void cast(HitRecord *r, Camera *camera, unsigned long id, unsigned int seed){

	// Generate a 2D random coordinate with a uniform distribution
	curandState rngState;
	curand_init(crs::WangHash(seed) + id, 0, 0, &rngState);
	glm::vec2 xy = crs::RandUniformSquare(&rngState);

	// Calculate direction, starting with pixel indices
	float x_index = fmod( (float)id, camera->width );
	float y_index = id / camera->width;

	float u = (((xy.x - 0.5f) + x_index) - (camera->width * 0.5f)) / camera->resolution;
	float v = (((xy.y - 0.5f) + y_index) - (camera->height * 0.5f)) / camera->resolution;
	float z = -camera->focusplane / camera->resolution;

	vec2 dof = (crs::RandUniformDisc(&rngState) * camera->aperture) / camera->resolution;

	// construct the local ray
	glm::vec4 l = vec4(u + dof.x, -(v + dof.y), z, 0.0f);

	//transform to world cordinates and normalize
	glm::vec4 n = glm::normalize(l);
	glm::vec4 t = n * camera->matrix;

	r->wi.origin = camera->position;
	r->wi.direction.x  = t.x;
	r->wi.direction.y  = t.y;
	r->wi.direction.z  = t.z;

	r->wi.frequency = 0.0f;
	r->wi.length = FLT_MAX;
}

// Generates rays
__global__ void crs::KERNEL_CAST_CAMERA_RAYS(HitRecord *hitrecords, Camera *camera, unsigned int seed){
	unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned long threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	if (threadId >= camera->width * camera->height) return;

	// Cast
	cast(&hitrecords[threadId], camera, threadId, seed);
}
