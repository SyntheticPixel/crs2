#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define GLM_FORCE_CUDA
#include <glm.hpp>

#include "camera.h"
#include "rand.h"

using namespace std;
using namespace crs;

crs::Camera::Camera(){
	width = 100;
	height = 100;
	resolution = 1.0f;
	focus = 1.0f;
}

crs::Camera::Camera(int w, int h, int r, float f){
	width = w;
	height = h;
	resolution = r;
	if(resolution == 0.0f) resolution = 1.0f;
	focus = f;
}

crs::Camera::~Camera(){

}

__device__ void cast(HitRecord *r, Camera *camera, unsigned long id, unsigned int seed){

	// Generate a 2D random coordinate with a uniform distribution
	curandState rngState;
	curand_init(crs::WangHash(seed) + id, 0, 0, &rngState);
	glm::vec2 xy = crs::RandUniformSquare(&rngState);

	// Calculate direction, starting with pixel indices
	float x_index = fmod( (float)id, (float)camera->width );
	float y_index = id / camera->width;

	float u = (((xy.x - 0.5f) + x_index) - (camera->width * 0.5f)) / camera->resolution;
	float v = (((xy.y - 0.5f) + y_index) - (camera->height * 0.5f)) / camera->resolution;
	float z = camera->focus / camera->resolution;

	r->wi.origin = camera->position;

	r->wi.direction.x = u;
	r->wi.direction.y = -v;
	r->wi.direction.z = -z;

	vec3 n = glm::normalize(r->wi.origin + r->wi.direction);
	r->wi.direction = n;

	r->wi.frequency = 0.0f;
	r->wi.length = FLT_MAX;
}

// Generates rays in camera space
__global__ void crs::KERNEL_CAST_CAMERA_RAYS(HitRecord *hitrecords, Camera *camera, unsigned int seed){
	unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned long threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	if (threadId >= camera->width * camera->height) return;

	// Cast
	cast(&hitrecords[threadId], camera, threadId, seed);
}
