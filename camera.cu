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
	resolution_x = 100;
	resolution_y = 100;

	fov = 90.0f;			// default 90° vertical fov
	focusplane = 100;
	imageplane = 100;
	aperture = 0.0f;
}

__host__ __device__ crs::Camera::~Camera(){

}

__host__ __device__ void crs::Camera::update(){

	// half the aperture
	aperture *= 0.5f;

	// update FOV
	if(fov <= 0.0f) fov = 0.1f;
	if(fov >= 180.0f) fov = 179.9f;

	imageplane = resolution_y * tan( glm::radians((180.0f - fov) * 0.5f) );

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

	// Calculate the direction
	float half_width = camera->resolution_x * 0.5f;
	float half_height = camera->resolution_y * 0.5f;
	float u = (sample.x + x_index) - half_width;
	float v = (sample.y + y_index) - half_height;
	float z = camera->imageplane;

	// Circle of Confusion
	vec2 coc = crs::RandUniformDisc(&rngState) * camera->aperture;

	// construct the local ray
	glm::vec4 l = vec4(coc, 0.0f, 0.0f) + vec4(u, -v, -z, 0.0f) ;

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

	if (threadId >= camera->resolution_x * camera->resolution_y) return;

	// Cast
	cast(&hitrecords[threadId], camera, threadId, seed);
}
