#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define GLM_FORCE_CUDA
#include <glm.hpp>

#include "bxdf.h"
#include "rand.h"

using namespace std;
using namespace crs;

__host__ int crs::BxdfTable::getBxdfIdbyName(std::string bxdfname) {
	unsigned int i = 0;

	for (i = 0; i < size; i++) {
		if (bxdfname == toc[i].bxdf_name) return i;
	}

	return i;
}

__device__ void crs::bxdf_NOHIT(Bxdf *b, HitRecord *r) {
	// calculate color
	float t = 0.5 * (r->wi.direction.y + 1.0f);
	vec3 C = ((1.0f - t) * glm::vec3(4.0f)) + (t * b->kd);
	
	// accumulate the bounce
	r->accumulator.color += C*(float)(1.0f / M_PI);

	// terminate the path
	r->terminated = true;
}

__device__ void crs::bxdf_NORMAL(HitRecord *r) {
	// accumulate color
	vec3 C = 0.5f * (r->normal + vec3(1.0f, 1.0f, 1.0f));
	
	// accumulate the bounce
	r->accumulator.color += C;

	// terminate the path
	r->terminated = true;
}

// S for scatter : Lambertian bxdf
__device__ void crs::bxdf_BSDF(Bxdf *b, HitRecord *r, unsigned int seed, unsigned int tid) {
	
	// calculate the color
	float NdL = glm::dot(r->normal, r->wi.direction);
	vec3 C = (glm::vec3(1.0f) - b->kd) * NdL;
	
	r->accumulator.color += (C * (float)(1.0f / M_PI));

	// rng state
	curandState rngState;
	curand_init(crs::WangHash(seed)+tid, 0, 0, &rngState);

	// generate a point within a unit sphere and transform according to location and normal
	vec3 t = crs::RandUniformInSphere(&rngState);
	vec3 target = t + r->normal + r->location;

	// construct the new ray for the next bounce
	r->wi.origin = r->location;
	r->wi.direction = glm::normalize(target - r->location);
	r->wi.length = FLT_MAX;
}

// R for Reflect : conductor/metal bxdf
__device__ void crs::bxdf_BRDF(Bxdf *b, HitRecord *r, unsigned int seed, unsigned int tid) {

	// calculate the color
	float NdL = glm::dot(r->normal, r->wi.direction);
	vec3 C = (glm::vec3(1.0f) - b->kd) * NdL;

	r->accumulator.color += (C * (float)(1.0f / M_PI));

	// rng state
	curandState rngState;
	curand_init(crs::WangHash(seed)+tid, 0, 0, &rngState);

	// generate a point within a unit sphere and transform according to location and normal
	vec3 t = crs::RandUniformInSphere(&rngState);
	vec3 target = t + r->normal + r->location;

	// reflect the incoming ray
	vec3 ref = r->wi.direction - (2.0f * glm::dot(r->wi.direction, r->normal) * r->normal);

	// shininess factor
	vec3 f = (t * b->sh) + ref;

	// construct the new ray for the next bounce
	r->wi.origin = r->location;
	r->wi.direction = glm::normalize(f);
	r->wi.length = FLT_MAX;
}

// T for Transmit : dielectric/glass bxdf
__device__ void crs::bxdf_BTDF(Bxdf *b, HitRecord *r, unsigned int seed, unsigned int tid) {
}

// SS for Subsurface : subsurface bxdf
__device__ void crs::bxdf_BSSDF(Bxdf *b, HitRecord *r, unsigned int seed, unsigned int tid) {
}

// C for constant : return a constant color
__device__ void crs::bxdf_CONSTANT(Bxdf *b, HitRecord *r) {

	// accumulate the bounce
	r->accumulator.color += b->kd;

	// terminate the path
	r->terminated = true;
}

__device__ void crs::evaluateBxdf(Bxdf *bxdfList, HitRecord *r, PixelBuffer *p, int pathlength, unsigned int seed, unsigned int tid) {
	
	// early exit
	if (r->terminated) return;

	int bid = r->bxdf;

	switch (bxdfList[bid].type) {
	case crs::NOHIT:
		bxdf_NOHIT(&bxdfList[0], r);
		break;
	case crs::NORMAL:
		bxdf_NORMAL(r);
		break;
	case crs::BSDF:
		bxdf_BSDF(&bxdfList[bid], r, seed, tid);
		break;
	case crs::BRDF:
		bxdf_BRDF(&bxdfList[bid], r, seed, tid);
		break;
	case crs::BTDF:
		bxdf_BTDF(&bxdfList[bid], r, seed, tid);
		break;
	case crs::BSSDF:
		bxdf_BSSDF(&bxdfList[bid], r, seed, tid);
		break;
	case crs::CONSTANT:
		bxdf_CONSTANT(&bxdfList[bid], r);
		break;
	default:
		// no valid bxdf assigned
		bxdf_NOHIT(&bxdfList[0], r);
		break;
	}
		
	// increase sample count
	r->accumulator.samples++;

	// mark path terminated if we reached our depth
	if (r->accumulator.samples >= pathlength) r->terminated = true;

	// reset the bxdf for the next bounce
	r->bxdf = crs::NOHIT;
}

__global__ void crs::KERNEL_BXDF(Bxdf *bxdfList, HitRecord *hitRecords, PixelBuffer *pixelBuffer, int width, int height, int pathlength, unsigned int seed) {
	unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned long threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	if (threadId >= width * height) return;
	
	// Evaluate
	evaluateBxdf(bxdfList, &hitRecords[threadId], &pixelBuffer[threadId], pathlength, seed, threadId);
}
