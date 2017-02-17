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

__device__ void crs::bxdf_NOHIT(Bxdf *b, HitRecord *r, PixelBuffer *p, int pathlength) {
	// calculate and accumulate color
	float t = 0.5 * (r->in.direction.y + 1.0f);
	vec3 C = ((1.0f - t) * b->kd + (t * b->ka)) / (float)pathlength;
	p->color += C;
}

__device__ void crs::bxdf_NORMAL(HitRecord *r, PixelBuffer *p, int pathlength) {
	// accumulate color
	vec3 C = (0.5f * (r->normal + vec3(1.0f, 1.0f, 1.0f))) / (float)pathlength;
	p->color += C;
}

// S for scatter : Lambertian bxdf
__device__ void crs::bxdf_BSDF(Bxdf *b, HitRecord *r, PixelBuffer *p, int pathlength, unsigned int seed, unsigned int tid) {

	/*
	//TODO
	float dist = distance(g_light_position, position);
	float att = 1.0f / (a + dist * b + dist * dist * c);

	// and then:
	float lum_final = dot(...) * att;
	*/

	// accumulate the color
	float NdL = glm::dot(r->normal, r->in.direction);
	vec3 C = ( b->ka * b->kd * r->in.attenuation * NdL) / (float)pathlength;
	p->color += C;

	// rng state
	curandState rngState;
	curand_init(crs::WangHash(seed)+tid, 0, 0, &rngState);

	// generate a point within a unit sphere and transform according to location and normal
	vec3 t = crs::RandUniformInSphere(&rngState);
	vec3 target = t + r->normal + r->location;

	// construct the new ray for the next bounce
	r->in.origin = r->location;
	r->in.direction = glm::normalize(target - r->location);
	r->in.attenuation = glm::length(t);

	// reset the bxdf for the next bounce
	r->bxdf = NOHIT;
}

// R for Reflect : conductor/metal bxdf
__device__ void crs::bxdf_BRDF(Bxdf *b, HitRecord *r, PixelBuffer *p, int pathlength, unsigned int seed, unsigned int tid) {

	// accumulate the color
	float NdL = glm::dot(r->normal, r->in.direction);
	vec3 C = ( b->ka * b->kd * r->in.attenuation * NdL) / (float)pathlength;
	p->color += C;

	// rng state
	curandState rngState;
	curand_init(crs::WangHash(seed)+tid, 0, 0, &rngState);

	// generate a point within a unit sphere and transform according to location and normal
	vec3 t = crs::RandUniformInSphere(&rngState);
	vec3 target = t + r->normal + r->location;

	// reflect the incoming ray
	vec3 ref = r->in.direction - (2.0f * glm::dot(r->in.direction, r->normal) * r->normal);

	// shininess factor
	vec3 f = t * b->sh + ref;

	// construct the new ray for the next bounce
	r->in.origin = r->location;
	r->in.direction = glm::normalize(f);
	r->in.attenuation = glm::length(f);

	// reset the bxdf for the next bounce
	r->bxdf = NOHIT;

}

// T for Transmit : dielectric/glass bxdf
__device__ void crs::bxdf_BTDF(Bxdf *b, HitRecord *r, PixelBuffer *p, int pathlength, unsigned int seed, unsigned int tid) {
}

// SS for Subsurface : subsurface bxdf
__device__ void crs::bxdf_BSSDF(Bxdf *b, HitRecord *r, PixelBuffer *p, int pathlength, unsigned int seed, unsigned int tid) {
}

// C for constant : return a constant color
__device__ void crs::bxdf_CONSTANT(Bxdf *b, HitRecord *r, PixelBuffer *p, int pathlength) {
	p->color += b->kd / (float)pathlength;
}

__device__ void crs::evaluateBxdf(Bxdf *bxdfList, HitRecord *r, PixelBuffer *p, int pathlength, unsigned int seed, unsigned int tid) {
	
	// early exit if the hitrecord is marked as terminated
	if(r->is_terminated){
		//r->reset();
		return;
	}

	// retrieve the bxdf at the intersection
	int bid = r->bxdf;

	switch (bxdfList[bid].type) {
	case crs::NOHIT:
		bxdf_NOHIT(&bxdfList[bid], r, p, pathlength);
		break;
	case crs::NORMAL:
		bxdf_NORMAL(r, p, pathlength);
		break;
	case crs::BSDF:
		bxdf_BSDF(&bxdfList[bid], r, p, pathlength, seed, tid);
		break;
	case crs::BRDF:
		bxdf_BRDF(&bxdfList[bid], r, p, pathlength, seed, tid);
		break;
	case crs::BTDF:
		bxdf_BTDF(&bxdfList[bid], r, p, pathlength, seed, tid);
		break;
	case crs::BSSDF:
		bxdf_BSSDF(&bxdfList[bid], r, p, pathlength, seed, tid);
		break;
	case crs::CONSTANT:
		bxdf_CONSTANT(&bxdfList[bid], r, p, pathlength);
		break;
	default:
		// no valid bxdf assigned
		bxdf_NOHIT(&bxdfList[bid], r, p, pathlength);
		break;
	}

	// mark another bounce
	r->pathcounter++;

	// increase sample count if we reached the end of our path
	if(r->pathcounter >= pathlength){
		// reset the hit record for the next sample
		r->reset();
		p->samples += 1;
	}
}

__global__ void crs::KERNEL_BXDF(Bxdf *bxdfList, HitRecord *hitRecords, PixelBuffer *pixelBuffer, int width, int height, int pathlength, unsigned int seed) {
	unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned long threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	if (threadId >= width * height) return;
	
	// Evaluate
	evaluateBxdf(bxdfList, &hitRecords[threadId], &pixelBuffer[threadId], pathlength, seed, threadId);
}
