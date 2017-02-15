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

__device__ void crs::bxdf_NOHIT(HitRecord *r, PixelBuffer *p, int pathlength) {
	float t = 0.5f*(r->in.direction.y + 1.0f);
	// accumulate color
	vec3 C = ((vec3(1.0f, 1.0f, 1.0f) * (1.0f - t)) + (t * vec3(0.5f, 0.7f, 1.0f))) / (float)pathlength;
	p->color += C;
}

__device__ void crs::bxdf_NORMAL(HitRecord *r, PixelBuffer *p, int pathlength) {
	// accumulate color
	vec3 C = (0.5f * (r->normal + vec3(1.0f, 1.0f, 1.0f))) / (float)pathlength;
	p->color += C;
}

__device__ void crs::bxdf_BSDF(Bxdf *b, HitRecord *r, PixelBuffer *p, int pathlength, unsigned int seed) {
	// accumulate the cosine weighted color
	float c = glm::dot(r->normal, r->in.direction);
	float absorption = 0.25f;
	p->color += (b->ka * (1.0f - absorption) * c) / (float)pathlength;
	
	// construct a new ray for the next bounce
	curandState rngState;
	curand_init(crs::WangHash(seed), 0, 0, &rngState);

	r->in.origin = r->location;
	r->in.direction = glm::normalize(r->normal + crs::RandUniformSphere(&rngState, 1.0f));
	r->in.length = FLT_MAX;
}

__device__ void crs::bxdf_BRDF(Bxdf *b, HitRecord *r, PixelBuffer *p, int pathlength, unsigned int seed) {
}

__device__ void crs::bxdf_BTDF(Bxdf *b, HitRecord *r, PixelBuffer *p, int pathlength, unsigned int seed) {
}

__device__ void crs::bxdf_BSSDF(Bxdf *b, HitRecord *r, PixelBuffer *p, int pathlength, unsigned int seed) {
}

__device__ void crs::bxdf_CONSTANT(Bxdf *b, HitRecord *r, PixelBuffer *p, int pathlength) {
	p->color += b->ka / (float)pathlength;
}

__device__ void crs::evaluateBxdf(Bxdf *bxdfList, HitRecord *r, PixelBuffer *p, int pathlength, unsigned int seed) {
	
	// early exit if the hitrecord is marked as terminated
	if(r->is_terminated){
		//r->reset();
		return;
	}

	// retrieve the bxdf at intersection
	int id = r->bxdf;

	switch (bxdfList[id].type) {
	case crs::NORMAL:
		bxdf_NORMAL(r, p, pathlength);
		break;
	case crs::BSDF:
		bxdf_BSDF(&bxdfList[id], r, p, pathlength, seed);
		break;
	case crs::BRDF:
		bxdf_BRDF(&bxdfList[id], r, p, pathlength, seed);
		break;
	case crs::BTDF:
		bxdf_BTDF(&bxdfList[id], r, p, pathlength, seed);
		break;
	case crs::BSSDF:
		bxdf_BSSDF(&bxdfList[id], r, p, pathlength, seed);
		break;
	case crs::CONSTANT:
		bxdf_CONSTANT(&bxdfList[id], r, p, pathlength);
		break;
	default:
		// no valid bxdf assigned
		bxdf_NOHIT(r, p, pathlength);
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
	evaluateBxdf(bxdfList, &hitRecords[threadId], &pixelBuffer[threadId], pathlength, seed);
}
