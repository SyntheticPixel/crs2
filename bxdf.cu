#include <math.h>

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

__device__ void crs::bxdf_NOHIT(HitRecord *r) {
	// calculate color
	r->accumulator.color += glm::vec3(0.0f, 0.0f, 0.0f);

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
__device__ void crs::bxdf_LAMBERT(Bxdf *b, HitRecord *r, unsigned int seed, unsigned int tid) {
	
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

	// calculate the color
	vec3 C = ((vec3(1.0f) - b->kd)) * (-1.0f / (float)M_PI);

	// add the result
	r->accumulator.color += C;
}

// ON for Oren-Nayar : Oren-Nayar bxdf
__device__ void crs::bxdf_OREN_NAYAR(Bxdf *b, HitRecord *r, unsigned int seed, unsigned int tid) {

	// simplified Oren-Nayar from http://ruh.li/GraphicsOrenNayar.html
	vec3 incoming = r->wi.direction;
	vec3 surface_normal = r->normal;

	// rng state
	curandState rngState;
	curand_init(crs::WangHash(seed) + tid, 0, 0, &rngState);

	// generate a point within a unit sphere and transform according to location and normal
	vec3 t = crs::RandUniformInSphere(&rngState);
	vec3 target = t + surface_normal + r->location;
	vec3 outgoing = glm::normalize(target - r->location);

	// calculate intermediary values
	float NdL = dot(surface_normal, outgoing);
	float NdV = glm::dot(surface_normal, incoming);

	float angleVN = acos(NdV);
	float angleLN = acos(NdL);

	float alpha = max(angleVN, angleLN);
	float beta = min(angleVN, angleLN);
	float gamma = dot(incoming - surface_normal * dot(incoming, surface_normal), outgoing - surface_normal * dot(outgoing, surface_normal));

	float sigmaSquared = 1.0f;

	// calculate A and B
	float A = 1.0 - 0.5 * (sigmaSquared / (sigmaSquared + 0.33));
	float B = 0.45 * (sigmaSquared / (sigmaSquared + 0.09));
	float C = sin(alpha) * tan(beta);

	// put it all together
	float L1 = max(0.0f, NdL) * (A + B * max(0.0f, gamma) * C);

	// get the final color 
	vec3 Color = b->kd * L1 * (1.0f / (float)M_PI);

	// add the result
	r->accumulator.color += Color;

	// construct the new ray for the next bounce
	r->wi.origin = r->location;
	r->wi.direction = outgoing;
	r->wi.length = FLT_MAX;
}

// R for Reflect : conductor/metal bxdf
__device__ void crs::bxdf_CONDUCTOR(Bxdf *b, HitRecord *r, unsigned int seed, unsigned int tid) {

	// calculate the color
	float NdL = glm::dot(r->normal, r->wi.direction);
	vec3 C = (glm::vec3(1.0f) - b->kd) * NdL;

	// add the result
	r->accumulator.color += C;

	// rng state
	curandState rngState;
	curand_init(crs::WangHash(seed)+tid, 0, 0, &rngState);

	// generate a point within a unit sphere and transform according to location and normal
	vec3 t = crs::RandUniformInSphere(&rngState);
	vec3 target = t + r->normal + r->location;

	// reflect the incoming ray
	vec3 ref = r->wi.direction - (2.0f * glm::dot(r->wi.direction, r->normal) * r->normal);

	// perturbate the reflected ray
	vec3 f = (t * b->rpt) + ref;

	// construct the new ray for the next bounce
	r->wi.origin = r->location;
	r->wi.direction = glm::normalize(f);
	r->wi.length = FLT_MAX;
}

// T for Transmit : dielectric/glass bxdf
__device__ void crs::bxdf_DIELECTRIC(Bxdf *b, HitRecord *r, unsigned int seed, unsigned int tid) {

	float	ni_over_nt;			// refraction index
	vec3	refracted;			// refracted ray
	vec3	reflected;			// reflected ray
	vec3	final;				// final ray
	vec3	no;					// outward normal
	bool	refl = false;		// refracted or reflected?
	
	float	cos;				// cosine for schlick approximation
	float	schlick;			// schlick approximation

	float dt = glm::dot(r->wi.direction, r->normal);

	if( dt > 0.0f){
		no = -r->normal;
		ni_over_nt = b->ior;
		cos = b->ior * dt / r->wi.length;

	}else{
		no = r->normal;
		ni_over_nt = 1.0f / b->ior;
		cos = -dt / r->wi.length;
	}

	// reflected or refracted?
	float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - (dt*dt));
	if(discriminant > 0){
		refracted = ni_over_nt * (r->wi.direction - (r->normal * dt)) - r->normal * sqrt(discriminant);

		// calculate schlick approximation, aka the probability of reflection
		float r0 = ( 1.0f - b->ior) / ( 1.0f + b->ior);
		r0 = r0*r0;
		schlick = r0 + (1 - r0)*pow((1.0f - cos), 5.0f);
	}else{
		schlick = 1.0;
	}

	// rng state
	curandState rngState;
	curand_init(crs::WangHash(seed) + tid, 0, 0, &rngState);
	if (curand_uniform(&rngState) < schlick) refl = true;

	if(refl){
		// calculate reflected ray
		reflected = r->wi.direction - (2.0f * glm::dot(r->wi.direction, r->normal) * r->normal);
		final = reflected;
	}else{
		final = refracted;
	}

	// perturbate the resulting ray
	vec3 t = crs::RandUniformInSphere(&rngState);
	vec3 target = t + no + r->location;
	vec3 f = (t * b->rpt) + final;

	// construct the new ray for the next bounce
	r->wi.origin = r->location;
	r->wi.direction = glm::normalize(f);
	r->wi.length = FLT_MAX;

	// calculate the color
	float NdL = glm::dot(r->normal, r->wi.direction);
	vec3 C = (glm::vec3(1.0f) - b->kd) * NdL;

	r->accumulator.color += C;
}

// E for Emission : energy emitting bxdf
__device__ void crs::bxdf_EMISSION(Bxdf *b, HitRecord *r, unsigned int seed, unsigned int tid) {
	// accumulate the bounce
	r->accumulator.color += b->kd;

	// terminate the path
	r->terminated = true;
}

// SS for Subsurface : subsurface bxdf
__device__ void crs::bxdf_SUBSURFACE(Bxdf *b, HitRecord *r, unsigned int seed, unsigned int tid) {
}

// C for constant : return a constant color
__device__ void crs::bxdf_CONSTANT(Bxdf *b, HitRecord *r) {

	// accumulate the bounce
	r->accumulator.color += b->kd;

	// terminate the path
	r->terminated = true;
}

__device__ void crs::bxdf_SIMPLE_SKY(Bxdf *b, HitRecord *r){
	// calculate color
	float t = 0.5 * (r->wi.direction.y + 1.0f);
	vec3 C = ((1.0f - t) * glm::vec3(b->ior)) + (t * b->kd);

	// accumulate the bounce
	r->accumulator.color += C;

	// terminate the path
	r->terminated = true;
}

__device__ void crs::evaluateBxdf(Bxdf *bxdfList, HitRecord *r, PixelBuffer *p, int pathlength, unsigned int seed, unsigned int tid) {
	
	// early exit
	if (r->terminated) return;

	int bid = r->bxdf;
	Bxdf d = bxdfList[0];

	switch (bxdfList[bid].type) {
	case crs::NOHIT:
		bxdf_NOHIT(r);
		break;
	case crs::NORMAL:
		bxdf_NORMAL(r);
		break;
	case crs::LAMBERT:
		bxdf_LAMBERT(&bxdfList[bid], r, seed, tid);
		break;
	case crs::OREN_NAYAR:
		bxdf_OREN_NAYAR(&bxdfList[bid], r, seed, tid);
		break;
	case crs::CONDUCTOR:
		bxdf_CONDUCTOR(&bxdfList[bid], r, seed, tid);
		break;
	case crs::DIELECTRIC:
		bxdf_DIELECTRIC(&bxdfList[bid], r, seed, tid);
		break;
	case crs::EMISSION:
		bxdf_EMISSION(&bxdfList[bid], r, seed, tid);
		break;
	case crs::SUBSURFACE:
		bxdf_SUBSURFACE(&bxdfList[bid], r, seed, tid);
		break;
	case crs::CONSTANT:
		bxdf_CONSTANT(&bxdfList[bid], r);
		break;
	case crs::SIMPLE_SKY:
		bxdf_SIMPLE_SKY(&bxdfList[bid], r);
		break;
	default:
		// no valid bxdf assigned
		if( d.type == crs::SIMPLE_SKY ){
			bxdf_SIMPLE_SKY(&d, r);
		}else{
			bxdf_NOHIT(r);
		}
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
