#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define GLM_FORCE_CUDA
#include <glm.hpp>

#include "sphere.h"

using namespace std;
using namespace crs;

__device__ float crs::SphereHit(Sphere *s, Ray *r){

	vec3 oc = r->origin - s->center;
	float a = glm::dot(r->direction, r->direction);
	float b = 2.0f * glm::dot(oc, r->direction);
	float c = glm::dot(oc, oc) - s->radius * s->radius;
	float discriminant = b*b - 4*a*c;

	if(discriminant < 0){
		return -1.0f;
	}else{
		return ( -b - glm::sqrt(discriminant)) / (2.0f * a);
	}

}

__device__ void crs::TestSphereIntersections(Sphere *sphere, unsigned int c, HitRecord *r){

	// early exit
	if (r->terminated) return;

	// loop over every sphere
	unsigned int i;
	for(i=0; i < c; i++){
		// local copy
		Sphere s = sphere[i];

		float t = SphereHit(&s, &r->wi);

		// make sure we keep the closest intersection
		if (t >= r->wi.length){
			return;
		}else{
			// we have a hit
			if(t > 0.0001f){
				r->wi.length = t;
				r->location = r->wi.evaluate();
				r->normal = glm::normalize(r->location - s.center);
				r->bxdf = s.bxdf;
			}
		}
	}
}

__global__ void crs::KERNEL_SPHEREINTERSECT(Sphere *spheres, unsigned int count, HitRecord *hitrecords, int w, int h){

	unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned long threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	if (threadId >= w * h) return;

	// test for sphere intersections
	TestSphereIntersections(spheres, count, &hitrecords[threadId]);
}

crs::Sphere::Sphere(){
	center = vec3(0.0f, 0.0f, 1.0f);		// default position in world units
	radius = 1.0f;							// default radius in world units
	bxdf = 0;								// we're assuming there's at least one bxdf
}

crs::Sphere::~Sphere(){
}
