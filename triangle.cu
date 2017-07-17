#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define GLM_FORCE_CUDA
#include <glm.hpp>

#include "triangle.h"

using namespace std;
using namespace crs;

__device__ float crs::TriangleHit(Triangle *tris, Ray *r){
		// Muller-Trumbore ray/triangle intersection test
		// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm

		vec3 e1 = tris->v1 - tris->v0;
	    vec3 e2 = tris->v2 - tris->v0;

	    vec3 dir = r->direction;
	    vec3 orig = r->origin;

	    // Calculate planes normal vector
	    vec3 pvec = glm::cross(dir, e2);
	    float det = glm::dot(e1, pvec);

	    // Ray is parallel to plane
	    if (det < 1e-8 && det > -1e-8) {
	        return 0;
	    }

	    float inv_det = 1.0f / det;
	    vec3 tvec = orig - tris->v0;
	    float u = glm::dot(tvec, pvec) * inv_det;
	    if (u < 0 || u > 1) {
	        return 0.0f;
	    }

	    vec3 qvec = glm::cross(tvec, e1);
	    float v = glm::dot(dir, qvec) * inv_det;
	    if (v < 0 || u + v > 1) {
	        return 0.0f;
	    }

	    return glm::dot(e2, qvec) * inv_det;
}

__device__ void crs::TestTriangleIntersections(Triangle *trisList, unsigned int c, HitRecord *r){
	// early exit
	if (r->terminated) return;

	// loop over every sphere
	unsigned int i;
	for(i=0; i < c; i++){
		// local copy
		Triangle t = trisList[i];

		float d = TriangleHit(&t, &r->wi);

		// make sure we keep the closest intersection
		if (d >= r->wi.length){
			return;
		}else{
			// we have a hit
			if(d > 0.00001f){
				// calculate normal
				vec3 n = glm::cross(t.v1 - t.v0, t.v2 - t.v1);

				r->wi.length = d;
				r->location = r->wi.evaluate();
				r->normal = glm::normalize(n);
				r->bxdf = t.bxdf;
			}
		}
	}

}

__global__ void crs::KERNEL_TRIANGLEINTERSECT(Triangle *trisList, unsigned int count, HitRecord *hitrecords, int w, int h){

	unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned long threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	if (threadId >= w * h) return;

	// test for sphere intersections
	TestTriangleIntersections(trisList, count, &hitrecords[threadId]);
}

crs::Triangle::Triangle(){
	v0 = vec3(0.0f, 0.0f, 0.0f);		// v0
	v1 = vec3(1.0f, 1.0f, 0.0f);		// v1
	v2 = vec3(1.0f, 0.0f, 0.0f);		// v2
	bxdf = 0;							// we're assuming there's at least one bxdf
}

crs::Triangle::~Triangle(){
}
