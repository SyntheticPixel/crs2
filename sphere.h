/*
 * sphere.h
 *
 *  Created on: 07 Feb 2017
 *      Author: erik
 */

#ifndef SPHERE_H_
#define SPHERE_H_


#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm.hpp>

#include "crs.h"
#include "ray.h"
#include "hit_record.h"

using namespace std;
using namespace glm;
using namespace crs;

namespace crs{
	class Sphere{
	public:
		vec3 		center;			// position of centerin world units
		float 		radius;			// radius in world units
		int			bxdf;			// material index

		__host__ __device__ Sphere();
		__host__ __device__ ~Sphere();

	};

	__device__ float SphereHit(Sphere *s, Ray *r);
	__device__ void TestSphereIntersections(Sphere *sphere, unsigned int c, HitRecord *r);
	__global__ void KERNEL_SPHEREINTERSECT(Sphere *spheres, unsigned int count, HitRecord *hitrecords, int w, int h);

}

#endif /* SPHERE_H_ */
