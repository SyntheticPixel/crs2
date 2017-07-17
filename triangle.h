/*
 * triangle.h
 *
 *  Created on: 17 Jul 2017
 *      Author: erik
 */

#ifndef TRIANGLE_H_
#define TRIANGLE_H_

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
	class Triangle{
	public:
		vec3 		v0;			// vertex 0
		vec3 		v1;			// vertex 1
		vec3 		v2;			// vertex 2
		int			bxdf;			// material index

		__host__ __device__ Triangle();
		__host__ __device__ ~Triangle();

	};

	__device__ float TriangleHit(Triangle *t, Ray *r);
	__device__ void TestTriangleIntersections(Triangle *tris, unsigned int c, HitRecord *r);
	__global__ void KERNEL_TRIANGLEINTERSECT(Triangle *trisList, unsigned int count, HitRecord *hitrecords, int w, int h);

}



#endif /* TRIANGLE_H_ */
