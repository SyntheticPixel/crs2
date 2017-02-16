/*
 * hit.h
 *
 *  Created on: 06 Feb 2017
 *      Author: erik
 */

#ifndef HIT_H_
#define HIT_H_

#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm.hpp>

#include "crs.h"
#include "ray.h"

using namespace std;

namespace crs{
	class HitRecord{
	public:
		//vec3				result;				// accumulator for each bounce

		crs::Ray 			in;					// incoming ray
		vec3				location;			// location of the hit
		vec3				normal;				// normal at location

		int					hits;				// number of hits; 0 = no intersections recorded
		int					bxdf;				// material index

		int					pathcounter;		// current bounce
		bool				is_terminated;		// termination flag

		__host__ __device__ HitRecord(){
			in = Ray();
			location = vec3(0.0f, 0.0f, 0.0f);
			normal = vec3(0.0f, 0.0f, 0.0f);
			hits = 0;
			bxdf = -1;
			pathcounter = 0;
			is_terminated = false;
		};

		__host__ __device__ ~HitRecord(){};

		__host__ __device__ void reset(){
			//result = vec3(0.0f, 0.0f, 0.0f);
			in.origin = vec3(0.0f, 0.0f, 0.0f);
			in.direction = vec3(0.0f, 0.0f, 0.0f);
			in.frequency = 0.0f;
			in.time = 0.0f;
			in.length = FLT_MAX;
			location = vec3(0.0f, 0.0f, 0.0f);
			normal = vec3(0.0f, 0.0f, 0.0f);
			hits = 0;
			bxdf = 0;		// in the worst case, we hit the default bxdf
			pathcounter = 0;
			is_terminated = false;
		};

	};
}

#endif /* HIT_H_ */
