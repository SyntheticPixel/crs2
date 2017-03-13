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
#include "pixelbuffer.h"

using namespace std;

namespace crs{
	class HitRecord{
	public:
		crs::Ray 			wi;					// incoming ray
		vec3				location;			// location of the hit
		vec3				normal;				// normal at location
		int					bxdf;				// material index
		PixelBuffer			accumulator;		// accumulator for the color/bounce
		bool				terminated;			// termination flag

		__host__ __device__ HitRecord(){
			wi = Ray();
			location = vec3(0.0f, 0.0f, 0.0f);
			normal = vec3(0.0f, 0.0f, 0.0f);
			bxdf = crs::NOHIT;
			accumulator.color = vec3(1.0f, 1.0f, 1.0f);
			accumulator.samples = 0;
			terminated = false;
		};

		__host__ __device__ ~HitRecord(){};

	};
}

#endif /* HIT_H_ */
