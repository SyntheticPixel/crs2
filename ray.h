/*
 * ray.h
 *
 *  Created on: 02 Feb 2017
 *      Author: erik
 */

#pragma once
#ifndef RAY_H_
#define RAY_H_

#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm.hpp>

#include "crs.h"

using namespace std;
using namespace glm;
using namespace crs;

namespace crs{

	/*
	@brief
	Define a ray using it's origin, normalized direction and parameter t (length)
	We need to do something like this : P' = o + d*t
	*/
	class Ray{
	public:
		vec3 	origin;				// ray origin
		vec3 	direction;			// ray direction, normalized
		float	length;				// ray length
		float	time;				// time of casting
		float	frequency;			// lambda, light frequency of the ray

		__host__ __device__ Ray(){
			origin = vec3(0.0f, 0.0f, 0.0f);
			direction = vec3(0.0f, 0.0f, 0.0f);
			length = FLT_MAX;		// initialize to max length
			time = 0.0f;
			frequency = 0.0f;
		};

		inline __host__ __device__ vec3  evaluate(){
			return origin + ( direction * length );

		}

		inline __host__ __device__ vec3  evaluate(float l){
			return origin + ( direction * l );

		}

	};

}

#endif /* RAY_H_ */
