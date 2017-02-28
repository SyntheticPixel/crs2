/*
 * pixelbuffer.h
 *
 *  Created on: 02 Feb 2017
 *      Author: erik
 */

#pragma once
#ifndef PIXELBUFFER_H_
#define PIXELBUFFER_H_

#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm.hpp>

#include "crs.h"

using namespace std;
using namespace glm;
using namespace crs;

namespace crs{
	struct PixelBuffer{
		vec3 color;
		unsigned int samples;

		__host__ __device__ PixelBuffer(){
			color = vec3(1.0f, 1.0f, 1.0f);
			samples = 0;
		}

		__host__ __device__ ~PixelBuffer(){
		}

		__host__ __device__  vec3 getAverage(){
			// return result
			if(samples != 0){
				return color / (float)samples;
			}else{
				return color;
			}
		}

	};
}

#endif /* PIXELBUFFER_H_ */
