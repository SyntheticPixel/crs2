/*
 * color.h
 *
 *  Created on: 13 Mar 2017
 *      Author: erik
 */

#ifndef COLOR_H_
#define COLOR_H_

#pragma once
#ifndef __RAND_H_
#define __RAND_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define GLM_FORCE_CUDA
#include <glm.hpp>

#include <math.h>

#include "crs.h"
#include "ray.h"

namespace crs{

	__device__ __host__ vec3 RGBtoXYZ( vec3 rgb );
	__device__ __host__ vec3 XYZtoRGB( vec3 xyz );
	__device__ __host__ vec3 SPECTRALtoXYZ( float l );

}

#endif /* COLOR_H_ */
