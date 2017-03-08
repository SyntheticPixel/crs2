/*
 * camera.h
 *
 *  Created on: 02 Feb 2017
 *      Author: erik
 */

#pragma once
#ifndef CAMERA_H_
#define CAMERA_H_

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

	class Camera{
	public:
		float		resolution_x;		// sensor width in pixels
		float 		resolution_y;		// sensor height in pixels

		float		fov;				// vertical field of view
		float 		focusplane;			// distance to the focus plane in camera space
		float 		imageplane;			// distance to the image plane in camera space
		float		aperture;			// aperture size

		vec3 		position;			// world position
		vec3 		lookat;				// look at coordinate in world
		vec3 		up;					// up vector (usually y-up : 0,1,0)
		mat4 		matrix;				// camera matrix

		__host__ __device__ Camera();
		__host__ __device__ ~Camera();

		__host__ __device__ void update();			// recalculate FOV and Matrix
	};

	__global__ void KERNEL_CAST_CAMERA_RAYS(HitRecord *hitrecords, Camera *camera, unsigned int seed);

}

#endif /* CAMERA_H_ */
