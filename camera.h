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
#define GLM_RIGHT_HANDED
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
		float		resolution_x;			// sensor width in pixels
		float 		resolution_y;			// sensor height in pixels

		float		fov;					// vertical field of view
		float 		focus_distance;			// distance to the focus plane in world units
		float 		image_plane;			// distance to the image plane in camera space
		float		aperture_radius;		// aperture radius

		vec3 		position;				// camera position in world units
		vec3 		lookat;					// look-at coordinate in world units
		vec3 		up;						// up vector (usually y-up : 0,1,0)
		mat4 		matrix;					// camera matrix

		__host__ __device__ Camera();
		__host__ __device__ ~Camera();

		__host__ __device__ void update();			// recalculate FOV and Matrix
	};

	__global__ void KERNEL_CAST_CAMERA_RAYS(HitRecord *hitrecords, Camera *camera, unsigned int seed);

}

#endif /* CAMERA_H_ */
