/*
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
CRAYON RENDER
Experiments in bidirectional pathtracing
Copyright Erik Veldeman
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
*/

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

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880168872420969808
#endif

#include "crs.h"

using namespace std;
using namespace glm;
using namespace crs;

namespace crs{
	/*
	@brief
	Wanghash for seeds
	*/
	inline __host__ __device__ unsigned int WangHash(unsigned int a) {
		a = (a ^ 61) ^ (a >> 16);
		a = a + (a << 3);
		a = a ^ (a >> 4);
		a = a * 0x27d4eb2d;
		a = a ^ (a >> 15);
		return a;
	}

	/*
	Generate a uniform distribution within a unit square
	*/
	inline __device__ glm::vec2 RandUniformSquare(curandState *s){
		glm::vec2 v;
		v.x = curand_uniform(s);
		v.y = curand_uniform(s);
		return v;
	}

	/*
	Generate a uniform distribution on a unit disc
	*/
	inline __device__ glm::vec2 RandUniformDisc(curandState *s){
		glm::vec2 p;

		float t = curand_uniform(s) * 2 * (float)M_PI;
		float r = 2 * curand_uniform(s);
		if(r > 2){r -= 2.0f;}

		p.x = cos(t)*r;
		p.y = sin(t)*r;

		return p;
	}

	/*
	Given a normal, generate a cosine weighted distribution on a unit hemisphere
	*/
	inline __device__ glm::vec3 RandCosineHemisphere(curandState *s, glm::vec3 normal){
		glm::vec3 p;

		float x = curand_uniform(s) * 2 * (float)M_PI;
		float y = curand_uniform(s);

		vec3 t = abs(normal.x) > 0.1 ? vec3(-normal.z, 0, normal.x) : vec3(0, -normal.z, normal.y);
		vec3 u = glm::normalize(t);
		vec3 v = glm::cross(normal, u);

		p = (u*cos(x) + v*sin(x))*sqrt(y) + normal*sqrt(1-y);

		return p;
	}

	/*
	Generate a uniform distribution within a unit sphere
	*/
	inline __device__ glm::vec3 RandUniformInSphere(curandState *s){
		glm::vec3 p;

		// generate a point in a unit sphere using the rejection method
		do{
			p.x = 2.0f * curand_uniform(s) - 1.0f;
			p.y = 2.0f * curand_uniform(s) - 1.0f;
			p.z = 2.0f * curand_uniform(s) - 1.0f;
		} while( glm::length(p) >= 1.0f );

		return p;
	}

	/*
	 * Given a radius, generate random point on a sphere
	 */
	inline __device__ glm::vec3 RandUniformOnSphere(curandState *s, float radius){
		glm::vec3 p;

		float u = curand_uniform(s);
		float v = curand_uniform(s);

		float theta = u * 2.0f * (float)M_PI;
		float phi = acos(2.0f * v - 1.0f);

		p.x = radius * sin(phi) * cos(theta);
		p.y = radius * sin(phi) * sin(theta);
		p.z = radius * cos(phi);

		return p;
	}

}

#endif
