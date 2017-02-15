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
#define M_PI 3.1415927410125732421875
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
	Generate a uniform distribution within a a 1D range of [0, 1)
	*/
	inline __device__ float RandUniform1D(curandState *s){
		return 1.0f - curand_uniform(s);
	}

	/*
	Generate a uniform distribution within a unit square
	*/
	inline __device__ glm::vec2 RandUniformSquare(curandState *s){
		glm::vec2 v;
		v.x = 1.0f - curand_uniform(s);
		v.y = 1.0f - curand_uniform(s);
		return v;
	}

	/*
	Generate a uniform distribution on a unit disc
	*/
	inline __device__ glm::vec2 RandUniformDisc(curandState *s){
		glm::vec2 p;

		float t = curand_uniform(s) * 2 * M_PI;
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

		float x = curand_uniform(s) * 2 * M_PI;
		float y = curand_uniform(s);

		vec3 t = abs(normal.x) > 0.1 ? vec3(-normal.z, 0, normal.x) : vec3(0, -normal.z, normal.y);
		vec3 u = glm::normalize(t);
		vec3 v = glm::cross(normal, u);

		p = (u*cos(x) + v*sin(x))*sqrt(y) + normal*sqrt(1-y);

		return p;
	}

	/*
	Given a radius, generate a unform distribution over a sphere
	*/
	inline __device__ glm::vec3 RandUniformSphere(curandState *s, float radius){
		glm::vec3 p;

		float u = curand_uniform(s);
		float v = curand_uniform(s);

		float theta = u * 2.0f * M_PI;
		float phi = acos(2.0f * v - 1.0f);

		p.x = radius * sin(phi) * cos(theta);
		p.y = radius * sin(phi) * sin(theta);
		p.z = radius * cos(phi);

		return p;
	}

}

#endif
