/*
* main.h
*
*  Created on: 12 Feb 2017
*      Author: erik
*/

#pragma once
#ifndef BXDF_H_
#define BXDF_H_

#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm.hpp>

#include "crs.h"
#include "ray.h"
#include "hit_record.h"
#include "pixelbuffer.h"
#include "rand.h"

using namespace std;
using namespace glm;

namespace crs {

	struct Bxdf {
		BXDFTYPE		type;			// bxdf type
		vec3			kd;				// diffuse reflection/absorption
		float			rpt;			// ray perturbation : 0.0 = mirror, 1.0 = lambert
		float			ior;			// index of reflection/refraction


		__host__ __device__ Bxdf() {
			type = NOHIT;
			kd = vec3(0.0f, 0.0f, 0.0f);
			rpt = 0.0f;
			ior = 1.0f;
		}

		__host__ __device__ ~Bxdf() {}
		
	};

	struct BxdfTocEntry {
		std::string		bxdf_name;		// key
		int				id;				// value

		BxdfTocEntry(){
			bxdf_name = "default";
			id = 0;
		}

		~BxdfTocEntry() {}
	};

	struct BxdfTable {
		unsigned int		size;			// number of bxdfs
		BxdfTocEntry		*toc;			// table of contents of the bxdfs

		BxdfTable() {
			size = 0;
			toc = NULL;
		}

		~BxdfTable() {
			if (toc != NULL) delete[] toc;

		}

		int getBxdfIdbyName(std::string bxdfname);
	};

	__device__ void bxdf_NOHIT(HitRecord *r);
	__device__ void bxdf_NORMAL(HitRecord *r);
	__device__ void bxdf_LAMBERT(Bxdf *b, HitRecord *r, unsigned int seed, unsigned int tid);
	__device__ void bxdf_OREN_NAYAR(Bxdf *b, HitRecord *r, unsigned int seed, unsigned int tid);
	__device__ void bxdf_CONDUCTOR(Bxdf *b, HitRecord *r, unsigned int seed, unsigned int tid);
	__device__ void bxdf_DIELECTRIC(Bxdf *b, HitRecord *r, unsigned int seed, unsigned int tid);
	__device__ void bxdf_EMISSION(Bxdf *b, HitRecord *r, unsigned int seed, unsigned int tid);
	__device__ void bxdf_SUBSURFACE(Bxdf *b, HitRecord *r, unsigned int seed, unsigned int tid);
	__device__ void bxdf_CONSTANT(Bxdf *b, HitRecord *r);
	__device__ void bxdf_SIMPLE_SKY(Bxdf *b, HitRecord *r);

	__device__ void evaluateBxdf(Bxdf *bxdfList, HitRecord *r, PixelBuffer *p, int pathlength, unsigned int seed, unsigned int tid);
	__global__ void KERNEL_BXDF(Bxdf *bxdfList, HitRecord *hitRecord, PixelBuffer *pixelBuffer, int width, int height, int pathlength, unsigned int seed);
}

#endif /* BXDF_H_ */
