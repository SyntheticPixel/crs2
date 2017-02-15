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
	typedef enum crs_Bxdf_Types {
		// CRAYON BXDF types
		NOHIT = 0x00,				// default bxdf, returns a constant color when no hit has been recorded
		NORMAL,						// returns a color based on the normal at the intersection						
		BSDF,						// bidirectional scattering distribution function
		BRDF,						// bidirectional reflectance distribution function
		BTDF,						// bidirectional transmittance distribution function
		BSSDF,						// bidirectional scattering surface (subsurface) distribution function
		CONSTANT,					// returns a constant color

	}BXDFTYPE;

	struct Bxdf {
		BXDFTYPE		type;			// bxdf type
		vec3			ka;				// albedo color

		__host__ __device__ Bxdf() {
			type = NOHIT;
			ka = vec3(0.0f, 0.0f, 0.0f);
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

	__device__ void bxdf_NOHIT(HitRecord *r, PixelBuffer *p, int pathlength);
	__device__ void bxdf_NORMAL(HitRecord *r, PixelBuffer *p, int pathlength);
	__device__ void bxdf_BSDF(Bxdf *b, HitRecord *r, PixelBuffer *p, int pathlength, unsigned int seed);
	__device__ void bxdf_BRDF(Bxdf *b, HitRecord *r, PixelBuffer *p, int pathlength, unsigned int seed);
	__device__ void bxdf_BTDF(Bxdf *b, HitRecord *r, PixelBuffer *p, int pathlength, unsigned int seed);
	__device__ void bxdf_BSSDF(Bxdf *b, HitRecord *r, PixelBuffer *p, int pathlength, unsigned int seed);
	__device__ void bxdf_CONSTANT(Bxdf *b, HitRecord *r, PixelBuffer *p, int pathlength);

	__device__ void evaluateBxdf(Bxdf *bxdfList, HitRecord *r, PixelBuffer *p, int pathlength, unsigned int seed);
	__global__ void KERNEL_BXDF(Bxdf *bxdfList, HitRecord *hitRecord, PixelBuffer *pixelBuffer, int width, int height, int pathlength, unsigned int seed);
}

#endif /* BXDF_H_ */
