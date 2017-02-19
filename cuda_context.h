/*
 * cuda_context.h
 *
 *  Created on: 02 Feb 2017
 *      Author: erik
 */

#pragma once
#ifndef CUDA_CONTEXT_H_
#define CUDA_CONTEXT_H_

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_occupancy.h>
#include <cuda_profiler_api.h>

#include "crs.h"
#include "pixelbuffer.h"
#include "ray.h"
#include "hit_record.h"
#include "bxdf.h"

using namespace std;
using namespace crs;

namespace crs{

	enum ProblemDimension{
		k1D,
		k2D,
		k3D,

	};

	/*
	 * Global threadID functions
	 */

	// 1D grid of 1D blocks
	__device__ inline int getGlobalIdx_1D_1D() { return blockIdx.x * blockDim.x + threadIdx.x; }

	// 1D grid of 2D blocks
	__device__ inline int getGlobalIdx_1D_2D() { return blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; }

	// 1D grid of 3D blocks
	__device__ inline int getGlobalIdx_1D_3D() { return blockIdx.x * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x; }

	// 2D grid of 1D blocks
	__device__ inline int getGlobalIdx_2D_1D() {
		int blockId = blockIdx.y * gridDim.x + blockIdx.x;
		int threadId = blockId * blockDim.x + threadIdx.x;
		return threadId;
	}

	// 2D grid of 2D blocks
	__device__ inline int getGlobalIdx_2D_2D() {
		int blockId = blockIdx.x + blockIdx.y * gridDim.x;
		int threadId = blockId * (blockDim.x * blockDim.y)
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
		return threadId;
		}

	// 2D grid of 3D blocks
	__device__ inline int getGlobalIdx_2D_3D() {
		int blockId = blockIdx.x + blockIdx.y * gridDim.x;
		int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
		+ (threadIdx.z * (blockDim.x * blockDim.y))
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
		return threadId;
		}

	// 3D grid of 1D blocks
	__device__ inline int getGlobalIdx_3D_1D() {
		int blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
		int threadId = blockId * blockDim.x + threadIdx.x;
		return threadId;
		}

	// 3D grid of 2D blocks
	__device__ inline int getGlobalIdx_3D_2D() {
		int blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
		int threadId = blockId * (blockDim.x * blockDim.y)
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
		return threadId;
	}

	// 3D grid of 3D blocks
	__device__ inline int getGlobalIdx_3D_3D() {
		int blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
		int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
		+ (threadIdx.z * (blockDim.x * blockDim.y))
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
		return threadId;
	}

	class CudaContext{
	public:
		int 				cuda_device_count;
		cudaError_t 		cuda_error;
		cudaDeviceProp 		*cuda_device_props;

		size_t 				free_memory;				// Free memory available on startup
		size_t 				total_memory;				// Total memory on device
		size_t 				host_occupied_memory;		// Counter for host memory
		size_t 				device_occupied_memory;		// Counter for device memory
		size_t				pixel_buffer_size;			// Size of the pixel buffer
		size_t				hitRecord_buffer_size;		// Size of the hit/intersection buffer
		size_t				bxdf_buffer_size;			// Size of the bxdf buffer

		PixelBuffer			*host_pixels;
		PixelBuffer			*device_pixels;
		HitRecord			*device_hitRecords;
		BxdfTable			*host_bxdfs;
		BxdfTable			*device_bxdfs;

		STATUS				state;

		dim3 				blockSize;
		dim3 				gridSize;

		int 				width;
		int 				height;
		int 				samples;
		int 				depth;

		ProblemDimension 	dimension;

		CudaContext();
		~CudaContext();

		// Evaluate the current state
		void EvaluateState();

		// Get some device stats
		void GetDeviceProps();

		// Check for a CUDA error
		void CheckCudaError(std::string successMessage);

		// Print some environment stats
		void PrintCudaContext();

		// Calculate kernel launchparameters
		void CalculateLaunchParamaters();

		// Assign host memory
		void SetupHostMemory();

		// Assign device memory
		void SetupDeviceMemory();

		// Copy pixelbuffer from device to host
		void CopyPixelBufferFromDeviceToHost();

		// Clean up all device memory assignments
		void CleanupDevice();

		// Clean up all host memory assignments
		void CleanupHost();

	};

	__device__ void BufferInit(HitRecord *r, PixelBuffer *p);
	__global__ void KERNEL_INIT(HitRecord *hitrecords, PixelBuffer *buffer, int w, int h);
	__global__ void KERNEL_ACCUMULATE(HitRecord *hitrecords, PixelBuffer *buffer, int w, int h);

}

#endif /* CUDA_CONTEXT_H_ */
