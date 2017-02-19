#include "cuda_context.h"

using namespace std;
using namespace crs;

__device__ void crs::BufferInit(HitRecord *r, PixelBuffer *p){
	// init hitrecord buffer
	*r = HitRecord();

	// init all pixel buffer
	*p = PixelBuffer();

}

__global__ void crs::KERNEL_INIT(HitRecord *hitrecords, PixelBuffer *buffer, int w, int h){

	unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned long threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	if (threadId >= w * h) return;

	// init buffers
	BufferInit(&hitrecords[threadId], &buffer[threadId]);
}

__global__ void crs::KERNEL_ACCUMULATE(HitRecord *hitrecords, PixelBuffer *buffer, int w, int h) {

	unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned long threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	if (threadId >= w * h) return;

	// accumulate
	buffer[threadId].color += hitrecords[threadId].accumulator.color;
	buffer[threadId].samples++;

	// reset hitrecords for next sample
	hitrecords[threadId].reset();

}

CudaContext::CudaContext(){
	cuda_device_count = 0;
	cuda_device_props = NULL;

	cuda_error = cudaSuccess;

	width = 1;
	height = 1;
	samples = 1;
	depth = 1;
	dimension = k2D;

	host_pixels = NULL;
	device_pixels = NULL;

	free_memory = 0;
	total_memory = 0;
	host_occupied_memory = 0;
	device_occupied_memory = 0;
	pixel_buffer_size = 0;
	hitRecord_buffer_size = 0;

	state = crs::CRS_NO_ERROR;

	blockSize = dim3(1,1,1);
	gridSize = 1;
}

CudaContext::~CudaContext(){
	if(cuda_device_props != NULL) delete cuda_device_props;
}

void CudaContext::EvaluateState(){
	switch(state){
	case crs::CRS_NO_ERROR:
		break;
	case crs::CRS_ABORT:
		CleanupDevice();
		CleanupHost();
		cudaDeviceReset();
		exit(EXIT_FAILURE);
		break;
	case crs::CRS_NO_CUDA_DEVICE:
		cout << " CRS ERROR: NO_CUDA_DEVICE" << endl;
		break;
	case crs::CRS_NOT_ENOUGH_MEMORY:
		cout << " CRS ERROR: NOT_ENOUGH_MEMORY" << endl;
		cout << " Available :" << free_memory / (1024 * 1024) << " Mb" << std::endl;
		break;
	case crs::CRS_KERNEL_LAUNCH_ERROR:
		cout << " CRS ERROR: KERNEL_LAUNCH_ERROR" << endl;
		break;
	case crs::CRS_ERROR_DEVICE_CLEANUP:
		cout << " CRS ERROR: ERROR_DEVICE_CLEANUP" << endl;
		break;
	case crs::CRS_ERROR_DEVICE_MEMORY:
		cout << " CRS ERROR: ERROR_DEVICE_MEMORY" << endl;
		break;
	case crs::CRS_ERROR_HOST_CLEANUP:
		cout << " CRS ERROR: ERROR_HOST_CLEANUP" << endl;
		break;
	case crs::CRS_ERROR_HOST_MEMORY:
		cout << " CRS ERROR: ERROR_HOST_MEMORY" << endl;
		break;
	case crs::CRS_ERROR_UNKNOWN:
		cout << " CRS ERROR: ERROR_UNKNOWN" << endl;
		break;
	default:
		break;
	}
	return;
}

void CudaContext::CheckCudaError(std::string successMessage){
	cuda_error = cudaGetLastError();
	if (cuda_error != cudaSuccess){
		const char *errorString = cudaGetErrorString(cuda_error);
		cout << " !ERROR! CUDA Reported : " << errorString << std::endl;
	}

	// if there's a success message, show it
	if (successMessage != "") cout << successMessage << std::endl;
}

void CudaContext::GetDeviceProps(){
	// Make sure we have a cuda device present. If not, exit
	cuda_error = cudaGetDeviceCount(&cuda_device_count);
	CheckCudaError("");

	if (cuda_device_count == 0){
		state = crs::CRS_NO_CUDA_DEVICE;
	}

	cuda_device_props = (cudaDeviceProp*) malloc( sizeof(cudaDeviceProp)*cuda_device_count );

	int device;
	for (device = 0; device < cuda_device_count; device++){
		cudaSetDevice(device);
		cudaGetDeviceProperties(&cuda_device_props[device], device);
	}

	// check for available memory
	cuda_error = cudaMemGetInfo( &free_memory, &total_memory ) ;
	if(free_memory < MIN_REQUIRED_MEMORY) state = crs::CRS_NOT_ENOUGH_MEMORY;
}

void CudaContext::CalculateLaunchParamaters(){

	switch(dimension){
	case k1D:
		blockSize = dim3(256, 1, 1);	// 256 threads
		gridSize = dim3((width + blockSize.x - 1) / blockSize.x, 1, 1);
		break;
	case k2D:
		blockSize = dim3(16, 16, 1);	// 256 threads
		gridSize = dim3((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, 1);
		break;
	case k3D:
		blockSize = dim3(8, 8, 8);	// 512 threads
		gridSize = dim3((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, (samples + blockSize.z - 1) / blockSize.z);
		break;
	}
}

void CudaContext::PrintCudaContext(){

	// For each device, get the attributes
	int device, driverVersion = 0, runtimeVersion = 0;
	for (device = 0; device < cuda_device_count; device++){
		cudaSetDevice(device);
		cudaGetDeviceProperties(&cuda_device_props[device], device);
		cout << "--------------------------------------------------------------------------------------------- " << std::endl;
		cout << " CUDA Device found : " << cuda_device_props[device].name << std::endl;

		cuda_error = cudaDriverGetVersion(&driverVersion);
		CheckCudaError("");

		cuda_error = cudaRuntimeGetVersion(&runtimeVersion);
		CheckCudaError("");

		cout << "--------------------------------------------------------------------------------------------- " << std::endl;
		cout << " CUDA Driver version : " << driverVersion << std::endl;
		cout << " CUDA Runtime version : " << runtimeVersion << std::endl;
		cout << " CUDA Capabilities : " << cuda_device_props[device].major << "." << cuda_device_props[device].minor << std::endl;
		cout << "--------------------------------------------------------------------------------------------- " << std::endl;
		cout << " Total memory on CUDA device: " << total_memory / (1024 * 1024) << " Mb" << std::endl;
		cout << " Available memory on CUDA device: " << free_memory / (1024 * 1024) << " Mb" << std::endl;
		cout << " Available SM processors : " << cuda_device_props[device].multiProcessorCount << std::endl;
		cout << " Max. shared memory per SM processor: " << cuda_device_props[device].sharedMemPerMultiprocessor / 1024 << " Kb" << std::endl;
		cout << " Max. shared memory per Block : " << cuda_device_props[device].sharedMemPerBlock / 1024 << " Kb" << std::endl;
		cout << " Max. number of threads per Block : " << cuda_device_props[device].maxThreadsPerBlock << std::endl;
		cout << " Max. Warp size : " << cuda_device_props[device].warpSize << " threads" << std::endl;
		cout << " Max. size X dimension of a Grid : " << cuda_device_props[device].maxGridSize[0] << std::endl;
		cout << " Max. size Y dimension of a Grid : " << cuda_device_props[device].maxGridSize[1] << std::endl;
		cout << " Max. size Z dimension of a Grid : " << cuda_device_props[device].maxGridSize[2] << std::endl;
		cout << " Max. size X dimension of a Block : " << cuda_device_props[device].maxThreadsDim[0] << std::endl;
		cout << " Max. size Y dimension of a Block : " << cuda_device_props[device].maxThreadsDim[1] << std::endl;
		cout << " Max. size Z dimension of a Block : " << cuda_device_props[device].maxThreadsDim[2] << std::endl;
		cout << " Max. concurrent Kernels : " << cuda_device_props[device].concurrentKernels << std::endl;
		cout << "--------------------------------------------------------------------------------------------- " << std::endl;
		cout << "--------------------------------------------------------------------------------------------- " << std::endl;
	}
}

void CudaContext::SetupHostMemory(){
	// Pixelbuffer
	pixel_buffer_size = sizeof(crs::PixelBuffer)*width*height;
	cuda_error = cudaHostAlloc((void**)&host_pixels, pixel_buffer_size, cudaHostAllocWriteCombined);
	if (cuda_error != cudaSuccess){
		state = crs::CRS_NOT_ENOUGH_MEMORY;
	}else{
		cout << " Required memory for the pixel buffer " << pixel_buffer_size/(1024 * 1024) << " Mb" << std::endl;
		cout << " Assigned pixel buffer on host..." << std::endl;
	}

}

void CudaContext::SetupDeviceMemory(){
	// Pixel Buffer
	pixel_buffer_size = sizeof(crs::PixelBuffer)*width*height;
	cuda_error = cudaMalloc((void**)&device_pixels, pixel_buffer_size);
	if (cuda_error != cudaSuccess){
		state = crs::CRS_ERROR_DEVICE_MEMORY;
	}else{
		cout << " Assigned pixel buffer on device..." << std::endl;
		//cudaMemset((void**)&device_pixels, 0, pixel_buffer_size);
	}

	// HitRecord Buffer
	hitRecord_buffer_size = sizeof(crs::HitRecord)*width*height;
	cuda_error = cudaMalloc((void**)&device_hitRecords, hitRecord_buffer_size);
	if (cuda_error != cudaSuccess) {
		state = crs::CRS_ERROR_DEVICE_MEMORY;
	}
	else {
		cout << " Required memory for the hitRecord buffer " << hitRecord_buffer_size / (1024 * 1024) << " Mb" << std::endl;
		cout << " Assigned hit buffer on device..." << std::endl;
		//cudaMemset((void**)&device_rays, 0, hitRecord_buffer_size);
	}

	// Bxdfs
}

void CudaContext::CopyPixelBufferFromDeviceToHost(){
	cuda_error = cudaMemcpy(host_pixels, device_pixels, pixel_buffer_size, cudaMemcpyDeviceToHost);
	if( cuda_error != cudaSuccess){
		state = crs::CRS_ERROR_COPY_FROM_DEVICE;
	}else{
		cout << " Copied the pixel buffer from the device..." << std::endl;
	}

}

void CudaContext::CleanupDevice(){

	// ON THE DEVICE
	// -------------

	// delete the pixel buffer
	if( device_pixels != NULL ){
		cuda_error = cudaFree(device_pixels);
		if( cuda_error != cudaSuccess){
			state = crs::CRS_ERROR_DEVICE_CLEANUP;
		}else{
			cout << " Deleted the pixel buffer on the device..." << std::endl;
			device_pixels = NULL;
		}
	}

	// delete the hitrecord buffer
	if (device_hitRecords != NULL) {
		cuda_error = cudaFree(device_hitRecords);
		if (cuda_error != cudaSuccess) {
			state = crs::CRS_ERROR_DEVICE_CLEANUP;
		}
		else {
			cout << " Deleted the hitRecord buffer on the device..." << std::endl;
			device_hitRecords = NULL;
		}
	}
}

void CudaContext::CleanupHost(){
	// ON THE HOST
	// -----------

	// delete the PixelBuffer
	if( host_pixels != NULL ){
		cuda_error = cudaFreeHost(host_pixels);
		if( cuda_error != cudaSuccess){
			state = crs::CRS_ERROR_HOST_CLEANUP;
		}else{
			cout << " Deleted the pixel buffer on the host..." << std::endl;
			host_pixels = NULL;
		}
	}
}
