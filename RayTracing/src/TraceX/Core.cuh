#pragma once

#ifdef DLL_BUILD_TX
#define TRACEX_API __declspec(dllexport)
#else
#define TRACEX_API __declspec(dllimport)
#endif

#define PI 3.14159265359
#define SQRT3 1.732050808 
#define RUINT_T register unsigned int

#ifdef RELEASE
	#define ASSERT(x)
#else
	#define ASSERT(x) if(!x) __debugbreak();
#endif

//Cuda Defines

#define __CUDAGLOBAL __device__ __host__

//Simple IS
#define CUDA_CALL(x) if(x != cudaSuccess) {std::cout << "[Cuda error]: " << x; __debugbreak();}
#define CUDA_FREE(x) CUDA_CALL(cudaFree(x))
#define CUDA_CPY2DEV(dst, src, size) CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice))
#define CUDA_CPY2HOST(dst, src, size) CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost))
#define CUDA_ALLOC(dst, size) CUDA_CALL(cudaMalloc((void**)&dst, size))
#define CUDA_GETDEVPROP(buffer) buffer = new cudaDeviceProp;\
								CUDA_CALL(cudaGetDeviceProperties(buffer, 0))
#define $(x, y) <<<x, y>>> 

//Complex IS
#define CUDA_CREATE(dst, src, size) CUDA_ALLOC(dst, size);\
									CUDA_CPY2DEV(dst, src, size)

#define CUDA_SHOWDEVPROP() cudaDeviceProp* ______data_______;\
						   CUDA_GETDEVPROP(______data_______);\
std::cout << "[MAX DIMGRID]: " << ______data_______->maxGridSize[0] << ", " << ______data_______->maxGridSize[1] << ", " << ______data_______->maxGridSize[2] << "\n";\
std::cout << "[MAX DIMBLOCK]: " << ______data_______->maxThreadsPerBlock << "\n";\
std::cout << "[NUMBER OF SM]: " << ______data_______->multiProcessorCount << "\n";\
std::cout << "[MAX THREADS PER SM]: " << ______data_______->maxThreadsPerMultiProcessor << "\n";\
std::cout << "[MAX BLOCKS PER SM]: " << ______data_______->maxBlocksPerMultiProcessor << "\n";\
std::cout << "[MAX WARPS PER SM]: " << ______data_______->maxThreadsPerMultiProcessor / ______data_______->warpSize << "\n";\
std::cout << "[WARP SIZE]: " << ______data_______->warpSize << "\n";\
std::cout << "[PER BLOCK REGS]: " << ______data_______->regsPerBlock << "\n";\
std::cout << "[SHARED MEM PER BLOCK]: " << ______data_______->sharedMemPerBlock << "\n";\
delete ______data_______