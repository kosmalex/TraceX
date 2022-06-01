#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Core.cuh"
#include "Ray.cuh"
#include "Scene.cuh"

#include <functional>

namespace TraceX {
	enum TRACEX_API nChannels { R = 1, RGB = 3, RGBA = 4 };

	template<class _Ty>
	inline void init_arr(_Ty* begin, uint64_t length, int max_val = 100, int flag = 0) {
		for (register uint64_t i = 0; i < length; i++)
			if (!flag) {
				if (max_val <= 0)
					begin[i] = max_val;
				else
					begin[i] = rand() % max_val;
			}
			else
			{
				begin[i] = max_val;
			}
	}

	template<>
	void TRACEX_API init_arr<double>(double* begin, uint64_t length, int max_val, int flag);

	void TRACEX_API cudaCastRays(void* framebuffer, const Basis& b, const RayGridData* GridData, const Scene* scene, size_t blck_size);
}