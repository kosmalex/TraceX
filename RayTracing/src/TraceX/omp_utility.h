#pragma once
#include "utility.cuh"


namespace TraceX {
	void TRACEX_API castRays(void* framebuffer, const Basis& b, const RayGridData* GridData, const Scene& scene);
	void TRACEX_API ompCastRays(void* framebuffer, const Basis& b, const RayGridData* GridData, const Scene& scene);
}