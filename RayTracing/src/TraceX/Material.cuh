#pragma once
#include "Core.cuh"
#include "Vector.cuh"

namespace TraceX {
	struct TRACEX_API Material {
		Vector3d color;
		float Ka, Kd, Ks, N;

		Material() :Ka(0.1f), Kd(0.6f), Ks(1.0f), N(32.0f) {}
		//Material(double r, double g, double b) : color(r, g, b), Ka(0.1f), Kd(0.6f), Ks(1.0f), N(32.0f) {}
		Material(Vector3d color) : color(color), Ka(0.1f), Kd(0.6f), Ks(1.0f), N(32.0f) {}
	};
}