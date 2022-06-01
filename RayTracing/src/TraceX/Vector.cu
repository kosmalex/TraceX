#include "Vector.cuh"

namespace TraceX {
	__host__ __device__ TRACEX_API inline float Dot(const Vector3d& src, const Vector3d& dst) {
		return (src.x * dst.x + src.y * dst.y + src.z * dst.z);
	}

	TRACEX_API __host__ __device__ inline  Vector3d Cross(const Vector3d& thumb, const Vector3d& index) {
		return { (thumb.y * index.z - index.y * thumb.z), (index.x * thumb.z - thumb.x * index.z),\
			(thumb.x * index.y - index.x * thumb.y) };
	}

	TRACEX_API __host__ __device__ inline  float Length(const Vector3d& vec) {
		return sqrt((vec.x) * (vec.x) + (vec.y) * (vec.y) + (vec.z) * (vec.z));
	}

	TRACEX_API __host__ __device__ float Distance(const Vector3d& src, const Vector3d& dst)
	{
		return Length(src - dst);
	}

	TRACEX_API __host__ __device__ inline Vector3d Normalize(const Vector3d& vec) {
		if (vec == 0) return { 0.0 };
		return { vec.x / Length(vec), vec.y / Length(vec), vec.z / Length(vec) };
	}

	TRACEX_API __host__ __device__ inline Vector3d Flip(const Vector3d& vec) {
		return { -vec.x, -vec.y, -vec.z };
	}

	TRACEX_API __host__ __device__ inline  Vector3d Reflect(const Vector3d& vec, const Vector3d& normal) {
		return vec - normal * 2 * Dot(vec, normal);
	}

	TRACEX_API std::ostream& operator<<(std::ostream& stream, const Vector3d& src) {
		stream << "[Vector]: {" << src.x << ", " << src.y << ", " << src.z << "}\n";
		return stream;
	}
}