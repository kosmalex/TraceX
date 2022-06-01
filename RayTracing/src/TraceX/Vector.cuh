#pragma once

#include "Core.cuh"

#include <iostream>
#include <math.h>
#include "cuda.h"
#include "cuda_runtime.h"

namespace TraceX {
	class TRACEX_API Vector3d {
	public:
		float x, y, z;

		__host__ __device__ Vector3d() :x(0), y(0), z(0), m_Length(0) {}
		__host__ __device__ Vector3d(float val) : x(val), y(val), z(val), m_Length(SQRT3* val) {}
		__host__ __device__ Vector3d(float x_r, float y_g, float z_b) : x(x_r), y(y_g), z(z_b), \
			m_Length(sqrt((this->x)* (this->x) + (this->y) * (this->y) + (this->z) * (this->z))) {}

		//out <- Vector on Vector'
		__host__ __device__ inline Vector3d operator+(const Vector3d& src) const {
			return { this->x + src.x, this->y + src.y , this->z + src.z };
		}

		__host__ __device__ inline Vector3d operator-(const Vector3d& src) const {
			return { this->x - src.x, this->y - src.y , this->z - src.z };
		}

		__host__ __device__ inline Vector3d operator*(const Vector3d& src) const {
			return { this->x * src.x, this->y * src.y, this->z * src.z };
		}

		//out <- Vector on Val
		__host__ __device__ inline Vector3d operator+(float val) const {
			return { this->x + val, this->y + val, this->z + val };
		}

		__host__ __device__ inline  Vector3d operator-(float val) const {
			return { this->x - val, this->y - val, this->z - val };
		}

		__host__ __device__ inline Vector3d operator*(float val) const {
			return { this->x * val, this->y * val, this->z * val };
		}

		__host__ __device__ inline Vector3d operator/(float val) const {
			ASSERT(val != 0);
			return { this->x / val, this->y / val, this->z / val };
		}

		//Vector <- Vector on Vector'
		__host__ __device__ inline void operator+=(const Vector3d& src) {
			*this = { this->x + src.x, this->y + src.y , this->z + src.z };
			this->updateLength();
		}

		__host__ __device__ inline void operator-=(const Vector3d& src) {
			*this = { this->x - src.x, this->y - src.y , this->z - src.z };
			this->updateLength();
		}

		__host__ __device__ inline void operator*=(const Vector3d& src) {
			*this = { this->x * src.x, this->y * src.y, this->z * src.z };
			this->updateLength();
		}

		//Vector <- Vector on Val
		__host__ __device__ inline void operator+=(float val) {
			*this = { this->x + val, this->y + val, this->z + val };
			this->updateLength();
		}

		__host__ __device__ inline void operator-=(float val) {
			*this = { this->x - val, this->y - val, this->z - val };
			this->updateLength();
		}

		__host__ __device__ inline void operator*=(float val) {
			*this = { this->x * val, this->y * val, this->z * val };
			this->updateLength();
		}

		__host__ __device__ inline void operator/=(float val) {
			ASSERT(val != 0);
			*this = { this->x / val, this->y / val, this->z / val };
			this->updateLength();
		}

		//Other ops
		__host__ __device__ inline void operator=(const Vector3d& src) {
			this->x = src.x;
			this->y = src.y;
			this->z = src.z;
		}

		//Condition Ops
		__host__ __device__ inline bool operator==(int i) const {
			return (this->x == i && this->y == i && this->z == i);
		}

		__host__ __device__ inline float getLength() { return m_Length; }

		__host__ __device__ inline float Dot(const Vector3d& dst)  const {
			return (x * dst.x + y * dst.y + z * dst.z);
		}

		__host__ __device__ inline  Vector3d Cross(const Vector3d& index) const{
			return { (y * index.z - index.y * z), (index.x * z - x * index.z),\
				(x * index.y - index.x * y) };
		}

		__host__ __device__ inline  float Length() const {
			return sqrt(x * x + y * y + z * z);
		}

		__host__ __device__ float Distance(const Vector3d& dst) const
		{
			return (*this - dst).Length();
		}

		__host__ __device__ inline Vector3d Normalize() const {
			if (*this == 0) return { 0.0 };
			float L = this->Length();
			return { x / L, y / L, z / L };
		}

		__host__ __device__ inline Vector3d Flip() const {
			return { -x, -y, -z };
		}

		__host__ __device__ inline  Vector3d Reflect(const Vector3d& normal) const{
			return *this - normal * 2 * this->Dot(normal);
		}

	protected:
		float m_Length;

		__host__ __device__ inline void updateLength() {
			m_Length = sqrt((this->x) * (this->x) + (this->y) * (this->y) + (this->z) * (this->z));
		}
	};

	struct TRACEX_API Basis {
		Vector3d t, v, u;

		__host__ __device__ Basis() :t({ 0, 0, 1 }), v({ 0, 1, 0 }), u({ 1, 0, 0 }) {}
		__host__ __device__ Basis(Vector3d t, Vector3d v, Vector3d u) : t(t), v(v), u(u) {}
	};

	//Vector ops
	/*TRACEX_API __host__ __device__ float Dot(const Vector3d& src, const Vector3d& dst);
	TRACEX_API __host__ __device__ Vector3d Cross(const Vector3d& thumb, const Vector3d& index);
	TRACEX_API __host__ __device__ float Length(const Vector3d& vec);
	TRACEX_API __host__ __device__ float Distance(const Vector3d& src, const Vector3d& dst);
	TRACEX_API __host__ __device__ Vector3d Normalize(const Vector3d& vec);
	TRACEX_API __host__ __device__ Vector3d Flip(const Vector3d& vec);
	TRACEX_API __host__ __device__ Vector3d Reflect(const Vector3d& vec, const Vector3d& normal);*/

	//Other operators
	TRACEX_API std::ostream& operator<<(std::ostream& stream, const Vector3d& src);
}