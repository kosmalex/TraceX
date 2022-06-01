#pragma once
#include "Entity.cuh"
#include "Scene.cuh"
#include <vector>

namespace TraceX {
	struct TRACEX_API RayGridData {
		Vector3d eye = Vector3d(-999);
		float FOV = -999;
		unsigned int hPixels = -1;
		unsigned int vPixels = -1;
		int d = -999; // distance from EYE
	};

	class TRACEX_API Ray3d {
	public:
		struct CollisionData {
			bool hit = false;
			Vector3d p1;
			Vector3d Normal;
			Vector3d color;
			float N = 1;
			int hostID = -1;
		};

		__host__ __device__ Ray3d() :m_Origin(0.0f), m_IsShadowRay(false), m_HostObjectID(-1) {};
		__host__ __device__ Ray3d(const Vector3d& vec, const Vector3d& org) :m_Direction(vec.Normalize()), m_Origin(org), m_IsShadowRay(false), m_HostObjectID(-1) {};
		__host__ __device__ Ray3d(double dir_x, double dir_y, double dir_z)
			:m_Origin(0.0f), m_Direction(dir_x, dir_y, dir_z), m_IsShadowRay(false), m_HostObjectID(-1)
		{
			float Length = m_Direction.Length();
			ASSERT(Length == 0);

			m_Direction.x /= Length;
			m_Direction.y /= Length;
			m_Direction.z /= Length;
		}

		__host__ __device__ inline void setDirection(const Vector3d dir) { m_Direction = dir.Normalize(); }
		__host__ __device__ inline void setOrigin(const Vector3d& org) { m_Origin = org; }
		__host__ __device__ inline void setToShadowRay() { m_IsShadowRay = true; }
		__host__ __device__ inline void setToNormalRay() { m_IsShadowRay = false; }
		__host__ __device__ inline void detachFromObject() { m_HostObjectID = -1; }
		__host__ __device__ inline void resetRay() { //In case it is needed
			detachFromObject();
			setOrigin(Vector3d(0.0f));
		}

		__host__ __device__ inline Vector3d getDirection() const { return m_Direction; }
		__host__ __device__ inline Vector3d getOrigin() const { return m_Origin; }
		__host__ __device__ inline bool isShadowRay() const { return m_IsShadowRay; }
		__host__ __device__ uint32_t getHostObjectID() { return m_HostObjectID; }

		__host__ __device__ CollisionData hasCollided(const Sphere& e) { //Change input to Entity
			TraceX::Vector3d v = m_Origin - e.pos;
			float v_dot_r = v.Dot(m_Direction);

			float v_L = v.getLength();
			float v_2 = v_L * v_L;

			float expr = (v_dot_r * v_dot_r) - (v_2 - (e.radius * e.radius));
			float sqrt_expr, t1;
			sqrt_expr = sqrt(expr);
			t1 = -v_dot_r - sqrt_expr;

			Vector3d p1 = { m_Origin + m_Direction * t1 };

			bool hit = m_HostObjectID != e.getID() && expr >= 0 && t1 > 0;
			return { hit, p1, (p1 - e.pos).Normalize(), e.color, e.N, (int)e.getID()}; //Be carefull may cause errors in the future
		}

		__host__ __device__ void updateRay(const CollisionData& data) {
			this->setOrigin(data.p1);
			this->m_HostObjectID = data.hostID;
		}

		friend std::ostream& operator<<(std::ostream& stream, const Ray3d& src);

	private:
		Vector3d m_Direction, m_Origin;
		int m_HostObjectID;
		bool m_IsShadowRay;
	};

	std::ostream& operator<<(std::ostream& stream, const Ray3d& src);
}