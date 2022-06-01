#pragma once

#include "Material.cuh"
#include "Vector.cuh"

namespace TraceX {
	enum class TRACEX_API ObjectType {
		VOID,
		SPHERE,
		SQUARE
	};

	class TRACEX_API Entity {
	public:
		Vector3d pos;

		Entity() {}
		Entity(float x, float y, float z) :pos({ x, y, z }) {}
		Entity(Vector3d v) :pos(v) {}
	};

	class TRACEX_API Object : public Entity {
	public:
		Vector3d color;
		float N;

		Object() :N(4), m_ID(s_IdGenerator++) { }
		Object(float x, float y, float z) :Entity(x, y, z), N(4), m_ID(s_IdGenerator++) {}
		Object(Vector3d v) :Entity(v), m_ID(s_IdGenerator++), N(4) {}

		virtual inline ObjectType getType() { return ObjectType::VOID; }
		__host__ __device__ inline uint32_t getID() const { return m_ID; }

		std::ostream& operator<<(std::ostream& stream);
	protected:
		uint32_t m_ID;

	private:
		static uint32_t s_IdGenerator;
	};

	class TRACEX_API Sphere : public Object {
	public:
		float radius;

		Sphere() :radius(0) {}
		Sphere(float x, float y, float z, float r) : Object(x, y, z), radius(r) {}
		Sphere(Vector3d v, float r) : Object(v), radius(r) {}

		virtual inline ObjectType getType() { return ObjectType::SPHERE; }
	};
}

