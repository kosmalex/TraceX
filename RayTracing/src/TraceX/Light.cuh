#pragma once
#include "Entity.cuh"

namespace TraceX {
	enum LightType { DIRECTIONAL = 0, POINT = 1 };

	class TRACEX_API LightSource : public Entity {
	public:
		LightSource() = delete;
		LightSource(double x, double y, double z, Vector3d m_Color) :Entity(x, y, z), m_Color(m_Color) {}
		LightSource(const Vector3d v, Vector3d color) :Entity(v), m_Color(color) {}

		//static inline LightType staticGetSrcType() { return }
		inline Vector3d getColor() { return m_Color; }

	protected:
		Vector3d m_Color;
		//LightType m_Type;
	};
}