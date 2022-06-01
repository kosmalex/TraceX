#include "Ray.cuh"

namespace TraceX {
	std::ostream& operator<<(std::ostream& stream, const Ray3d& src) {
		stream << "[Ray]:{\n";
		stream << "\t[Origin]: {" << src.m_Origin.x << ", " << src.m_Origin.y << ", " << src.m_Origin.z << "}\n";
		stream << "\t[Direction]: {" << src.m_Direction.x << ", " << src.m_Direction.y << ", " << src.m_Direction.z << "}\n";
		stream << "}\n";

		return stream;
	}
}