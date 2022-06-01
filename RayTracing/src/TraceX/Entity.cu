#include "Entity.cuh"

namespace TraceX {
	TRACEX_API uint32_t Object::s_IdGenerator = 0;

	std::ostream& Object::operator<<(std::ostream& stream) {
		stream << "[Object]:{\n";
		stream << "\t[ID]: " << this->getID() << "\n";
		stream << "\t[Position]: " << this->pos << "\n";
		stream << "\t[Color]: " << this->color << "\n";
		stream << "}\n";

		return stream;
	}
}