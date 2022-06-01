#include "VertexArrayLayout.h"
#include <GL/glew.h>

namespace TraceX {
	uint32_t Element::GetTypeSize(uint32_t type)
	{
		switch (type)
		{
			case GL_FLOAT:         return 4;
			case GL_UNSIGNED_INT:  return 4;
			case GL_UNSIGNED_BYTE: return 1;
			default:
				return 0;
				break;
		}
	}

	VertexArrayLayout::VertexArrayLayout() :m_Stride(0) {}
}
