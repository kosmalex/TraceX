#include "EBO.h"

namespace TraceX {

	EBO::EBO(uint32_t* data, size_t size)
		:m_Count(size / sizeof(GLuint))
	{
		glGenBuffers(1, &m_ID);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ID);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
	}

	EBO::~EBO() { glDeleteBuffers(1, &m_ID); }
}