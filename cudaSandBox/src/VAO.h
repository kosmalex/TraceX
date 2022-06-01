#pragma once

#include <GL/glew.h>

#include "VertexArrayLayout.h"
#include "VBO.h"

namespace TraceX {
	class VAO {
	public:

		VAO();
		~VAO();

		void AddBuffer(const VBO& vbo, const VertexArrayLayout& layout);

		inline unsigned int ID() { return m_ID; }
		inline void Bind() const { glBindVertexArray(m_ID); }
		inline void unBind() const { glBindVertexArray(0); }
	private:
		unsigned int m_ID;
	};
}