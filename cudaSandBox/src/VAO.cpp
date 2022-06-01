#include "TraceX.h"
#include "VAO.h"
#include <GL/glew.h>

namespace TraceX {
	VAO::VAO() {
		glGenVertexArrays(1, &m_ID);
		glBindVertexArray(m_ID);
	}

	VAO::~VAO() {
		glDeleteVertexArrays(1, &m_ID);
	}

	void VAO::AddBuffer(const VBO& vbo, const VertexArrayLayout& layout)
	{
		int offset = 0; //The offset of an Vertex Attribute from the beggining
		const auto& elements = layout.GetElements(); //Get the Elements array

		for (RUINT_T i = 0; i < elements.size(); i++)
		{
			const auto& element = elements[i];

			glEnableVertexAttribArray(i);
			glVertexAttribPointer(i, element.count, element.type, element.normalized, layout.GetStride(), (const void*)offset);
			offset += element.count * Element::GetTypeSize(element.type);
		}
	}
}