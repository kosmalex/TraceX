#include "VBO.h"
#include "GL/glew.h"

#include <iostream>

namespace TraceX {

	VBO::VBO(const void* data, int size)
	{
		glGenBuffers(1, &m_ID); //create a Buffer with m_ID
		glBindBuffer(GL_ARRAY_BUFFER, m_ID); //Set the state of GL_ARRAY_BUFFER to be m_ID
		glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW); //Set the data to the buffer
		//Static_draw means that the data store contents will be modified once and 
			//used many times as the source for GL drawing commands
	}
}