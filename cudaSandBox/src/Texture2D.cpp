#include "Texture2D.h"
#include "Debug.h"

#include <GL/glew.h>

namespace TraceX {
	Texture2D::Texture2D(void* framebuffer, size_t hPixels, size_t vPixels)
		: m_Width(hPixels), m_Height(vPixels)
	{
		glGenTextures(1, &m_ID);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, m_ID);

		GL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, hPixels, vPixels, 0, GL_RGBA, GL_UNSIGNED_BYTE, framebuffer));
		glGenerateMipmap(GL_TEXTURE_2D);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR);
	}
}