#include <string>
#include <GL/glew.h> 
#include "Debug.h"

namespace TraceX {
	class Texture2D {
	public:
		Texture2D(void* framebuffer, size_t hPixels, size_t vPixels);

		inline uint32_t ID() const { return m_ID; }
		inline void Bind() const { glBindTexture(GL_TEXTURE_2D, m_ID); }
		inline void setTexture(void* framebuffer) const { 
			GL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_Width, m_Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, framebuffer));
			glGenerateMipmap(GL_TEXTURE_2D);
		}

	private:
		uint32_t m_ID;
		size_t m_Width, m_Height;
	};
}
