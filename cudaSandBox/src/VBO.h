#include <GL/glew.h>

namespace TraceX {
	class VBO {
	public:
		VBO(const void* data, int size);
		VBO():m_ID(-1) {};
		~VBO() { glDeleteBuffers(1, &m_ID); };

		inline unsigned int ID() { return m_ID; }
		inline void Bind() const { glBindVertexArray(m_ID); }
		inline void unBind() const { glBindVertexArray(0); }

	private:
		unsigned int m_ID;
	};
}

