#include "GL/glew.h"
#include "TraceX.h"

namespace TraceX
{
	class EBO {
	public:
		EBO(uint32_t* data, size_t size);
		~EBO();
		EBO() :m_ID(-1), m_Count(-1) {};

		inline unsigned int ID() { return m_ID; }
		inline int Count() { return m_Count; }
		inline void Bind() const { glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ID); }
		inline void unBind() const { glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); }

	private:
		unsigned int m_ID;
		int m_Count;
	};
}

