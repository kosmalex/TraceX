#include<vector>
#include "GL/glew.h"

namespace TraceX
{
	struct Element {
		uint32_t type;
		int count;
		uint32_t normalized;

		static uint32_t GetTypeSize(uint32_t type);
	};

	class VertexArrayLayout {
	private:
		std::vector<Element> m_Elements;
		int m_Stride;

	public:
		VertexArrayLayout();


		template<typename T> void Push(int count) {}

		template<> void Push<float>(int count) {
			m_Elements.push_back({ GL_FLOAT,count,GL_FALSE });
			m_Stride += count * Element::GetTypeSize(GL_FLOAT);
		}


		inline const std::vector<Element> GetElements() const { return m_Elements; }
		inline int GetStride() const { return m_Stride; };
	};
}

