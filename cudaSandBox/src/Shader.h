#pragma once

#include <string>
#include <unordered_map>
#include <GL/glew.h>

namespace TraceX
{
	class Shader {
	public:
		Shader() = delete;
		Shader(const char* vertexPath, const char* fragmentPath, const char* geometryPath = nullptr);
		~Shader();

		inline void use() const { glUseProgram(m_ID); }
		inline unsigned int ID() const { return m_ID; }

		inline void setBool(const std::string& name, bool value) const {
			glUniform1i(getUniform(name), (int)value);
		}

		inline void setInt(const std::string& name, int value) const {
			glUniform1i(getUniform(name), value);
		}

		inline void setFloat(const std::string& name, float value) const {
			glUniform1f(getUniform(name), value);
		}

		inline void setVec2(const std::string& name, float x, float y) const {
			glUniform2f(getUniform(name), x, y);
		}

		inline void setVec3(const std::string& name, float x, float y, float z) const {
			glUniform3f(getUniform(name), x, y, z);
		}

		inline void setVec4(const std::string& name, float x, float y, float z, float w) const {
			glUniform4f(getUniform(name), x, y, z, w);
		}

	private:
		unsigned int m_ID;
		mutable std::unordered_map<std::string, int> m_UniformCache;

		int getUniform(const std::string& name) const;
	};
}