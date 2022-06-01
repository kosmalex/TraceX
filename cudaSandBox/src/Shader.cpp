#include "Shader.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>
#include <GL/glew.h>

namespace TraceX {
	static const char* extractString(const char* file_path) {
		std::ifstream file(file_path);

		if (!file.is_open()) {
			std::cout << "[Failed to open file]\n";
			return NULL;
		}

		std::stringstream ss;
		ss << file.rdbuf();
		std::string s = ss.str();
		unsigned int size = s.size();

		char* dst = new char[s.size() + 1];
		std::memmove((void*)dst, (const void*)s.c_str(), size * sizeof(char));
		dst[s.size()] = '\0';

		return dst;
	}
	static bool compileShader(unsigned int id, const char* src_code, unsigned int type) {
		glShaderSource(id, 1, &src_code, NULL);
		glCompileShader(id);

		int success;
		glGetShaderiv(id, GL_COMPILE_STATUS, &success);
		if (!success) {
			int loglength = 0;
			glGetShaderiv(id, GL_INFO_LOG_LENGTH, &loglength);

			char* buffer = new char[loglength];
			glGetShaderInfoLog(id, loglength, &loglength, buffer);
			std::cout << (type ? "[FragmentError]: ": "[VertexError]: ") << buffer << "\n";

			delete[] buffer;

			return false;
		}

		return true;
	}

	Shader::Shader(const char* vertexPath, const char* fragmentPath, const char* geometryPath)
	{
		const char* vs_source = nullptr;
		const char* fs_source = nullptr;
		vs_source = extractString(vertexPath);
		fs_source = extractString(fragmentPath);

		unsigned int vs_id = glCreateShader(GL_VERTEX_SHADER);
		unsigned int fs_id = glCreateShader(GL_FRAGMENT_SHADER);
		compileShader(vs_id, vs_source, 0);
		compileShader(fs_id, fs_source, 1);

		delete vs_source;
		delete fs_source;

		m_ID = glCreateProgram();
		glAttachShader(m_ID, vs_id);
		glAttachShader(m_ID, fs_id);
		glLinkProgram(m_ID);

		int success;
		glGetProgramiv(m_ID, GL_LINK_STATUS, &success);
		if (!success) {
			int loglength = 0;
			glGetProgramiv(m_ID, GL_INFO_LOG_LENGTH, &loglength);

			char* buffer = new char[loglength];
			glGetProgramInfoLog(m_ID, loglength, &loglength, buffer);
			std::cout << "[Program Error]: " << buffer << "\n";

			delete[] buffer;
		}

		glDetachShader(m_ID, vs_id);
		glDetachShader(m_ID, fs_id);

		glDeleteShader(vs_id);
		glDeleteShader(fs_id);
	}

	Shader::~Shader() {}

	/*void Shader::setVec2(const std::string &name, const glm::vec2 &value)
	{
		glUniform2fv(getUniform(name), 1, &value[0]);
	}

	void Shader::setVec3(const std::string &name, const glm::vec3 &value) const
	{
		glUniform3fv(getUniform(name), 1, &value[0]);
	}*/

	/*void Shader::setVec4(const std::string &name, const glm::vec4 &value) const
	{
		glUniform4fv(getUniform(name), 1, &value[0]);
	}*/

	/*void Shader::setMat2(const std::string &name, const glm::mat2 &mat) const
	{
		glUniformMatrix2fv(getUniform(name), 1, GL_FALSE, &mat[0][0]);
	}

	void Shader::setMat3(const std::string &name, const glm::mat3 &mat) const
	{
		glUniformMatrix3fv(getUniform(name), 1, GL_FALSE, &mat[0][0]);
	}

	void Shader::setMat4(const std::string &name, const glm::mat4 &mat) const
	{
		glUniformMatrix4fv(getUniform(name), 1, GL_FALSE, &mat[0][0]);
	}*/

	GLint Shader::getUniform(const std::string& name) const
	{
		if (m_UniformCache.find(name) != m_UniformCache.end())
			return m_UniformCache[name];

		GLint location = glGetUniformLocation(m_ID, name.c_str());
		m_UniformCache[name] = location;

		return location;
	}
}