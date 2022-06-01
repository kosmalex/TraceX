#pragma once
#include "GL/glew.h"
#include <iostream>
#include "TraceX/Core.cuh"

#define GL_CALL(x) glClearError();\
					x;\
					ASSERT(glLogError());

static void glClearError() {
	while (glGetError() != GL_NO_ERROR);
}

static bool glLogError() {
	while (GLenum err = glGetError()) {
		std::cout << std::hex << "[Opengl Error]: " << err << "\n";
		return false;
	}

	return true;
}