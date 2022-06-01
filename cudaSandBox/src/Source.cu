#include "TraceX.h"
#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include <cuda_gl_interop.h>
#define __CUDACC__
#include "cuda_texture_types.h"

#include "Texture2D.h"
#include "Shader.h"
#include "VAO.h"
#include "EBO.h"

#include "Debug.h"

#include <iomanip>
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"


#define VISUAL
#define CUDA
//#define OMP
//#define SERIAL

#define RAD(x) x * PI / 180.0

static unsigned int s_Width = 500;
static unsigned int s_Height = s_Width * 9 >> 4;
static const unsigned int s_Scale = 3;
static const char* s_Title = "Ray Tracing";

static double speed = 100.0f;

static void updateBasis(TraceX::Basis& b, GLFWwindow* window, double deltaTime, bool reset);
static bool handleInput(GLFWwindow* window, TraceX::Basis& b, TraceX::Vector3d& eye, double speed, double deltaTime, bool& cur_locked, bool ImGui);

static void genMaterials(TraceX::Scene& scene) {

	TraceX::Vector3d colors[] = {
		{1.0f, 0.0f, 0.0f}, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f },
		{1.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 1.0f}, {0.2, 1.0, 0.5},
		{0.0f, 1.0f, 1.0f}, {0.8, 0.3, 0.1}
	};

	int shine[] = {
		128, 4, 32, 64, 16, 8, 2
	};

	for (RUINT_T i = 0; i < scene.nObjects; i++) {
		int randIdx = abs(rand()) % 7;

		TraceX::Material mat(colors[randIdx]);
		mat.N = shine[randIdx];
		scene.setMat(i, mat);
	}
}

int main(int argc, char** argv) {
#pragma region Init

#ifdef VISUAL
	if (glfwInit() != GLFW_TRUE) return -1;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(s_Scale * s_Width, s_Scale * s_Height, s_Title, NULL, NULL);
	glfwMakeContextCurrent(window);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");

	if (glewInit() != GLEW_OK) {
		glfwTerminate();
		fprintf(stderr, "Error: %s\n", glewGetErrorString(glewInit()));
		return -1;
	}

#endif // VISUAL

	TraceX::Sphere sph1(TraceX::Vector3d(0.0, 0.0, 4.0), 1.0);
	TraceX::Sphere sph2(TraceX::Vector3d(2.0, 3.0, 7.0), 1.0);
	TraceX::Sphere sph3(TraceX::Vector3d(0, 0, 9.0), 2);
	TraceX::Sphere sph4(TraceX::Vector3d(2, 0, 4.0), 1.0);
	TraceX::Sphere sph5(TraceX::Vector3d(4, -3, 15.0), 1);
	TraceX::Sphere sph6(TraceX::Vector3d(2, 1, 25.0), 2);
	TraceX::Sphere sph7(TraceX::Vector3d(8, 3, 13.0), 4);
	TraceX::Sphere sph8(TraceX::Vector3d(-1, -4, 9.0), 2);

	TraceX::LightSource lt({ 0.0, -1.0, 0.5 }, { 1.0, 1.0, 1.0 });

	TraceX::Scene scene;
	scene.addObj(sph1);
	scene.addObj(sph2);
	scene.addObj(sph3);
	scene.addObj(sph4);
	scene.addObj(sph5);
	scene.addObj(sph6);
	scene.addObj(sph7);
	scene.addObj(sph8);
	scene.addLight(lt);

	genMaterials(scene);

	TraceX::Vector3d eye(0.0f);

#ifdef VISUAL
	unsigned int hPixels = 2048;
	unsigned int vPixels = (hPixels * 9) >> 4;

	TraceX::RayGridData grid_data = { eye, PI / 2, hPixels, vPixels };
	uint8_t* framebuffer = new uint8_t[hPixels * vPixels * TraceX::RGBA];
#endif // VISUAL
#pragma endregion


#ifndef VISUAL
	TraceX::Basis basis;
#ifdef OMP
	std::cout << "\n/////////////////////////// OMP ///////////////////////////\n";
	for (size_t N = 64; N <= 32768; N <<= 1) {
		unsigned int hPixels = N;
		unsigned int vPixels = (hPixels * 9) >> 4;

		TraceX::RayGridData grid_data = { eye, PI / 2, hPixels, vPixels };
		uint8_t* framebuffer = new uint8_t[hPixels * vPixels * TraceX::RGBA];
		TraceX::ompCastRays(framebuffer, basis, &grid_data, scene);

		delete[] framebuffer;
	}
#endif

#ifdef CUDA
	std::cout << "\n/////////////////////////// CUDA ///////////////////////////\n";
	for (size_t N = 64; N <= 32768; N <<= 1) {
		unsigned int hPixels = N;
		unsigned int vPixels = (hPixels * 9) >> 4;

		TraceX::RayGridData grid_data = { eye, PI / 2, hPixels, vPixels };
		uint8_t* framebuffer = new uint8_t[hPixels * vPixels * TraceX::RGBA];

		for (size_t b_size = 2; b_size <= 32; b_size <<= 1)
			TraceX::cudaCastRays(framebuffer, basis, &grid_data, &scene, b_size);

		delete[] framebuffer;
	}
#endif

#ifdef SERIAL
	std::cout << "/////////////////////////// SERIAL///////////////////////////\n";
	for (size_t N = 64; N <= 32768; N <<= 1) {
		unsigned int hPixels = N;
		unsigned int vPixels = (hPixels * 9) >> 4;

		TraceX::RayGridData grid_data = { eye, PI / 2, hPixels, vPixels };
		uint8_t* framebuffer = new uint8_t[hPixels * vPixels * TraceX::RGBA];
		TraceX::castRays(framebuffer, basis, &grid_data, scene);

		delete[] framebuffer;
	}
#endif

#endif // !VISUAL


#ifdef VISUAL
#pragma region Draw
	TraceX::Texture2D glframe(framebuffer, hPixels, vPixels);

	TraceX::Shader shader("shader/main.vs", "shader/main.fs");
	shader.use();
	shader.setInt("frame", 2);

	float data[] = {
		-1, -1, 0.0, 0.0, 0.0,
		1, -1, 0.0,  1.0, 0.0,
		1, 1, 0.0,   1.0, 1.0,
		-1, 1, 0.0,   0.0, 1.0
	};

	uint32_t indeces[] = {
		0, 3, 1,
		3, 2, 1
	};

	TraceX::VAO vao;
	TraceX::EBO ebo(indeces, sizeof(indeces));

	TraceX::VBO vbo(data, sizeof(data));
	TraceX::VertexArrayLayout lo;

	lo.Push<float>(3);
	lo.Push<float>(2);

	vao.AddBuffer(vbo, lo);
#pragma endregion
#endif // VISUAL

#ifdef VISUAL
	uint8_t mode = 0;
	bool cur_locked = false;
	bool ImGuiTurn = false;
	double deltaTime = 0.0;
	static bool FirstTime = true;
	GL_CALL(glClearColor(0.2, 0.3, 0.4, 1.0));
	while (!glfwWindowShouldClose(window)) {
		GL_CALL(glClear(GL_COLOR_BUFFER_BIT));

#pragma region ImGui
		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.
		if (ImGui::Button("mode"))  mode < 2 ? mode++ : mode = 0;
		ImGui::SameLine();
		ImGui::Text("%d", mode);

		ImGuiTurn = ImGui::IsWindowFocused();

		ImGui::End();
#pragma endregion

		double now = glfwGetTime();
		deltaTime = now - deltaTime;
		deltaTime = now;

		TraceX::Basis b;
		bool drawNewFrame = handleInput(window, b, grid_data.eye, speed, deltaTime, cur_locked, ImGuiTurn);

		if (drawNewFrame || FirstTime) {
			FirstTime = false;
			switch (mode) {
			case 0:
			{
				TraceX::cudaCastRays(framebuffer, b, &grid_data, &scene, 16);
				break;
			}
			case 1: {
				TraceX::ompCastRays(framebuffer, b, &grid_data, scene);
				break;
			}
			case 2: {
				TraceX::castRays(framebuffer, b, &grid_data, scene);
				break;
			}
			default:
				break;
			}

			glframe.setTexture(framebuffer);
		}


		GL_CALL(glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0));


		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		GL_CALL(glfwPollEvents());
		GL_CALL(glfwSwapBuffers(window));

	}
#endif

	glfwDestroyWindow(window);
	glfwTerminate();

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	return 0;
}

void updateBasis(TraceX::Basis& b, GLFWwindow* window, double deltaTime, bool reset)
{
	static bool first_Time = true;
	static double x_old = 0.0, y_old = 0.0;
	static double yaw = 90, pitch = 0;

	if (reset) first_Time = true;

	double x, y;
	glfwGetCursorPos(window, &x, &y);

	double dx, dy;
	if (first_Time) {
		dx = 0;
		dy = 0;
		x_old = x;
		y_old = y;
		first_Time = false;
	}
	else {
		dx = -(x - x_old) * deltaTime;
		dy = -(y - y_old) * deltaTime;
		x_old = x;
		y_old = y;
	}

	yaw += dx * 0.001f;
	pitch += dy * 0.001f;

	b.t.x = cos(RAD(yaw)) * cos(RAD(pitch));
	b.t.y = sin(RAD(pitch));
	b.t.z = sin(RAD(yaw)) * cos(RAD(pitch));

	b.u = b.t.Cross({ 0.0, 1.0, 0.0 }).Normalize() * -1; //CONVERT TO LEFT-HAND-SIDE

	b.v = b.u.Cross(b.t).Normalize() * -1; //CONVERT TO LEFT-HAND-SIDE
}

bool handleInput(GLFWwindow* window, TraceX::Basis& b, TraceX::Vector3d& eye, double speed, double deltaTime, bool& cur_locked, bool ImGui)
{
	static bool resetCamRot = false;
	bool newFrame = false;

	if (cur_locked && !ImGui) {
		updateBasis(b, window, deltaTime, resetCamRot);
		newFrame = true;
		resetCamRot = false;
	}

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		eye += ((glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) ? b.v : b.t) * 0.0001f * speed * deltaTime;
		newFrame = true;
	}

	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		eye -= ((glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) ? b.v : b.t) * 0.0001f * speed * deltaTime;
		newFrame = true;
	}

	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) { eye -= b.u * 0.0001f * speed * deltaTime; newFrame = true; }
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) { eye += b.u * 0.0001f * speed * deltaTime; newFrame = true; }

	if (cur_locked && glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		cur_locked = false;
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}

	if (glfwGetWindowAttrib(window, GLFW_HOVERED) && !ImGui) {
		if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS) {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			cur_locked = true;
			resetCamRot = true;
		}
	}

	return newFrame;
}
