#include "utility.cuh"
#include "Vector.cuh"
#include <fstream>
#include "iomanip"

namespace TraceX {
	__device__  void illum(Vector3d& sink, const Vector3d& eye, const Ray3d::CollisionData& col_data, bool render, bool isInShadow);

	__global__ void cudaCastRaysKernel(void* frambuffer, const Vector3d* vecs,
		const RayGridData* gData, const Sphere* scene, int* nObjects);

	void TRACEX_API cudaCastRays(void* framebuffer, const Basis& b, const RayGridData* GridData, const Scene* scene, size_t blck_size)
	{
#pragma region InitCudaEvent
		float mem_cpy_time, buffer;
		float kernel_time, tot_time;

		cudaEvent_t start, stop;
		cudaEventCreate(&start, 0);
		cudaEventCreate(&stop, 0);
#pragma endregion 

#pragma region Preparation
		float k = (float)GridData->hPixels;
		float m = (float)GridData->vPixels;

		float aspect_ratio = (m - 1) / (k - 1);
		float gx = tan(GridData->FOV / 2);
		float gy = gx * aspect_ratio;

		Vector3d vecs[] = {
			b.u * (2 * gx / (k - 1)),
			b.v * (2 * gy / (m - 1)),
			b.t - b.u * gx - b.v * gy
		};
#pragma endregion

		uint8_t* d_Fb;
		Sphere* d_Scene;
		RayGridData* d_Gdata;
		Vector3d* d_Vecs;
		int* d_Nobjects;

#pragma region Alloc
		cudaEventRecord(start, 0);

		CUDA_CREATE(d_Vecs, vecs, sizeof(Vector3d) * 3);
		CUDA_CREATE(d_Scene, scene->getElements(), scene->nObjects * sizeof(Sphere));
		CUDA_CREATE(d_Gdata, GridData, sizeof(RayGridData));
		CUDA_CREATE(d_Nobjects, &scene->nObjects, sizeof(int));

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&buffer, start, stop);

		CUDA_ALLOC(d_Fb, sizeof(uint8_t) * GridData->hPixels * GridData->vPixels * 4);
#pragma endregion

		dim3 dimBlock(blck_size, blck_size);
		dim3 dimGrid(GridData->hPixels / dimBlock.x, GridData->vPixels / dimBlock.y);


		cudaEventRecord(start);
		cudaCastRaysKernel $(dimGrid, dimBlock) (d_Fb, d_Vecs, d_Gdata, d_Scene, d_Nobjects);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&kernel_time, start, stop);

		cudaEventRecord(start);

		CUDA_CPY2HOST(framebuffer, d_Fb, sizeof(uint8_t) * GridData->hPixels * GridData->vPixels * 4);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&mem_cpy_time, start, stop);
		mem_cpy_time += buffer;

		tot_time = mem_cpy_time + kernel_time;

#pragma region WriteResOut
		std::ofstream file("C:/users/kosma/desktop/cudaPrj/res/cuda.csv", std::ios::app);
		if (!file.is_open()) std::cout << "[ERROR]: FILE DID NOT OPEN!!!\n";

		std::cout << std::setw(10) << GridData->hPixels << ", " << blck_size << ", " << mem_cpy_time << ", " << kernel_time << ", " << tot_time << "\n";
		file << GridData->hPixels << ", " << blck_size << ", " << mem_cpy_time << ", " << kernel_time << ", " << tot_time << "\n";

		file.close();
#pragma endregion

#pragma region Free
		CUDA_FREE(d_Fb);
		CUDA_FREE(d_Vecs);
		CUDA_FREE(d_Gdata);
		CUDA_FREE(d_Scene);
		CUDA_FREE(d_Nobjects);
#pragma endregion
	}

	__global__ void cudaCastRaysKernel(void* framebuffer, const Vector3d* vecs, const RayGridData* gData,
		const Sphere* scene, int* nObjects)
	{
		unsigned int threadIDx = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int threadIDy = blockIdx.y * blockDim.y + threadIdx.y;
		unsigned int threadID = threadIDx + threadIDy * (gridDim.x * blockDim.x);

		//vecs[2] -> d_00;
		//vecs[0] -> u
		//vecs[1] -> v

		__shared__ Vector3d sh_Vecs[3];
		__shared__ int sh_nObjects;

		if ((threadIdx.x + threadIdx.y * blockDim.x) < 3) sh_Vecs[threadIdx.x + threadIdx.y * blockDim.x] = vecs[threadIdx.x + threadIdx.y * blockDim.x];

		if ((threadIdx.x + threadIdx.y * blockDim.x) == 0) sh_nObjects = *nObjects;

		__syncthreads();

		Vector3d d_ij = sh_Vecs[2] + sh_Vecs[0] * threadIDx + sh_Vecs[1] * threadIDy;
		Ray3d ray(d_ij, gData->eye);

		Vector3d color(0.2f);
		Ray3d::CollisionData cdata;

		float d_nearest = 100; //a large value for deapth testing
		int d_nearestID = -1;
		bool isInShadow = false;

		for (int k = 0; k < sh_nObjects; k++) { //for each object in the scene
			auto temp = ray.hasCollided(scene[k]);

			if (!temp.hit) continue;

			isInShadow = ray.isShadowRay() && (d_nearestID == ray.getHostObjectID());

			//depth testing
			float cur_d = gData->eye.Distance(temp.p1);
			if (cur_d <= d_nearest && !isInShadow) {
				cdata = temp;
				d_nearest = cur_d;
			}
		}

		//for shadow depth-testing
		ray.updateRay(cdata);
		d_nearestID = cdata.hostID;

		//set the ray as shadow ray
		ray.setToShadowRay();
		ray.setDirection(Vector3d(0.0, -1.0, 0.5).Flip());

		for (int k = 0; k < sh_nObjects; k++) { //for each object in the scene
			auto temp = ray.hasCollided(scene[k]);

			if (!temp.hit) continue;

			isInShadow = ray.isShadowRay() && (d_nearestID == ray.getHostObjectID());

			//depth testing
			float cur_d = gData->eye.Distance(temp.p1);
			if (cur_d <= d_nearest && !isInShadow) {
				cdata = temp;
				d_nearest = cur_d;
			}
		}

		illum(color, gData->eye, cdata, d_nearest < 100, isInShadow);

		((uint8_t*)framebuffer)[RGBA * threadID] = (int)(color.x * 0xff);
		((uint8_t*)framebuffer)[RGBA * threadID + 1] = (int)(color.y * 0xff);
		((uint8_t*)framebuffer)[RGBA * threadID + 2] = (int)(color.z * 0xff);
		((uint8_t*)framebuffer)[RGBA * threadID + 3] = 0xff;
	}

	__constant__ float Ka = 0.2f;
	__constant__ float Kd = 0.65f;
	__constant__ float Ks = 1.0f;

	__device__ void illum(Vector3d& sink, const Vector3d& eye, const Ray3d::CollisionData& col_data, bool render, bool isInShadow)
	{
		Vector3d Lpos(0.0f, -1.0f, 0.5f);

		bool illuminate = render && !isInShadow;
		sink = Vector3d(0.2f) * !render;

		//Ambient
		sink += Vector3d(1.0f) * Ka * render;

		//Diffuse
		float N_dot_L = Lpos.Normalize().Flip().Dot(col_data.Normal);
		sink += Vector3d(1.0f) * Kd * N_dot_L * (N_dot_L >= 0) * illuminate;

		//Specular
		float V_dot_R = (col_data.p1 - eye).Normalize().Flip().Dot(Vector3d(0.0, -1.0, 0.5).Normalize().Reflect(col_data.Normal));
		sink += Vector3d(1.0f) * Ks * pow(V_dot_R * (V_dot_R >= 0), col_data.N) * illuminate;

		sink *= col_data.color * render + !render;

		sink.x = sink.x * (sink.x <= 1.0f) + 1.0f * (sink.x > 1.0f);
		sink.y = sink.y * (sink.y <= 1.0f) + 1.0f * (sink.y > 1.0f);
		sink.z = sink.z * (sink.z <= 1.0f) + 1.0f * (sink.z > 1.0f);
	}

	template<>
	void TRACEX_API init_arr<double>(double* begin, uint64_t length, int max_val, int flag) {
		for (register uint64_t i = 0; i < length; i++)
			if (!flag) {
				if (max_val <= 0)
					begin[i] = max_val;
				else
					begin[i] = rand() % max_val;
			}
			else
			{
				begin[i] = max_val;
			}
	}
}



