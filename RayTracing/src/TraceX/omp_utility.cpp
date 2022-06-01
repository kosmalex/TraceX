#include "omp_utility.h"
#include <omp.h>
#include <fstream>
#include <iomanip>

namespace TraceX {
	double max(double a, double b) { if (a <= b) return b; return a; }

	double min(double a, double b) { if (a <= b) return a; return b; }
	
	void hostIllum(Vector3d& sink, const Vector3d& eye, const Ray3d::CollisionData& col_data, bool render, bool isInShadow) {
		double Ka = 0.2;
		double Kd = 0.65;
		double Ks = 1.0;
		Vector3d Lpos = { 0.0, -1.0, 0.5 };

		bool illuminate = render && !isInShadow;
		sink = Vector3d(0.2f) * !render;

		//Ambient
		sink += Vector3d(1.0f) * Ka * render;

		//Diffuse
		double N_dot_L = Lpos.Normalize().Flip().Dot(col_data.Normal);
		sink += Vector3d(1.0f) * Kd * max(N_dot_L, 0) * illuminate;

		//Specular
		double V_dot_R = (col_data.p1 - eye).Normalize().Flip().Dot(Vector3d(0.0, -1.0, 0.5).Normalize().Reflect(col_data.Normal));
		sink += Vector3d(1.0f) * Ks * pow(max(V_dot_R, 0), col_data.N) * illuminate;

		sink *= col_data.color * render + !render;

		sink.x = min(sink.x, 1.0f);
		sink.y = min(sink.y, 1.0f);
		sink.z = min(sink.z, 1.0f);
	}
	
	void ompCastRays(void* framebuffer, const Basis& b, const RayGridData* gData, const Scene& scene) {
		ASSERT(framebuffer != NULL);

		Vector3d color(0.2, 0.2, 0.2);
		Ray3d::CollisionData cdata;
		double d_nearest = 100; //a large value for deapth testing
		int d_nearestID = -1;
		bool isInShadow = false;

		float start = omp_get_wtime();
		
		double k = (double)gData->hPixels;
		double m = (double)gData->vPixels;

		double aspect_ratio = (m - 1) / (k - 1);
		double gx = tan(gData->FOV / 2);
		double gy = gx * aspect_ratio;

		Vector3d qx = b.u * (2 * gx / (k - 1));
		Vector3d qy = b.v * (2 * gy / (m - 1));
		Vector3d p1m = b.t - b.u * gx - b.v * gy;

#pragma omp parallel for private(color, cdata, d_nearest, d_nearestID, isInShadow) schedule(static) collapse(2)
		for (int j = 0; j < gData->vPixels; j++) {
			for (int i = 0; i < gData->hPixels; i++) {

				Vector3d temp = p1m + qx * i + qy * j;
				Ray3d ray(temp, gData->eye);
				d_nearest = 100;

				for (int bounce = 0; bounce < 2; bounce++) { //2 bounces
					for (RUINT_T k = 0; k < scene.nObjects; k++) { //for each object in the scene
						auto temp = ray.hasCollided(scene.getObject(k));

						if (!temp.hit) continue;

						isInShadow = ray.isShadowRay() && (d_nearestID == ray.getHostObjectID());

						//depth testing
						double cur_d = gData->eye.Distance(temp.p1);;
						if (cur_d <= d_nearest && !isInShadow) {
							cdata = temp;
							d_nearest = cur_d;
						}
					}

					//for shadow depth-testing
					ray.updateRay(cdata);
					d_nearestID = cdata.hostID;

					//set the ray as shadow ray
					if (!cdata.hit && !ray.isShadowRay()) break;
					ray.setToShadowRay();
					ray.setDirection(Vector3d(0.0, -1.0, 0.5).Flip());
				}

				hostIllum(color, gData->eye, cdata, d_nearest < 100, isInShadow);

				((uint8_t*)framebuffer)[RGBA * (i + j * gData->hPixels)] = (int)(color.x * 0xff);
				((uint8_t*)framebuffer)[RGBA * (i + j * gData->hPixels) + 1] = (int)(color.y * 0xff);
				((uint8_t*)framebuffer)[RGBA * (i + j * gData->hPixels) + 2] = (int)(color.z * 0xff);
				((uint8_t*)framebuffer)[RGBA * (i + j * gData->hPixels) + 3] = 0xff;
			}
		}
		
		float total_time = omp_get_wtime() - start;

		std::ofstream file("C:/users/kosma/desktop/cudaPrj/res/omp.csv", std::ios::app);
		if (!file.is_open()) std::cout << "[ERROR]: FILE DID NOT OPEN!!!\n";

		std::cout << std::setw(10) << gData->hPixels << ", " << total_time * 1000.0f << "\n";
		file << std::setw(10) << gData->hPixels << ", " << total_time * 1000.0f << "\n";

		file.close();
	}

	void castRays(void* framebuffer, const Basis& b, const RayGridData* gData, const Scene& scene) {
		float start = omp_get_wtime();

		double k = (double)gData->hPixels;
		double m = (double)gData->vPixels;

		double aspect_ratio = (m - 1) / (k - 1);
		double gx = tan(gData->FOV / 2);
		double gy = gx * aspect_ratio;

		Vector3d qx = b.u * (2 * gx / (k - 1));
		Vector3d qy = b.v * (2 * gy / (m - 1));
		Vector3d p1m = b.t - b.u * gx - b.v * gy;

		for (int j = 0; j < gData->vPixels; j++) {
			for (int i = 0; i < gData->hPixels; i++) {
				
				Vector3d temp = p1m + qx * i + qy * j;
				Ray3d ray(temp, gData->eye);

				Vector3d color(0.2, 0.2, 0.2);
				Ray3d::CollisionData cdata;

				double d_nearest = 100; //a large value for deapth testing
				int d_nearestID = -1;
				bool isInShadow = false;

				for (int bounce = 0; bounce < 2; bounce++) { //2 bounces
					for (RUINT_T k = 0; k < scene.nObjects; k++) { //for each object in the scene
						auto temp = ray.hasCollided(scene.getObject(k));

						if (!temp.hit) continue;

						isInShadow = ray.isShadowRay() && (d_nearestID == ray.getHostObjectID());

						//depth testing
						double cur_d = gData->eye.Distance(temp.p1);;
						if (cur_d <= d_nearest && !isInShadow) {
							cdata = temp;
							d_nearest = cur_d;
						}
					}

					//for shadow depth-testing
					ray.updateRay(cdata);
					d_nearestID = cdata.hostID;

					//set the ray as shadow ray
					if (!cdata.hit && !ray.isShadowRay()) break;
					ray.setToShadowRay();
					ray.setDirection(Vector3d(0.0, -1.0, 0.5).Flip());
				}

				hostIllum(color, gData->eye, cdata, d_nearest < 100, isInShadow);

				((uint8_t*)framebuffer)[RGBA * (i + j * gData->hPixels)] = (int)(color.x * 0xff);
				((uint8_t*)framebuffer)[RGBA * (i + j * gData->hPixels) + 1] = (int)(color.y * 0xff);
				((uint8_t*)framebuffer)[RGBA * (i + j * gData->hPixels) + 2] = (int)(color.z * 0xff);
				((uint8_t*)framebuffer)[RGBA * (i + j * gData->hPixels) + 3] = 0xff;
			}
		}

		float total_time = omp_get_wtime() - start;

		std::ofstream file("C:/users/kosma/desktop/cudaPrj/res/serial.csv", std::ios::app);
		if (!file.is_open()) std::cout << "[ERROR]: FILE DID NOT OPEN!!!\n";

		std::cout << std::setw(10) << gData->hPixels << ", " << total_time * 1000.0f << "\n";
		file << std::setw(10) << gData->hPixels << ", " << total_time * 1000.0f << "\n";

		file.close();
	}
}