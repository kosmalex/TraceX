#pragma once
#include "Entity.cuh"
#include "Light.cuh"

#include <vector>

namespace TraceX {

	//Needs to be generalized to entities
	class Scene
	{
	public:
		uint32_t nObjects, nLights;

		Scene() :nObjects(0), nLights(0) {}

		inline void addObj(const Sphere& e) { m_Entities.push_back(e); nObjects++; }
		inline void rmObj(int i) { m_Entities.erase(m_Entities.begin() + i); nObjects--; }
		inline void setMat(unsigned int i, const Material& mat) { m_Entities[i].color = mat.color; m_Entities[i].N = mat.N; }

		inline void addLight(const LightSource& e) { m_Lights.push_back(e); nLights++; }
		inline void rmLight(int i) { m_Lights.erase(m_Lights.begin() + i); nLights--; }

		inline bool isEmpty() const { return nObjects; }

		inline const Sphere& getObject(unsigned int i) const { return m_Entities[i]; }
		inline const LightSource& getLight(unsigned int i) const { return m_Lights[i]; }
		inline const Sphere* getElements() const { return &m_Entities[0]; }

	private:
		std::vector<Sphere> m_Entities;
		std::vector<LightSource> m_Lights;
	};
}

