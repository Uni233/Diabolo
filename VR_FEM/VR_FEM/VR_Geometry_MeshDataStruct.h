#ifndef _VR_GEOMETRY_MESHDATASTRUCT_H
#define _VR_GEOMETRY_MESHDATASTRUCT_H

#include "VR_Global_Define.h"
#include <vector>
#include <string>
using std::vector;
#include <glm/glm.hpp>
using glm::vec3;
using glm::vec2;
using glm::vec4;

namespace YC
{
	namespace Geometry
	{
		struct MeshDataStruct
		{
			std::string fileName;
			MyFloat m_maxDiameter,m_translation_x,m_translation_y,m_translation_z;
			vector<vec3>  points;
			vector<vec3>  normals;
			vector<vec2> texCoords;
			vector<vec4> tangents;
			vector<int> faces;
			vector<vec3> displacedVertices;

			void loadOBJ(const char* lpszFileName, bool loadTex);
			void printMeshDataStructInfo( std::ostream& out);

			void loadPLY(const char* lpszFileName);
			vec3 calculateNormal(const vec3& vtx0,const vec3& vtx1,const vec3& vtx2);
		};
	}
}

#endif//_VR_GEOMETRY_MESHDATASTRUCT_H