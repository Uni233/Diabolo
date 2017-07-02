#ifndef _VR_GEOMETRY_MESHDATASTRUCT_H
#define _VR_GEOMETRY_MESHDATASTRUCT_H

#include "vrBase/vrBase.h"
#include <boost/shared_ptr.hpp>
namespace VR
{
	namespace Geometry
	{
		struct MeshDataStruct
		{
			vrString fileName;
			vrFloat m_maxDiameter,m_translation_x,m_translation_y,m_translation_z;
			std::vector< vrGLMVec3 >  points;
			std::vector< vrGLMVec3 >  normals;
			std::vector< vrGLMVec2 > texCoords;
			std::vector< vrGLMVec4 > tangents;
			//std::vector<vrInt> faces;
			//std::vector<vrInt> faces_group;
			std::vector<vrVec3I> facesVec3I;
			std::vector<vrInt> facesVec3I_group;

			std::vector< vrGLMVec3 > displacedVertices;

			void loadOBJ(vrLpsz lpszFileName, bool loadTex, bool unify);
			void unitMesh();
			void clear();
			void printMeshDataStructInfo( std::ostream& out);			
			vrGLMVec3 calculateNormal(const vrGLMVec3& vtx0, const vrGLMVec3& vtx1, const vrGLMVec3& vtx2);
		};

		typedef boost::shared_ptr<MeshDataStruct> MeshDataStructPtr;
	}//Geometry
}//VR

#endif//_VR_GEOMETRY_MESHDATASTRUCT_H