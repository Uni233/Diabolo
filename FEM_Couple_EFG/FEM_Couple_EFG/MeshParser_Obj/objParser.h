#ifndef _ObjParser_H
#define _ObjParser_H

#include "../VR_Global_Define.h"
#include "MeshParser.h"
#include <vector>

namespace VR_FEM
{
	struct ObjParserData
	{
		ObjParserData()
			:m_translation_x(MyZero),m_translation_y(MyZero),m_translation_z(MyZero),m_maxDiameter(1.f)
			,m_nCoordCount(MyZero),m_nVerticeCount(MyZero),m_nVerticeNormalCount(MyZero),m_nTriangleCount(MyZero)
			,m_hasCoord(false),m_hasVerticeNormal(false)
		{}
		std::vector< MyDenseVector > vertices;
		std::vector< MyDenseVector > verticeNormals;
		std::vector< MyDenseVector > normalizeVertices;
		std::vector< std::pair<MyFloat,MyFloat> > coords;
		std::vector< MyVectorI > face_indicies;
		std::vector< MyVectorI > vertexNormal_indicies;
		std::vector< MyVectorI > coord_indicies;

		MyFloat m_translation_x;
		MyFloat m_translation_y;
		MyFloat m_translation_z;
		MyFloat m_maxDiameter;

		MyInt m_nCoordCount;
		MyInt m_nVerticeCount;
		MyInt m_nVerticeNormalCount;
		MyInt m_nTriangleCount;

		bool m_hasCoord;
		bool m_hasVerticeNormal;
	};

	class objParser /*: public MeshParser*/
	{
	public:
		objParser(bool hasCoord,bool hasVerticeNormal,ObjParserData* dataPtr):m_hasCoord(false),m_hasVerticeNormal(false),m_dataPtr(dataPtr)/*,
					m_arrayCoord(MyNull),m_arrayVertice(MyNull),m_arrayVerticeNormal(MyNull),
					m_arrayTriangleIndice(MyNull),m_arrayCoordIndice(MyNull),m_arrayVerticeNormalIndice(MyNull),*/
					{
						Q_ASSERT(MyNull != m_dataPtr);
		}
		~objParser(){clear();}

	public:
		virtual bool parser(const char* lpszMeshPath);
		virtual void clear();
		static void makeScalarSamplePoint(ObjParserData* dataPtr,const std::vector< MyPoint >& vecNativePointSet , std::vector< MyPoint >& vecScalarPointSet);
		/*virtual MyInt getCoordCount()const{return m_nCoordCount;}
		virtual MyInt getVerticeCount()const{return m_nVerticeCount;}
		virtual MyInt getVerticeNormalCount()const{return m_nVerticeNormalCount;}
		virtual MyInt getTriangleCount()const{return m_nTriangleCount;}

		virtual MyInt getCoordStep()const{return 2;}
		virtual MyInt getVerticeStep()const{return 3;}
		virtual MyInt getVerticeNormalStep()const{return 3;}

		virtual MyFloat* getVerticeArray()const{return m_arrayVertice;}
		virtual MyFloat* getCoordArray()const{return m_arrayCoord;}
		virtual MyFloat* getVerticeNormalArray()const{return m_arrayVerticeNormal;}

		virtual MyInt* getTriangleIndiceArray()const{return m_arrayTriangleIndice;}
		virtual MyInt* getCoordIndiceArray()const{return m_arrayCoordIndice;}
		virtual MyInt* getVerticeNormalIndiceArray()const{return m_arrayVerticeNormalIndice;}*/

		/*void getNormalizeParameter(MyFloat& translation_x,MyFloat& translation_y,MyFloat& translation_z,MyFloat& maxDiameter)
		{
			translation_x = m_translation_x;
			translation_y = m_translation_y;
			translation_z = m_translation_z;
			maxDiameter   = m_maxDiameter;
		}*/

	private:
		MyFloat minIn3(MyFloat a, MyFloat b, MyFloat c)const;
		MyFloat maxIn3(MyFloat a, MyFloat b, MyFloat c)const;
	private:
		

		/*MyFloat* m_arrayCoord;
		MyFloat* m_arrayVertice;
		MyFloat* m_arrayVerticeNormal;

		MyInt* m_arrayTriangleIndice;
		MyInt* m_arrayCoordIndice;
		MyInt* m_arrayVerticeNormalIndice;*/

		bool m_hasCoord;
		bool m_hasVerticeNormal;
		ObjParserData* m_dataPtr;
		
	public:
		
		//std::vector< MyPoint > vecSamplePointSet;
	};
}
#endif//_ObjParser_H