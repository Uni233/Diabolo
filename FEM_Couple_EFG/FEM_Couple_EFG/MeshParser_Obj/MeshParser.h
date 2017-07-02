#ifndef _MeshParser_H
#define _MeshParser_H

#include "../VR_Global_Define.h"
namespace VR_FEM
{
	class MeshParser
	{
	public:
		virtual ~MeshParser(){}
	public:
		virtual bool parser(const char* lpszMeshPath)=0;
		virtual void clear()=0;
		virtual MyInt getCoordCount()const=0;
		virtual MyInt getVerticeCount()const=0;
		virtual MyInt getVerticeNormalCount()const=0;
		virtual MyInt getTriangleCount()const=0;

		virtual MyInt getCoordStep()const=0;
		virtual MyInt getVerticeStep()const=0;
		virtual MyInt getVerticeNormalStep()const=0;

		virtual MyFloat* getVerticeArray()const=0;
		virtual MyFloat* getCoordArray()const=0;
		virtual MyFloat* getVerticeNormalArray()const=0;

		virtual MyInt* getTriangleIndiceArray()const=0;
		virtual MyInt* getCoordIndiceArray()const=0;
		virtual MyInt* getVerticeNormalIndiceArray()const=0;
	};
}
#endif//_MeshParser_H