#ifndef _VR_GPU_GEOMETRY_OCTREELEVELINFO_H
#define _VR_GPU_GEOMETRY_OCTREELEVELINFO_H

#include "VR_MACRO.h"
#include <vector_types.h>
namespace YC
{
	namespace Geometry
	{
		namespace GPU
		{
			struct LevelInfo
			{
				int nLevel;
				int nBase;
				int nCount;
				float radius;
				float radiusX2;
				int ElementCountPerDim;
			};

			struct OctreeInfo
			{
				//geometry attribution
				//int nId;//start 0
				int nId4CurrentLevel;
				int nLevel;
				float radius;
				//int   spaceIndexes[MyDIM];
				float centerPos[MyDIM];

				//state attribution
				int isContainMeshTriangle;//-1 : no; 0 : yes
				int isLeaf;//MyYES MyNO
				int isInside;//MyYES MyNO
				int nearestMeshTriangleId;//start 0
				
				float distance;
				float3 center2nearestPtDirect;

				int nId;
				int parentId;
				int childId[8];
			};
		}
	}
}
#endif//_VR_GPU_GEOMETRY_OCTREELEVELINFO_H