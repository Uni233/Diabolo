#ifndef _VR_GPU_GEOMETRY_OCTREE_H
#define _VR_GPU_GEOMETRY_OCTREE_H
//__device__ inline const RenderInput& getInput(void)

#include "VR_MACRO.h"
#include "VR_Geometry_MeshDataStruct.h"
#include "VR_GPU_Geometry_OctreeLevelInfo.h"
#if USE_CUDA
#include "vector_types.h"
#include "MyVBOLineSet.h"
namespace YC
{
	namespace Geometry
	{
		namespace GPU
		{
			class VR_Octree
			{
			public:
				enum{ChildrenCount = 8};
				
			public:
				VR_Octree()
				{
					m_OctreeInfoElement_Cpu = MyNull;
					m_VertexPosElement_Cpu = MyNull;				
					m_FaceIdxElement_Cpu = MyNull;				
					m_NormalDirectElement_Cpu = MyNull;
				}
				virtual ~VR_Octree(){}

				int spliteCube(const MeshDataStruct& objMeshInfo, int nLevel);
				void parserOnCpu(const int nHexCount,const MeshDataStruct& objMeshInfo);
				void parserInside(const int nHexCount,const MeshDataStruct& objMeshInfo);
				void mergeInside(const std::vector<LevelInfo>& vecLevelInfo);
				MyVBOLineSet * makeVBOLineSet();
				MyVBOLineSet * makeVBOLineSet(const int nLevel);

				void exportOctreeGrid(const char* lpszGridFile, const int nLevel);
			private:
				int m_OctreeInfoElement_Count;
				OctreeInfo * m_OctreeInfoElement_Cpu;
							
				float3* m_VertexPosElement_Cpu;
				int3*	m_FaceIdxElement_Cpu;
				float3* m_NormalDirectElement_Cpu;
			};
		}
	}
}
#endif

#endif//_VR_GPU_GEOMETRY_OCTREE_H