#ifndef _VR_GPU_PHYSIC_STRUCTINFO_H
#define _VR_GPU_PHYSIC_STRUCTINFO_H

#include "VR_MACRO.h"
#include <vector_types.h>

struct MeshVertex2CellInfo
{
	float3 m_vertexPos;
	float  m_dist;
	int  m_cellIdBelong;
};

struct Cell2MeshVertexInfo
{
	float3 m_centerPos;
	float  m_radius;
	int    m_nDofs[24];
};
#endif//_VR_GPU_PHYSIC_STRUCTINFO_H