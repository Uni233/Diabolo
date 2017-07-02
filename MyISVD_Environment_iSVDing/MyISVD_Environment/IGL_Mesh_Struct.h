#ifndef _IGL_MESH_STRUCT_H_
#define _IGL_MESH_STRUCT_H_

#include "VR_Global_Define.h"

namespace YC
{
	struct IGL_Mesh_Struct
	{
		YC::MyMatrix m_original_vtx_pos;
		YC::MyIntMatrix m_original_vtx_dofs;
		YC::MyMatrix m_current_vtx_pos;
		YC::MyMatrix m_current_vtx_disp;
		YC::MyIntMatrix m_original_face_struct;
		YC::MyIntVector m_current_face_color_id;
		YC::MyMatrix m_current_face_color;
		YC::MyMatrix m_current_face_color_subspace;
		YC::MyInt m_nVtxSize;
		YC::MyInt m_nFaceSize;
	};
}

#endif//_IGL_MESH_STRUCT_H_