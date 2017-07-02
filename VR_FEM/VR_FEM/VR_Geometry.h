#ifndef _VR_GEOMETRY_H
#define _VR_GEOMETRY_H

namespace YC
{

	namespace Geometry
	{

		const int shape_Function_Count_In_FEM = 8;
		const int gauss_Sample_Point = 8;
		const int n_tensor_pols = 8;
		const int dofs_per_cell = 24;
		const int dofs_per_cell_8 = 8;
		const int vertexs_per_cell = 8;
		const int dimensions_per_vertex = 3;
		const int first_dof_idx = 0;
		const int max_dofs_per_face = 12;
		const int faces_per_cell = 6;
		const int vertexs_per_face = 4;
		const int sons_per_cell = 8;
		const int lines_per_quad = 4;
		const int lines_per_cell = 12;
		const int vertexes_per_line = 2;
		const int subQuads_per_quad = 4;
		const int subLines_per_line = 2;

		static MyDenseVector Cell_Vertex_Sequence[Geometry::vertexs_per_cell] = {MyDenseVector(-1,-1,-1), MyDenseVector(1,-1,-1),
			MyDenseVector(-1,1,-1)	, MyDenseVector(1,1,-1),
			MyDenseVector(-1,-1,1)	, MyDenseVector(1,-1,1),
			MyDenseVector(-1,1,1)	, MyDenseVector(1,1,1)};
	}//namespace Geometry
}//namespace YC
#endif//_VR_GEOMETRY_H