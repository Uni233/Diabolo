#include "stdafx.h"
#include "MeshGenerate.h"

void initCellMatrixOnCuda(int nCount,float * localStiffnessMatrixOnCpu,float * localMassMatrixOnCpu,float * localRhsVectorOnCpu);
void initFEMShapeValueOnCuda(int nCount, FEMShapeValue* femShapeValuePtr);

namespace VR_FEM
{
	void MeshGenerate::assembleFEMonCPU()
	{
		
	}

	void MeshGenerate::assembleFEMPreComputeMatrix()
	{
		const int nStiffnessMatrixSize = Cell::vec_cell_stiffness_matrix.size();
		const int nMassMatrixSize = Cell::vec_cell_mass_matrix.size();
		const int nRhsVectorSize = Cell::vec_cell_rhs_matrix.size();
		const int nFEMValueSize = Cell::vec_FEM_ShapeValue.size();

		Q_ASSERT( (nStiffnessMatrixSize == nMassMatrixSize)&&(nMassMatrixSize == nRhsVectorSize)&&(nRhsVectorSize == nFEMValueSize) );

		
		const int nSize = nStiffnessMatrixSize;
		MyFloat * localStiffnessMatrixOnCpu = new MyFloat[nSize * Geometry::dofs_per_cell * Geometry::dofs_per_cell];
		MyFloat * localMassMatrixOnCpu		= new MyFloat[nSize * Geometry::dofs_per_cell * Geometry::dofs_per_cell];
		MyFloat * localRhsVectorOnCpu       = new MyFloat[nSize * Geometry::dofs_per_cell];
		FEMShapeValue * localFEMShapeValueOnCpu = new FEMShapeValue [nFEMValueSize]; 

		memset(localStiffnessMatrixOnCpu	,'\0'	,nSize * Geometry::dofs_per_cell * Geometry::dofs_per_cell);
		memset(localMassMatrixOnCpu			,'\0'	,nSize * Geometry::dofs_per_cell * Geometry::dofs_per_cell);
		memset(localRhsVectorOnCpu			,'\0'	,nSize * Geometry::dofs_per_cell );
		memset(localFEMShapeValueOnCpu		,'\0'	,nFEMValueSize * sizeof(FEMShapeValue));

		//assemble stiffness
		for (unsigned idx=0;idx<nSize;++idx)
		{
			const MyMatrix& curStiffnessMatrix = Cell::vec_cell_stiffness_matrix[idx].matrix;
			const MyMatrix& curMassMatrix = Cell::vec_cell_mass_matrix[idx].matrix;
			const MyVector& curRhsVector = Cell::vec_cell_rhs_matrix[idx].vec;
			for (int row=0;row < Geometry::dofs_per_cell;++row)
			{
				for(int col=0;col < Geometry::dofs_per_cell;++col)
				{
					localStiffnessMatrixOnCpu[ idx*Geometry::dofs_per_cell * Geometry::dofs_per_cell + row * Geometry::dofs_per_cell + col] = curStiffnessMatrix.coeff(row,col);
					localMassMatrixOnCpu     [ idx*Geometry::dofs_per_cell * Geometry::dofs_per_cell + row * Geometry::dofs_per_cell + col] = curMassMatrix.coeff(row,col);
				}
				localRhsVectorOnCpu[idx*Geometry::dofs_per_cell + row] = curRhsVector.coeff(row,0);
			}	

			for (unsigned v=0;v<Geometry::vertexs_per_cell;++v)
			{
				for (unsigned i=0;i<Geometry::vertexs_per_cell;++i)
				{
					localFEMShapeValueOnCpu[idx].shapeFunctionValue_8_8[v][i] = Cell::vec_FEM_ShapeValue[idx].shapeFunctionValue_8_8[v][i];
					localFEMShapeValueOnCpu[idx].shapeDerivativeValue_8_8_3[v][i][0] = Cell::vec_FEM_ShapeValue[idx].shapeDerivativeValue_8_8_3[v][i][0];
					localFEMShapeValueOnCpu[idx].shapeDerivativeValue_8_8_3[v][i][1] = Cell::vec_FEM_ShapeValue[idx].shapeDerivativeValue_8_8_3[v][i][1];
					localFEMShapeValueOnCpu[idx].shapeDerivativeValue_8_8_3[v][i][2] = Cell::vec_FEM_ShapeValue[idx].shapeDerivativeValue_8_8_3[v][i][2];
				}
			}
		}

		initCellMatrixOnCuda(nSize,localStiffnessMatrixOnCpu,localMassMatrixOnCpu,localRhsVectorOnCpu);
		initFEMShapeValueOnCuda(nSize,localFEMShapeValueOnCpu);
	}
}