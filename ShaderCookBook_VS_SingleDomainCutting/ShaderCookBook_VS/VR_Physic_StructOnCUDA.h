#ifndef _VR_PHYSIC_STRUCTONCUDA_H
#define _VR_PHYSIC_STRUCTONCUDA_H

#define CUSP_CG_DEBUG (0)
#include "VR_MACRO.h"
#include <cusp/hyb_matrix.h>
#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg.h>
#include <iostream>
#include <map>
#include <cusp/print.h>
#include <cusp/precond/diagonal.h>
#include <cusp/blas.h>
#include <cusp/multiply.h>
#include <cusp/coo_matrix.h>
#include <cusp/print.h>
#include "cuda_runtime.h"
#include <cusp/hyb_matrix.h>
#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg.h>
#include <cusp/elementwise.h>
#include <cusp/linear_operator.h>



typedef int IndexType;
typedef float ValueType;
typedef int* IndexTypePtr;
typedef float * ValueTypePtr;
typedef float EigenValueType;
typedef EigenValueType * EigenValueTypePtr;
typedef cusp::device_memory MemorySpace;

#define DEBUG_TIME (0)
#if DEBUG_TIME
int nCurrentTick,nLastTick;
#endif

typedef cusp::array1d<ValueType, MemorySpace> CuspVec;
typedef CuspVec::view MyCuspVecView;
typedef cusp::array1d<ValueType, cusp::host_memory> CuspVecOnCpu;
typedef cusp::csr_matrix<int,ValueType,MemorySpace> CuspMatrix;

#include "VR_Physic_StructOnCUDA_CPU.h"

#if USE_CUDA_STREAM
cudaStream_t streamForCalc;
cudaStream_t streamForSkin;
#endif

#define STENCIL_DEFINITION (1)
#if STENCIL_DEFINITION
__global__ 	void make_diagonal(int nDofs, EigenValueTypePtr diagonal,
								IndexTypePtr outerIndexPtr,
								IndexTypePtr innerIndexPtr,
								EigenValueTypePtr valuePtr,
								IndexTypePtr  bcPtr)
{
	int currentDof = threadIdx.x + blockIdx.x * blockDim.x;

	if (currentDof < nDofs)
	{
		int base = currentDof*nMaxNonZeroSize;//outerIndexPtr[currentDof];
		int nCount = nMaxNonZeroSize;//outerIndexPtr[currentDof+1] - outerIndexPtr[currentDof];

		if (1 == bcPtr[currentDof])
		{
			diagonal[currentDof] = 0.f;
		}
		else
		{


			IndexType colDof;
			EigenValueType colVal=0.0f;
			for (int v=0;v<nCount;++v)
			{
				colDof = innerIndexPtr[base + v];
				if (colDof == currentDof)
				{
					colVal += valuePtr[base + v];
				}
			}

			diagonal[currentDof] = colVal;
		}
	}
}

__global__ 	void make_diagonal_plus(int nDofs, EigenValueTypePtr diagonal,
	IndexTypePtr outerIndexPtr,
	IndexTypePtr innerIndexPtr,
	EigenValueTypePtr valuePtr,
	IndexTypePtr  bcPtr)
{
	int currentDof = threadIdx.x + blockIdx.x * blockDim.x;

	if (currentDof < nDofs)
	{
		int base = currentDof*nMaxNonZeroSize;//outerIndexPtr[currentDof];
		int nCount = nMaxNonZeroSize;//outerIndexPtr[currentDof+1] - outerIndexPtr[currentDof];

		if (1 == bcPtr[currentDof])
		{
			diagonal[currentDof] = 0.f;
		}
		else
		{


			IndexType colDof;
			EigenValueType colVal=0.0f;
			for (int v=0;v<nCount;++v)
			{
				colDof = innerIndexPtr[base + v];
				if (colDof == currentDof)
				{
					colVal += valuePtr[base + v];
				}
			}

			diagonal[currentDof] = 1.0f/colVal;
		}
	}
}

__global__ 
void stencil_kernel_plusplus(int nDofs, const float * x, float * y,
					IndexTypePtr /*outerIndexPtr*/,
					IndexTypePtr innerIndexPtr,
					EigenValueTypePtr valuePtr,
					IndexTypePtr  bcPtr,
					EigenValueTypePtr diagonal,
					bool bNeedDiag)

{
	__shared__ float vals [MAX_KERNEL_PER_BLOCK];
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x ; // global thread index
	int warp_id = thread_id / 32; // global warp index
	int lane = thread_id & (32 - 1); // thread index within the warp
	// one warp per row
	int row = warp_id ;
	if ( row < nDofs ){
		
		if (bNeedDiag && 1 == bcPtr[row])
		{
			y[row] = diagonal[row] * x[row];
			return ;
		}
		
		y[ row ] = 0.0f;
		int row_start = row*nMaxNonZeroSize;//outerIndexPtr[row];
		int row_end = (row+1)*nMaxNonZeroSize;//outerIndexPtr [ row +1];
		// compute running sum per thread
		vals [ threadIdx.x ] = 0;

		/*if (83 == row )
		{
			CUPRINTF("nDofs=%d,row=%d,bNeedDiag=%d,bcPtr[row]=%d,row_start=%d,row_end=%d\n",nDofs,row,(int)bNeedDiag,bcPtr[row],row_start,row_end);
		}*/

		for ( int jj = row_start + lane ; jj < row_end ; jj += 32)
		{
			vals [ threadIdx.x ] += valuePtr [jj] * x[ innerIndexPtr [jj ]];

			/*if (83 == row )
			{
				CUPRINTF("row=%d,col=%d,value=%f\n",row,innerIndexPtr [jj ],valuePtr [jj]);
			}*/
		}
		// parallel reduction in shared memory
		if ( lane < 16) vals [ threadIdx.x ] += vals [ threadIdx.x + 16];
		if ( lane < 8) vals [ threadIdx.x ] += vals [ threadIdx.x + 8];
		if ( lane < 4) vals [ threadIdx.x ] += vals [ threadIdx.x + 4];
		if ( lane < 2) vals [ threadIdx.x ] += vals [ threadIdx.x + 2];
		if ( lane < 1) vals [ threadIdx.x ] += vals [ threadIdx.x + 1];
		// first thread writes the result
		if ( lane == 0)
		{
			y[ row ] += vals [ threadIdx.x ];
		}
		
	}
}

class stencil : public cusp::linear_operator<float,cusp::device_memory>
{
public:
	typedef cusp::linear_operator<float,cusp::device_memory> super;

	int nDofs;
	IndexTypePtr outerIndexPtr;
	IndexTypePtr innerIndexPtr;
	EigenValueTypePtr valuePtr;
	MyCuspVecView diagonalVec;
	IndexTypePtr isBoundaryCondition;
	EigenValueTypePtr diagonalPtr;
	bool bNeedDiag;


	// constructor
	stencil()
		: super(0,0),bNeedDiag(false),isBoundaryCondition(0),diagonalPtr(0) {}

	void setNeedDiag(bool needDiag)
	{
		bNeedDiag = needDiag;
	}
	void initialize(int _dofs,IndexTypePtr outer, IndexTypePtr inner, EigenValueTypePtr value)
	{

		nDofs = _dofs;
		outerIndexPtr = outer;
		innerIndexPtr = inner;
		valuePtr = value;
		resize(nDofs,nDofs,0);

		if (!isBoundaryCondition)
		{
			cudaMalloc( (void**)&isBoundaryCondition,_nExternalMemory * (nDofs) * sizeof(IndexType));
			cudaMemset( (void*)isBoundaryCondition,0,_nExternalMemory * (nDofs) * sizeof(IndexType));
		}
		
		if (bNeedDiag)
		{
			if (!diagonalPtr)
			{
				HANDLE_ERROR( cudaMalloc( (void**)&diagonalPtr, _nExternalMemory * nDofs * sizeof(ValueType)));
				HANDLE_ERROR( cudaMemset( (void*)diagonalPtr,0, _nExternalMemory * nDofs * sizeof(ValueType)));
				{
					thrust::device_ptr<ValueType> wrapped_myOptimize_Array_Displacement(diagonalPtr);
					diagonalVec = MyCuspVecView(wrapped_myOptimize_Array_Displacement,wrapped_myOptimize_Array_Displacement+_nExternalMemory * nDofs);
					diagonalVec.resize(nDofs);
				}
			}

#if USE_CUDA_STREAM
			make_diagonal_plus<<< GRIDCOUNT(nDofs,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK ,0,streamForSkin>>>
#else
			make_diagonal_plus<<< GRIDCOUNT(nDofs,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK >>>
#endif
				(nDofs,diagonalPtr,outerIndexPtr, innerIndexPtr, valuePtr, isBoundaryCondition);

			/*diagonalVec.resize(nDofs,0);
			cudaMalloc( (void**)&diagonalPtr,(nDofs) * sizeof(ValueType));
			cudaMemset( (void*)diagonalPtr,0,(nDofs) * sizeof(ValueType));

			if (nDofs >= MAX_DOFS_COUNT)
			{
				LogInfo("nDofs(%d) >= %d\n",nDofs,MAX_DOFS_COUNT);
				MyError("");
			}
			
			
			make_diagonal<<<GRIDCOUNT(nDofs,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK>>>(nDofs,diagonalPtr,outerIndexPtr, innerIndexPtr, valuePtr, isBoundaryCondition);
			

			thrust::device_ptr<ValueType> wrapped_diagonal(diagonalPtr);

			cusp::array1d_view<thrust::device_ptr<ValueType>> cusp_diagonal (wrapped_diagonal, wrapped_diagonal + nDofs);

			CuspVec::view diagonalView(diagonalVec.begin(),diagonalVec.end());

			cusp::copy(cusp_diagonal, diagonalView);

			cudaFree(diagonalPtr);*/
		}
	}

	void setBoundaryCondition(IndexType idx)
	{
		static IndexType one[2]={1,0};
		cudaMemcpy( (void *)(&isBoundaryCondition[idx]),&one[0], sizeof(IndexType),cudaMemcpyHostToDevice );
	}
	// linear operator y = A*x
	template <typename VectorType1,
		typename VectorType2>
		void operator()(const VectorType1& x, VectorType2& y) const
	{
		// obtain a raw pointer to device memory
		const float * x_ptr = thrust::raw_pointer_cast(&x[0]);
		float * y_ptr = thrust::raw_pointer_cast(&y[0]);

#if USE_CUDA_STREAM
		stencil_kernel_plusplus<<< GRIDCOUNT(nDofs*WRAP_SIZE,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK ,0,streamForSkin>>>
#else
		stencil_kernel_plusplus<<< GRIDCOUNT(nDofs*WRAP_SIZE,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK >>>
#endif
		(nDofs, x_ptr, y_ptr, outerIndexPtr, innerIndexPtr, valuePtr, isBoundaryCondition, diagonalPtr, bNeedDiag);
	}

	/*CuspVec::iterator		diagonal_begin(void){return diagonalVec.begin();}
	CuspVec::const_iterator diagonal_begin(void) const{return diagonalVec.begin();}

	CuspVec::iterator		diagonal_end(void){return diagonalVec.end();}
	CuspVec::const_iterator diagonal_end(void) const{return diagonalVec.end();}*/
};

//namespace MyKrylov
//{
//	template <typename Matrix/*, typename Array*/>
//	void my_extract_diagonal(const Matrix& A, MyCuspVecView& output, cusp::unknown_format)
//	{
//		output = A.diagonalVec;
//	}
//}//namespace cusp

#endif

#define  MATRIX_DEBUG (0)

#define  PhysicsContextDefinition (1)
#if PhysicsContextDefinition

#include "VR_Physic_CuttingCheckStructOnCuda.h"

struct PhysicsContext
{
	MyCuspVecView cusp_Array_Incremental_displace;
	MyCuspVecView cusp_Array_Displacement;ValueTypePtr myOptimize_Array_Displacement;
	MyCuspVecView cusp_Array_R_rhs;
	MyCuspVecView cusp_Array_Mass_rhs;ValueTypePtr myOptimize_Array_Mass_rhs;
	MyCuspVecView cusp_Array_Damping_rhs;ValueTypePtr myOptimize_Array_Damping_rhs;
	MyCuspVecView cusp_Array_Velocity;ValueTypePtr myOptimize_Array_Velocity;
	MyCuspVecView cusp_Array_Acceleration;ValueTypePtr myOptimize_Array_Acceleration;
	MyCuspVecView cusp_Array_Rhs;ValueTypePtr g_systemRhsPtr_MF;//matrix-free
	MyCuspVecView cusp_Array_Old_Acceleration;ValueTypePtr myOptimize_Old_Acceleration;
	MyCuspVecView cusp_Array_Old_Displacement;ValueTypePtr myOptimize_Old_Displacement;
	MyCuspVecView cusp_Array_BladeForce;ValueTypePtr myOptimize_BladeForce;ValueTypePtr myOptimize_BladeForce_In8_MF;
	//CuspVec tmpRhs;
#if USE_CO_RATION
	MyCuspVecView cusp_Array_R_rhs_tmp4Corotaion;ValueTypePtr myOptimize_Array_R_rhs_tmp4Corotaion;
	MyCuspVecView cusp_Array_R_rhs_Corotaion;ValueTypePtr g_systemRhsPtr_MF_Rotation;//for cutting resize
	ValueTypePtr cuda_RKR;
	ValueTypePtr cuda_RKRtPj;	
	ValueTypePtr g_systemRhsPtr_In8_MF_Rotation;//for cutting resize
#endif
	IndexType nBcCount;
	IndexTypePtr cusp_boundaryCondition;
	IndexType nForceCount;
	IndexTypePtr cuda_forceCondition;

	//ValueTypePtr diagnosticValue;

	IndexType g_nDofs;
	IndexType g_nDofsLast;
	ValueTypePtr displacementOnCuda;ValueTypePtr displacementOnCuda4SpeedUp;
	ValueTypePtr rhsOnCuda;
	IndexTypePtr cuda_boundaryCondition;
	ValueTypePtr cuda_diagnosticValue;

	IndexType funcMatrixCount;
	ValueType * localStiffnessMatrixOnCuda;
	ValueType * localMassMatrixOnCuda;
	//ValueType * localRhsVectorOnCuda;
	IndexType nFEMShapeValueCount;
	FEMShapeValue* FEMShapeValueOnCuda;

	stencil SystemMatrix;
	stencil MassMatrix;
	stencil DampingMatrix;

	CommonCellOnCuda* CellOnCudaPtr;
	CommonCellOnCuda* CellOnCudaPtrOnCPU;
	
	/*int g_LinesElementCount;
	float * g_LinesElement;*/

#if USE_CUTTING
	int * topLevelCellInMain_OnHost;
	int * topLevelCellInMain_OnCuda;
	int * g_CellCollisionFlag_onCuda;
	int * g_CellCollisionFlag;
	int * beClonedObjectFlag_OnHost;
	int * beClonedObjectFlag_OnCuda;

	int g_nCuttingLineSetCount;
	CuttingLinePair * g_CuttingLineSet;
	char *beCuttingLinesFlagElement;//size = g_LinesElement is g_LinesElementCount

	int * beClonedVertexFlag_OnHost;
	int * beClonedVertexFlag_OnCuda;
#endif	

	int * cuda_invalidCellFlagPtr;
	int nCellOnCudaCount;
	int nCellOnCudaCountLast;

	//for cutting awareness


	IndexTypePtr g_globalDof_MF;//matrix-free
	ValueTypePtr g_globalValue_MF;////matrix-free
	IndexTypePtr g_globalDof_Mass_MF;//matrix-free
	ValueTypePtr g_globalValue_Mass_MF;//matrix-free
	IndexTypePtr g_globalDof_Damping_MF;//matrix-free
	ValueTypePtr g_globalValue_Damping_MF;//matrix-free
	IndexTypePtr g_globalDof_System_MF;//matrix-free
	ValueTypePtr g_globalValue_System_MF;//matrix-free
#if MATRIX_DEBUG
	IndexTypePtr g_globalDof_MF_CPU;
	ValueTypePtr g_globalValue_MF_CPU;
	IndexTypePtr g_globalDof_Mass_MF_CPU;
	ValueTypePtr g_globalValue_Mass_MF_CPU;
	IndexTypePtr g_globalDof_Damping_MF_CPU;
	ValueTypePtr g_globalValue_Damping_MF_CPU;
	IndexTypePtr g_globalDof_System_MF_CPU;
	ValueTypePtr g_globalValue_System_MF_CPU;
#endif
	
	

	float *materialParams;
	int   *materialIndex;
	float *materialValue;
	float *g_externForce;
	float g_lai;
	float g_Density;

	//
	CuspVec g_globalExternalForce;
	ValueTypePtr g_systemRhsPtr_In8_MF;//matrix-free
	
	ValueTypePtr g_globalExternalOnCuda_x8;
	int g_isApplyExternalForceAtCurrentFrame;
	int g_isApplyBladeForceCurrentFrame;


	int g_nVertexOnCudaCount;
	VertexOnCuda* g_VertexOnCudaPtr;
	VertexOnCuda* g_VertexOnCudaPtrOnCPU;
	IndexTypePtr g_linePairOnCuda;/*12X2=24*/

	MyCuspVecView cg_y;
	MyCuspVecView cg_z;
	MyCuspVecView cg_r;
	MyCuspVecView cg_p;
	ValueTypePtr cg_ptr_y;
	ValueTypePtr cg_ptr_z;
	ValueTypePtr cg_ptr_r;
	ValueTypePtr cg_ptr_p;
};
#endif
#endif//_VR_PHYSIC_STRUCTONCUDA_H