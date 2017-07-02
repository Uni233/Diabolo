//#include "../elasticdeformed.h"
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

//#include <cusp/precond/smoothed_aggregation.h>
#include <iostream>
using namespace std;

#include "simplePrintf.cu"
#include "../CellStructOnCuda.h"
#include <helper_math.h>//normalize cross
#include <vector_functions.h>
#include "../MeshCuttingStructureOnCuda.h"


//extern cudaGraphicsResource *resource;

//extern const long g_bufferVetexCount;
//extern float3* devPtr;
//extern size_t  size;
extern float* devPtr;
extern int g_bcMinCount;
extern int g_bcMaxCount;
extern float g_externalForceFactor;

//#define KERNEL_COUNT 220000
//#define BLOCK_COUNT  128
#define InValidIdx (-1)
#define KERNEL_COUNT 100000
#define KERNEL_COUNTx5 500000
#define BLOCK_COUNT  128
#define KERNEL_COUNT_TMP 30000
#define BLOCK_COUNT_TMP  256
#define MASS_MATRIX_COEFF (0)//(16384000000UL)
//#define MASS_MATRIX_COEFF_2 (1000000UL)
#define MASS_MATRIX_COEFF_2 (1000UL)
#define OBLIGATE   (10000)
#define MATRIX_NONZERO (MaxDofCount*EFGInflunceMaxSize*3)
#define LocalMaxDofCount_YC (24)
#define DofMaxAssociateCellCount_YC (8)
#define CloneScalarFactor (0.91230f)

#define VertxMaxInflunceCellCount (8)
#define Dim_Count (3)
#define CellMaxInflunceVertexCount (8)
#define nMaxNonZeroSize  (VertxMaxInflunceCellCount*Dim_Count*CellMaxInflunceVertexCount)
#define nMaxNonZeroSizeInFEM  (192 * 1)
#define LocalMaxDofCount_YC (24)
#define MaxCellDof4EFG   (Dim_Count*CellMaxInflunceVertexCount)

#define MY_DEBUG
#define MY_PAUSE while(1){}
#define EFG_BasisNb_ (4)

#define CellTypeFEM (0)
#define CellTypeEFG (1)
#define CellTypeCouple (2)
#define BladeTriangleSize (1024)
#define BladeEdgeSize (BladeTriangleSize * 3)
#define BladeVertexSize (BladeTriangleSize*3)

#define USE_CORATIONAL (0)
 
 static void HandleError( cudaError_t err,
                          const char *file,
                          int line ) {
     if (err != cudaSuccess) {
         printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                 file, line );
         exit( EXIT_FAILURE );
     }
 }

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
// where to perform the computation
typedef cusp::device_memory MemorySpace;

// which floating point type to use
typedef float EigenValueType;
typedef EigenValueType * EigenValueTypePtr;
typedef float ValueType;
typedef ValueType * ValueTypePtr;
typedef int    IndexType;
typedef IndexType * IndexTypePtr;

typedef cusp::csr_matrix<int,ValueType,MemorySpace> CuspMatrix;
typedef cusp::array1d<ValueType, MemorySpace> CuspVec;

CuspMatrix cusp_CsrMt_System;
CuspMatrix cusp_CsrMt_Mass;
CuspMatrix cusp_CsrMt_Damping;

CuspVec cusp_Array_Incremental_displace;
CuspVec cusp_Array_Displacement;
CuspVec cusp_Array_R_rhs;
#if USE_CORATIONAL
CuspVec cusp_Array_R_rhs_tmp4Corotaion,cusp_Array_R_rhs_Corotaion;
////////////////////////////////////////    Corotation    /////////////////////////////////////////////////////////
float Signs_on_cpu[8*9] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,
							1,-1,-1,1,-1,-1,1,-1,-1,
						   -1,1,-1,-1,1,-1,-1,1,-1,
							1,1,-1,1,1,-1,1,1,-1,					  
							-1,-1,1,-1,-1,1,-1,-1,1,					  
							1,-1,1, 1,-1,1,1,-1,1,					  
							-1,1,1,-1,1,1,-1,1,1,					  
							1,1,1,1,1,1,1,1,1};
ValueTypePtr Signs_on_cuda = 0;
//ValueTypePtr cuda_RKR=0;
//ValueTypePtr cuda_RKRtPj=0;
void compareRocotation();
////////////////////////////////////////    Corotation    /////////////////////////////////////////////////////////
#endif
CuspVec cusp_Array_Mass_rhs;
CuspVec cusp_Array_Damping_rhs;
CuspVec cusp_Array_Velocity;
CuspVec cusp_Array_Acceleration;
CuspVec cusp_Array_Rhs;
CuspVec cusp_Array_Old_Acceleration;
CuspVec cusp_Array_Old_Displacement;

CuspVec cusp_Array_NewMarkConstant;
 
IndexType nBcCount;
IndexTypePtr cusp_boundaryCondition = 0;
IndexType nForceCount;
IndexTypePtr cuda_forceCondition = 0;

 ValueTypePtr diagnosticValue = 0;
 ValueType first_nonzero_diagonal_entry = 1;

IndexTypePtr g_lineDofPtr = 0;
IndexType g_lineDofSize;

IndexType g_nDofs,g_nDofsLast;
ValueTypePtr displacementOnCuda = 0;
ValueTypePtr rhsOnCuda = 0;
IndexTypePtr cuda_boundaryCondition = 0;
//ValueTypePtr cuda_boundaryConditionVal = 0;
ValueTypePtr cuda_diagnosticValue = 0;

int g_nLineCount4Display;


/////////////////////////   VBO DataStruct //////////////
IndexTypePtr vbo_line_vertex_pair = 0;
IndexType vbo_lineCount;
IndexTypePtr vbo_vertex_dofs = 0;
ValueTypePtr vbo_vertex_pos = 0;
IndexType vbo_vertexCount;
/////////////////////////   VBO DataStruct //////////////

/////////////////////////     triangle mesh /////////////////////
ValueTypePtr cuda_elemVertexPos = 0;
ValueTypePtr cuda_elemVertexMaxForce = 0;
IndexType g_nVertexPosLen,g_nTriangleMeshVertexCount;
ValueTypePtr cuda_elemVertexTriLinearWeight = 0;
IndexType g_nVertexTriLinearWeightLen;
IndexTypePtr cuda_elemVertexRelatedDofs = 0;
IndexType g_nVertexRelatedDofsLen;
ValueTypePtr cuda_elemVertexNormal = 0;
IndexType g_nVertexNormalLen,g_nTriangleMeshVertexNormalCount;
IndexTypePtr cuda_elemFaceVertexIdx = 0;
IndexType g_nFaceLen,g_nTriangleMeshFaceCount;
IndexTypePtr cuda_elemVertexNormalIdx = 0;
IndexType g_nVertexNormalIdxLen,g_nTriangleMeshVNCount;

////////////////////////      triangle mesh  /////////////////////

/////////////////////////////////////////   EFG Cell Structure ///////////////////////////
int    nExternalMemory = 2;
EFGCellOnCuda * EFG_CellOnCudaPtr=0;
EFGCellOnCuda_Ext * EFG_CellOnCuda_Ext_Ptr=0;
int * cuda_invalidCellFlagPtr = 0;
int EFG_CellOnCudaElementCount,EFG_CellOnCudaElementCount_Last;
/////////////////////////////////////////   EFG Cell Structure ///////////////////////////
//////////////////////////    collision detection //////////
int g_LinesElementCount;
float * g_LinesElement=0;
int * g_CellCollisionFlag=0;
ValueTypePtr g_BladeElementOnCuda=0;
ValueTypePtr g_BladeNormalOnCuda=0;

int g_nUpBladeVertexSize=0,g_nDownBladeVertexSize=0;
int g_nUpBladeEdgeSize=0,g_nDownBladeEdgeSize=0;
int g_nUpBladeSurfaceSize=0,g_nDownBladeSurfaceSize=0;
MC_Vertex_Cuda*		g_UpBlade_MC_Vertex_Cuda=0,*g_UpBlade_MC_Vertex_Cpu=0;
MC_Edge_Cuda*		g_UpBlade_MC_Edge_Cuda=0,*g_UpBlade_MC_Edge_Cpu=0;
MC_Surface_Cuda*	g_UpBlade_MC_Surface_Cuda=0,*g_UpBlade_MC_Surface_Cpu=0;

MC_Vertex_Cuda*		g_DownBlade_MC_Vertex_Cuda=0,*g_DownBlade_MC_Vertex_Cpu=0;
MC_Edge_Cuda*		g_DownBlade_MC_Edge_Cuda=0,*g_DownBlade_MC_Edge_Cpu=0;
MC_Surface_Cuda*	g_DownBlade_MC_Surface_Cuda=0,*g_DownBlade_MC_Surface_Cpu=0;
MC_CuttingEdge_Cuda * g_BladeEdgeVSSpliteFace = 0;//size equals blade edge size
//////////////////////////    collision detection //////////
////////////////////////////////////////  Vertex Structure /////////////////////////
int g_nVertexOnCudaCount;
VertexOnCuda* g_VertexOnCudaPtr = 0;
////////////////////////////////////////  Vertex Structure /////////////////////////
////////////////////////////////////////  Line Structure   ////////////////////////
IndexTypePtr g_linePairOnCuda = 0;
////////////////////////////////////////  Line Structure   ////////////////////////

//////////////////////////////////////////   Matrix-Free Structure  /////////////////////////
void assembleSystemOnCuda_EFG_RealTime();
IndexTypePtr g_globalDof_MF = 0;
ValueTypePtr g_globalValue_MF = 0;
IndexTypePtr g_globalDof_Mass_MF = 0;
ValueTypePtr g_globalValue_Mass_MF = 0;
IndexTypePtr g_globalDof_Damping_MF = 0;
ValueTypePtr g_globalValue_Damping_MF = 0;
IndexTypePtr g_globalDof_System_MF = 0;
ValueTypePtr g_globalValue_System_MF = 0;
ValueTypePtr g_systemRhsPtr_MF = 0;
ValueTypePtr g_systemRhsPtr_In8_MF = 0;

#if 0
ValueTypePtr g_globalNonZeroValue_System_MF = 0;
ValueTypePtr g_globalNonZeroValue_Mass_MF = 0;
ValueTypePtr g_globalNonZeroValue_Damping_MF = 0;
IndexTypePtr g_globalInnerIdx_MF = 0;
IndexTypePtr g_systemOuterIdxPtr_MF = 0;
IndexTypePtr g_systemOuterIdxPtrOnHost_MF = 0;
#endif
//////////////////////////////////////////   Matrix-Free Structure  /////////////////////////

/////////////////////////////////////////   matrix free   //////////////////////////////////////////////////////////

////////////////////////////////////////    FEM PreCompute Matrix /////////////////////
int funcMatrixCount;
ValueType * localStiffnessMatrixOnCuda=0;
ValueType * localMassMatrixOnCuda=0;
ValueType * localRhsVectorOnCuda=0;
int nFEMShapeValueCount=0;
FEMShapeValue* FEMShapeValueOnCuda=0;
////////////////////////////////////////    FEM PreCompute Matrix /////////////////////

///////////////////////////////////////////  Mesh Cutting  /////////////////////////////
int g_nMCVertexSize=0,g_nMaxMCVertexSize=0,g_nLastVertexSize=0;
int g_nMCEdgeSize=0,g_nMaxMCEdgeSize=0,g_nLastEdgeSize=0;
int g_nMCSurfaceSize=0,g_nMaxMCSurfaceSize=0,g_nLastSurfaceSize=0;
int g_nVertexNormalSize = 0;
MC_Vertex_Cuda*		g_MC_Vertex_Cuda=0;
MC_Edge_Cuda*		g_MC_Edge_Cuda=0;
MC_Surface_Cuda*	g_MC_Surface_Cuda=0;
float * g_elementVertexNormal = 0;//g_nVertexNormalSize * 3
///////////////////////////////////////////  Mesh Cutting  /////////////////////////////
#define MAX_BLOCK_SZ 256

__global__ 
void make_diagonal(int nDofs, EigenValueTypePtr diagonal,
					IndexTypePtr outerIndexPtr,
					IndexTypePtr innerIndexPtr,
					EigenValueTypePtr valuePtr)
{
	int currentDof = threadIdx.x + blockIdx.x * blockDim.x;
    
	if (currentDof < nDofs)
	{
		int base = currentDof*nMaxNonZeroSize;//outerIndexPtr[currentDof];
		int nCount = nMaxNonZeroSize;//outerIndexPtr[currentDof+1] - outerIndexPtr[currentDof];

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


__global__ 
void stencil_kernel(int nDofs, const float * x, float * y,
					IndexTypePtr outerIndexPtr,
					IndexTypePtr innerIndexPtr,
					EigenValueTypePtr valuePtr)
{
	int currentDof = threadIdx.x + blockIdx.x * blockDim.x;
    
	if (currentDof < nDofs)
	{
		int base = outerIndexPtr[currentDof];
		//int globalDofBase = currentDof * nMaxNonZeroSize;
		int nCount = outerIndexPtr[currentDof+1] - outerIndexPtr[currentDof];

		y[currentDof] = 0.0f;
		for (int v=0;v<nCount;++v)
		{
			const IndexType colDof = innerIndexPtr[base + v];
			const EigenValueType colVal = valuePtr[base + v];
			y[currentDof] += x[colDof] * colVal;
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
	__shared__ float vals [MAX_BLOCK_SZ];
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
		for ( int jj = row_start + lane ; jj < row_end ; jj += 32)
		vals [ threadIdx.x ] += valuePtr [jj] * x[ innerIndexPtr [jj ]];
		// parallel reduction in shared memory
		if ( lane < 16) vals [ threadIdx.x ] += vals [ threadIdx.x + 16];
		if ( lane < 8) vals [ threadIdx.x ] += vals [ threadIdx.x + 8];
		if ( lane < 4) vals [ threadIdx.x ] += vals [ threadIdx.x + 4];
		if ( lane < 2) vals [ threadIdx.x ] += vals [ threadIdx.x + 2];
		if ( lane < 1) vals [ threadIdx.x ] += vals [ threadIdx.x + 1];
		// first thread writes the result
		if ( lane == 0)
		y[ row ] += vals [ threadIdx.x ];
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
	CuspVec diagonalVec;
	IndexTypePtr isBoundaryCondition;
	EigenValueTypePtr diagonalPtr;
	bool bNeedDiag;
	

    // constructor
    stencil(bool needDiag)
        : super(0,0),bNeedDiag(needDiag),isBoundaryCondition(0),diagonalPtr(0) {}

	
	void initialize(int N,IndexTypePtr outer, IndexTypePtr inner, EigenValueTypePtr value)
	{

		 nDofs = N;
		 outerIndexPtr = outer;
		 innerIndexPtr = inner;
		 valuePtr = value;
		 resize(nDofs,nDofs,0);
		 
		 if (!isBoundaryCondition)
		 {
			HANDLE_ERROR( cudaMalloc( (void**)&isBoundaryCondition,nExternalMemory * (nDofs) * sizeof(IndexType))) ;
			HANDLE_ERROR( cudaMemset( (void*)isBoundaryCondition,0,nExternalMemory * (nDofs) * sizeof(IndexType))) ;
		 }

		 if (bNeedDiag)
		 {
			 

			 diagonalVec.resize(nDofs,0);
			 HANDLE_ERROR( cudaMalloc( (void**)&diagonalPtr,(nDofs) * sizeof(ValueType))) ;
			 HANDLE_ERROR( cudaMemset( (void*)diagonalPtr,0,(nDofs) * sizeof(ValueType))) ;

			 //printf("call make_diagonal begin\n");
			 make_diagonal<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(nDofs,diagonalPtr,outerIndexPtr, innerIndexPtr, valuePtr);
			 //printf("call make_diagonal end\n");

			 thrust::device_ptr<ValueType> wrapped_diagonal(diagonalPtr);

			 cusp::array1d_view<thrust::device_ptr<ValueType>> cusp_diagonal (wrapped_diagonal, wrapped_diagonal + nDofs);

			 CuspVec::view diagonalView(diagonalVec.begin(),diagonalVec.end());

			 cusp::copy(cusp_diagonal, diagonalView);

			 HANDLE_ERROR( cudaFree(diagonalPtr));
		 }
	}

	void setBoundaryCondition(IndexType idx)
	{
		static IndexType one[2]={1,0};
		HANDLE_ERROR( cudaMemcpy( (void *)(&isBoundaryCondition[idx]),&one[0], sizeof(IndexType),cudaMemcpyHostToDevice ) );
	}
    // linear operator y = A*x
    template <typename VectorType1,
              typename VectorType2>
    void operator()(const VectorType1& x, VectorType2& y) const
    {
        // obtain a raw pointer to device memory
        const float * x_ptr = thrust::raw_pointer_cast(&x[0]);
		float * y_ptr = thrust::raw_pointer_cast(&y[0]);

		stencil_kernel_plusplus<<< KERNEL_COUNT_TMP,MAX_BLOCK_SZ >>>(nDofs, x_ptr, y_ptr, outerIndexPtr, innerIndexPtr, valuePtr, isBoundaryCondition, diagonalPtr, bNeedDiag);
    }

	CuspVec::iterator		diagonal_begin(void){return diagonalVec.begin();}
    CuspVec::const_iterator diagonal_begin(void) const{return diagonalVec.begin();}

	CuspVec::iterator		diagonal_end(void){return diagonalVec.end();}
    CuspVec::const_iterator diagonal_end(void) const{return diagonalVec.end();}
};

stencil SystemMatrix(true);
stencil MassMatrix(false);
stencil DampingMatrix(false);
//stencil StiffnessMatrix(false);
/////////////////////////////////////////   matrix free   //////////////////////////////////////////////////////////

bool isZero(ValueType var)
{
    const  ValueType EPSINON = 0.000001;


    if(var < EPSINON  && var > -EPSINON)
    {
            return true;
    }
    else
    {
        return false;
    }
}
void setCuspMatrix(CuspMatrix & destMatrix,
				   IndexType nRows,
				   IndexType nCols,
				   IndexType nonZeros,
				   IndexTypePtr outerIndexPtr,
				   IndexType outerSize,
				   IndexTypePtr innerIndexPtr,
				   IndexType innerSize,
				   EigenValueTypePtr valuePtr,
				   IndexType valueSize)
{
	{
		cusp::csr_matrix<int,ValueType,cusp::host_memory> destMatrixHost;
		printf("%d,%d,%d \n",nRows,nCols,nonZeros);
		destMatrixHost.resize(nRows,nCols,nonZeros);

		printf("outerSize is %d \n",outerSize);
		for (IndexType v=0;v<outerSize;++v)
		{
			destMatrixHost.row_offsets[v] = outerIndexPtr[v];
		}

		printf("innerSize is %d \n",innerSize);
		for (IndexType v=0;v < innerSize;++v)
		{
			destMatrixHost.column_indices[v] = innerIndexPtr[v];
		}

		printf("valueSize is %d \n",valueSize);
		for (IndexType v = 0;v < valueSize;++v)
		{
			destMatrixHost.values[v] = (ValueType)valuePtr[v];
		}

		destMatrix = destMatrixHost;
	}
}

void setCuspVector(CuspVec& destVector,IndexType nRows,ValueTypePtr valuePtr)
{
	destVector.resize(nRows,0.);
	for (IndexType v = 0; v < nRows;++v)
	{
		destVector[v] = (ValueType)valuePtr[v];
	}
}


void initial_Cusp(const IndexType nDofs,

	              const IndexType nonZerosOfSystem,
					IndexTypePtr system_outerIndexPtr,
					IndexType system_outerSize,
					IndexTypePtr system_innerIndexPtr,
					IndexType system_innerSize,
					EigenValueTypePtr system_valuePtr,
					IndexType system_valueSize,

				  const IndexType nonZerosOfMass,
					IndexTypePtr mass_outerIndexPtr,
					IndexType mass_outerSize,
					IndexTypePtr mass_innerIndexPtr,
					IndexType mass_innerSize,
					EigenValueTypePtr mass_valuePtr,
					IndexType mass_valueSize,

				  const IndexType nonZerosOfDamping,
					IndexTypePtr damping_outerIndexPtr,
					IndexType damping_outerSize,
					IndexTypePtr damping_innerIndexPtr,
					IndexType damping_innerSize,
					EigenValueTypePtr damping_valuePtr,
					IndexType damping_valueSize,

					ValueTypePtr rhsValue,

				  EigenValueType dbNewMarkConstant[8],
				  IndexTypePtr boundaryconditionPtr,
				  IndexType bcCount,
				  
				  IndexTypePtr forceconditionPtr,
				  IndexType forceCount/*,
				  
				  ValueTypePtr nativeLocationPtr,
				  IndexType nativeLocationCount,
				  
				  IndexTypePtr lineDofPtr,
				  IndexType lineDofSize*/)
{
	g_nDofs = nDofs;
	g_nDofsLast = nDofs;
	nBcCount = bcCount;
	nForceCount = forceCount;
	printf("9  nBcCount = %d  nBcCount * sizeof(IndexType) is %d\n",nBcCount,nBcCount * sizeof(IndexType));
	
	cusp_boundaryCondition = (IndexTypePtr)malloc( nBcCount * sizeof(IndexType) );
	//cusp_boundaryConditionVal = (ValueTypePtr)malloc( nBcCount * sizeof(ValueType) );
	diagnosticValue = (ValueTypePtr)malloc( nDofs * sizeof(ValueType) );

	HANDLE_ERROR( cudaMalloc( (void**)&displacementOnCuda,nDofs * sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMalloc( (void**)&rhsOnCuda,nDofs * sizeof(ValueType))) ;

	HANDLE_ERROR( cudaMalloc( (void**)&cuda_boundaryCondition,nBcCount * sizeof(IndexType))) ;
	HANDLE_ERROR( cudaMalloc( (void**)&cuda_forceCondition,nForceCount * sizeof(IndexType))) ;
	HANDLE_ERROR( cudaMalloc( (void**)&cuda_diagnosticValue,nDofs * sizeof(ValueType))) ;
	
	
	memcpy(cusp_boundaryCondition,boundaryconditionPtr,nBcCount * sizeof(IndexType));
	//memcpy(cusp_boundaryConditionVal,boundaryconditionValPtr,nBcCount * sizeof(ValueType));

	HANDLE_ERROR( cudaMemcpy( (void *)cuda_boundaryCondition,boundaryconditionPtr,nBcCount * sizeof(IndexType),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( (void *)cuda_forceCondition,forceconditionPtr,nForceCount * sizeof(IndexType),cudaMemcpyHostToDevice ) );

	
	printf("10 \n");

	printf("5 \n");
	setCuspMatrix(cusp_CsrMt_System,nDofs,nDofs,nonZerosOfSystem,system_outerIndexPtr,system_outerSize,system_innerIndexPtr,system_innerSize,system_valuePtr,system_valueSize);
	//cusp::print(cusp_CsrMt_System);
	//exit(0);
	printf("6 \n");
	setCuspMatrix(cusp_CsrMt_Mass,nDofs,nDofs,nonZerosOfMass,mass_outerIndexPtr,mass_outerSize,mass_innerIndexPtr,mass_innerSize,mass_valuePtr,mass_valueSize);
	printf("7 \n");
	setCuspMatrix(cusp_CsrMt_Damping,nDofs,nDofs,nonZerosOfDamping,damping_outerIndexPtr,damping_outerSize,damping_innerIndexPtr,damping_innerSize,damping_valuePtr,damping_valueSize);
	printf("8 \n");
	setCuspVector(cusp_Array_Rhs,nDofs,rhsValue);


	cusp_Array_Incremental_displace.resize(nDofs,0);
	cusp_Array_Displacement.resize(nDofs,0);
	cusp_Array_R_rhs.resize(nDofs,0);
	cusp_Array_Mass_rhs.resize(nDofs,0);
	cusp_Array_Damping_rhs.resize(nDofs,0);
	cusp_Array_Velocity.resize(nDofs,0);
	cusp_Array_Acceleration.resize(nDofs,0);
	//cusp_Array_Rhs.resize(nDofs,0);
	cusp_Array_Old_Acceleration.resize(nDofs,0);
	cusp_Array_Old_Displacement.resize(nDofs,0);

	cusp_Array_NewMarkConstant.resize(8);
	for (int v=0;v < 8;++v)
	{
		cusp_Array_NewMarkConstant[v] = (ValueType)dbNewMarkConstant[v];
	}
	
}

void initVBODataStruct(IndexTypePtr line_vertex_pair,IndexType lineCount,IndexTypePtr vertex_dofs,ValueTypePtr vertex_pos,IndexType vertexCount)
{
	vbo_lineCount = lineCount;
	vbo_vertexCount = vertexCount;

	printf("initVBODataStruct lineCount is %d  vertexCount is %d \n",lineCount,vertexCount);

	HANDLE_ERROR( cudaMalloc( (void**)&vbo_line_vertex_pair,lineCount * 2 * sizeof(IndexType))) ;
	HANDLE_ERROR( cudaMalloc( (void**)&vbo_vertex_dofs,vertexCount * 3 * sizeof(IndexType))) ;
	HANDLE_ERROR( cudaMalloc( (void**)&vbo_vertex_pos,vertexCount * 3 * sizeof(ValueType))) ;

	HANDLE_ERROR( cudaMemcpy( (void *)vbo_line_vertex_pair,line_vertex_pair,lineCount * 2 * sizeof(IndexType),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( (void *)vbo_vertex_dofs,vertex_dofs,vertexCount * 3 * sizeof(IndexType),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( (void *)vbo_vertex_pos,vertex_pos,vertexCount * 3 * sizeof(ValueType),cudaMemcpyHostToDevice ) );
}

void initVBODataStruct_triangleMesh(IndexType nTriangleMeshVertexCount,
									IndexType nTriangleMeshVertexNormalCount,
									IndexType nTriangleMeshFaceCount,
									IndexType nTriangleMeshVNCount,
									ValueTypePtr elemVertexPos,IndexType nVertexPosLen,
									ValueTypePtr elemVertexTriLinearWeight,IndexType nVertexTriLinearWeightLen,
									IndexTypePtr elemVertexRelatedDofs,IndexType nVertexRelatedDofsLen,
									ValueTypePtr elemVertexNormal,IndexType nVertexNormalLen,
									IndexTypePtr elemFaceVertexIdx,IndexType nFaceLen,
									IndexTypePtr elemVertexNormalIdx,IndexType nVertexNormalIdxLen)

{
	g_nTriangleMeshVertexCount = nTriangleMeshVertexCount;
	g_nTriangleMeshVertexNormalCount = nTriangleMeshVertexNormalCount;
	g_nTriangleMeshFaceCount = nTriangleMeshFaceCount;
	g_nTriangleMeshVNCount = nTriangleMeshVNCount;

	g_nVertexPosLen = nVertexPosLen;
	g_nVertexTriLinearWeightLen = nVertexTriLinearWeightLen;
	g_nVertexRelatedDofsLen = nVertexRelatedDofsLen;
	g_nVertexNormalLen = nVertexNormalLen;
	g_nFaceLen = nFaceLen;
	g_nVertexNormalIdxLen = nVertexNormalIdxLen;

	HANDLE_ERROR( cudaFree(cuda_elemVertexPos));
	HANDLE_ERROR( cudaFree(cuda_elemVertexMaxForce));
	HANDLE_ERROR( cudaFree( cuda_elemVertexTriLinearWeight ));
	HANDLE_ERROR( cudaFree( cuda_elemVertexRelatedDofs ));
	HANDLE_ERROR( cudaFree( cuda_elemVertexNormal ));
	HANDLE_ERROR( cudaFree( cuda_elemFaceVertexIdx ));
	HANDLE_ERROR( cudaFree( cuda_elemVertexNormalIdx ));

	HANDLE_ERROR( cudaMalloc( (void**)&cuda_elemVertexPos               ,nVertexPosLen				* sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMalloc( (void**)&cuda_elemVertexMaxForce          ,nVertexPosLen				* sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMalloc( (void**)&cuda_elemVertexTriLinearWeight	,nVertexTriLinearWeightLen	* sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMalloc( (void**)&cuda_elemVertexRelatedDofs		,nVertexRelatedDofsLen		* sizeof(IndexType))) ;
	HANDLE_ERROR( cudaMalloc( (void**)&cuda_elemVertexNormal			,nVertexNormalLen			* sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMalloc( (void**)&cuda_elemFaceVertexIdx			,nFaceLen					* sizeof(IndexType))) ;
	HANDLE_ERROR( cudaMalloc( (void**)&cuda_elemVertexNormalIdx			,nVertexNormalIdxLen		* sizeof(IndexType))) ;

	HANDLE_ERROR( cudaMemcpy( (void *)cuda_elemVertexPos			,elemVertexPos				,nVertexPosLen * sizeof(ValueType),	cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemset( (void *)cuda_elemVertexMaxForce		,0							,nVertexPosLen * sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMemcpy( (void *)cuda_elemVertexTriLinearWeight,elemVertexTriLinearWeight	,nVertexTriLinearWeightLen	* sizeof(ValueType),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( (void *)cuda_elemVertexRelatedDofs	,elemVertexRelatedDofs		,nVertexRelatedDofsLen		* sizeof(IndexType),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( (void *)cuda_elemVertexNormal			,elemVertexNormal			,nVertexNormalLen			* sizeof(ValueType),	cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( (void *)cuda_elemFaceVertexIdx		,elemFaceVertexIdx			,nFaceLen					* sizeof(IndexType),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( (void *)cuda_elemVertexNormalIdx		,elemVertexNormalIdx		,nVertexNormalIdxLen		* sizeof(IndexType),cudaMemcpyHostToDevice ) );
}

__global__ void cudaMoveMesh(float4* pos, uchar4 *colorPos,IndexTypePtr line_vertex_pair,IndexType lineCount,IndexTypePtr vertex_dofs,ValueTypePtr vertex_pos,IndexType vertexCount,ValueTypePtr displacement,IndexType displacementCount)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < lineCount)
	{
		int lineIdx = tid * 2;
	

		int BeginVertexId = line_vertex_pair[lineIdx];
		int EndVertexId = line_vertex_pair[lineIdx+1];
		int BeginVertexIdx = BeginVertexId * 3;
		int EndVertexIdx = EndVertexId * 3;

		pos[lineIdx ]    = make_float4(vertex_pos[BeginVertexIdx] + displacement[ vertex_dofs[BeginVertexIdx] ],vertex_pos[BeginVertexIdx+1] + displacement[ vertex_dofs[BeginVertexIdx+1] ],vertex_pos[BeginVertexIdx+2]+ displacement[ vertex_dofs[BeginVertexIdx+2] ],1.0f);
		pos[lineIdx + 1] = make_float4(vertex_pos[EndVertexIdx  ] + displacement[ vertex_dofs[EndVertexIdx  ] ],vertex_pos[EndVertexIdx  +1] + displacement[ vertex_dofs[EndVertexIdx  +1] ],vertex_pos[EndVertexIdx  +2]+ displacement[ vertex_dofs[EndVertexIdx  +2] ],1.0f);
		/*colorPos[lineIdx] = make_uchar4(0,85,128,0);
		colorPos[lineIdx + 1] = make_uchar4(0,85,128,0);*/

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void cudaMoveMeshOptimal(float4* pos, uchar4 *colorPos,
									int nCellOnCudaCount, EFGCellOnCuda * cellOnCuda, 
									VertexOnCuda* VertexOnCudaPtr,
									ValueTypePtr displacement,IndexType displacementCount, IndexTypePtr linePair)
{
	int currentThreadId = threadIdx.x + blockIdx.x * blockDim.x;
	int currentCellIdx = currentThreadId /12;
	int currentLineIdx = currentThreadId % 12;
	if (currentCellIdx < nCellOnCudaCount)
	{
		int v2 = 2*currentLineIdx;
		int lineIdx = currentCellIdx * 24;
		int p0,p1;
		int vertexId0,vertexId1;
		int *vertexDof0,*vertexDof1;
		if (true == cellOnCuda[currentCellIdx].m_bLeaf)
		{
			//int dofOrder[8] = {2,3,6,7,0,1,4,5};//{4,5,0,1,6,7,2,3};
			//for (int v=0;v<12;++v)
			{
				p0 = linePair[v2];
				p1 = linePair[v2+1];
				vertexId0 = cellOnCuda[currentCellIdx].vertexId[p0];
				vertexId1 = cellOnCuda[currentCellIdx].vertexId[p1];
				vertexDof0 = &VertexOnCudaPtr[vertexId0].m_nDof[0];
				vertexDof1 = &VertexOnCudaPtr[vertexId1].m_nDof[0];

				
				pos[lineIdx+v2] = make_float4( VertexOnCudaPtr[vertexId0].local[0] + displacement[ vertexDof0[0] ],
												VertexOnCudaPtr[vertexId0].local[1] + displacement[ vertexDof0[1]],
												VertexOnCudaPtr[vertexId0].local[2] + displacement[ vertexDof0[2]],
												1.f);

				
				pos[lineIdx+v2+1] = make_float4(   VertexOnCudaPtr[vertexId1].local[0] + displacement[ vertexDof1[0] ],
													VertexOnCudaPtr[vertexId1].local[1] + displacement[ vertexDof1[1]],
													VertexOnCudaPtr[vertexId1].local[2] + displacement[ vertexDof1[2]],
													1.f);
				/*colorPos[lineIdx+v2] = make_uchar4(0,85,128,0);
				colorPos[lineIdx+v2+1] = make_uchar4(0,85,128,0);*/
			}
		}
		else
		{
			//for (int v=0;v<12;++v)
			{
				pos[lineIdx+v2] = make_float4(0.f,0.f,0.f,1.f);
				pos[lineIdx+v2+1] = make_float4(0.f,0.f,0.f,1.f);
				/*colorPos[lineIdx+v2] = make_uchar4(0,85,128,0);
				colorPos[lineIdx+v2+1] = make_uchar4(0,85,128,0);*/
			}
		}
	}
}

__global__ void cudaMoveMesh_TriangleMesh(float4* pos,float3 * vn,
										  ValueTypePtr elemVertexPos,
										  ValueTypePtr elemVertexTriLinearWeight,
										  IndexTypePtr elemVertexRelatedDofs,
										  ValueTypePtr elemVertexNormal,
										  IndexTypePtr elemFaceVertexIdx,
										  IndexType	   nFaceCount,
										  IndexTypePtr elemVertexNormalIdx,
										  ValueTypePtr displacement)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < nFaceCount)
	{
		int faceIdx = tid * 3;

		int Vertex0Id = elemFaceVertexIdx[faceIdx + 0];
		int Vertex1Id = elemFaceVertexIdx[faceIdx + 1];
		int Vertex2Id = elemFaceVertexIdx[faceIdx + 2];
		int VN0Id = elemVertexNormalIdx[faceIdx + 0];
		int VN1Id = elemVertexNormalIdx[faceIdx + 1];
		int VN2Id = elemVertexNormalIdx[faceIdx + 2];

		int dof_base[3]={Vertex0Id * 24,Vertex1Id * 24,Vertex2Id * 24};
		int pos_base[3] = {Vertex0Id * 3,Vertex1Id * 3,Vertex2Id * 3};
		int deta_base[3] = {Vertex0Id * 8,Vertex1Id * 8,Vertex2Id * 8};
		int vn_base[3] = {VN0Id*3,VN1Id*3,VN2Id*3};

		float p[2][2][2];
		float curDisplace[3];

		for (int i = 0; i < 3; ++i)
		{
			//int step = i;
			for (int step = 0; step < 3; ++step)
			{
				p[0][0][0] = displacement[ elemVertexRelatedDofs[dof_base[i] + 0*3 + step] ];
				p[1][0][0] = displacement[ elemVertexRelatedDofs[dof_base[i] + 1*3 + step] ];
				p[0][1][0] = displacement[ elemVertexRelatedDofs[dof_base[i] + 2*3 + step] ];
				p[1][1][0] = displacement[ elemVertexRelatedDofs[dof_base[i] + 3*3 + step] ];
				p[0][0][1] = displacement[ elemVertexRelatedDofs[dof_base[i] + 4*3 + step] ];
				p[1][0][1] = displacement[ elemVertexRelatedDofs[dof_base[i] + 5*3 + step] ];
				p[0][1][1] = displacement[ elemVertexRelatedDofs[dof_base[i] + 6*3 + step] ];
				p[1][1][1] = displacement[ elemVertexRelatedDofs[dof_base[i] + 7*3 + step] ];

				
				curDisplace[step] = p[0][0][0] * elemVertexTriLinearWeight[deta_base[i]+0] + 
								    p[1][0][0] * elemVertexTriLinearWeight[deta_base[i]+1] + 
								    p[0][1][0] * elemVertexTriLinearWeight[deta_base[i]+2] + 
								    p[1][1][0] * elemVertexTriLinearWeight[deta_base[i]+3] + 
								    p[0][0][1] * elemVertexTriLinearWeight[deta_base[i]+4] + 
								    p[1][0][1] * elemVertexTriLinearWeight[deta_base[i]+5] + 
								    p[0][1][1] * elemVertexTriLinearWeight[deta_base[i]+6] + 
								    p[1][1][1] * elemVertexTriLinearWeight[deta_base[i]+7] ;
				//curDisplace[step] = 0.f;
			}
	
			pos[faceIdx + i] = make_float4(elemVertexPos[pos_base[i] + 0] + curDisplace[0] - 0.5f,
										   elemVertexPos[pos_base[i] + 1] + curDisplace[1] - 0.5f,
										   elemVertexPos[pos_base[i] + 2] + curDisplace[2] - 0.5f,
										   1.0f);
			//colorPos[faceIdx + i] = make_uchar4(0,85,128,0);;
			vn[faceIdx + i].x = elemVertexNormal[vn_base[i] + 0];//make_float4( elemVertexNormal[vn_base[i] + 0],elemVertexNormal[vn_base[i] + 1],elemVertexNormal[vn_base[i] + 2],1.0f);
			vn[faceIdx + i].y = elemVertexNormal[vn_base[i] + 1];
			vn[faceIdx + i].z = elemVertexNormal[vn_base[i] + 2];
		}

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void cudaMoveMesh_TriangleMesh4MeshCutting(float4* pos,float3 * vn, uchar4 *colorPos,
													  const int nVertexSize,MC_Vertex_Cuda* curVertexSet,
													  const int nLineSize,MC_Edge_Cuda* curLineSet,
													  const int nTriSize,MC_Surface_Cuda* curFaceSet,
													  const int nVertexNormalSize,float* vecVN,
													  ValueTypePtr displacement,ValueTypePtr rhs,const int nLastTriSize)

{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < nTriSize)
	{
		const int faceIdx = tid * 3;
		MC_Surface_Cuda& curface = curFaceSet[tid];
		if (!curface.m_isValid)
		{
			for (int i = 0; i < 3; ++i)
			{
				pos[faceIdx + i] = make_float4(1000.f, 1000.f, 1000.f, 1.0f);
				colorPos[faceIdx + i] = make_uchar4(0,85,128,0);
			
				vn[faceIdx + i].x = vecVN[curface.m_VertexNormal[i] * 3 + 0];
				vn[faceIdx + i].y = vecVN[curface.m_VertexNormal[i] * 3 + 1];
				vn[faceIdx + i].z = vecVN[curface.m_VertexNormal[i] * 3 + 2];

			}
		}
		else
		{
			float p[2][2][2];
			float curDisplace[3],curForce[3];

			for (int i = 0; i < 3; ++i)//pt0,pt1,pt2
			{
				//if (curface.m_Vertex[i] >= 0 && curface.m_Vertex[i] <nVertexSize){}else{CUPRINTF("vertex invalid(Tri[%d]) \n",tid);}
				MC_Vertex_Cuda& curVertex = curVertexSet[curface.m_Vertex[i]];
				//int step = i;
				for (int step = 0; step < 3; ++step)//x,y,z
				{
				
					p[0][0][0] = displacement[ curVertex.m_elemVertexRelatedDofs[0*3 + step] ];
					p[1][0][0] = displacement[ curVertex.m_elemVertexRelatedDofs[1*3 + step] ];
					p[0][1][0] = displacement[ curVertex.m_elemVertexRelatedDofs[2*3 + step] ];
					p[1][1][0] = displacement[ curVertex.m_elemVertexRelatedDofs[3*3 + step] ];
					p[0][0][1] = displacement[ curVertex.m_elemVertexRelatedDofs[4*3 + step] ];
					p[1][0][1] = displacement[ curVertex.m_elemVertexRelatedDofs[5*3 + step] ];
					p[0][1][1] = displacement[ curVertex.m_elemVertexRelatedDofs[6*3 + step] ];
					p[1][1][1] = displacement[ curVertex.m_elemVertexRelatedDofs[7*3 + step] ];

				
					curDisplace[step] = p[0][0][0] * curVertex.m_TriLinearWeight[0] + 
										p[1][0][0] * curVertex.m_TriLinearWeight[1] + 
										p[0][1][0] * curVertex.m_TriLinearWeight[2] + 
										p[1][1][0] * curVertex.m_TriLinearWeight[3] + 
										p[0][0][1] * curVertex.m_TriLinearWeight[4] + 
										p[1][0][1] * curVertex.m_TriLinearWeight[5] + 
										p[0][1][1] * curVertex.m_TriLinearWeight[6] + 
										p[1][1][1] * curVertex.m_TriLinearWeight[7] ;
#if 0
					p[0][0][0] = rhs[ curVertex.m_elemVertexRelatedDofs[0*3 + step] ];
					p[1][0][0] = rhs[ curVertex.m_elemVertexRelatedDofs[1*3 + step] ];
					p[0][1][0] = rhs[ curVertex.m_elemVertexRelatedDofs[2*3 + step] ];
					p[1][1][0] = rhs[ curVertex.m_elemVertexRelatedDofs[3*3 + step] ];
					p[0][0][1] = rhs[ curVertex.m_elemVertexRelatedDofs[4*3 + step] ];
					p[1][0][1] = rhs[ curVertex.m_elemVertexRelatedDofs[5*3 + step] ];
					p[0][1][1] = rhs[ curVertex.m_elemVertexRelatedDofs[6*3 + step] ];
					p[1][1][1] = rhs[ curVertex.m_elemVertexRelatedDofs[7*3 + step] ];

					curForce[step] =  p[0][0][0] * curVertex.m_TriLinearWeight[0] + 
									p[1][0][0] * curVertex.m_TriLinearWeight[1] + 
									p[0][1][0] * curVertex.m_TriLinearWeight[2] + 
									p[1][1][0] * curVertex.m_TriLinearWeight[3] + 
									p[0][0][1] * curVertex.m_TriLinearWeight[4] + 
									p[1][0][1] * curVertex.m_TriLinearWeight[5] + 
									p[0][1][1] * curVertex.m_TriLinearWeight[6] + 
									p[1][1][1] * curVertex.m_TriLinearWeight[7] ;
#endif
					//curDisplace[step] = 0.f;
				}
#if 0
				ValueType currentForce = (curForce[0]*curForce[0]) + (curForce[1]*curForce[1]) + (curForce[2]*curForce[2]);
				//CUPRINTF(" curForce(%f)\n",currentForce);
				unsigned char curColor[3];
				ValueType dbTmp = sqrtf(currentForce);
				const int currentIdx = (dbTmp / 50000.0);
				double curr_weight;
			
				int currentIdxTwoLevel;
				if ( 0 == currentIdx)
				{
					currentIdxTwoLevel = int(dbTmp / 1600);
					curr_weight = (0.25/1600) * ((long)dbTmp % 1600);
				}
				else
				{
					currentIdxTwoLevel =  16;
					curr_weight = (0.25/250) * (dbTmp / 1600);
				
				}
				ValueType Color[21][3] = { {0,0,0},{0,0,0.25},{0,0,0.5},{0,0,0.75},{0,0,1},
									{0,0.25,1},{0,0.5,1},{0,0.75,1},{0,1,1},{0,1,0.75},
									{0,1,0.5},{0,1,0.25},{0,1,0},{0.25,1,0},{0.5,1,0},
									{0.75,1,0},{1,1,0},{1,0.5,0},{1,0,0},{1,0,0.5},{1,0,1}
									};

				ValueType ColorTemplate[21][3] = {{0.0,0.0,1.0},{0.0,0.0,1.0},{0.0,0.0,1.0},
										{0.0,0.0,1.0},{0.0,1.0,0.0},{0.0,1.0,0.0},
										{0.0,1.0,0.0},{0.0,1.0,0.0},{0.0,0.0,0.0},
										{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0},
										{1.0,0.0,0.0},{1.0,0.0,0.0},{1.0,0.0,0.0},
										{1.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0},
										{0.0,0.0,1.0},{0.0,0.0,1.0}};

				curColor[0] = (curr_weight * ColorTemplate[currentIdxTwoLevel][0] + Color[currentIdxTwoLevel][0])*255;
				curColor[1] = (curr_weight * ColorTemplate[currentIdxTwoLevel][1] + Color[currentIdxTwoLevel][1])*255;
				curColor[2] = (curr_weight * ColorTemplate[currentIdxTwoLevel][2] + Color[currentIdxTwoLevel][2])*255;
				colorPos[faceIdx + i] = make_uchar4( curColor[0],curColor[1],curColor[2],0);;//
#endif
				pos[faceIdx + i] = make_float4(curVertex.m_VertexPos[0] + curDisplace[0]/* - 0.5f*/,
											   curVertex.m_VertexPos[1] + curDisplace[1]/* - 0.5f*/,
											   curVertex.m_VertexPos[2] + curDisplace[2]/* - 0.5f*/,
											   1.0f);
				colorPos[faceIdx + i] = make_uchar4(211,255,9,0);;
			
				vn[faceIdx + i].x = vecVN[curface.m_VertexNormal[i] * 3 + 0];
				vn[faceIdx + i].y = vecVN[curface.m_VertexNormal[i] * 3 + 1];
				vn[faceIdx + i].z = vecVN[curface.m_VertexNormal[i] * 3 + 2];
			}

		}

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void cudaMoveMesh_TriangleMesh4MeshCutting_Face(float4* pos,float3 * vn, uchar4 *colorPos,
													  const int nVertexSize,MC_Vertex_Cuda* curVertexSet,
													  const int nLineSize,MC_Edge_Cuda* curLineSet,
													  const int nTriSize,MC_Surface_Cuda* curFaceSet,
													  const int nVertexNormalSize,float* vecVN,
													  ValueTypePtr displacement,const int nTriBase)

{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < nTriSize)
	{
		const int faceIdx = tid * 3 + nTriBase*3;
		MC_Surface_Cuda& curface = curFaceSet[tid];
		if (!curface.m_isValid)
		{
			for (int i = 0; i < 3; ++i)
			{
				pos[faceIdx + i] = make_float4(1000.f, 1000.f, 1000.f, 1.0f);
				colorPos[faceIdx + i] = make_uchar4(0,85,128,0);;
				vn[faceIdx + i].x = vecVN[curface.m_VertexNormal[i] * 3 + 0];
				vn[faceIdx + i].y = vecVN[curface.m_VertexNormal[i] * 3 + 1];
				vn[faceIdx + i].z = vecVN[curface.m_VertexNormal[i] * 3 + 2];

			}
		}
		else
		{
			//CUPRINTF("cudaMoveMesh_TriangleMesh4MeshCutting_Face\n");
			float p[2][2][2];
			float curDisplace[3];

			for (int i = 0; i <3 ; ++i)//pt0,pt1,pt2
			{
				MC_Vertex_Cuda& curVertex = curVertexSet[curface.m_Vertex[i]];
				for (int step = 0; step < 3; ++step)//x,y,z
				{				
					p[0][0][0] = displacement[ curVertex.m_elemVertexRelatedDofs[0*3 + step] ];
					p[1][0][0] = displacement[ curVertex.m_elemVertexRelatedDofs[1*3 + step] ];
					p[0][1][0] = displacement[ curVertex.m_elemVertexRelatedDofs[2*3 + step] ];
					p[1][1][0] = displacement[ curVertex.m_elemVertexRelatedDofs[3*3 + step] ];
					p[0][0][1] = displacement[ curVertex.m_elemVertexRelatedDofs[4*3 + step] ];
					p[1][0][1] = displacement[ curVertex.m_elemVertexRelatedDofs[5*3 + step] ];
					p[0][1][1] = displacement[ curVertex.m_elemVertexRelatedDofs[6*3 + step] ];
					p[1][1][1] = displacement[ curVertex.m_elemVertexRelatedDofs[7*3 + step] ];

				
					curDisplace[step] = p[0][0][0] * curVertex.m_TriLinearWeight[0] + 
										p[1][0][0] * curVertex.m_TriLinearWeight[1] + 
										p[0][1][0] * curVertex.m_TriLinearWeight[2] + 
										p[1][1][0] * curVertex.m_TriLinearWeight[3] + 
										p[0][0][1] * curVertex.m_TriLinearWeight[4] + 
										p[1][0][1] * curVertex.m_TriLinearWeight[5] + 
										p[0][1][1] * curVertex.m_TriLinearWeight[6] + 
										p[1][1][1] * curVertex.m_TriLinearWeight[7] ;
				}
	
				pos[faceIdx + i] = make_float4(curVertex.m_VertexPos[0] + curDisplace[0] /*- 0.5f*/,
											   curVertex.m_VertexPos[1] + curDisplace[1] /*- 0.5f*/,
											   curVertex.m_VertexPos[2] + curDisplace[2] /*- 0.5f*/,
											   1.0f);
				colorPos[faceIdx + i] = make_uchar4(128,0,0,0);;
				vn[faceIdx + i].x = vecVN[curface.m_VertexNormal[i] * 3 + 0];
				vn[faceIdx + i].y = vecVN[curface.m_VertexNormal[i] * 3 + 1];
				vn[faceIdx + i].z = vecVN[curface.m_VertexNormal[i] * 3 + 2];

				//CUPRINTF("cudaMoveMesh_TriangleMesh4MeshCutting_Face (%f,%f,%f)\n",curVertex.m_VertexPos[0] + curDisplace[0],curVertex.m_VertexPos[1] + curDisplace[1],curVertex.m_VertexPos[2] + curDisplace[2] );

				//pos[faceIdx + 2*i+1] = make_float4(curVertex.m_VertexPos[0] + curDisplace[0] /*- 0.5f*/,
				//							   curVertex.m_VertexPos[2] + curDisplace[2] /*- 0.5f*/,
				//							   curVertex.m_VertexPos[1] + curDisplace[1] /*- 0.5f*/,
				//							   1.0f);
				//colorPos[faceIdx + 2*i+1] = make_uchar4(211,255,9,0);;
				//vn[faceIdx + 2*i+1].x = -1.*vecVN[curface.m_VertexNormal[i] * 3 + 0];
				//vn[faceIdx + 2*i+1].y = -1.*vecVN[curface.m_VertexNormal[i] * 3 + 1];
				//vn[faceIdx + 2*i+1].z = -1.*vecVN[curface.m_VertexNormal[i] * 3 + 2];
			}

		}

		tid += blockDim.x * gridDim.x;
	}
}

void solve_cusp_cg_inner(float4* pos, uchar4 *colorPos, float4* pos_Triangles, float3* vertexNormal)
{

	{
		cusp::verbose_monitor<float> monitor(cusp_Array_R_rhs, 1000, 0.005f);
		cusp::precond::diagonal<ValueType, MemorySpace> M(SystemMatrix);
		//printf("begin cg\n");
		cusp::krylov::cg(SystemMatrix, cusp_Array_Incremental_displace, cusp_Array_R_rhs, monitor,M);
		//printf("end cg\n");
		thrust::device_ptr<ValueType> wrapped_displacement(displacementOnCuda);

		cusp::array1d_view<thrust::device_ptr<ValueType>> cusp_dX (wrapped_displacement, wrapped_displacement + g_nDofs);

		CuspVec::view displaceView(cusp_Array_Incremental_displace.begin(),cusp_Array_Incremental_displace.end());

		cusp::copy(displaceView, cusp_dX);
		{
			thrust::device_ptr<ValueType> wrapped_RHS(rhsOnCuda);
			cusp::array1d_view<thrust::device_ptr<ValueType>> cusp_dx_Rhs (wrapped_RHS, wrapped_RHS + g_nDofs);

			CuspVec::view rhsView(cusp_Array_R_rhs.begin(),cusp_Array_R_rhs.end());

			// copy the first half to the last half
			cusp::copy(rhsView, cusp_dx_Rhs);
		}
#if 0
		cudaMoveMesh<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(pos,colorPos,vbo_line_vertex_pair,vbo_lineCount,vbo_vertex_dofs,vbo_vertex_pos,vbo_vertexCount,displacementOnCuda,g_nDofs);
		cudaMoveMesh_TriangleMesh<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(
											  pos_Triangles,
											  vertexNormal,
											  cuda_elemVertexPos,
											  cuda_elemVertexTriLinearWeight,
											  cuda_elemVertexRelatedDofs,
											  cuda_elemVertexNormal,
											  cuda_elemFaceVertexIdx,
											  g_nTriangleMeshFaceCount,
											  cuda_elemVertexNormalIdx,
											  displacementOnCuda);
#endif
		
		//printf("cudaMoveMeshOptimal begin\n");
		cudaMoveMeshOptimal<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(
		pos,colorPos,
		EFG_CellOnCudaElementCount,EFG_CellOnCudaPtr,
		g_VertexOnCudaPtr,
		displacementOnCuda,g_nDofs, g_linePairOnCuda);
		

		cudaMoveMesh_TriangleMesh4MeshCutting<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>
											(pos_Triangles, vertexNormal,colorPos,
											  g_nMCVertexSize,g_MC_Vertex_Cuda,
											  g_nMCEdgeSize,g_MC_Edge_Cuda,
											  g_nMCSurfaceSize,g_MC_Surface_Cuda,
											  g_nVertexNormalSize,g_elementVertexNormal,
											  displacementOnCuda,rhsOnCuda,g_nLastSurfaceSize);
		
		

		cudaMoveMesh_TriangleMesh4MeshCutting_Face<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>
											(pos_Triangles, vertexNormal,colorPos,
											 g_nDownBladeVertexSize,g_DownBlade_MC_Vertex_Cuda,
											 g_nDownBladeEdgeSize,g_DownBlade_MC_Edge_Cuda,
											 g_nDownBladeSurfaceSize,g_DownBlade_MC_Surface_Cuda,
											 g_nVertexNormalSize,g_elementVertexNormal,
											 displacementOnCuda,g_nMCSurfaceSize);
		//return ;
		cudaMoveMesh_TriangleMesh4MeshCutting_Face<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>
											(pos_Triangles, vertexNormal,colorPos,
											 g_nUpBladeVertexSize,g_UpBlade_MC_Vertex_Cuda,
											 g_nUpBladeEdgeSize,g_UpBlade_MC_Edge_Cuda,
											 g_nUpBladeSurfaceSize,g_UpBlade_MC_Surface_Cuda,
											 g_nVertexNormalSize,g_elementVertexNormal,
											 displacementOnCuda,g_nMCSurfaceSize + g_nDownBladeSurfaceSize);
		//printf("cudaMoveMeshOptimal end\n");
		return ;
	}
}

void solve_cusp_cg(int size,float * pdisplacement,float * prhs)
{
	printf(" 3 \n");
	//displace
	cusp::array1d<ValueType, MemorySpace> displace(size, 0);
	//cusp::array1d<ValueType, cusp::host_memory> tmpDisplace(size, 0);

	//rhs
	//cusp::array1d<ValueType, cusp::host_memory> tmpRhs(size,0);
	cusp::array1d<ValueType, MemorySpace> rhs(size, 0);
	for (int v=0;v < size;++v)
	{
		//tmpRhs[v] = (float)prhs[v];
		rhs[v] = prhs[v];
	}
	printf(" 4 \n");
	
	//cusp::copy(tmpRhs,rhs);
	
    

    // set stopping criteria:
    //  iteration_limit    = 100
    //  relative_tolerance = 1e-3
    //cusp::verbose_monitor<ValueType> monitor(rhs, 100, 1e-3);
	cusp::verbose_monitor<ValueType> monitor(rhs, 1000, 1e-3);
	printf(" 5 \n");

    // set preconditioner (identity)
    //cusp::identity_operator<ValueType, MemorySpace> M(g_cuspSystem.num_rows, g_cuspSystem.num_rows);
	cusp::precond::diagonal<ValueType, MemorySpace> M(cusp_CsrMt_System);
	printf(" 6 \n");

    // solve the linear system A * x = b with the Conjugate Gradient method
	cusp::print(cusp_CsrMt_System);
	cusp::print(displace);
	cusp::print(rhs);
	exit(0);
    cusp::krylov::cg(cusp_CsrMt_System, displace, rhs, monitor, M);

	//cusp::copy(displace,tmpDisplace);

	printf(" 7 \n");
	for (int v=0;v<size;++v)
	{
		pdisplacement[v] = displace[v];
		//pdisplacement[v] = (double)tmpDisplace[v];
	}
	printf(" 8 \n");
}

__global__ void cudaApplyExternalForce(IndexType nBCCount, IndexTypePtr displaceBoundaryConditionDofs, ValueTypePtr rhs_on_cuda ,ValueType scaled)
{
	int nCurrentIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (nCurrentIdx < nBCCount)
	{
		rhs_on_cuda[ displaceBoundaryConditionDofs[nCurrentIdx] ] = scaled * 345.772125;
	}
}

void update_rhs(unsigned nCount)
{
	cusp::blas::axpbypcz(cusp_Array_Displacement,cusp_Array_Velocity,cusp_Array_Acceleration,
		                 cusp_Array_Mass_rhs,
						 cusp_Array_NewMarkConstant[0],cusp_Array_NewMarkConstant[2],cusp_Array_NewMarkConstant[3]);

	 
	cusp::blas::axpbypcz(cusp_Array_Displacement,cusp_Array_Velocity,cusp_Array_Acceleration,
		                 cusp_Array_Damping_rhs,
		                 cusp_Array_NewMarkConstant[1],cusp_Array_NewMarkConstant[4],cusp_Array_NewMarkConstant[5]);

	
	cusp::multiply(MassMatrix, cusp_Array_Mass_rhs, cusp_Array_Old_Acceleration);

	
	cusp::multiply(DampingMatrix,cusp_Array_Damping_rhs,cusp_Array_Old_Displacement);

	cusp::blas::axpbypcz(cusp_Array_Rhs,cusp_Array_Old_Acceleration,cusp_Array_Old_Displacement,
		                 cusp_Array_R_rhs,
		                 ValueType(1),ValueType(1),ValueType(1));

	CuspVec tmpRhs(cusp_Array_Rhs);
	if ( nCount >= g_bcMinCount && nCount < g_bcMaxCount)
	{		
		thrust::device_ptr<ValueType> wrapped_rhs(rhsOnCuda);
		cusp::array1d_view<thrust::device_ptr<ValueType>> cusp_rhs (wrapped_rhs, wrapped_rhs + g_nDofs);
		CuspVec::view rhsView(tmpRhs.begin(),tmpRhs.end());
		cusp::copy(rhsView, cusp_rhs);
		
		cudaApplyExternalForce<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(nForceCount, cuda_forceCondition, rhsOnCuda,g_externalForceFactor);
		cusp::copy(cusp_rhs,rhsView);
	}

#if USE_CORATIONAL

	cusp::blas::axpbypcz(tmpRhs,cusp_Array_Old_Acceleration,cusp_Array_Old_Displacement,
		                 cusp_Array_R_rhs_tmp4Corotaion,
		                 ValueType(1),ValueType(1),ValueType(1));
	cudaDeviceSynchronize();
	cusp::blas::axpby(cusp_Array_R_rhs_tmp4Corotaion,cusp_Array_R_rhs_Corotaion,
					  cusp_Array_R_rhs,
					  ValueType(1),ValueType(1));
#else
	cusp::blas::axpbypcz(tmpRhs,cusp_Array_Old_Acceleration,cusp_Array_Old_Displacement,
		                 cusp_Array_R_rhs,
		                 ValueType(1),ValueType(1),ValueType(1));
#endif
}


 void update_u_v_a()
{
	const CuspVec& solu = cusp_Array_Incremental_displace;
    CuspVec& disp_vec = cusp_Array_Displacement;
    CuspVec& vel_vec = cusp_Array_Velocity;
    CuspVec& acc_vec = cusp_Array_Acceleration;
    CuspVec& old_acc = cusp_Array_Old_Acceleration;
    CuspVec& old_solu = cusp_Array_Old_Displacement;

    /*old_solu = disp_vec;*/
	cusp::copy(disp_vec,old_solu);
    /*disp_vec = solu;*/
	cusp::copy(solu,disp_vec);
    /*old_acc  = acc_vec;*/
	cusp::copy(acc_vec,old_acc);


    /*acc_vec *= (-1 * m_db_NewMarkConstant[3]);*/
	cusp::blas::scal(acc_vec,(-1 * cusp_Array_NewMarkConstant[3]) );
    /*acc_vec += (disp_vec * m_db_NewMarkConstant[0]); */
	cusp::blas::axpy(disp_vec,acc_vec,cusp_Array_NewMarkConstant[0]);
    /*acc_vec += (old_solu * (-1 * m_db_NewMarkConstant[0]));*/
	cusp::blas::axpy(old_solu,acc_vec,(-1*cusp_Array_NewMarkConstant[0]) );
    /*acc_vec += (vel_vec * (-1 * m_db_NewMarkConstant[2]));*/
	cusp::blas::axpy(vel_vec,acc_vec,(-1*cusp_Array_NewMarkConstant[2]) );

    /*vel_vec += (old_acc * m_db_NewMarkConstant[6]);*/
	cusp::blas::axpy(old_acc,vel_vec,cusp_Array_NewMarkConstant[6]);
    /*vel_vec += (acc_vec * m_db_NewMarkConstant[7]);*/
	cusp::blas::axpy(acc_vec,vel_vec,cusp_Array_NewMarkConstant[7]);

#if USE_CORATIONAL
	computeRotationMatrix();
#endif
}

 void setMatrixRowZero( const int  rowIdx )
{
	for (int idx = cusp_CsrMt_System.row_offsets[rowIdx];idx < cusp_CsrMt_System.row_offsets[rowIdx+1];++idx)
	{
		if (cusp_CsrMt_System.column_indices[idx] != rowIdx)
		{
			cusp_CsrMt_System.values[idx] = 0.;
		}
	}
}

 
__global__ void cuda_apply_boundary_values(IndexType nbcCount,IndexTypePtr param_boundaryCondition,ValueTypePtr param_diagnosticValue/*,ValueTypePtr para_boundaryConditionVal*/,ValueTypePtr para_Array_R_rhs,ValueTypePtr para_Array_Incremental_displace)
{
	//(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT
	int v = threadIdx.x + blockIdx.x * blockDim.x;
	while (v < nbcCount)
	{
		const int dof_number = param_boundaryCondition[v];
		;

		const  ValueType EPSINON = 0.000001;


		/*if( !(param_diagnosticValue[dof_number] < EPSINON  && param_diagnosticValue[dof_number] > -EPSINON) )
		{
			
		}*/
		para_Array_R_rhs[dof_number] = 0.f;//para_boundaryConditionVal[v] * param_diagnosticValue[dof_number];
		para_Array_Incremental_displace[dof_number] = 0.f;//para_boundaryConditionVal[v];

		v += blockDim.x * gridDim.x;
	}
}
 


void apply_boundary_values(int *dofsList,int nListCount,bool bInit)
{
	
	//const unsigned int n_dofs = g_nDofs;
	//return ;
	if (bInit)
	{
		
		for (int v=0;v < nBcCount;++v)
		{
			const int dof_number = cusp_boundaryCondition[v];

			SystemMatrix.setBoundaryCondition(dof_number);
		}
		return ;
	}
	else
	{
		//printf("thrust::device_ptr<ValueType> wrapped_rhs(rhsOnCuda); \n");
		thrust::device_ptr<ValueType> wrapped_rhs(rhsOnCuda);
		//printf("cusp::array1d_view<thrust::device_ptr<ValueType>> cusp_rhs (wrapped_rhs, wrapped_rhs + g_nDofs); \n");
		cusp::array1d_view<thrust::device_ptr<ValueType>> cusp_rhs (wrapped_rhs, wrapped_rhs + g_nDofs);
		//printf("CuspVec::view rhsView(cusp_Array_R_rhs.begin(),cusp_Array_R_rhs.end()); \n");
		CuspVec::view rhsView(cusp_Array_R_rhs.begin(),cusp_Array_R_rhs.end());
		//printf("cusp::copy(rhsView, cusp_rhs); \n");
		cusp::copy(rhsView, cusp_rhs);

		//printf("thrust::device_ptr<ValueType> wrapped_displacement(displacementOnCuda); \n");
		thrust::device_ptr<ValueType> wrapped_displacement(displacementOnCuda);

		//printf("cusp::array1d_view<thrust::device_ptr<ValueType>> cusp_displace (wrapped_displacement, wrapped_displacement + g_nDofs); \n");
		cusp::array1d_view<thrust::device_ptr<ValueType>> cusp_displace (wrapped_displacement, wrapped_displacement + g_nDofs);

		//printf("CuspVec::view displaceView(cusp_Array_Incremental_displace.begin(),cusp_Array_Incremental_displace.end()); \n");
		CuspVec::view displaceView(cusp_Array_Incremental_displace.begin(),cusp_Array_Incremental_displace.end());
		//printf("cusp::copy(displaceView, cusp_displace); \n");
		cusp::copy(displaceView, cusp_displace);
		//printf("cuda_apply_boundary_values<<<(KERNEL_COUNT + BLOCK_COUNT - 1 \n");
		cuda_apply_boundary_values<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(nBcCount,cuda_boundaryCondition,cuda_diagnosticValue/*,cuda_boundaryConditionVal*/,rhsOnCuda,displacementOnCuda);

		//if (nCount < 8)
		//{
		//	//printf("cuda_apply_boundary_values_displace<<<(KERNEL_COUNT + BLOCK_COUNT - 1) \n");
		//	cuda_apply_boundary_values_displace<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(nCount,nbcCountDisplace,cuda_boundaryconditionDisplacementPtr,
		//		cuda_diagnosticValue,cuda_boundaryconditionDisplacementValPtr,rhsOnCuda,displacementOnCuda);
		//}
		//printf("cusp::copy(cusp_rhs,rhsView); \n");
		cusp::copy(cusp_rhs,rhsView);
		//printf("cusp::copy(cusp_displace,displaceView); \n");
		cusp::copy(cusp_displace,displaceView);
	}
}

void initBoundaryCondition()
{
	apply_boundary_values(cusp_boundaryCondition,nBcCount,true);
	
}

__device__ bool checkLineTri(float3 L1,float3 L2,float3 PV1,float3 PV2,float3 PV3 )
{
	float3 VIntersect;
	float3 VNorm,tmpVec3;

	tmpVec3 = cross(make_float3(PV2.x - PV1.x,PV2.y - PV1.y,PV2.z - PV1.z),make_float3(PV3.x - PV1.x,PV3.y - PV1.y,PV3.z - PV1.z));
	VNorm = normalize(tmpVec3);
	float fDst1 = dot( make_float3(L1.x - PV1.x,L1.y - PV1.y,L1.z - PV1.z),VNorm );
	float fDst2 = dot( make_float3(L2.x - PV1.x,L2.y - PV1.y,L2.z - PV1.z),VNorm );
	if ( (fDst1 * fDst2) >= 0.0f) return false;  // line doesn't cross the triangle.
	if ( fDst1 == fDst2) {return false;} // line and plane are parallel
	// Find point on the line that intersects with the plane
	VIntersect = L1 + (L2-L1) * ( -fDst1/(fDst2-fDst1) );
	float3 VTest;
	VTest = cross(VNorm,PV2-PV1);
	if ( dot(VTest,VIntersect-PV1) < 0.0f ) return false;
	VTest = cross(VNorm,PV3-PV2);
	if ( dot(VTest,VIntersect-PV2) < 0.0f ) return false;
	VTest = cross(VNorm, PV1-PV3);
	if ( dot(VTest,VIntersect-PV1) < 0.0f ) return false;
	return true;
}

__device__ bool checkLineTriWithIntersect(float3 L1,float3 L2,float3 PV1,float3 PV2,float3 PV3,float* HitP )
{
	float3 VIntersect;
	float3 VNorm,tmpVec3;

	tmpVec3 = cross(make_float3(PV2.x - PV1.x,PV2.y - PV1.y,PV2.z - PV1.z),make_float3(PV3.x - PV1.x,PV3.y - PV1.y,PV3.z - PV1.z));
	VNorm = normalize(tmpVec3);
	float fDst1 = dot( make_float3(L1.x - PV1.x,L1.y - PV1.y,L1.z - PV1.z),VNorm );
	float fDst2 = dot( make_float3(L2.x - PV1.x,L2.y - PV1.y,L2.z - PV1.z),VNorm );
	if ( (fDst1 * fDst2) >= 0.0f) return false;  // line doesn't cross the triangle.
	if ( fDst1 == fDst2) {return false;} // line and plane are parallel
	// Find point on the line that intersects with the plane
	VIntersect = L1 + (L2-L1) * ( -fDst1/(fDst2-fDst1) );
	float3 VTest;
	VTest = cross(VNorm,PV2-PV1);
	if ( dot(VTest,VIntersect-PV1) < 0.0f ) return false;
	VTest = cross(VNorm,PV3-PV2);
	if ( dot(VTest,VIntersect-PV2) < 0.0f ) return false;
	VTest = cross(VNorm, PV1-PV3);
	if ( dot(VTest,VIntersect-PV1) < 0.0f ) return false;
	HitP[0] = VIntersect.x;
	HitP[1] = VIntersect.y;
	HitP[2] = VIntersect.z;
	return true;
}

__device__ float checkPointPlaneOnCuda(float destPointX, float destPointY, float destPointZ,
									   float bladeX, float bladeY, float bladeZ, 
									   float bladeNormalX, float bladeNormalY, float bladeNormalZ)
{
	return (bladeNormalX)*(bladeX-destPointX)+(bladeNormalY)*(bladeY-destPointY)+(bladeNormalZ)*(bladeZ-destPointZ);
}

__global__ void collisionDetection_onMain(int nCellCount,EFGCellOnCuda * cellOnCuda,float * LinesElement,int * CellCollisionFlag,
										  int nTimeStep,ValueTypePtr bladeElement,int nBladeBase,
										  IndexTypePtr topLevelCellInMain_OnCuda, VertexOnCuda* VertexOnCudaPtr, ValueTypePtr bladeNormal)
{
	int currentDof = threadIdx.x + blockIdx.x * blockDim.x;
	if (currentDof < nCellCount)
	{
		float3 PV1 = make_float3(bladeElement[nBladeBase+0],bladeElement[nBladeBase+1],bladeElement[nBladeBase+2]);
		float3 PV2 = make_float3(bladeElement[nBladeBase+3],bladeElement[nBladeBase+4],bladeElement[nBladeBase+5]);
		float3 PV3 = make_float3(bladeElement[nBladeBase+6],bladeElement[nBladeBase+7],bladeElement[nBladeBase+8]);
		float * lineBase = LinesElement + cellOnCuda[currentDof].m_nLinesBaseIdx;
		int curVertexId;
		//cellOnCuda[currentDof].m_nLinesCount;
		
		CellCollisionFlag[currentDof] = 0;
		topLevelCellInMain_OnCuda[currentDof] = 0;
		
		//CUPRINTF("cellOnCuda[%d].m_nLinesCount = %d ;cellOnCuda[%d].m_nLinesBaseIdx = %d\n",currentDof,cellOnCuda[currentDof].m_nLinesCount,currentDof,cellOnCuda[currentDof].m_nLinesBaseIdx);
		if (cellOnCuda[currentDof].m_bLeaf)
		{
			cellOnCuda[currentDof].m_bTopLevelOctreeNodeList = false;
			for (int v=0;v<cellOnCuda[currentDof].m_nLinesCount;++v)
			{
			
				if (true == checkLineTri( make_float3(lineBase[0],lineBase[1],lineBase[2]), make_float3(lineBase[3],lineBase[4],lineBase[5]),  PV1,PV2,PV3) )
				{
					//CUPRINTF("currentDof is %d collision \n%f %f %f %f %f %f\n",currentDof,lineBase[0],lineBase[1],lineBase[2],lineBase[3],lineBase[4],lineBase[5]);
					
					if (cellOnCuda[currentDof].m_nGhostCellCount>0)
					{
						cellOnCuda[currentDof].m_bLeaf=false;
						CellCollisionFlag[currentDof] = cellOnCuda[currentDof].m_nGhostCellCount;
						
					}
					else
					{
						topLevelCellInMain_OnCuda[currentDof] = 1;
						cellOnCuda[currentDof].m_bTopLevelOctreeNodeList = true;
						for (int p=0;p<8;++p)
						{
							curVertexId = cellOnCuda[currentDof].vertexId[p];
							cellOnCuda[currentDof].nPointPlane[p] = checkPointPlaneOnCuda(
								VertexOnCudaPtr[ curVertexId ].local[0]	, VertexOnCudaPtr[ curVertexId ].local[1]	, VertexOnCudaPtr[ curVertexId ].local[2],
								bladeElement[nBladeBase+0]				,bladeElement[nBladeBase+1]					,bladeElement[nBladeBase+2],
								bladeNormal[0]							,bladeNormal[1]								,bladeNormal[2]);
						}
					}
					break;
				}
				lineBase += 6;
			}
		}
		
	}
}

__global__ void ReInitCellOnCuda(int nCellCount,EFGCellOnCuda * cellOnCuda, VertexOnCuda* VertexOnCudaPtr, IndexTypePtr beClonedObjectFlag, IndexTypePtr beCloneVertexFlag )
{
	int currentCellIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (currentCellIdx < nCellCount && true == cellOnCuda[currentCellIdx].m_bLeaf)
	{
		/*int nCurId;
		int dofOrder[8] = {4,5,0,1,6,7,2,3};
		for (int v=0;v<8;++v)
		{
			int vv=dofOrder[v];
			nCurId = cellOnCuda[currentCellIdx].vertexId[vv];
			cellOnCuda[currentCellIdx].globalDofs[v*3+0] = VertexOnCudaPtr[nCurId].m_nDof[0];
			cellOnCuda[currentCellIdx].globalDofs[v*3+1] = VertexOnCudaPtr[nCurId].m_nDof[1];
			cellOnCuda[currentCellIdx].globalDofs[v*3+2] = VertexOnCudaPtr[nCurId].m_nDof[2];
			cellOnCuda[currentCellIdx].localDofs[v*3+0] = v*3+0;
			cellOnCuda[currentCellIdx].localDofs[v*3+1] = v*3+1;
			cellOnCuda[currentCellIdx].localDofs[v*3+2] = v*3+2;
		}*/
		if (cellOnCuda[currentCellIdx].m_bTopLevelOctreeNodeList)
		{
			beClonedObjectFlag[currentCellIdx] = 1;
			beCloneVertexFlag[ cellOnCuda[currentCellIdx].vertexId[0] ] = 1;
			beCloneVertexFlag[ cellOnCuda[currentCellIdx].vertexId[1] ] = 1;
			beCloneVertexFlag[ cellOnCuda[currentCellIdx].vertexId[2] ] = 1;
			beCloneVertexFlag[ cellOnCuda[currentCellIdx].vertexId[3] ] = 1;
			beCloneVertexFlag[ cellOnCuda[currentCellIdx].vertexId[4] ] = 1;
			beCloneVertexFlag[ cellOnCuda[currentCellIdx].vertexId[5] ] = 1;
			beCloneVertexFlag[ cellOnCuda[currentCellIdx].vertexId[6] ] = 1;
			beCloneVertexFlag[ cellOnCuda[currentCellIdx].vertexId[7] ] = 1;
		}
		else
		{
			beClonedObjectFlag[currentCellIdx] = 0;
		}
	}
}

__global__ void cloneVertexOnCuda(int nVertexOnCudaCount, VertexOnCuda* VertexOnCudaPtr, IndexTypePtr beCloneVertexFlag, int nDofBase)
{
	int currentVertexIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (currentVertexIdx < nVertexOnCudaCount && (beCloneVertexFlag[currentVertexIdx+1] - beCloneVertexFlag[currentVertexIdx])>0 )
	{
		int cloneVertexIdx = nVertexOnCudaCount + beCloneVertexFlag[currentVertexIdx];
		int cloneDofBase = nDofBase + beCloneVertexFlag[currentVertexIdx] * 3;

		VertexOnCudaPtr[cloneVertexIdx] = VertexOnCudaPtr[currentVertexIdx];
		VertexOnCudaPtr[cloneVertexIdx].local[0] = VertexOnCudaPtr[currentVertexIdx].local[0];
		VertexOnCudaPtr[cloneVertexIdx].local[1] = VertexOnCudaPtr[currentVertexIdx].local[1];
		VertexOnCudaPtr[cloneVertexIdx].local[2] = VertexOnCudaPtr[currentVertexIdx].local[2];
		VertexOnCudaPtr[cloneVertexIdx].m_createTimeStamp = 1;
		VertexOnCudaPtr[cloneVertexIdx].m_nDof[0] = cloneDofBase;
		VertexOnCudaPtr[cloneVertexIdx].m_nDof[1] = cloneDofBase+1;
		VertexOnCudaPtr[cloneVertexIdx].m_nDof[2] = cloneDofBase+2;
		VertexOnCudaPtr[cloneVertexIdx].m_nId = cloneVertexIdx;
		VertexOnCudaPtr[currentVertexIdx].m_nCloneId = cloneVertexIdx;
	}
}

__global__ void cloneCellOnCuda(int nCellOnCudaCount, EFGCellOnCuda * cellOnCuda, VertexOnCuda* VertexOnCudaPtr, IndexTypePtr beCloneCellFlag)
{
	int currentCellIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (currentCellIdx < nCellOnCudaCount && (beCloneCellFlag[currentCellIdx+1] - beCloneCellFlag[currentCellIdx])>0)
	{
		int cloneCellIdx = nCellOnCudaCount + beCloneCellFlag[currentCellIdx];

		cellOnCuda[cloneCellIdx] = cellOnCuda[currentCellIdx];
		cellOnCuda[cloneCellIdx].m_bLeaf = true;//cellOnCuda[currentCellIdx].m_bLeaf;
		cellOnCuda[currentCellIdx].m_nCloneCellIdx = cloneCellIdx;
		cellOnCuda[cloneCellIdx].m_nCloneCellIdx = -1;
#if 0
		cellOnCuda[cloneCellIdx].cellType = cellOnCuda[currentCellIdx].cellType;
		cellOnCuda[cloneCellIdx].m_bLeaf = true;//cellOnCuda[currentCellIdx].m_bLeaf;
		cellOnCuda[cloneCellIdx].m_nLinesCount = cellOnCuda[currentCellIdx].m_nLinesCount;
		cellOnCuda[cloneCellIdx].m_nLinesBaseIdx = cellOnCuda[currentCellIdx].m_nLinesBaseIdx;
		cellOnCuda[cloneCellIdx].m_bNewOctreeNodeList = cellOnCuda[currentCellIdx].m_bNewOctreeNodeList;
		cellOnCuda[cloneCellIdx].m_bTopLevelOctreeNodeList = cellOnCuda[currentCellIdx].m_bTopLevelOctreeNodeList;
		cellOnCuda[cloneCellIdx].m_nGhostCellCount = cellOnCuda[currentCellIdx].m_nGhostCellCount;
		cellOnCuda[cloneCellIdx].m_nGhostCellIdxInVec = cellOnCuda[currentCellIdx].m_nGhostCellIdxInVec;
		cellOnCuda[cloneCellIdx].m_nRhsIdx = cellOnCuda[currentCellIdx].m_nRhsIdx;
		cellOnCuda[cloneCellIdx].m_nMassMatrixIdx = cellOnCuda[currentCellIdx].m_nMassMatrixIdx;
		cellOnCuda[cloneCellIdx].m_nStiffnessMatrixIdx = cellOnCuda[currentCellIdx].m_nStiffnessMatrixIdx;
		cellOnCuda[cloneCellIdx].m_nLevel = cellOnCuda[currentCellIdx].m_nLevel;
		cellOnCuda[cloneCellIdx].m_nCellInfluncePointSize = cellOnCuda[currentCellIdx].m_nCellInfluncePointSize;
		cellOnCuda[cloneCellIdx].m_nJxW = cellOnCuda[currentCellIdx].m_nJxW;
		/************************************************************************/
		/* Common data                                                                     */
		/************************************************************************/

		float m_EFGGobalGaussPoint[8][3];//initialize on cpu
	
		int m_influncePointList[8][EFGInflunceMaxSize];
		int m_cellInflunceVertexList[EFGInflunceMaxSize];
		float  N_[8][EFGInflunceMaxSize];
		float  D1N_[8][EFGInflunceMaxSize][dim];
		float m_ShapeFunction_R[8];
		float m_ShapeDeriv_R[8][dim];
#endif	

		int nativeVertexId,cloneVertexId;

		for (int v=0;v<8;++v)
		{
			nativeVertexId = cellOnCuda[currentCellIdx].vertexId[v];
			cloneVertexId = VertexOnCudaPtr[ nativeVertexId ].m_nCloneId;

			
			if (cellOnCuda[currentCellIdx].nPointPlane[v] < 0)
			{
				cellOnCuda[currentCellIdx].vertexId[v] = nativeVertexId;
				cellOnCuda[cloneCellIdx].vertexId[v]   = cloneVertexId;

				cellOnCuda[currentCellIdx].m_cellInflunceVertexList[v] = nativeVertexId;
				cellOnCuda[cloneCellIdx].m_cellInflunceVertexList[v]   = cloneVertexId;
				
			} 
			else
			{
				cellOnCuda[currentCellIdx].vertexId[v] = cloneVertexId;
				cellOnCuda[cloneCellIdx].vertexId[v]   = nativeVertexId;

				cellOnCuda[currentCellIdx].m_cellInflunceVertexList[v] = cloneVertexId;
				cellOnCuda[cloneCellIdx].m_cellInflunceVertexList[v]   = nativeVertexId;
			}

			
		}
	}
}

void updateCudaStructDof(int newAddDof)
{
	//printf("updateCudaStructDof new dof is %d \n",newAddDof);
	g_nDofs = newAddDof;

	free(diagnosticValue);
	diagnosticValue = (ValueTypePtr)malloc( g_nDofs * sizeof(ValueType) );

	HANDLE_ERROR( cudaFree(cuda_diagnosticValue));
	HANDLE_ERROR( cudaMalloc( (void**)&cuda_diagnosticValue,g_nDofs * sizeof(ValueType))) ;

	HANDLE_ERROR( cudaFree(displacementOnCuda));
	HANDLE_ERROR( cudaMalloc( (void**)&displacementOnCuda,g_nDofs * sizeof(ValueType))) ;

	HANDLE_ERROR( cudaFree(rhsOnCuda));
	HANDLE_ERROR( cudaMalloc( (void**)&rhsOnCuda,g_nDofs * sizeof(ValueType))) ;
}

int nBladeBase = 0;
void makeVertex2BladeRelationShipOnCuda();
 void do_loop(int nCount,float4* pos_Lines, uchar4 *colorPos, float4* pos_Triangles, float3* vertexNormal)
{
	if ( false && (20 == nCount || 400000 == nCount) )
	{
		EFG_CellOnCudaElementCount_Last = EFG_CellOnCudaElementCount;
		printf("begin cutting\n");
		if (400000 == nCount) 
		{
			nBladeBase += 9;
			HANDLE_ERROR( cudaMemcpy( (void *)g_BladeNormalOnCuda	,&g_BladeNormalOnCuda[3]	,3 * sizeof(ValueType),cudaMemcpyDeviceToDevice ) );
		}

		int nLastTick = GetTickCount();
		int nCurrentTick;

		int * topLevelCellInMain_OnHost/*,*topLevelCellInGhost_OnHost*/;
		HANDLE_ERROR( cudaHostAlloc( (void**)&topLevelCellInMain_OnHost	,EFG_CellOnCudaElementCount * sizeof(IndexType),cudaHostAllocMapped)) ;
		//HANDLE_ERROR( cudaHostAlloc( (void**)&topLevelCellInGhost_OnHost	,EFG_CellOnCudaElementCount * sizeof(IndexType),cudaHostAllocMapped)) ;

		
		int * distributeDofFlag_OnHost,*distributeDofFlag_OnCuda;
		HANDLE_ERROR( cudaHostAlloc( (void**)&distributeDofFlag_OnHost	,g_nVertexOnCudaCount * sizeof(IndexType),cudaHostAllocMapped)) ;
		HANDLE_ERROR( cudaHostGetDevicePointer((void **)&distributeDofFlag_OnCuda,(void *)distributeDofFlag_OnHost,0));
		HANDLE_ERROR( cudaMemset( (void*)distributeDofFlag_OnCuda,0,g_nVertexOnCudaCount * sizeof(IndexType))) ;

		int * topLevelCellInMain_OnCuda/*,*topLevelCellInGhost_OnCuda*/;
		HANDLE_ERROR(cudaHostGetDevicePointer((void **)&topLevelCellInMain_OnCuda,(void *)topLevelCellInMain_OnHost,0));
		//HANDLE_ERROR(cudaHostGetDevicePointer((void **)&topLevelCellInGhost_OnCuda,(void *)topLevelCellInGhost_OnHost,0));

		
		IndexTypePtr g_CellCollisionFlag_onCuda/*,g_CellCudaCollisionFlag_onCuda*/;
		HANDLE_ERROR(cudaHostGetDevicePointer((void **)&g_CellCollisionFlag_onCuda,(void *)g_CellCollisionFlag,0));
		//HANDLE_ERROR(cudaHostGetDevicePointer((void **)&g_CellCudaCollisionFlag_onCuda,(void *)g_CellCudaCollisionFlag,0));
		//HANDLE_ERROR( cudaMemset( (void*)g_CellCudaCollisionFlag_onCuda,0,cellGhostOnCudaElementCount	* sizeof(IndexType))) ;
		HANDLE_ERROR( cudaMemset( (void*)g_CellCollisionFlag_onCuda,	0,EFG_CellOnCudaElementCount		* sizeof(IndexType))) ;
		cudaDeviceSynchronize();


		collisionDetection_onMain<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>
			(EFG_CellOnCudaElementCount,EFG_CellOnCudaPtr,
			 g_LinesElement,g_CellCollisionFlag_onCuda,nCount,
			 g_BladeElementOnCuda,nBladeBase,
			 topLevelCellInMain_OnCuda,g_VertexOnCudaPtr,g_BladeNormalOnCuda);

		cudaDeviceSynchronize();

		thrust::exclusive_scan(g_CellCollisionFlag		, g_CellCollisionFlag+EFG_CellOnCudaElementCount+1			, g_CellCollisionFlag); //generate ghost cell count
		int nNewCellInMain = g_CellCollisionFlag[EFG_CellOnCudaElementCount];

		

		//current nNewCellInMain is zero in steak;
#ifdef MY_DEBUG
		if (nNewCellInMain)
		{
			do{}while(true);
		}
#endif
		EFG_CellOnCudaElementCount += nNewCellInMain;

		int * beClonedObjectFlag_OnHost,*beClonedObjectFlag_OnCuda;
			HANDLE_ERROR( cudaHostAlloc( (void**)&beClonedObjectFlag_OnHost	,EFG_CellOnCudaElementCount * sizeof(IndexType),cudaHostAllocMapped)) ;
			HANDLE_ERROR( cudaHostGetDevicePointer((void **)&beClonedObjectFlag_OnCuda,(void *)beClonedObjectFlag_OnHost,0));

			int * beClonedVertexFlag_OnHost,*beClonedVertexFlag_OnCuda;
			HANDLE_ERROR( cudaHostAlloc( (void**)&beClonedVertexFlag_OnHost	,g_nVertexOnCudaCount * sizeof(IndexType),cudaHostAllocMapped)) ;
			HANDLE_ERROR( cudaHostGetDevicePointer((void **)&beClonedVertexFlag_OnCuda,(void *)beClonedVertexFlag_OnHost,0));
			HANDLE_ERROR( cudaMemset( (void*)beClonedVertexFlag_OnCuda,0,g_nVertexOnCudaCount * sizeof(IndexType))) ;

		ReInitCellOnCuda<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(EFG_CellOnCudaElementCount,EFG_CellOnCudaPtr,g_VertexOnCudaPtr, beClonedObjectFlag_OnCuda, beClonedVertexFlag_OnCuda);
		cudaDeviceSynchronize();

		thrust::exclusive_scan(beClonedObjectFlag_OnHost, beClonedObjectFlag_OnHost+EFG_CellOnCudaElementCount+1	, beClonedObjectFlag_OnHost); //generate ghost cell count

		thrust::exclusive_scan(beClonedVertexFlag_OnHost, beClonedVertexFlag_OnHost+g_nVertexOnCudaCount+1	, beClonedVertexFlag_OnHost); //generate ghost cell count

		nNewCellInMain = beClonedObjectFlag_OnHost[EFG_CellOnCudaElementCount];//be clone cell size
		int nNewVertexCount = beClonedVertexFlag_OnHost[g_nVertexOnCudaCount];//be clone vertex size
		printf("nNewCellInMain(%d) nNewVertexCount(%d)\n",nNewCellInMain,nNewVertexCount);
		int nNewDofCount = nNewVertexCount * 3;

		cloneVertexOnCuda<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(g_nVertexOnCudaCount, g_VertexOnCudaPtr, beClonedVertexFlag_OnCuda, g_nDofs);
		g_nVertexOnCudaCount += nNewVertexCount;
		g_nDofs += nNewDofCount;

		cloneCellOnCuda<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(EFG_CellOnCudaElementCount, EFG_CellOnCudaPtr, g_VertexOnCudaPtr, beClonedObjectFlag_OnCuda);
		EFG_CellOnCudaElementCount += nNewCellInMain;

		updateCudaStructDof(g_nDofs);
		//assembleSystemOnCuda_EFG_RealTime();	

		makeVertex2BladeRelationShipOnCuda();

		/*
		g_nLastVertexSize = g_nMCVertexSize;
		g_nLastEdgeSize = g_nMCEdgeSize;
		g_nLastSurfaceSize = g_nMCSurfaceSize;
		*/

		/*nCurrentTick = GetTickCount();
		printf("assembleSystemOnCuda %d \n",nCurrentTick - nLastTick);*/
		nLastTick = nCurrentTick;
	}
	//for (int v=0;v < nCount;++v)
	{
		assembleSystemOnCuda_EFG_RealTime();
        //printf("begin update_rhs \n");
        update_rhs(nCount);
		 
        //printf("begin set_boundary_condition \n");
        apply_boundary_values(cusp_boundaryCondition,nBcCount,false);

        //printf("begin solve_linear_problem \n");
		solve_cusp_cg_inner(pos_Lines,colorPos,pos_Triangles,vertexNormal);

        //printf("begin update_u_v_a \n");
        update_u_v_a();

		
	}
}

void solve_cusp_cg_inner(/*float4* pos, uchar4 *colorPos*/)
{
	cusp::verbose_monitor<ValueType> monitor(cusp_Array_R_rhs, 1000, 0.005/*1e-3*/);

	cusp::precond::diagonal<ValueType, MemorySpace> M(cusp_CsrMt_System);

    cusp::krylov::cg(cusp_CsrMt_System, cusp_Array_Incremental_displace, cusp_Array_R_rhs, monitor, M);


	thrust::device_ptr<ValueType> wrapped_displacement(displacementOnCuda);
	cusp::array1d_view<thrust::device_ptr<ValueType>> cusp_dX (wrapped_displacement, wrapped_displacement + g_nDofs);

	CuspVec::view displaceView(cusp_Array_Incremental_displace.begin(),cusp_Array_Incremental_displace.end());

    cusp::copy(displaceView, cusp_dX);
	//cudaMoveMesh<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(pos,colorPos,vbo_line_vertex_pair,vbo_lineCount,vbo_vertex_dofs,vbo_vertex_pos,vbo_vertexCount,displacementOnCuda,g_nDofs);
}


void intCellElementOnCuda_EFG(int nCount,EFGCellOnCuda * cellElementOnCpu,int linesCount,float * linesPtrOnCpu)
{
	EFG_CellOnCudaElementCount = nCount;
	HANDLE_ERROR( cudaMalloc( (void**)&EFG_CellOnCudaPtr,  nExternalMemory * EFG_CellOnCudaElementCount * sizeof(EFGCellOnCuda)	));
	HANDLE_ERROR( cudaMemcpy( (void *)EFG_CellOnCudaPtr,  cellElementOnCpu,  EFG_CellOnCudaElementCount * sizeof(EFGCellOnCuda), cudaMemcpyHostToDevice	));

	g_LinesElementCount = linesCount;
	HANDLE_ERROR( cudaMalloc( (void**)&g_LinesElement,  nExternalMemory * g_LinesElementCount * sizeof(float) )	) ;
	HANDLE_ERROR( cudaMemcpy( (void *)g_LinesElement,  linesPtrOnCpu,  g_LinesElementCount * sizeof(float),  cudaMemcpyHostToDevice ));

	HANDLE_ERROR( cudaHostAlloc( (void**)&g_CellCollisionFlag, nExternalMemory * EFG_CellOnCudaElementCount * sizeof(IndexType), cudaHostAllocMapped  )) ;

//#if USE_CORATIONAL
//	HANDLE_ERROR( cudaMalloc( (void**)&cuda_RKR	,nExternalMemory * EFG_CellOnCudaElementCount * Geometry_dofs_per_cell*Geometry_dofs_per_cell * sizeof(ValueType))) ;
//	HANDLE_ERROR( cudaMalloc( (void**)&cuda_RKRtPj	,nExternalMemory * EFG_CellOnCudaElementCount * Geometry_dofs_per_cell * sizeof(ValueType))) ;
//#endif
}
#if USE_CORATIONAL
#define MatrixRows (3)

__device__ float determinant(float* src_element)
{
	return  src_element[0*MatrixRows+0]*src_element[1*MatrixRows+1]*src_element[2*MatrixRows+2]+
			src_element[0*MatrixRows+1]*src_element[1*MatrixRows+2]*src_element[2*MatrixRows+0]+
			src_element[0*MatrixRows+2]*src_element[1*MatrixRows+0]*src_element[2*MatrixRows+1]-
			src_element[2*MatrixRows+0]*src_element[1*MatrixRows+1]*src_element[0*MatrixRows+2]-
			src_element[1*MatrixRows+0]*src_element[0*MatrixRows+1]*src_element[2*MatrixRows+2]-
			src_element[0*MatrixRows+0]*src_element[2*MatrixRows+1]*src_element[1*MatrixRows+2];
}

__global__ void update_u_v_a_init_corotaion(int nCellCount, EFGCellOnCuda * cellOnCudaPointer, VertexOnCuda* vecVertexOnCuda, int nDofCount, ValueTypePtr global_incremental_displacement,ValueTypePtr para_Signs_on_cuda)
{
	const int currentCellIdx = blockIdx.x;
	const int matrixIdx3x3 = threadIdx.x;/*0-8*/
	const int component_row = threadIdx.x / 3;
	const int component_col = threadIdx.x % 3;

	if(currentCellIdx < nCellCount  && true == cellOnCudaPointer[currentCellIdx].m_bLeaf)
	{
		if (component_row == component_col)
		{
			cellOnCudaPointer[currentCellIdx].RotationMatrix[matrixIdx3x3] = 1;
		}
		else
		{
			cellOnCudaPointer[currentCellIdx].RotationMatrix[matrixIdx3x3] = 0;
		}
		//return ;
		float weight = 0.25f * (1.f / cellOnCudaPointer[currentCellIdx].radiusx2);

		for (unsigned i=0;i<8;++i)
		{
			const int curDof = vecVertexOnCuda[cellOnCudaPointer[currentCellIdx].vertexId[i] ].m_nDof[component_row];
			float u = weight * global_incremental_displacement[curDof] * para_Signs_on_cuda[i*9+matrixIdx3x3];
			cellOnCudaPointer[currentCellIdx].RotationMatrix[matrixIdx3x3] += u;
		}
	}
}


__global__ void update_u_v_a_iterator_corotaion(int nCellCount, EFGCellOnCuda * cellOnCudaPointer)
{
	const int currentCellIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if(currentCellIdx < nCellCount  && true == cellOnCudaPointer[currentCellIdx].m_bLeaf)
	{
		const int inverseIdx[3*3][4] = {{1*MatrixRows+1,2*MatrixRows+2,2*MatrixRows+1,1*MatrixRows+2},
										{0*MatrixRows+1,2*MatrixRows+2,2*MatrixRows+1,0*MatrixRows+2},
										{0*MatrixRows+1,1*MatrixRows+2,1*MatrixRows+1,0*MatrixRows+2},
										{1*MatrixRows+2,2*MatrixRows+0,2*MatrixRows+2,1*MatrixRows+0},
										{0*MatrixRows+2,2*MatrixRows+0,0*MatrixRows+0,2*MatrixRows+2},
										{0*MatrixRows+2,1*MatrixRows+0,0*MatrixRows+0,1*MatrixRows+2},
										{1*MatrixRows+0,2*MatrixRows+1,1*MatrixRows+1,2*MatrixRows+0},
										{0*MatrixRows+0,2*MatrixRows+1,0*MatrixRows+1,2*MatrixRows+0},
										{0*MatrixRows+0,1*MatrixRows+1,1*MatrixRows+0,0*MatrixRows+1}};
		const float inverseFlag[3*3] = {1.f,-1.f,1.f,1.f,-1.f,1.f,1.f,-1.f,1.f};
		const int transposeIdx[3*3] = {0,3,6,1,4,7,2,5,8};

		float* R = &cellOnCudaPointer[currentCellIdx].RotationMatrix[0];
		float* RIT = &cellOnCudaPointer[currentCellIdx].RotationMatrix4Inverse[0];
		for (unsigned i0=1;i0<5;++i0)
		{
			const float det = 1.0f/determinant(R);
	
			for (unsigned i=0;i<(3*3);++i)
			{
				RIT[transposeIdx[i]] = (det) * inverseFlag[i] * (R[inverseIdx[i][0]]*R[inverseIdx[i][1]]-
														 R[inverseIdx[i][2]]*R[inverseIdx[i][3]]);
			}

			for (unsigned i=0;i<(3*3);++i)
			{
				R[i] = 0.5f*(R[i] + RIT[i]);
			}
		}
	}
}

__global__ void update_u_v_a_corotaion_compute_R_Rt(int nCellCount, EFGCellOnCuda * cellOnCudaPointer, ValueTypePtr localStiffnessMatrixPtr)
{
	const int currentCellIdx = blockIdx.x;
	const int localRowIdx = threadIdx.x;/*0-23*/
	const int localColIdx = threadIdx.y;/*0-23*/

	if(currentCellIdx < nCellCount && true == cellOnCudaPointer[currentCellIdx].m_bLeaf )
	{
		float* R_24x24 = &cellOnCudaPointer[currentCellIdx].R[0];
		float* Rt_24x24 = &cellOnCudaPointer[currentCellIdx].Rt[0];
		float* R_3x3  = &cellOnCudaPointer[currentCellIdx].RotationMatrix[0];
		float* K_24X24 = &localStiffnessMatrixPtr[cellOnCudaPointer[currentCellIdx].m_nStiffnessMatrixIdx * Geometry_dofs_per_cell_squarte];
		float* RK_24X24 = &cellOnCudaPointer[currentCellIdx].RxK[0];

		//1. make R for 24*24
		const int blockIdxRow = localRowIdx /3;
		const int blockIdxCol = localColIdx /3;
		const int blockInnerRow = localRowIdx % 3;
		const int blockInnerCol = localColIdx % 3;
		if (blockIdxRow == blockIdxCol)
		{
			R_24x24[localRowIdx*24+localColIdx] = R_3x3[blockInnerRow*3+blockInnerCol];
			Rt_24x24[localColIdx*24+localRowIdx] = R_3x3[blockInnerRow*3+blockInnerCol];
		}
		else
		{
			R_24x24[localRowIdx*24+localColIdx] = 0.f;
			Rt_24x24[localColIdx*24+localRowIdx] = 0.f;
		}
	}
}

__global__ void update_u_v_a_corotaion_compute_RK(int nCellCount, EFGCellOnCuda * cellOnCudaPointer, ValueTypePtr localStiffnessMatrixPtr)
{
	const int currentCellIdx = blockIdx.x;
	const int localRowIdx = threadIdx.x;/*0-23*/
	const int localColIdx = threadIdx.y;/*0-23*/

	if(currentCellIdx < nCellCount && true == cellOnCudaPointer[currentCellIdx].m_bLeaf )
	{
		float* R_24x24 = &cellOnCudaPointer[currentCellIdx].R[0];
		float* Rt_24x24 = &cellOnCudaPointer[currentCellIdx].Rt[0];
		float* R_3x3  = &cellOnCudaPointer[currentCellIdx].RotationMatrix[0];
		float* K_24X24 = &localStiffnessMatrixPtr[cellOnCudaPointer[currentCellIdx].m_nStiffnessMatrixIdx * Geometry_dofs_per_cell_squarte];
		float* RK_24X24 = &cellOnCudaPointer[currentCellIdx].RxK[0];

		//2. compute RxK
		float Cvalue = 0;

		for (int e = 0; e < Geometry_dofs_per_cell; ++e)
		{
			Cvalue += R_24x24[localRowIdx * Geometry_dofs_per_cell + e]	* K_24X24[e * Geometry_dofs_per_cell + localColIdx];
		}

		RK_24X24[localRowIdx * Geometry_dofs_per_cell + localColIdx] = Cvalue;
	}
}

__global__ void update_u_v_a_corotaion_compute_RtPj(int nCellCount, EFGCellOnCuda * cellOnCudaPointer)
{
	const int currentCellIdx = blockIdx.x;
	const int localRowIdx = threadIdx.x;/*0-23*/
	//const int localColIdx = threadIdx.y;/*0-23*/
	if(currentCellIdx < nCellCount  && true == cellOnCudaPointer[currentCellIdx].m_bLeaf)
	{
		float* Rt_24x24 = &cellOnCudaPointer[currentCellIdx].Rt[0];
		float* Pj_24 = &cellOnCudaPointer[currentCellIdx].Pj[0];
		float* Rhs_24 = &cellOnCudaPointer[currentCellIdx].CorotaionRhs[0];
		float Cvalue = 0;

		for (int e = 0; e < Geometry_dofs_per_cell; ++e)
		{
			Cvalue += Rt_24x24[localRowIdx * Geometry_dofs_per_cell + e]	* Pj_24[e];
		}

		Rhs_24[localRowIdx] =  Pj_24[localRowIdx] - Cvalue;
	}
}

__global__ void update_u_v_a_corotaion_compute_RKR_RPj(int nCellCount, EFGCellOnCuda * cellOnCudaPointer)
{
	const int currentCellIdx = blockIdx.x;
	const int localRowIdx = threadIdx.x;/*0-23*/
	const int localColIdx = threadIdx.y;/*0-24*/

	if(currentCellIdx < nCellCount  && true == cellOnCudaPointer[currentCellIdx].m_bLeaf)
	{
		if (localColIdx < Geometry_dofs_per_cell)
		{
			float* RK_24X24 = &cellOnCudaPointer[currentCellIdx].RxK[0];
			float* RKR_24X24 = &cellOnCudaPointer[currentCellIdx].RKR[0];
			float* Rt_24x24 = &cellOnCudaPointer[currentCellIdx].Rt[0];

			float Cvalue = 0;

			for (int e = 0; e < Geometry_dofs_per_cell; ++e)
			{
				Cvalue += RK_24X24[localRowIdx * Geometry_dofs_per_cell + e]	* Rt_24x24[e * Geometry_dofs_per_cell + localColIdx];
			}

			RKR_24X24[localRowIdx * Geometry_dofs_per_cell + localColIdx] = Cvalue;
		}
		else
		{
			float* RK_24X24 = &cellOnCudaPointer[currentCellIdx].RxK[0];
			float* RtPj = &cellOnCudaPointer[currentCellIdx].CorotaionRhs[0];
			float* RKRtPj = &cellOnCudaPointer[currentCellIdx].RKRtPj[0];

			float Cvalue = 0;

			for (int e = 0; e < Geometry_dofs_per_cell; ++e)
			{
				Cvalue += RK_24X24[localRowIdx * Geometry_dofs_per_cell + e]	* RtPj[e];
			}

			RKRtPj[localRowIdx] = Cvalue;
		}
	}
}

void computeRotationMatrix()
{
	//return ;
	update_u_v_a_init_corotaion<<< KERNEL_COUNT_TMP,3*3 >>>(EFG_CellOnCudaElementCount,EFG_CellOnCudaPtr, g_VertexOnCudaPtr,g_nDofs,displacementOnCuda,Signs_on_cuda);
	cudaDeviceSynchronize();
	update_u_v_a_iterator_corotaion<<<KERNEL_COUNT_TMP,128>>>(EFG_CellOnCudaElementCount,EFG_CellOnCudaPtr);
	cudaDeviceSynchronize();
	

	dim3 threads4RK(24,24);
	update_u_v_a_corotaion_compute_R_Rt<<<KERNEL_COUNT_TMP,threads4RK>>>(EFG_CellOnCudaElementCount,EFG_CellOnCudaPtr,localStiffnessMatrixOnCuda);
	update_u_v_a_corotaion_compute_RK<<<KERNEL_COUNT_TMP,threads4RK>>>(EFG_CellOnCudaElementCount,EFG_CellOnCudaPtr,localStiffnessMatrixOnCuda);
	cudaDeviceSynchronize();
	
	update_u_v_a_corotaion_compute_RtPj<<<KERNEL_COUNT_TMP,24>>>(EFG_CellOnCudaElementCount,EFG_CellOnCudaPtr);
	cudaDeviceSynchronize();
	//exit(66);
	dim3 threads4RKR(24,25);
	update_u_v_a_corotaion_compute_RKR_RPj<<<KERNEL_COUNT_TMP,threads4RKR>>>(EFG_CellOnCudaElementCount,EFG_CellOnCudaPtr);
	cudaDeviceSynchronize();
}

__global__ void assemble_matrix_free_on_cuda_4_Corotation(int nCellCount,int /*funcMatrixCount*/,int nDofCount,EFGCellOnCuda * cellOnCudaPointer,
	
	ValueType * localStiffnessMatrixPtr,ValueType * localMassMatrixPtr,ValueType * /*localRhsVectorPtr*/,
	IndexTypePtr systemOuterIdxPtr,ValueTypePtr systemRhsPtr,

	IndexTypePtr g_globalDof,ValueTypePtr g_globalValue,
	IndexTypePtr globalDof_Mass,ValueTypePtr globalValue_Mass,
	IndexTypePtr globalDof_Damping,ValueTypePtr globalValue_Damping,
	IndexTypePtr globalDof_System,ValueTypePtr globalValue_System,
	ValueTypePtr tmp_blockShareRhs/*,IndexTypePtr livedCellFlag*/,ValueTypePtr tmp_blockShareRhs_compare,
	ValueTypePtr para_RKRtPj
	)
{
	const int currentCellIdx = blockIdx.x;
	const int localDofIdx = threadIdx.x;/*0-23*/
	const int localColIdx = threadIdx.y;/*0-23*/
	
	int *local_globalDof;
	float *value;

	int *local_globalDof_mass;
	float *value_mass;

	int *local_globalDof_damping;
	float *value_damping;

	int *local_globalDof_system;
	float *value_system;

	float rhsValue = 0.0f;
	float *blockShareRhs,*blockShareRhs_compare;

	
	
	if(currentCellIdx < nCellCount && true == cellOnCudaPointer[currentCellIdx].m_bLeaf )
	{
		/*livedCellFlag[currentCellIdx] = 1;*/
		const int currentDof = cellOnCudaPointer[currentCellIdx].globalDofs[localDofIdx];
		const int nStep = currentDof * nMaxNonZeroSize;
		local_globalDof = g_globalDof + nStep;
		value = g_globalValue + nStep;

		local_globalDof_mass = globalDof_Mass + nStep;
		value_mass = globalValue_Mass + nStep;

		local_globalDof_damping = globalDof_Damping + nStep;
		value_damping = globalValue_Damping + nStep;

		local_globalDof_system = globalDof_System + nStep;
		value_system = globalValue_System + nStep;

		blockShareRhs = tmp_blockShareRhs + currentDof*8;
		blockShareRhs_compare = tmp_blockShareRhs_compare + currentDof*8;

		const int global_row = currentDof;
		const int loc_row = cellOnCudaPointer[currentCellIdx].localDofs[localDofIdx];
		const int global_col = cellOnCudaPointer[currentCellIdx].globalDofs[localColIdx];
		const int loc_col = cellOnCudaPointer[currentCellIdx].localDofs[localColIdx];//localColIdx;
		const int idx_in_8 = loc_row / 3;

		const float col_val_mass =  localMassMatrixPtr[cellOnCudaPointer[currentCellIdx].m_nMassMatrixIdx * Geometry_dofs_per_cell_squarte + loc_row * Geometry_dofs_per_cell + loc_col] * MASS_MATRIX_COEFF_2;
#if 1
		const float col_val_stiffness = cellOnCudaPointer[currentCellIdx].RKR[loc_row * Geometry_dofs_per_cell + loc_col];//cellOnCudaPointer[currentCellIdx]localStiffnessMatrixPtr[cellOnCudaPointer[currentCellIdx].m_nStiffnessMatrixIdx * Geometry_dofs_per_cell_squarte + loc_row * Geometry_dofs_per_cell + loc_col];
		//const float col_val_stiffness_compare = localStiffnessMatrixPtr[currentCellIdx*Geometry_dofs_per_cell_squarte+loc_row * Geometry_dofs_per_cell + loc_col];
		//CUPRINTF("blockShareStiff[%d][%d][%f]--blockShareStiff_compare[%d][%d][%f]\n",global_row,global_col,col_val_stiffness,global_row,global_col,col_val_stiffness_compare);
#else
		const float col_val_stiffness = localStiffnessMatrixPtr[currentCellIdx*Geometry_dofs_per_cell_squarte+loc_row * Geometry_dofs_per_cell + loc_col];
#endif
		const int index = idx_in_8 * LocalMaxDofCount_YC + loc_col;
		local_globalDof_mass[index] = global_col;
		value_mass[index] = col_val_mass;
		local_globalDof[index] = global_col;
		value[index] = col_val_stiffness;
		local_globalDof_damping[index] = global_col;
		value_damping[index] = 0.183f * col_val_mass + 0.00128f * col_val_stiffness;
		local_globalDof_system[index] = global_col;
		value_system[index] = 16384 * value_mass[index] + 128 * value_damping[index] + value[index];

		if (0 == localColIdx)
		{	
#if 1
			blockShareRhs[idx_in_8] = cellOnCudaPointer[currentCellIdx].RKRtPj[loc_row];
			//blockShareRhs_compare[idx_in_8] = para_RKRtPj[currentCellIdx*Geometry_dofs_per_cell + loc_row];
			
			//CUPRINTF("blockShareRhs[%d][%f]--blockShareRhs_compare[%d][%f]\n",idx_in_8,blockShareRhs[idx_in_8],idx_in_8,blockShareRhs_compare[idx_in_8]);
#else
			blockShareRhs[idx_in_8] = para_RKRtPj[currentCellIdx*Geometry_dofs_per_cell + loc_row];
			/*if (0 == currentCellIdx)
			{
				CUPRINTF("RKRtPj[%d]=%f\n",loc_row,blockShareRhs[idx_in_8]);
			}*/
#endif
		}
	}
	
	return ;
}

#endif

void initVertexOnCuda(int nCount,VertexOnCuda * VertexOnCudaPtr)
{
	g_nVertexOnCudaCount = nCount;
	HANDLE_ERROR( cudaMalloc( (void**)&g_VertexOnCudaPtr, nExternalMemory * g_nVertexOnCudaCount * sizeof(VertexOnCuda))) ;
	HANDLE_ERROR( cudaMemset( (void*)g_VertexOnCudaPtr,0, nExternalMemory * g_nVertexOnCudaCount * sizeof(VertexOnCuda))) ;
	HANDLE_ERROR( cudaMemcpy( (void*)g_VertexOnCudaPtr	, VertexOnCudaPtr, g_nVertexOnCudaCount * sizeof(VertexOnCuda), cudaMemcpyHostToDevice));
}


void initLinePair()
{
	//int linePair[12][2] = {{0,2},{4,6},{0,4},{2,6},{1,3},{5,7},{1,5},{3,7},{0,1},{4,5},{2,3},{6,7}};
	IndexType linePair[24] = {0,2,4,6,0,4,2,6,1,3,5,7,1,5,3,7,0,1,4,5,2,3,6,7};
	HANDLE_ERROR( cudaMalloc( (void**)&g_linePairOnCuda,24 * sizeof(IndexType))) ;
	HANDLE_ERROR( cudaMemcpy( (void *)g_linePairOnCuda	,&linePair[0]	,24 * sizeof(IndexType),cudaMemcpyHostToDevice ) );
}

int deviceQuery() 
{
    printf("CUDA Device Query (Runtime API) version (CUDART static linking)\n");

    int deviceCount;

    cudaGetDeviceCount(&deviceCount);

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
        printf("There is no device supporting CUDA\n");
    int dev;
    for (dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0) {
			// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                printf("There is no device supporting CUDA.\n");
            else if (deviceCount == 1)
                printf("There is 1 device supporting CUDA\n");
            else
                printf("There are %d devices supporting CUDA\n", deviceCount);
        }
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
    #if CUDART_VERSION >= 2020
		int driverVersion = 0, runtimeVersion = 0;
		cudaDriverGetVersion(&driverVersion);
		printf("  CUDA Driver Version:                           %d.%d\n", driverVersion/1000, driverVersion%100);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, runtimeVersion%100);
    #endif

        printf("  CUDA Capability Major revision number:         %d\n", deviceProp.major);
        printf("  CUDA Capability Minor revision number:         %d\n", deviceProp.minor);

		printf("  Total amount of global memory:                 %u bytes\n", deviceProp.totalGlobalMem);
    #if CUDART_VERSION >= 2000
        printf("  Number of multiprocessors:                     %d\n", deviceProp.multiProcessorCount);
        printf("  Number of cores:                               %d\n", 8 * deviceProp.multiProcessorCount);
    #endif
        printf("  Total amount of constant memory:               %u bytes\n", deviceProp.totalConstMem); 
        printf("  Total amount of shared memory per block:       %u bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %u bytes\n", deviceProp.memPitch);
        printf("  Texture alignment:                             %u bytes\n", deviceProp.textureAlignment);
        printf("  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
    #if CUDART_VERSION >= 2000
        printf("  Concurrent copy and execution:                 %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
    #endif
    #if CUDART_VERSION >= 2020
        printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated:                                    %s\n", deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Compute mode:                                  %s\n", deviceProp.computeMode == cudaComputeModeDefault ?
			                                                            "Default (multiple host threads can use this device simultaneously)" :
		                                                                deviceProp.computeMode == cudaComputeModeExclusive ?
																		"Exclusive (only one host thread at a time can use this device)" :
		                                                                deviceProp.computeMode == cudaComputeModeProhibited ?
																		"Prohibited (no host thread can use this device)" :
																		"Unknown");
    #endif
}
    printf("\nTest PASSED\n");

    //CUT_EXIT(argc, argv);
	return 0;
}


void initial_Cuda(const IndexType nDofs,
				  EigenValueType dbNewMarkConstant[8],
				  IndexTypePtr boundaryconditionPtr,
				  IndexType bcCount,
				  IndexTypePtr boundaryconditionDisplacementPtr,
				  IndexType bcCountDisplace
				  )
{
#if 0
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0); 
		if (!prop.canMapHostMemory ) 
		{
			printf("prop.canMapHostMemory can't support. \n");
			exit(0); 
		}
		if (prop.maxThreadsPerBlock < BLOCK_COUNT || prop.maxThreadsPerBlock < BLOCK_COUNT_TMP)
		{
			printf("prop.maxThreadsPerBlock can't support. \n");
			exit(0); 
		}

		deviceQuery();
		///*if (prop.maxGridSize < ((KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT) )
		//{gpu高性能编程cuda实战
		//	printf("prop.maxThreadsPerBlock can't support. \n");
		//	exit(0); 
		//}*/
		 cudaSetDeviceFlags(cudaDeviceMapHost); 
	}
#endif
	g_nDofs = nDofs;
	nBcCount = bcCount;

	cusp_Array_Incremental_displace.resize(nDofs,0);
	printf("9  nBcCount = %d  nBcCount * sizeof(IndexType) is %d\n",nBcCount,nBcCount * sizeof(IndexType));	
	
	cusp_boundaryCondition = (IndexTypePtr)malloc( nBcCount * sizeof(IndexType) );
	diagnosticValue = (ValueTypePtr)malloc( nDofs * sizeof(ValueType) );

	HANDLE_ERROR( cudaMalloc( (void**)&displacementOnCuda,nDofs * sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMalloc( (void**)&rhsOnCuda,nDofs * sizeof(ValueType))) ;

	HANDLE_ERROR( cudaMalloc( (void**)&cuda_boundaryCondition,nBcCount * sizeof(IndexType))) ;
	HANDLE_ERROR( cudaMalloc( (void**)&cuda_diagnosticValue,nDofs * sizeof(ValueType))) ;

	nForceCount = bcCountDisplace;
	HANDLE_ERROR( cudaMalloc( (void**)&cuda_forceCondition,nForceCount * sizeof(IndexType))) ;
	
	memcpy(cusp_boundaryCondition,boundaryconditionPtr,nBcCount * sizeof(IndexType));

	HANDLE_ERROR( cudaMemcpy( (void *)cuda_boundaryCondition,boundaryconditionPtr,nBcCount * sizeof(IndexType),cudaMemcpyHostToDevice ) );

	HANDLE_ERROR( cudaMemcpy( (void *)cuda_forceCondition,boundaryconditionDisplacementPtr,nForceCount * sizeof(IndexType),cudaMemcpyHostToDevice ) );

	printf("5* \n");

	cusp_Array_Incremental_displace.resize(nDofs,0);
	printf("5 1 \n");
	cusp_Array_Displacement.resize(nDofs,0);
	cusp_Array_R_rhs.resize(nDofs,0);
	cusp_Array_Mass_rhs.resize(nDofs,0);
	cusp_Array_Damping_rhs.resize(nDofs,0);
	cusp_Array_Velocity.resize(nDofs,0);
	cusp_Array_Acceleration.resize(nDofs,0);

#if USE_CORATIONAL
	cusp_Array_R_rhs_tmp4Corotaion.resize(nDofs,0);
	cusp_Array_R_rhs_Corotaion.resize(nDofs,0);
	HANDLE_ERROR( cudaMalloc( (void**)&Signs_on_cuda	,8*9 * sizeof(ValueType)));
	HANDLE_ERROR( cudaMemcpy( (void *)Signs_on_cuda		, Signs_on_cpu,8*9 * sizeof(ValueType)	,cudaMemcpyHostToDevice ));
#endif
	//cusp_Array_Rhs.resize(nDofs,0);
	cusp_Array_Old_Acceleration.resize(nDofs,0);
	cusp_Array_Old_Displacement.resize(nDofs,0);
	printf("5 1 \n");
	cusp_Array_NewMarkConstant.resize(8);
	for (int v=0;v < 8;++v)
	{
		cusp_Array_NewMarkConstant[v] = (ValueType)dbNewMarkConstant[v];
	}
	printf("5 2 \n");


	HANDLE_ERROR( cudaMalloc( (void**)&g_globalDof_MF	,nExternalMemory *(g_nDofs)*nMaxNonZeroSize * sizeof(IndexType))) ;
	HANDLE_ERROR( cudaMemset((void *)g_globalDof_MF,  0, nExternalMemory *(g_nDofs)*nMaxNonZeroSize * sizeof(IndexType)));

	HANDLE_ERROR( cudaMalloc( (void**)&g_globalValue_MF	,nExternalMemory *(g_nDofs)*nMaxNonZeroSize * sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMemset((void *)g_globalValue_MF,  0, nExternalMemory *(g_nDofs)*nMaxNonZeroSize * sizeof(ValueType)));

	HANDLE_ERROR( cudaMalloc( (void**)&g_globalDof_Mass_MF	,nExternalMemory *(g_nDofs)*nMaxNonZeroSize * sizeof(IndexType))) ;
	HANDLE_ERROR( cudaMemset((void *)g_globalDof_Mass_MF,  0, nExternalMemory *(g_nDofs)*nMaxNonZeroSize * sizeof(IndexType)));

	HANDLE_ERROR( cudaMalloc( (void**)&g_globalValue_Mass_MF	,nExternalMemory *(g_nDofs)*nMaxNonZeroSize * sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMemset((void *)g_globalValue_Mass_MF,  0, nExternalMemory *(g_nDofs)*nMaxNonZeroSize * sizeof(ValueType)));

	HANDLE_ERROR( cudaMalloc( (void**)&g_globalDof_Damping_MF	,nExternalMemory *(g_nDofs)*nMaxNonZeroSize * sizeof(IndexType))) ;
	HANDLE_ERROR( cudaMemset((void *)g_globalDof_Damping_MF,  0, nExternalMemory *(g_nDofs)*nMaxNonZeroSize * sizeof(IndexType)));

	HANDLE_ERROR( cudaMalloc( (void**)&g_globalValue_Damping_MF	,nExternalMemory *(g_nDofs)*nMaxNonZeroSize * sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMemset((void *)g_globalValue_Damping_MF,  0, nExternalMemory *(g_nDofs)*nMaxNonZeroSize * sizeof(ValueType)));

	HANDLE_ERROR( cudaMalloc( (void**)&g_globalDof_System_MF	,nExternalMemory *(g_nDofs)*nMaxNonZeroSize * sizeof(IndexType))) ;
	HANDLE_ERROR( cudaMemset((void *)g_globalDof_System_MF,  0, nExternalMemory *(g_nDofs)*nMaxNonZeroSize * sizeof(IndexType)));

	HANDLE_ERROR( cudaMalloc( (void**)&g_globalValue_System_MF	,nExternalMemory *(g_nDofs)*nMaxNonZeroSize * sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMemset((void *)g_globalValue_System_MF,  0, nExternalMemory *(g_nDofs)*nMaxNonZeroSize * sizeof(ValueType)));
#if 0
	HANDLE_ERROR( cudaMalloc( (void**)&g_globalNonZeroValue_System_MF	,MATRIX_NONZERO * sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMemset((void *)g_globalNonZeroValue_System_MF,  0, MATRIX_NONZERO * sizeof(ValueType)));

	HANDLE_ERROR( cudaMalloc( (void**)&g_globalNonZeroValue_Mass_MF	,MATRIX_NONZERO * sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMemset((void *)g_globalNonZeroValue_Mass_MF,  0, MATRIX_NONZERO * sizeof(ValueType)));

	HANDLE_ERROR( cudaMalloc( (void**)&g_globalNonZeroValue_Damping_MF	,MATRIX_NONZERO * sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMemset((void *)g_globalNonZeroValue_Damping_MF,  0, MATRIX_NONZERO * sizeof(ValueType)));	

	HANDLE_ERROR( cudaMalloc( (void**)&g_globalInnerIdx_MF	,MATRIX_NONZERO * sizeof(IndexType))) ;
	HANDLE_ERROR( cudaMemset((void *)g_globalInnerIdx_MF,  0, MATRIX_NONZERO * sizeof(IndexType)));	
	
	printf("5 3 \n");

	HANDLE_ERROR( cudaMalloc( (void**)&g_systemOuterIdxPtr_MF	,nExternalMemory *(g_nDofs + 1) * sizeof(IndexType))) ;
	HANDLE_ERROR( cudaMemset((void *)g_systemOuterIdxPtr_MF,  0, nExternalMemory *(g_nDofs + 1) * sizeof(IndexType))); 

	HANDLE_ERROR( cudaHostAlloc( (void**)&g_systemOuterIdxPtrOnHost_MF	,nExternalMemory *(g_nDofs + 1) * sizeof(IndexType),cudaHostAllocMapped)) ;
#endif
	HANDLE_ERROR( cudaMalloc( (void**)&g_systemRhsPtr_MF	,nExternalMemory *(g_nDofs) * sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMemset((void *)g_systemRhsPtr_MF,  0, nExternalMemory *(g_nDofs) * sizeof(ValueType)));

	HANDLE_ERROR( cudaMalloc( (void**)&g_systemRhsPtr_In8_MF	,nExternalMemory *(g_nDofs) * VertxMaxInflunceCellCount * sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMemset((void *)g_systemRhsPtr_In8_MF,  0, nExternalMemory *(g_nDofs) * VertxMaxInflunceCellCount * sizeof(ValueType)));
	//
	printf("5*5 \n");
}

__device__ float my_abs(float val)
{
	return (val > 0 ? val : -1*val);
}


__device__ float my_sign(float val)
{
	return (val > 0 ? 1.f : -1.f);
}


__device__ void WeightFun(float pCenterPt[],float Radius_, float pEvPoint[], float Weight[])
{
	float difx = ( pEvPoint[0] - pCenterPt[0] );
	float dify = ( pEvPoint[1] - pCenterPt[1] );
	float difz = ( pEvPoint[2] - pCenterPt[2] );

	float drdx = my_sign(difx)/Radius_;
	float drdy = my_sign(dify)/Radius_;
	float drdz = my_sign(difz)/Radius_;

	float rx = my_abs(difx)/Radius_;
	float ry = my_abs(dify)/Radius_;
	float rz = my_abs(difz)/Radius_;

	float wx,wy,wz,dwx,dwy,dwz,ddwx,ddwy,ddwz;

	wx = 1 - 6*rx*rx + 8*rx*rx*rx - 3*rx*rx*rx*rx;
	wy = 1 - 6*ry*ry + 8*ry*ry*ry - 3*ry*ry*ry*ry;
	wz = 1 - 6*rz*rz + 8*rz*rz*rz - 3*rz*rz*rz*rz;

	
	/*if( DerOrder == 0 )
		return wx*wy*wz;*/
	Weight[0] = wx*wy*wz;

	dwx = ( -12*rx+24*rx*rx-12*rx*rx*rx ) * drdx;
	dwy = ( -12*ry+24*ry*ry-12*ry*ry*ry ) * drdy;
	dwz = ( -12*rz+24*rz*rz-12*rz*rz*rz ) * drdz;

	/*if( DerOrder == 1 )
		return wy*dwx*wz;*/
	Weight[1] = wy*dwx*wz;

	/*if( DerOrder == 2 )
		return wx*dwy*wz;*/
	Weight[2] = wx*dwy*wz;

	/*if( DerOrder == 3 )
		return wx*wy*dwz;*/
	Weight[3] = wx*wy*dwz;

	ddwx = ( -12+48*rx-36*rx*rx )*drdx*drdx;
	ddwy = ( -12+48*ry-36*ry*ry )*drdy*drdy;
	ddwz = ( -12+48*rz-36*rz*rz )*drdz*drdz;

	/*if( DerOrder == 4 )
		return ddwx*wy*wz;*/
	Weight[4] = ddwx*wy*wz;
	/*if( DerOrder == 5 )
		return wx*ddwy*wz;*/
	Weight[5] = wx*ddwy*wz;
	/*if( DerOrder == 6 )
		return wx*wy*ddwz;*/
	Weight[6] = wx*wy*ddwz;
	/*if( DerOrder == 7 )
		return dwx*dwy*wz;*/
	Weight[7] = dwx*dwy*wz;
	/*if( DerOrder == 8 )
		return dwx*wy*dwz;*/
	Weight[8] = dwx*wy*dwz;
	/*if( DerOrder == 9 )
		return wx*dwy*dwz;*/
	Weight[9] = wx*dwy*dwz;
	/*else
		return 0.0;*/
}

#if 0
__device__ bool InvertMatrix4(const float m_0,const float m_1,const float m_2,const float m_3,const float m_4,const float m_5,const float m_6,const float m_7,const float m_8,const float m_9,const float m_10,const float m_11,const float m_12,const float m_13,const float m_14,const float m_15, float invOut[16])
{
	float inv[16], det;
	int i;

	inv[0] = m_5  * m_10 * m_15 - 
		m_5  * m_11 * m_14 - 
		m_9  * m_6  * m_15 + 
		m_9  * m_7  * m_14 +
		m_13 * m_6  * m_11 - 
		m_13 * m_7  * m_10;

	inv[4] = -m_4  * m_10 * m_15 + 
		m_4  * m_11 * m_14 + 
		m_8  * m_6  * m_15 - 
		m_8  * m_7  * m_14 - 
		m_12 * m_6  * m_11 + 
		m_12 * m_7  * m_10;

	inv[8] = m_4  * m_9 * m_15 - 
		m_4  * m_11 * m_13 - 
		m_8  * m_5 * m_15 + 
		m_8  * m_7 * m_13 + 
		m_12 * m_5 * m_11 - 
		m_12 * m_7 * m_9;

	inv[12] = -m_4  * m_9 * m_14 + 
		m_4  * m_10 * m_13 +
		m_8  * m_5 * m_14 - 
		m_8  * m_6 * m_13 - 
		m_12 * m_5 * m_10 + 
		m_12 * m_6 * m_9;

	inv[1] = -m_1  * m_10 * m_15 + 
		m_1  * m_11 * m_14 + 
		m_9  * m_2 * m_15 - 
		m_9  * m_3 * m_14 - 
		m_13 * m_2 * m_11 + 
		m_13 * m_3 * m_10;

	inv[5] = m_0  * m_10 * m_15 - 
		m_0  * m_11 * m_14 - 
		m_8  * m_2 * m_15 + 
		m_8  * m_3 * m_14 + 
		m_12 * m_2 * m_11 - 
		m_12 * m_3 * m_10;

	inv[9] = -m_0  * m_9 * m_15 + 
		m_0  * m_11 * m_13 + 
		m_8  * m_1 * m_15 - 
		m_8  * m_3 * m_13 - 
		m_12 * m_1 * m_11 + 
		m_12 * m_3 * m_9;

	inv[13] = m_0  * m_9 * m_14 - 
		m_0  * m_10 * m_13 - 
		m_8  * m_1 * m_14 + 
		m_8  * m_2 * m_13 + 
		m_12 * m_1 * m_10 - 
		m_12 * m_2 * m_9;

	inv[2] = m_1  * m_6 * m_15 - 
		m_1  * m_7 * m_14 - 
		m_5  * m_2 * m_15 + 
		m_5  * m_3 * m_14 + 
		m_13 * m_2 * m_7 - 
		m_13 * m_3 * m_6;

	inv[6] = -m_0  * m_6 * m_15 + 
		m_0  * m_7 * m_14 + 
		m_4  * m_2 * m_15 - 
		m_4  * m_3 * m_14 - 
		m_12 * m_2 * m_7 + 
		m_12 * m_3 * m_6;

	inv[10] = m_0  * m_5 * m_15 - 
		m_0  * m_7 * m_13 - 
		m_4  * m_1 * m_15 + 
		m_4  * m_3 * m_13 + 
		m_12 * m_1 * m_7 - 
		m_12 * m_3 * m_5;

	inv[14] = -m_0  * m_5 * m_14 + 
		m_0  * m_6 * m_13 + 
		m_4  * m_1 * m_14 - 
		m_4  * m_2 * m_13 - 
		m_12 * m_1 * m_6 + 
		m_12 * m_2 * m_5;

	inv[3] = -m_1 * m_6 * m_11 + 
		m_1 * m_7 * m_10 + 
		m_5 * m_2 * m_11 - 
		m_5 * m_3 * m_10 - 
		m_9 * m_2 * m_7 + 
		m_9 * m_3 * m_6;

	inv[7] = m_0 * m_6 * m_11 - 
		m_0 * m_7 * m_10 - 
		m_4 * m_2 * m_11 + 
		m_4 * m_3 * m_10 + 
		m_8 * m_2 * m_7 - 
		m_8 * m_3 * m_6;

	inv[11] = -m_0 * m_5 * m_11 + 
		m_0 * m_7 * m_9 + 
		m_4 * m_1 * m_11 - 
		m_4 * m_3 * m_9 - 
		m_8 * m_1 * m_7 + 
		m_8 * m_3 * m_5;

	inv[15] = m_0 * m_5 * m_10 - 
		m_0 * m_6 * m_9 - 
		m_4 * m_1 * m_10 + 
		m_4 * m_2 * m_9 + 
		m_8 * m_1 * m_6 - 
		m_8 * m_2 * m_5;

	det = m_0 * inv[0] + m_1 * inv[4] + m_2 * inv[8] + m_3 * inv[12];

	if (det == 0)
		return false;

	det = 1.0 / det;

	for (i = 0; i < 16; i++)
		invOut[i] = inv[i] * det;

	return true;
}
#else
__device__ bool InvertMatrix4(const int currentCellIdx ,const int gaussIdx,float *mm, float* invOut)
{
	double inv[16], det,m[16];
	int i;

	m[0] = (double)mm[0];	m[1] = (double)mm[1];	m[2] = (double)mm[2];	m[3] = (double)mm[3];
	m[4] = (double)mm[4];	m[5] = (double)mm[5];	m[6] = (double)mm[6];	m[7] = (double)mm[7];
	m[8] = (double)mm[8];	m[9] = (double)mm[9];	m[10] = (double)mm[10];	m[11] = (double)mm[11];
	m[12] = (double)mm[12];	m[13] = (double)mm[13];	m[14] = (double)mm[14];	m[15] = (double)mm[15];

	inv[0] = m[5]  * m[10] * m[15] - 
		m[5]  * m[11] * m[14] - 
		m[9]  * m[6]  * m[15] + 
		m[9]  * m[7]  * m[14] +
		m[13] * m[6]  * m[11] - 
		m[13] * m[7]  * m[10];

	inv[4] = -m[4]  * m[10] * m[15] + 
		m[4]  * m[11] * m[14] + 
		m[8]  * m[6]  * m[15] - 
		m[8]  * m[7]  * m[14] - 
		m[12] * m[6]  * m[11] + 
		m[12] * m[7]  * m[10];

	inv[8] = m[4]  * m[9] * m[15] - 
		m[4]  * m[11] * m[13] - 
		m[8]  * m[5] * m[15] + 
		m[8]  * m[7] * m[13] + 
		m[12] * m[5] * m[11] - 
		m[12] * m[7] * m[9];

	inv[12] = -m[4]  * m[9] * m[14] + 
		m[4]  * m[10] * m[13] +
		m[8]  * m[5] * m[14] - 
		m[8]  * m[6] * m[13] - 
		m[12] * m[5] * m[10] + 
		m[12] * m[6] * m[9];

	inv[1] = -m[1]  * m[10] * m[15] + 
		m[1]  * m[11] * m[14] + 
		m[9]  * m[2] * m[15] - 
		m[9]  * m[3] * m[14] - 
		m[13] * m[2] * m[11] + 
		m[13] * m[3] * m[10];

	inv[5] = m[0]  * m[10] * m[15] - 
		m[0]  * m[11] * m[14] - 
		m[8]  * m[2] * m[15] + 
		m[8]  * m[3] * m[14] + 
		m[12] * m[2] * m[11] - 
		m[12] * m[3] * m[10];

	inv[9] = -m[0]  * m[9] * m[15] + 
		m[0]  * m[11] * m[13] + 
		m[8]  * m[1] * m[15] - 
		m[8]  * m[3] * m[13] - 
		m[12] * m[1] * m[11] + 
		m[12] * m[3] * m[9];

	inv[13] = m[0]  * m[9] * m[14] - 
		m[0]  * m[10] * m[13] - 
		m[8]  * m[1] * m[14] + 
		m[8]  * m[2] * m[13] + 
		m[12] * m[1] * m[10] - 
		m[12] * m[2] * m[9];

	inv[2] = m[1]  * m[6] * m[15] - 
		m[1]  * m[7] * m[14] - 
		m[5]  * m[2] * m[15] + 
		m[5]  * m[3] * m[14] + 
		m[13] * m[2] * m[7] - 
		m[13] * m[3] * m[6];

	inv[6] = -m[0]  * m[6] * m[15] + 
		m[0]  * m[7] * m[14] + 
		m[4]  * m[2] * m[15] - 
		m[4]  * m[3] * m[14] - 
		m[12] * m[2] * m[7] + 
		m[12] * m[3] * m[6];

	inv[10] = m[0]  * m[5] * m[15] - 
		m[0]  * m[7] * m[13] - 
		m[4]  * m[1] * m[15] + 
		m[4]  * m[3] * m[13] + 
		m[12] * m[1] * m[7] - 
		m[12] * m[3] * m[5];

	inv[14] = -m[0]  * m[5] * m[14] + 
		m[0]  * m[6] * m[13] + 
		m[4]  * m[1] * m[14] - 
		m[4]  * m[2] * m[13] - 
		m[12] * m[1] * m[6] + 
		m[12] * m[2] * m[5];

	inv[3] = -m[1] * m[6] * m[11] + 
		m[1] * m[7] * m[10] + 
		m[5] * m[2] * m[11] - 
		m[5] * m[3] * m[10] - 
		m[9] * m[2] * m[7] + 
		m[9] * m[3] * m[6];

	inv[7] = m[0] * m[6] * m[11] - 
		m[0] * m[7] * m[10] - 
		m[4] * m[2] * m[11] + 
		m[4] * m[3] * m[10] + 
		m[8] * m[2] * m[7] - 
		m[8] * m[3] * m[6];

	inv[11] = -m[0] * m[5] * m[11] + 
		m[0] * m[7] * m[9] + 
		m[4] * m[1] * m[11] - 
		m[4] * m[3] * m[9] - 
		m[8] * m[1] * m[7] + 
		m[8] * m[3] * m[5];

	inv[15] = m[0] * m[5] * m[10] - 
		m[0] * m[6] * m[9] - 
		m[4] * m[1] * m[10] + 
		m[4] * m[2] * m[9] + 
		m[8] * m[1] * m[6] - 
		m[8] * m[2] * m[5];

	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

	

	if (det == 0)
		return false;

	det = 1.0 / det;

	/*if (0==currentCellIdx && 0 == gaussIdx)
	{
		CUPRINTF("{%f,%f,%f,%f}\n",inv[0],inv[1],inv[2],inv[3]);
		CUPRINTF("{%f,%f,%f,%f}\n",inv[4],inv[5],inv[6],inv[7]);
		CUPRINTF("{%f,%f,%f,%f}\n",inv[8],inv[9],inv[10],inv[11]);
		CUPRINTF("{%f,%f,%f,%f}\n",inv[12],inv[13],inv[14],inv[15]);
		CUPRINTF("det %f\n",det);
	}*/

	invOut[0] = (float)inv[0] * det;	invOut[1] = (float)inv[1] * det;	invOut[2] = (float)inv[2] * det;	invOut[3] = (float)inv[3] * det;
	invOut[4] = (float)inv[4] * det;	invOut[5] = (float)inv[5] * det;	invOut[6] = (float)inv[6] * det;	invOut[7] = (float)inv[7] * det;
	invOut[8] = (float)inv[8] * det;	invOut[9] = (float)inv[9] * det;	invOut[10] = (float)inv[10] * det;	invOut[11] = (float)inv[11] * det;
	invOut[12] = (float)inv[12] * det;	invOut[13] = (float)inv[13] * det;	invOut[14] = (float)inv[14] * det;	invOut[15] = (float)inv[15] * det;

	/*for (i = 0; i < 16; i++)
		invOut[i] = (float)inv[i] * det;*/

	return true;
}
#endif
__global__ void makeInfluncePointListCuda_ForCell_nonRealTime(const int nCellSize, EFGCellOnCuda * CellOnCudaPtrtt, EFGCellOnCuda_Ext * CellOnCuda_Ext_Ptr, const int nVertexSize, VertexOnCuda* VertexOnCudaPtr, const float SupportSize, const int validDomainId)
{
	int currentCellIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int tmpVertexPtr[EFGInflunceMaxSize];
	if (currentCellIdx < nCellSize)
	{
		EFGCellOnCuda& curCell = CellOnCudaPtrtt[currentCellIdx];
		curCell.m_nCellInfluncePointSize = 0;

		for (int v=0;v<nVertexSize;++v)
		{
			if (validDomainId == VertexOnCudaPtr[v].m_fromDomainId)
			{
				const float* supportPtsPos = &VertexOnCudaPtr[v].local[0];
				bool bFlag = true;
				
				for (unsigned ls=0;ls < 8;++ls)
				{
					const float gaussX = curCell.m_EFGGobalGaussPoint[ls][0];
					const float gaussY = curCell.m_EFGGobalGaussPoint[ls][1];
					const float gaussZ = curCell.m_EFGGobalGaussPoint[ls][2];

					if( my_abs( gaussX - supportPtsPos[0]  ) < SupportSize && 
						my_abs( gaussY - supportPtsPos[1]  ) < SupportSize && 
						my_abs( gaussZ - supportPtsPos[2]  ) < SupportSize)
					{
						bFlag = false;
						tmpVertexPtr[ls] = v;
					}
					else
					{
						tmpVertexPtr[ls] = InValidIdx;
						//CUPRINTF("tmpVertexPtr[ls] = InValidIdx\n");
					}
				}

				if (!bFlag)
				{
					curCell.m_influncePointList[0][curCell.m_nCellInfluncePointSize] = tmpVertexPtr[0];
					curCell.m_influncePointList[1][curCell.m_nCellInfluncePointSize] = tmpVertexPtr[1];
					curCell.m_influncePointList[2][curCell.m_nCellInfluncePointSize] = tmpVertexPtr[2];
					curCell.m_influncePointList[3][curCell.m_nCellInfluncePointSize] = tmpVertexPtr[3];
					curCell.m_influncePointList[4][curCell.m_nCellInfluncePointSize] = tmpVertexPtr[4];
					curCell.m_influncePointList[5][curCell.m_nCellInfluncePointSize] = tmpVertexPtr[5];
					curCell.m_influncePointList[6][curCell.m_nCellInfluncePointSize] = tmpVertexPtr[6];
					curCell.m_influncePointList[7][curCell.m_nCellInfluncePointSize] = tmpVertexPtr[7];
					curCell.m_cellInflunceVertexList[curCell.m_nCellInfluncePointSize] = v;
					curCell.m_nCellInfluncePointSize += 1;

					

#ifdef MY_DEBUG
					if (curCell.m_nCellInfluncePointSize > EFGInflunceMaxSize)
					{
						CUPRINTF("curCell.m_nCellInfluncePointSize > EFGInflunceMaxSize\n");
						MY_PAUSE;
					}
#endif

				}
				
			}
		}
	}
}

__global__ void makeInfluncePointListCuda_ForCell_Radius2_RealTime/*<<<nCellSize,GaussPointSize(8) * InfluncePointSize(8)>>>*/(const int nCellSize, EFGCellOnCuda * CellOnCudaPtrtt, EFGCellOnCuda_Ext * CellOnCuda_Ext_Ptr, const int nVertexSize, VertexOnCuda* VertexOnCudaPtr, const float SupportSize, const int validDomainId)
{
	const int currentCellIdx = blockIdx.x;
	const int gaussIdx = threadIdx.x;/*0-7*/
	const int influnceIdx = threadIdx.y;/*0-7*/

	
	if (currentCellIdx < nCellSize)
	{
		//CUPRINTF("printf test\n");
		EFGCellOnCuda& curCell = CellOnCudaPtrtt[currentCellIdx];
		

		curCell.m_influncePointList[gaussIdx][influnceIdx] = curCell.vertexId[influnceIdx];

		if (0 == gaussIdx )
		{
			if (0 == influnceIdx)
			{
				curCell.m_nCellInfluncePointSize = EFGInflunceMaxSize;
			}			
			curCell.m_cellInflunceVertexList[influnceIdx] = curCell.vertexId[influnceIdx];
		}
	}
}

__global__ void computeShapeFunctionOnCuda_RealTime(const int nCellSize, EFGCellOnCuda * CellOnCudaPtrtt, EFGCellOnCuda_Ext * CellOnCuda_Ext_Ptrtt,FEMShapeValue* FEMShapeValueOnCudaPtr , const int nVertexSize, VertexOnCuda* VertexOnCudaPtr, const float SupportSize, const int validDomainId)
{
	const int currentCellIdx = blockIdx.x;
	const int gaussIdx = threadIdx.x;/*0-7*/
	if (currentCellIdx < nCellSize && CellTypeFEM != CellOnCudaPtrtt[currentCellIdx].cellType)
	{
		EFGCellOnCuda& curCell = CellOnCudaPtrtt[currentCellIdx];
		FEMShapeValue& curFEMShapeValue = FEMShapeValueOnCudaPtr[curCell.m_nStiffnessMatrixIdx];
		EFGCellOnCuda_Ext& curCellExt = CellOnCuda_Ext_Ptrtt[currentCellIdx];
		const int refInflunceSize = curCell.m_nCellInfluncePointSize;

#ifdef MY_DEBUG
					if (refInflunceSize > EFGInflunceMaxSize)
					{
						CUPRINTF("refInflunceSize > EFGInflunceMaxSize\n");
						MY_PAUSE;
					}
#endif

		float WI[10];
		float pX[3] = {curCell.m_EFGGobalGaussPoint[gaussIdx][0],curCell.m_EFGGobalGaussPoint[gaussIdx][1],curCell.m_EFGGobalGaussPoint[gaussIdx][2]};
		float r[4],tmpD1P_[4];

		float* P_ = &curCellExt.P_[gaussIdx][0];
		float *PPt = &curCellExt.lltOfA[gaussIdx][0][0];
		float *A_ = &curCellExt.A_[gaussIdx][0][0];
		float *Ax_ = &curCellExt.Ax_[gaussIdx][0][0];
		float *Ay_ = &curCellExt.Ay_[gaussIdx][0][0];
		float *Az_ = &curCellExt.Az_[gaussIdx][0][0];
		float *B_ = &curCellExt.B_[gaussIdx][0][0];
		float *Bx_ = &curCellExt.Bx_[gaussIdx][0][0];
		float *By_ = &curCellExt.By_[gaussIdx][0][0];
		float *Bz_ = &curCellExt.Bz_[gaussIdx][0][0];
		float *D1P_ = &curCellExt.D1P_[gaussIdx][0][0];
		float* lltOfA = &curCellExt.lltOfA[gaussIdx][0][0];

		for (int i=0,v;i<refInflunceSize;++i)
		{
			v = curCell.m_influncePointList[gaussIdx][i];
			if (InValidIdx == v) continue;

			P_[0] = 1.f;
			P_[1] = VertexOnCudaPtr[v].local[0];
			P_[2] = VertexOnCudaPtr[v].local[1];
			P_[3] = VertexOnCudaPtr[v].local[2];		
			float pts_j[3] = {P_[1],P_[2],P_[3]};
					
			WeightFun(pts_j,SupportSize,pX,WI);
			const float cof = ( i == 0 ? 0.f:1.f );

			//float *PPt = &CellOnCudaPtr[currentCellIdx].lltOfA[gaussIdx][0][0];
			PPt[0] = 1.f		;PPt[1] = P_[1]			;PPt[2] = P_[2]			;PPt[3] = P_[3];
			PPt[4] = P_[1]		;PPt[5] = P_[1]*P_[1]	;PPt[6] = P_[1]*P_[2]	;PPt[7] = P_[1]*P_[3];
			PPt[8] = P_[2]		;PPt[9] = P_[2]*P_[1]	;PPt[10] = P_[2]*P_[2]	;PPt[11] = P_[2]*P_[3];
			PPt[12] = P_[3]		;PPt[13] = P_[3]*P_[1]	;PPt[14] = P_[2]*P_[3]	;PPt[15] = P_[3]*P_[3];

			//float *A_ = &CellOnCudaPtr[currentCellIdx].A_[gaussIdx][0][0];
			A_[0] = PPt[0] * WI[0] + cof * A_[0];A_[1] = PPt[1] * WI[0] + cof * A_[1];A_[2] = PPt[2] * WI[0] + cof * A_[2];A_[3] = PPt[3] * WI[0] + cof * A_[3];A_[4] = PPt[4] * WI[0] + cof * A_[4];A_[5] = PPt[5] * WI[0] + cof * A_[5];A_[6] = PPt[6] * WI[0] + cof * A_[6];A_[7] = PPt[7] * WI[0] + cof * A_[7];A_[8] = PPt[8] * WI[0] + cof * A_[8];A_[9] = PPt[9] * WI[0] + cof * A_[9];A_[10] = PPt[10] * WI[0] + cof * A_[10];A_[11] = PPt[11] * WI[0] + cof * A_[11];A_[12] = PPt[12] * WI[0] + cof * A_[12];A_[13] = PPt[13] * WI[0] + cof * A_[13];A_[14] = PPt[14] * WI[0] + cof * A_[14];A_[15] = PPt[15] * WI[0] + cof * A_[15];

			//float *Ax_ = &CellOnCudaPtr[currentCellIdx].Ax_[gaussIdx][0][0];
			Ax_[0] = PPt[0] * WI[1] + cof * Ax_[0];Ax_[1] = PPt[1] * WI[1] + cof * Ax_[1];Ax_[2] = PPt[2] * WI[1] + cof * Ax_[2];Ax_[3] = PPt[3] * WI[1] + cof * Ax_[3];Ax_[4] = PPt[4] * WI[1] + cof * Ax_[4];Ax_[5] = PPt[5] * WI[1] + cof * Ax_[5];Ax_[6] = PPt[6] * WI[1] + cof * Ax_[6];Ax_[7] = PPt[7] * WI[1] + cof * Ax_[7];Ax_[8] = PPt[8] * WI[1] + cof * Ax_[8];Ax_[9] = PPt[9] * WI[1] + cof * Ax_[9];Ax_[10] = PPt[10] * WI[1] + cof * Ax_[10];Ax_[11] = PPt[11] * WI[1] + cof * Ax_[11];Ax_[12] = PPt[12] * WI[1] + cof * Ax_[12];Ax_[13] = PPt[13] * WI[1] + cof * Ax_[13];Ax_[14] = PPt[14] * WI[1] + cof * Ax_[14];Ax_[15] = PPt[15] * WI[1] + cof * Ax_[15];

			//float *Ay_ = &CellOnCudaPtr[currentCellIdx].Ay_[gaussIdx][0][0];
			Ay_[0] = PPt[0] * WI[2] + cof * Ay_[0];Ay_[1] = PPt[1] * WI[2] + cof * Ay_[1];Ay_[2] = PPt[2] * WI[2] + cof * Ay_[2];Ay_[3] = PPt[3] * WI[2] + cof * Ay_[3];Ay_[4] = PPt[4] * WI[2] + cof * Ay_[4];Ay_[5] = PPt[5] * WI[2] + cof * Ay_[5];Ay_[6] = PPt[6] * WI[2] + cof * Ay_[6];Ay_[7] = PPt[7] * WI[2] + cof * Ay_[7];Ay_[8] = PPt[8] * WI[2] + cof * Ay_[8];Ay_[9] = PPt[9] * WI[2] + cof * Ay_[9];Ay_[10] = PPt[10] * WI[2] + cof * Ay_[10];Ay_[11] = PPt[11] * WI[2] + cof * Ay_[11];Ay_[12] = PPt[12] * WI[2] + cof * Ay_[12];Ay_[13] = PPt[13] * WI[2] + cof * Ay_[13];Ay_[14] = PPt[14] * WI[2] + cof * Ay_[14];Ay_[15] = PPt[15] * WI[2] + cof * Ay_[15];

			//float *Az_ = &CellOnCudaPtr[currentCellIdx].Az_[gaussIdx][0][0];
			Az_[0] = PPt[0] * WI[3] + cof * Az_[0];Az_[1] = PPt[1] * WI[3] + cof * Az_[1];Az_[2] = PPt[2] * WI[3] + cof * Az_[2];Az_[3] = PPt[3] * WI[3] + cof * Az_[3];Az_[4] = PPt[4] * WI[3] + cof * Az_[4];Az_[5] = PPt[5] * WI[3] + cof * Az_[5];Az_[6] = PPt[6] * WI[3] + cof * Az_[6];Az_[7] = PPt[7] * WI[3] + cof * Az_[7];Az_[8] = PPt[8] * WI[3] + cof * Az_[8];Az_[9] = PPt[9] * WI[3] + cof * Az_[9];Az_[10] = PPt[10] * WI[3] + cof * Az_[10];Az_[11] = PPt[11] * WI[3] + cof * Az_[11];Az_[12] = PPt[12] * WI[3] + cof * Az_[12];Az_[13] = PPt[13] * WI[3] + cof * Az_[13];Az_[14] = PPt[14] * WI[3] + cof * Az_[14];Az_[15] = PPt[15] * WI[3] + cof * Az_[15];

			//float *B_ = &CellOnCudaPtr[currentCellIdx].B_[gaussIdx][0][0];
			B_[EFGInflunceMaxSize_0+i] = P_[0] * WI[0];B_[EFGInflunceMaxSize_1+i] = P_[1] * WI[0];B_[EFGInflunceMaxSize_2+i] = P_[2] * WI[0];B_[EFGInflunceMaxSize_3+i] = P_[3] * WI[0];

			//float *Bx_ = &CellOnCudaPtr[currentCellIdx].Bx_[gaussIdx][0][0];
			Bx_[EFGInflunceMaxSize_0+i] = P_[0] * WI[1];Bx_[EFGInflunceMaxSize_1+i] = P_[1] * WI[1];Bx_[EFGInflunceMaxSize_2+i] = P_[2] * WI[1];Bx_[EFGInflunceMaxSize_3+i] = P_[3] * WI[1];

			//float *By_ = &CellOnCudaPtr[currentCellIdx].By_[gaussIdx][0][0];
			By_[EFGInflunceMaxSize_0+i] = P_[0] * WI[2];By_[EFGInflunceMaxSize_1+i] = P_[1] * WI[2];By_[EFGInflunceMaxSize_2+i] = P_[2] * WI[2];By_[EFGInflunceMaxSize_3+i] = P_[3] * WI[2];

			//float *Bz_ = &CellOnCudaPtr[currentCellIdx].Bz_[gaussIdx][0][0];
			Bz_[EFGInflunceMaxSize_0+i] = P_[0] * WI[3];Bz_[EFGInflunceMaxSize_1+i] = P_[1] * WI[3];Bz_[EFGInflunceMaxSize_2+i] = P_[2] * WI[3];Bz_[EFGInflunceMaxSize_3+i] = P_[3] * WI[3];
		}

		InvertMatrix4(currentCellIdx,gaussIdx,A_,lltOfA);
		

		P_[0] = 1.f;
		P_[1] = pX[0];
		P_[2] = pX[1];
		P_[3] = pX[2];

		r[0] = lltOfA[0] * P_[0] + lltOfA[1] * P_[1] + lltOfA[2] * P_[2] + lltOfA[3] * P_[3];
		r[1] = lltOfA[4] * P_[0] + lltOfA[5] * P_[1] + lltOfA[6] * P_[2] + lltOfA[7] * P_[3];
		r[2] = lltOfA[8] * P_[0] + lltOfA[9] * P_[1] + lltOfA[10] * P_[2] + lltOfA[11] * P_[3];
		r[3] = lltOfA[12] * P_[0] + lltOfA[13] * P_[1] + lltOfA[14] * P_[2] + lltOfA[15] * P_[3];

		float *N_ = &curCell.N_[gaussIdx][0];
		for (int i=0;i<refInflunceSize;++i)
		{
			N_[i] = r[0] * B_[EFGInflunceMaxSize_0 + i] + r[1] * B_[EFGInflunceMaxSize_1 + i] + r[2] * B_[EFGInflunceMaxSize_2 + i] + r[3] * B_[EFGInflunceMaxSize_3 + i];
		}

		D1P_[0] = -1.f * (Ax_[0] * r[0] + Ax_[1] * r[1] + Ax_[2] * r[2] + Ax_[3] * r[3]);
		D1P_[3] = 1.0f - (Ax_[4] * r[0] + Ax_[5] * r[1] + Ax_[6] * r[2] + Ax_[7] * r[3]);
		D1P_[6] = -1.f * (Ax_[8] * r[0] + Ax_[9] * r[1] + Ax_[10] * r[2] + Ax_[11] * r[3]);
		D1P_[9] = -1.f * (Ax_[12] * r[0] + Ax_[13] * r[1] + Ax_[14] * r[2] + Ax_[15] * r[3]);

		D1P_[1] = -1.f * (Ay_[0] * r[0] + Ay_[1] * r[1] + Ay_[2] * r[2] + Ay_[3] * r[3]);
		D1P_[4] = -1.f * (Ay_[4] * r[0] + Ay_[5] * r[1] + Ay_[6] * r[2] + Ay_[7] * r[3]);
		D1P_[7] = 1.0f - (Ay_[8] * r[0] + Ay_[9] * r[1] + Ay_[10] * r[2] + Ay_[11] * r[3]);
		D1P_[10] = -1.f * (Ay_[12] * r[0] + Ay_[13] * r[1] + Ay_[14] * r[2] + Ay_[15] * r[3]);

		D1P_[2] = -1.f * (Az_[0] * r[0] + Az_[1] * r[1] + Az_[2] * r[2] + Az_[3] * r[3]);
		D1P_[5] = -1.f * (Az_[4] * r[0] + Az_[5] * r[1] + Az_[6] * r[2] + Az_[7] * r[3]);
		D1P_[8] = -1.f * (Az_[8] * r[0] + Az_[9] * r[1] + Az_[10] * r[2] + Az_[11] * r[3]);
		D1P_[11] = 1.0f - (Az_[12] * r[0] + Az_[13] * r[1] + Az_[14] * r[2] + Az_[15] * r[3]);

		tmpD1P_[0] = lltOfA[0] * D1P_[0] + lltOfA[1] * D1P_[3] + lltOfA[2] * D1P_[6] + lltOfA[3] * D1P_[9];
		tmpD1P_[1] = lltOfA[4] * D1P_[0] + lltOfA[5] * D1P_[3] + lltOfA[6] * D1P_[6] + lltOfA[7] * D1P_[9];
		tmpD1P_[2] = lltOfA[8] * D1P_[0] + lltOfA[9] * D1P_[3] + lltOfA[10] * D1P_[6] + lltOfA[11] * D1P_[9];
		tmpD1P_[3] = lltOfA[12] * D1P_[0] + lltOfA[13] * D1P_[3] + lltOfA[14] * D1P_[6] + lltOfA[15] * D1P_[9];
		D1P_[0] = tmpD1P_[0];D1P_[3] = tmpD1P_[1];D1P_[6] = tmpD1P_[2];D1P_[9] = tmpD1P_[3];

		tmpD1P_[0] = lltOfA[0] * D1P_[1] + lltOfA[1] * D1P_[4] + lltOfA[2] * D1P_[7] + lltOfA[3] * D1P_[10];
		tmpD1P_[1] = lltOfA[4] * D1P_[1] + lltOfA[5] * D1P_[4] + lltOfA[6] * D1P_[7] + lltOfA[7] * D1P_[10];
		tmpD1P_[2] = lltOfA[8] * D1P_[1] + lltOfA[9] * D1P_[4] + lltOfA[10] * D1P_[7] + lltOfA[11] * D1P_[10];
		tmpD1P_[3] = lltOfA[12] * D1P_[1] + lltOfA[13] * D1P_[4] + lltOfA[14] * D1P_[7] + lltOfA[15] * D1P_[10];
		D1P_[1] = tmpD1P_[0];D1P_[4] = tmpD1P_[1];D1P_[7] = tmpD1P_[2];D1P_[10] = tmpD1P_[3];

		tmpD1P_[0] = lltOfA[0] * D1P_[2] + lltOfA[1] * D1P_[5] + lltOfA[2] * D1P_[8] + lltOfA[3] * D1P_[11];
		tmpD1P_[1] = lltOfA[4] * D1P_[2] + lltOfA[5] * D1P_[5] + lltOfA[6] * D1P_[8] + lltOfA[7] * D1P_[11];
		tmpD1P_[2] = lltOfA[8] * D1P_[2] + lltOfA[9] * D1P_[5] + lltOfA[10] * D1P_[8] + lltOfA[11] * D1P_[11];
		tmpD1P_[3] = lltOfA[12] * D1P_[2] + lltOfA[13] * D1P_[5] + lltOfA[14] * D1P_[8] + lltOfA[15] * D1P_[11];
		D1P_[2] = tmpD1P_[0];D1P_[5] = tmpD1P_[1];D1P_[8] = tmpD1P_[2];D1P_[11] = tmpD1P_[3];


		float *D1N_ = &curCell.D1N_[gaussIdx][0][0];//EFGInflunceMaxSize*dim
		for (int npt=0;npt < refInflunceSize;++npt)
		{
			D1N_[npt*3+0] = B_[EFGInflunceMaxSize_0 + npt] * D1P_[0] + B_[EFGInflunceMaxSize_1 + npt] * D1P_[3] + B_[EFGInflunceMaxSize_2 + npt] * D1P_[6] + B_[EFGInflunceMaxSize_3 + npt] * D1P_[9];
			D1N_[npt*3+1] = B_[EFGInflunceMaxSize_0 + npt] * D1P_[1] + B_[EFGInflunceMaxSize_1 + npt] * D1P_[4] + B_[EFGInflunceMaxSize_2 + npt] * D1P_[7] + B_[EFGInflunceMaxSize_3 + npt] * D1P_[10];
			D1N_[npt*3+2] = B_[EFGInflunceMaxSize_0 + npt] * D1P_[2] + B_[EFGInflunceMaxSize_1 + npt] * D1P_[5] + B_[EFGInflunceMaxSize_2 + npt] * D1P_[8] + B_[EFGInflunceMaxSize_3 + npt] * D1P_[11];
		}

		for (int npt=0;npt < refInflunceSize;++npt)
		{
			D1N_[npt*3+0] += Bx_[EFGInflunceMaxSize_0 + npt] * r[0] + Bx_[EFGInflunceMaxSize_1 + npt] * r[1] + Bx_[EFGInflunceMaxSize_2 + npt] * r[2] + Bx_[EFGInflunceMaxSize_3 + npt] * r[3];
			D1N_[npt*3+1] += By_[EFGInflunceMaxSize_0 + npt] * r[0] + By_[EFGInflunceMaxSize_1 + npt] * r[1] + By_[EFGInflunceMaxSize_2 + npt] * r[2] + By_[EFGInflunceMaxSize_3 + npt] * r[3];
			D1N_[npt*3+2] += Bz_[EFGInflunceMaxSize_0 + npt] * r[0] + Bz_[EFGInflunceMaxSize_1 + npt] * r[1] + Bz_[EFGInflunceMaxSize_2 + npt] * r[2] + Bz_[EFGInflunceMaxSize_3 + npt] * r[3];
		}

		if (CellTypeCouple == curCell.cellType)
		{
			const float RofGaussPt = curCell.m_ShapeFunction_R[gaussIdx];
			float NI;
			float FaiI;
			float Rderiv;
			float NIderiv;
			float FaiIderive;
			for (int npt=0;npt < refInflunceSize;++npt)
			{
				NI = curFEMShapeValue.shapeFunctionValue_8_8[gaussIdx][npt];
				FaiI = N_[npt];
				N_[npt] = (1-RofGaussPt)* NI + RofGaussPt * FaiI;

				Rderiv = curCell.m_ShapeDeriv_R[gaussIdx][0];
				NIderiv = curFEMShapeValue.shapeDerivativeValue_8_8_3[gaussIdx][npt][0];
				FaiIderive = D1N_[npt*3+0];
				D1N_[npt*3+0] = Rderiv * NI + (1-RofGaussPt) * NIderiv + Rderiv * FaiI + RofGaussPt * FaiIderive;

				Rderiv = curCell.m_ShapeDeriv_R[gaussIdx][1];
				NIderiv = curFEMShapeValue.shapeDerivativeValue_8_8_3[gaussIdx][npt][1];
				FaiIderive = D1N_[npt*3+1];
				D1N_[npt*3+1] = Rderiv * NI + (1-RofGaussPt) * NIderiv + Rderiv * FaiI + RofGaussPt * FaiIderive;

				Rderiv = curCell.m_ShapeDeriv_R[gaussIdx][2];
				NIderiv = curFEMShapeValue.shapeDerivativeValue_8_8_3[gaussIdx][npt][2];
				FaiIderive = D1N_[npt*3+2];
				D1N_[npt*3+2] = Rderiv * NI + (1-RofGaussPt) * NIderiv + Rderiv * FaiI + RofGaussPt * FaiIderive;
			}
		}
	}
}


void makeInfluncePointList(const float SupportSize, const int validDomainId)
{
	HANDLE_ERROR( cudaMalloc( (void**)&EFG_CellOnCuda_Ext_Ptr,  EFG_CellOnCudaElementCount * sizeof(EFGCellOnCuda_Ext) ) );
	HANDLE_ERROR( cudaMemset( (void*)EFG_CellOnCuda_Ext_Ptr,0,EFG_CellOnCudaElementCount * sizeof(EFGCellOnCuda_Ext)   ) ) ;
	
	//makeInfluncePointListCuda_ForCell_nonRealTime<<<MaxCellCount,8>>>(EFG_CellOnCudaElementCount,EFG_CellOnCudaPtr, EFG_CellOnCuda_Ext_Ptr,g_nVertexOnCudaCount,g_VertexOnCudaPtr,SupportSize,validDomainId);
	dim3 threads_24_24(CellMaxInflunceVertexCount,VertxMaxInflunceCellCount);
	makeInfluncePointListCuda_ForCell_Radius2_RealTime<<<MaxCellCount,threads_24_24>>>(EFG_CellOnCudaElementCount,EFG_CellOnCudaPtr, EFG_CellOnCuda_Ext_Ptr,g_nVertexOnCudaCount,g_VertexOnCudaPtr,SupportSize,validDomainId);
	computeShapeFunctionOnCuda_RealTime<<<MaxCellCount,8>>>(EFG_CellOnCudaElementCount,EFG_CellOnCudaPtr, EFG_CellOnCuda_Ext_Ptr,FEMShapeValueOnCuda,g_nVertexOnCudaCount,g_VertexOnCudaPtr,SupportSize,validDomainId);
	//makeInfluncePointListCuda<<<MaxCellCount,8>>>(EFG_CellOnCudaElementCount,EFG_CellOnCudaPtr, EFG_CellOnCuda_Ext_Ptr,g_nVertexOnCudaCount,g_VertexOnCudaPtr,SupportSize,validDomainId);


	HANDLE_ERROR( cudaFree(EFG_CellOnCuda_Ext_Ptr));
}

void debugCopyCellStruct(int & nCellSize,EFGCellOnCuda * CellOnCudaPtr,EFGCellOnCuda_Ext* CellOnCudaExtPtr)
{
	nCellSize = EFG_CellOnCudaElementCount;
	//CellOnCudaPtr = (EFGCellOnCuda*)malloc( nCellSize * sizeof(EFGCellOnCuda) );
	HANDLE_ERROR( cudaMemcpy( (void *)CellOnCudaPtr,EFG_CellOnCudaPtr,nCellSize * sizeof(EFGCellOnCuda),cudaMemcpyDeviceToHost ) );
	//HANDLE_ERROR( cudaMemcpy( (void *)CellOnCudaExtPtr,EFG_CellOnCuda_Ext_Ptr,nCellSize * sizeof(EFGCellOnCuda_Ext),cudaMemcpyDeviceToHost ) );
}




void callCudaHostAlloc()
{
	float * tmp=0;
	printf("need memory %d\n",EFG_CellOnCudaElementCount *8*108*108* sizeof(float));
	HANDLE_ERROR( cudaMalloc( (void**)&tmp,  EFG_CellOnCudaElementCount *8*108*108* sizeof(float) ) );
	HANDLE_ERROR( cudaMemset( (void*)EFG_CellOnCuda_Ext_Ptr,0,EFG_CellOnCudaElementCount * sizeof(EFGCellOnCuda_Ext)   ) ) ;
}

#if 1
void setCuspVector_deviceMemory(CuspVec& destVector,IndexType nRows,ValueTypePtr valuePtr)
{

	{
		destVector.resize(nRows,0.);
		thrust::device_ptr<ValueType> wrapped_value(valuePtr);
		cusp::array1d_view<thrust::device_ptr<ValueType>> cusp_value (wrapped_value, wrapped_value + nRows);
		CuspVec::view valueView(destVector.begin(),destVector.end());
		cusp::copy(cusp_value,valueView);
	}
}

float *materialParams=0;
int   *materialIndex =0;
float *materialValue =0;
float *g_externForce =0;
float g_lai;
float g_Density;

#define bbcoeff(r,_local,_dim) (materialParam[r*3+_dim] * D1N_[_local*3+materialIndex[r*3+_dim]] )

void makeGlobalIndexPara(float YoungModulus, float PossionRatio, float Density, float *externForce)
{
	//static MyFloat params[6][3] =	{{1,0,0},{0,1,0},{0,0,1},{1,1,0},{1,0,1},{0,1,1}};
	float params[6*3] =				 {1,0,0,  0,1,0,  0,0,1,  1,1,0,  1,0,1,  0,1,1};
	//static int index[6][3] = {{0,0,0},{0,1,0},{0,0,2},{1,0,0},{2,0,0},{0,2,1}};
	int   index[6*3] =			{0,0,0,  0,1,0,  0,0,2,  1,0,0,  2,0,0,  0,2,1};
	float E = YoungModulus / 3;
	float mu = PossionRatio;
	float G = E/(2*(1+mu));
	float lai = mu*E/((1+mu)*(1-2*mu));
	float material[6] = {lai+2*G,lai+2*G,lai+2*G,G,G,G};
	g_lai = lai;
	g_Density = Density;
	

	HANDLE_ERROR( cudaMalloc( (void**)&materialParams,  6*3* sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&materialIndex,   6*3* sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&materialValue,   6*   sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&g_externForce,   3*   sizeof(float) ) );

	HANDLE_ERROR( cudaMemcpy( (void *)materialParams,&params[0],6*3* sizeof(float),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( (void *)materialIndex ,&index[0] ,6*3* sizeof(int)  ,cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( (void *)materialValue ,&material[0],6* sizeof(float),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( (void *)g_externForce ,&externForce[0],3* sizeof(float),cudaMemcpyHostToDevice ) );
}



__global__ void assemble_matrix_free_on_cuda(
	int nCellCount,int /*funcMatrixCount*/,int nDofCount,EFGCellOnCuda * cellOnCudaPointer,VertexOnCuda *	VertexOnCudaPtr,
	IndexTypePtr /*systemOuterIdxPtr*/,ValueTypePtr systemRhsPtr,
	IndexTypePtr g_globalDof,ValueTypePtr g_globalValue,
	IndexTypePtr globalDof_Mass,ValueTypePtr globalValue_Mass,
	IndexTypePtr globalDof_Damping,ValueTypePtr globalValue_Damping,
	IndexTypePtr globalDof_System,ValueTypePtr globalValue_System,
	ValueTypePtr tmp_blockShareRhs,
	float* materialParam, int* materialIndex, float * materialValue,
	float lai, float Density, float *externForce,
	ValueType * localStiffnessMatrixPtr,ValueType * localMassMatrixPtr,ValueType * localRhsVectorPtr
	)
{
	const int currentCellIdx = blockIdx.x;
	const int localDofIdx = threadIdx.x;/*0-23*/
	const int localColIdx = threadIdx.y;/*0-23*/
	//MY_PAUSE;
	if(currentCellIdx < nCellCount && true == cellOnCudaPointer[currentCellIdx].m_bLeaf )
	{
		int *local_globalDof;
		float *value;

		int *local_globalDof_mass;
		float *value_mass;

		int *local_globalDof_damping;
		float *value_damping;

		int *local_globalDof_system;
		float *value_system;

		float rhsValue = 0.0f;
		float *blockShareRhs;

		EFGCellOnCuda& curEFGCell = cellOnCudaPointer[currentCellIdx];

		if (CellTypeFEM == curEFGCell.cellType)
		{
			//FEM
#if 1
			const int r_local = localDofIdx / Dim_Count;//[0-7]
			const int r_dim   = localDofIdx % Dim_Count;//[0-2]
			const int c_local = localColIdx / Dim_Count;//[0-7]
			const int c_dim   = localColIdx % Dim_Count;//[0-2]		

			const int global_row = VertexOnCudaPtr[ curEFGCell.vertexId[r_local] ].m_nDof[r_dim];
			const int global_col = VertexOnCudaPtr[ curEFGCell.vertexId[c_local] ].m_nDof[c_dim];

			
			const int nStep = global_row * nMaxNonZeroSizeInFEM;
			local_globalDof = g_globalDof + nStep;
			value = g_globalValue + nStep;

			local_globalDof_mass = globalDof_Mass + nStep;
			value_mass = globalValue_Mass + nStep;

			local_globalDof_damping = globalDof_Damping + nStep;
			value_damping = globalValue_Damping + nStep;

			local_globalDof_system = globalDof_System + nStep;
			value_system = globalValue_System + nStep;

			blockShareRhs = tmp_blockShareRhs + global_row*8;

			
			const int loc_row = localDofIdx;//cellOnCudaPointer[currentCellIdx].localDofs[localDofIdx];
			//const int global_col = cellOnCudaPointer[currentCellIdx].globalDofs[localColIdx];
			const int loc_col = localColIdx;//cellOnCudaPointer[currentCellIdx].localDofs[localColIdx];//localColIdx;
			const int idx_in_8 = loc_row / 3;

			const float col_val_mass =  localMassMatrixPtr[curEFGCell.m_nMassMatrixIdx * Geometry_dofs_per_cell_squarte + loc_row * Geometry_dofs_per_cell + loc_col]*MASS_MATRIX_COEFF_2;
			const float col_val_stiffness = localStiffnessMatrixPtr[curEFGCell.m_nStiffnessMatrixIdx * Geometry_dofs_per_cell_squarte + loc_row * Geometry_dofs_per_cell + loc_col];
				
			const int index = idx_in_8 * LocalMaxDofCount_YC + loc_col;
			local_globalDof_mass[index] = global_col;
			value_mass[index] = col_val_mass;
			local_globalDof[index] = global_col;
			value[index] = col_val_stiffness;
			local_globalDof_damping[index] = global_col;
			value_damping[index] = 0.183f * col_val_mass + 0.00128f * col_val_stiffness;
			local_globalDof_system[index] = global_col;
			value_system[index] = 16384 * value_mass[index] + 128 * value_damping[index] + value[index];

			if (0 == localColIdx)
			{			
				blockShareRhs[idx_in_8] = localRhsVectorPtr[curEFGCell.m_nRhsIdx * Geometry_dofs_per_cell + loc_row];
			}
#endif
		}
		else //if (1 == curEFGCell.cellType)
		{
			//EFG
#if 1
			const int r_local = localDofIdx / Dim_Count;//[0-7]
			const int r_dim   = localDofIdx % Dim_Count;//[0-2]
			const int c_local = localColIdx / Dim_Count;//[0-7]
			const int c_dim   = localColIdx % Dim_Count;//[0-2]		

			const float JxW = curEFGCell.m_nJxW;
			const float dbPara = JxW * Density;

			const int global_row = VertexOnCudaPtr[ curEFGCell.m_cellInflunceVertexList[r_local] ].m_nDof[r_dim];
			const int global_col = VertexOnCudaPtr[ curEFGCell.m_cellInflunceVertexList[c_local] ].m_nDof[c_dim];

			const int nStep = global_row * nMaxNonZeroSize;
			local_globalDof = g_globalDof + nStep;
			value = g_globalValue + nStep;

			local_globalDof_mass = globalDof_Mass + nStep;
			value_mass = globalValue_Mass + nStep;

			local_globalDof_damping = globalDof_Damping + nStep;
			value_damping = globalValue_Damping + nStep;

			local_globalDof_system = globalDof_System + nStep;
			value_system = globalValue_System + nStep;

			blockShareRhs = tmp_blockShareRhs + global_row*8;
		
			float col_val_mass = 0.0f;
			float col_val_stiffness = 0.0f;
			
				for (int i=0;i<8;++i)
				{
					float * N_ = &curEFGCell.N_[i][0];
					float * D1N_ = &curEFGCell.D1N_[i][0][0];
					float tmp = 0.0f;

					tmp += ( bbcoeff(0,r_local,r_dim) * bbcoeff(0,c_local,c_dim) * materialValue[0]); 
					tmp += ( bbcoeff(1,r_local,r_dim) * bbcoeff(1,c_local,c_dim) * materialValue[1]); 
					tmp += ( bbcoeff(2,r_local,r_dim) * bbcoeff(2,c_local,c_dim) * materialValue[2]); 
					tmp += ( bbcoeff(3,r_local,r_dim) * bbcoeff(3,c_local,c_dim) * materialValue[3]); 
					tmp += ( bbcoeff(4,r_local,r_dim) * bbcoeff(4,c_local,c_dim) * materialValue[4]); 
					tmp += ( bbcoeff(5,r_local,r_dim) * bbcoeff(5,c_local,c_dim) * materialValue[5]); 
					tmp *= JxW;
					tmp += 
							(bbcoeff(0,r_local,r_dim) * (bbcoeff(1,c_local,c_dim) + bbcoeff(2,c_local,c_dim)) + bbcoeff(1,r_local,r_dim) * bbcoeff(2,c_local,c_dim) +
							bbcoeff(0,c_local,c_dim) * (bbcoeff(1,r_local,r_dim) + bbcoeff(2,r_local,r_dim)) + bbcoeff(1,c_local,c_dim) * bbcoeff(2,r_local,r_dim)) * lai * JxW;
					col_val_stiffness += tmp;
				}
				for (int i=0;i<8;++i)
				{
					float * N_ = &curEFGCell.N_[i][0];
					col_val_mass += (r_dim == c_dim) ? ( N_[r_local] * N_[c_local] * dbPara ) : (0.f);
				}
				float * N_;
				if (0 == localColIdx)
				{			
					N_ = &curEFGCell.N_[0][0];
					blockShareRhs[r_local] = JxW * N_[r_local] * externForce[r_dim];
					N_ = &curEFGCell.N_[1][0];
					blockShareRhs[r_local] += JxW * N_[r_local] * externForce[r_dim];
					N_ = &curEFGCell.N_[2][0];
					blockShareRhs[r_local] += JxW * N_[r_local] * externForce[r_dim];
					N_ = &curEFGCell.N_[3][0];
					blockShareRhs[r_local] += JxW * N_[r_local] * externForce[r_dim];
					N_ = &curEFGCell.N_[4][0];
					blockShareRhs[r_local] += JxW * N_[r_local] * externForce[r_dim];
					N_ = &curEFGCell.N_[5][0];
					blockShareRhs[r_local] += JxW * N_[r_local] * externForce[r_dim];
					N_ = &curEFGCell.N_[6][0];
					blockShareRhs[r_local] += JxW * N_[r_local] * externForce[r_dim];
					N_ = &curEFGCell.N_[7][0];
					blockShareRhs[r_local] += JxW * N_[r_local] * externForce[r_dim];
				}
			//}

		
			const int index = r_local * MaxCellDof4EFG + localColIdx;
		
			local_globalDof[index] = global_col;
			local_globalDof_mass[index] = global_col;
			local_globalDof_damping[index] = global_col;
			local_globalDof_system[index] = global_col;

			value[index] = (float)col_val_stiffness;
			value_mass[index] = (float)col_val_mass;
			value_damping[index] = 0.183f * col_val_mass + 0.00128f * col_val_stiffness;
			value_system[index] = 16384 * value_mass[index] + 128 * value_damping[index] + value[index];
#endif
		}		
	}
	return ;
}

__global__ void assembleRhsValue_on_cuda(int nDofCount, ValueTypePtr systemRhsPtr,ValueTypePtr tmp_blockShareRhs)
{
	const int currentDof = threadIdx.x + blockIdx.x * blockDim.x;
	if (currentDof < nDofCount)
	{
		ValueTypePtr blockShareRhs = tmp_blockShareRhs + currentDof*8;
		systemRhsPtr[currentDof] = blockShareRhs[0] + blockShareRhs[1] + blockShareRhs[2] + blockShareRhs[3] + 
									blockShareRhs[4] + blockShareRhs[5] + blockShareRhs[6] + blockShareRhs[7];
	}
}

void assembleSystemOnCuda_EFG_RealTime()
{
	int nCurrentTick,nLastTick = GetTickCount()/*,nnCount = 0*/;
	int nCellCount = EFG_CellOnCudaElementCount;

	int nouse;
	int nDofCount = g_nDofs;
	int nonZeroCount/*,nonZeroCountMass*/;

	//printf("nCellCount[%d] nDofCount[%d] g_lai[%f] g_Density[%f]\n",nCellCount,nDofCount,g_lai,g_Density);

	EFGCellOnCuda * cellOnCudaPointer = EFG_CellOnCudaPtr;
	VertexOnCuda*   VertexOnCudaPtr = g_VertexOnCudaPtr;
	
	{
		HANDLE_ERROR( cudaMemset((void *)g_globalDof_MF,  0, (g_nDofs)*nMaxNonZeroSize * sizeof(IndexType)));
		HANDLE_ERROR( cudaMemset((void *)g_globalValue_MF,  0, (g_nDofs)*nMaxNonZeroSize * sizeof(ValueType)));
		HANDLE_ERROR( cudaMemset((void *)g_globalDof_Mass_MF,  0, (g_nDofs)*nMaxNonZeroSize * sizeof(IndexType)));
		HANDLE_ERROR( cudaMemset((void *)g_globalValue_Mass_MF,  0, (g_nDofs)*nMaxNonZeroSize * sizeof(ValueType)));
		HANDLE_ERROR( cudaMemset((void *)g_globalDof_Damping_MF,  0, (g_nDofs)*nMaxNonZeroSize * sizeof(IndexType)));
		HANDLE_ERROR( cudaMemset((void *)g_globalValue_Damping_MF,  0, (g_nDofs)*nMaxNonZeroSize * sizeof(ValueType)));
		HANDLE_ERROR( cudaMemset((void *)g_globalDof_System_MF,  0, (g_nDofs)*nMaxNonZeroSize * sizeof(IndexType)));
		HANDLE_ERROR( cudaMemset((void *)g_globalValue_System_MF,  0, (g_nDofs)*nMaxNonZeroSize * sizeof(ValueType)));
		HANDLE_ERROR( cudaMemset((void *)g_systemRhsPtr_MF,  0, (g_nDofs) * sizeof(ValueType)));
		HANDLE_ERROR( cudaMemset((void *)g_systemRhsPtr_In8_MF,  0, (g_nDofs)* VertxMaxInflunceCellCount * sizeof(ValueType)));
		
	}
	//return ;
	dim3 threads_24_24(MaxCellDof4EFG,MaxCellDof4EFG);
	//dim3 threads_24_24(24,24);
	assemble_matrix_free_on_cuda<<< KERNEL_COUNT_TMP,threads_24_24 >>>(
		nCellCount, nouse, nDofCount, cellOnCudaPointer, VertexOnCudaPtr,
		/*g_systemOuterIdxPtr*/(IndexTypePtr)0 , g_systemRhsPtr_MF,
		g_globalDof_MF,g_globalValue_MF,
		g_globalDof_Mass_MF,g_globalValue_Mass_MF,
		g_globalDof_Damping_MF,g_globalValue_Damping_MF,
		g_globalDof_System_MF,g_globalValue_System_MF,g_systemRhsPtr_In8_MF,
		materialParams,materialIndex,materialValue,
		g_lai,g_Density,g_externForce,
		localStiffnessMatrixOnCuda, localMassMatrixOnCuda, localRhsVectorOnCuda);

	
	assembleRhsValue_on_cuda<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(nDofCount,g_systemRhsPtr_MF,g_systemRhsPtr_In8_MF);

	g_nLineCount4Display = nCellCount*12;
	
	setCuspVector_deviceMemory(cusp_Array_Rhs,nDofCount,g_systemRhsPtr_MF);

	if (g_nDofsLast < nDofCount)
	{
		cusp_Array_Incremental_displace.resize(nDofCount,0);
		cusp_Array_Displacement.resize(nDofCount,0);
		cusp_Array_R_rhs.resize(nDofCount,0);
		cusp_Array_Mass_rhs.resize(nDofCount,0);
		cusp_Array_Damping_rhs.resize(nDofCount,0);
		cusp_Array_Velocity.resize(nDofCount,0);
		cusp_Array_Acceleration.resize(nDofCount,0);
		cusp_Array_Old_Acceleration.resize(nDofCount,0);
		cusp_Array_Old_Displacement.resize(nDofCount,0);
#if USE_CORATIONAL
		cusp_Array_R_rhs_tmp4Corotaion.resize(nDofCount,0);
		cusp_Array_R_rhs_Corotaion.resize(nDofCount,0);
#endif
	}
	//return ;
	MassMatrix.initialize(g_nDofs,/*g_systemOuterIdxPtr*/(IndexTypePtr)0, g_globalDof_Mass_MF, g_globalValue_Mass_MF);
	DampingMatrix.initialize(g_nDofs,/*g_systemOuterIdxPtr*/(IndexTypePtr)0, g_globalDof_Damping_MF, g_globalValue_Damping_MF);
	SystemMatrix.initialize(g_nDofs,/*g_systemOuterIdxPtr*/(IndexTypePtr)0, g_globalDof_System_MF, g_globalValue_System_MF);

	/*nCurrentTick = GetTickCount();
	printf("assembleSystemOnCuda %d \n",nCurrentTick - nLastTick);*/
	
	return ;
}

#if USE_CORATIONAL
void assembleMatrixForCorotation(IndexType startIdx)
{
	//return ;
	int nCurrentTick,nLastTick = GetTickCount()/*,nnCount = 0*/;
	int nCellCount = cellOnCudaElementCount;

	int nouse;
	int nDofCount = g_nDofs;
	int nonZeroCount/*,nonZeroCountMass*/;

	EFGCellOnCuda * cellOnCudaPointer = cellOnCudaPtr;
	ValueType * localStiffnessMatrixPtr = localStiffnessMatrixOnCuda;
	ValueType * localMassMatrixPtr = localMassMatrixOnCuda;
	ValueType * localRhsVectorPtr = localRhsVectorOnCuda;

	dim3 threads_24_24(LocalMaxDofCount_YC,LocalMaxDofCount_YC);
	//dim3 threads_8_24(DofMaxAssociateCellCount_YC,LocalMaxDofCount_YC);
	ValueTypePtr tmp_8,tmp_8_compare;
	HANDLE_ERROR( cudaMalloc( (void**)&tmp_8,(nDofCount * 8) * sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMemset( (void*)tmp_8,0,(nDofCount * 8) * sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMalloc( (void**)&tmp_8_compare,(nDofCount * 8) * sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMemset( (void*)tmp_8_compare,0,(nDofCount * 8) * sizeof(ValueType))) ;

	{
		HANDLE_ERROR( cudaMemset((void *)g_globalDof,  0, (g_nDofs)*nMaxNonZeroSize * sizeof(IndexType)));
		HANDLE_ERROR( cudaMemset((void *)g_globalValue,  0, (g_nDofs)*nMaxNonZeroSize * sizeof(ValueType)));
		HANDLE_ERROR( cudaMemset((void *)g_globalDof_Mass,  0, (g_nDofs)*nMaxNonZeroSize * sizeof(IndexType)));
		HANDLE_ERROR( cudaMemset((void *)g_globalValue_Mass,  0, (g_nDofs)*nMaxNonZeroSize * sizeof(ValueType)));
		HANDLE_ERROR( cudaMemset((void *)g_globalDof_Damping,  0, (g_nDofs)*nMaxNonZeroSize * sizeof(IndexType)));
		HANDLE_ERROR( cudaMemset((void *)g_globalValue_Damping,  0, (g_nDofs)*nMaxNonZeroSize * sizeof(ValueType)));
		HANDLE_ERROR( cudaMemset((void *)g_globalDof_System,  0, (g_nDofs)*nMaxNonZeroSize * sizeof(IndexType)));
		HANDLE_ERROR( cudaMemset((void *)g_globalValue_System,  0, (g_nDofs)*nMaxNonZeroSize * sizeof(ValueType)));
		HANDLE_ERROR( cudaMemset((void *)g_systemRhsPtr,  0, (g_nDofs) * sizeof(ValueType)));
	}

	assemble_matrix_free_on_cuda_4_Corotation<<< KERNEL_COUNT_TMP,threads_24_24 >>>(
		nCellCount, nouse, nDofCount, cellOnCudaPointer, 
		/*localStiffnessMatrixPtr*/cuda_RKR, localMassMatrixPtr, localRhsVectorPtr, 

		g_systemOuterIdxPtr , g_systemRhsPtr,
		g_globalDof,g_globalValue,
		g_globalDof_Mass,g_globalValue_Mass,
		g_globalDof_Damping,g_globalValue_Damping,
		g_globalDof_System,g_globalValue_System,tmp_8/*,livedCellFlag_OnCuda*/,tmp_8_compare,
		cuda_RKRtPj);

	assembleRhsValue_on_cuda<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(nDofCount,g_systemRhsPtr,tmp_8);
	HANDLE_ERROR( cudaFree(tmp_8));	
	setCuspVector_deviceMemory(cusp_Array_R_rhs_Corotaion,nDofCount,g_systemRhsPtr);

	CuspVec cusp_Array_R_rhs_Corotaion_tmp;
	assembleRhsValue_on_cuda<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(nDofCount,g_systemRhsPtr,tmp_8_compare);
	setCuspVector_deviceMemory(cusp_Array_R_rhs_Corotaion_tmp,nDofCount,g_systemRhsPtr);
	HANDLE_ERROR( cudaFree(tmp_8_compare));

	g_nLineCount4Display = nCellCount*12;

	MassMatrix.initialize(g_nDofs,g_systemOuterIdxPtr, g_globalDof_Mass, g_globalValue_Mass);
	DampingMatrix.initialize(g_nDofs,g_systemOuterIdxPtr, g_globalDof_Damping, g_globalValue_Damping);
	SystemMatrix.initialize(g_nDofs,g_systemOuterIdxPtr, g_globalDof_System, g_globalValue_System);
	
	return ;
}
#endif

void initCellMatrixOnCuda(int nCount,ValueType * localStiffnessMatrixOnCpu,ValueType * localMassMatrixOnCpu,ValueType * localRhsVectorOnCpu)
{
	funcMatrixCount = nCount;
	HANDLE_ERROR( cudaMalloc( (void**)&localStiffnessMatrixOnCuda	,nCount * Geometry_dofs_per_cell_squarte * sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMalloc( (void**)&localMassMatrixOnCuda		,nCount * Geometry_dofs_per_cell_squarte * sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMalloc( (void**)&localRhsVectorOnCuda			,nCount * Geometry_dofs_per_cell		 * sizeof(ValueType))) ;

	HANDLE_ERROR( cudaMemcpy( (void *)localStiffnessMatrixOnCuda	,localStiffnessMatrixOnCpu	,nCount * Geometry_dofs_per_cell_squarte * sizeof(ValueType),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( (void *)localMassMatrixOnCuda			,localMassMatrixOnCpu		,nCount * Geometry_dofs_per_cell_squarte * sizeof(ValueType),cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( (void *)localRhsVectorOnCuda			,localRhsVectorOnCpu		,nCount * Geometry_dofs_per_cell		 * sizeof(ValueType),cudaMemcpyHostToDevice ) );	
}


void initFEMShapeValueOnCuda(int nCount, FEMShapeValue* femShapeValuePtr)
{
	nFEMShapeValueCount = nCount;
	HANDLE_ERROR( cudaMalloc( (void**)&FEMShapeValueOnCuda	,nCount * sizeof(FEMShapeValue))) ;
	HANDLE_ERROR( cudaMemcpy( (void *)FEMShapeValueOnCuda	,femShapeValuePtr	,nCount * sizeof(FEMShapeValue),cudaMemcpyHostToDevice ) );
}

void initBlade(ValueTypePtr elementBlade,IndexType nElementCount, ValueTypePtr bladeNormal)
{
	HANDLE_ERROR( cudaMalloc( (void**)&g_BladeElementOnCuda	,nElementCount * sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMemcpy( (void *)g_BladeElementOnCuda	,elementBlade	,nElementCount * sizeof(ValueType),cudaMemcpyHostToDevice ) );
	
	HANDLE_ERROR( cudaMalloc( (void**)&g_BladeNormalOnCuda	,6 * sizeof(ValueType))) ;
	HANDLE_ERROR( cudaMemcpy( (void *)g_BladeNormalOnCuda	,bladeNormal	,6 * sizeof(ValueType),cudaMemcpyHostToDevice ) );
}
/*
int g_nUpBladeVertexSize=0,g_nDownBladeVertexSize=0;
int g_nUpBladeEdgeSize=0,g_nDownBladeEdgeSize=0;
int g_nUpBladeSurfaceSize=0,g_nDownBladeSurfaceSize=0;
MC_Vertex_Cuda*		g_UpBlade_MC_Vertex_Cuda=0;
MC_Edge_Cuda*		g_UpBlade_MC_Edge_Cuda=0;
MC_Surface_Cuda*	g_UpBlade_MC_Surface_Cuda=0;

MC_Vertex_Cuda*		g_DownBlade_MC_Vertex_Cuda=0;
MC_Edge_Cuda*		g_DownBlade_MC_Edge_Cuda=0;
MC_Surface_Cuda*	g_DownBlade_MC_Surface_Cuda=0;
*/

void initBlade_MeshCutting( int nUpBladeVertexSize,int nDownBladeVertexSize,int nUpBladeEdgeSize,
							int nDownBladeEdgeSize,int nUpBladeSurfaceSize,int nDownBladeSurfaceSize,
							MC_Vertex_Cuda*	UpBlade_MC_Vertex_Cuda,MC_Edge_Cuda* UpBlade_MC_Edge_Cuda,MC_Surface_Cuda*	UpBlade_MC_Surface_Cuda,
							MC_Vertex_Cuda*	DownBlade_MC_Vertex_Cuda,MC_Edge_Cuda* DownBlade_MC_Edge_Cuda,MC_Surface_Cuda*	DownBlade_MC_Surface_Cuda)
{
	g_nUpBladeVertexSize = nUpBladeVertexSize;
	g_nDownBladeVertexSize = nDownBladeVertexSize;
	g_nUpBladeEdgeSize = nUpBladeEdgeSize;
	g_nDownBladeEdgeSize = nDownBladeEdgeSize;
	g_nUpBladeSurfaceSize = nUpBladeSurfaceSize;
	g_nDownBladeSurfaceSize = nDownBladeSurfaceSize;

	//
	HANDLE_ERROR( cudaHostAlloc( (void**)&g_UpBlade_MC_Vertex_Cpu, g_nUpBladeVertexSize * 2 * sizeof(MC_Vertex_Cuda),cudaHostAllocMapped   )) ;
	HANDLE_ERROR( cudaHostGetDevicePointer((void **)&g_UpBlade_MC_Vertex_Cuda,(void *)g_UpBlade_MC_Vertex_Cpu,0));
	HANDLE_ERROR( cudaMemset( (void*)g_UpBlade_MC_Vertex_Cuda,	0,g_nUpBladeVertexSize * 2 * sizeof(MC_Vertex_Cuda))) ;
	memcpy((void *)g_UpBlade_MC_Vertex_Cpu	,UpBlade_MC_Vertex_Cuda	,g_nUpBladeVertexSize * sizeof(MC_Vertex_Cuda));

	HANDLE_ERROR( cudaHostAlloc( (void**)&g_DownBlade_MC_Vertex_Cpu, g_nDownBladeVertexSize * 2 * sizeof(MC_Vertex_Cuda),cudaHostAllocMapped   )) ;
	HANDLE_ERROR( cudaHostGetDevicePointer((void **)&g_DownBlade_MC_Vertex_Cuda,(void *)g_DownBlade_MC_Vertex_Cpu,0));
	HANDLE_ERROR( cudaMemset( (void*)g_DownBlade_MC_Vertex_Cuda,	0,g_nDownBladeVertexSize * 2 * sizeof(MC_Vertex_Cuda))) ;
	memcpy((void *)g_DownBlade_MC_Vertex_Cpu	,DownBlade_MC_Vertex_Cuda	,g_nDownBladeVertexSize * sizeof(MC_Vertex_Cuda));

	HANDLE_ERROR( cudaHostAlloc( (void**)&g_UpBlade_MC_Edge_Cpu, g_nUpBladeEdgeSize * 3 * sizeof(MC_Edge_Cuda),cudaHostAllocMapped   )) ;
	HANDLE_ERROR( cudaHostGetDevicePointer((void **)&g_UpBlade_MC_Edge_Cuda,(void *)g_UpBlade_MC_Edge_Cpu,0));
	HANDLE_ERROR( cudaMemset( (void*)g_UpBlade_MC_Edge_Cuda,	0,g_nUpBladeEdgeSize * 3 * sizeof(MC_Edge_Cuda))) ;
	memcpy((void *)g_UpBlade_MC_Edge_Cpu	,UpBlade_MC_Edge_Cuda	,g_nUpBladeEdgeSize * sizeof(MC_Edge_Cuda));

	HANDLE_ERROR( cudaHostAlloc( (void**)&g_DownBlade_MC_Edge_Cpu, g_nDownBladeEdgeSize * 3 * sizeof(MC_Edge_Cuda),cudaHostAllocMapped   )) ;
	HANDLE_ERROR( cudaHostGetDevicePointer((void **)&g_DownBlade_MC_Edge_Cuda,(void *)g_DownBlade_MC_Edge_Cpu,0));
	HANDLE_ERROR( cudaMemset( (void*)g_DownBlade_MC_Edge_Cuda,	0,g_nDownBladeEdgeSize * 3 * sizeof(MC_Edge_Cuda))) ;
	memcpy((void *)g_DownBlade_MC_Edge_Cpu	,DownBlade_MC_Edge_Cuda	,g_nDownBladeEdgeSize * sizeof(MC_Edge_Cuda));

	HANDLE_ERROR( cudaHostAlloc( (void**)&g_UpBlade_MC_Surface_Cpu, g_nUpBladeSurfaceSize * 3 * sizeof(MC_Surface_Cuda),cudaHostAllocMapped   )) ;
	HANDLE_ERROR( cudaHostGetDevicePointer((void **)&g_UpBlade_MC_Surface_Cuda,(void *)g_UpBlade_MC_Surface_Cpu,0));
	HANDLE_ERROR( cudaMemset( (void*)g_UpBlade_MC_Surface_Cuda,	0,g_nUpBladeSurfaceSize * 3 * sizeof(MC_Surface_Cuda))) ;
	memcpy((void *)g_UpBlade_MC_Surface_Cpu	,UpBlade_MC_Surface_Cuda	,g_nUpBladeSurfaceSize * sizeof(MC_Surface_Cuda));
	
	HANDLE_ERROR( cudaHostAlloc( (void**)&g_DownBlade_MC_Surface_Cpu, g_nDownBladeSurfaceSize * 3 * sizeof(MC_Surface_Cuda),cudaHostAllocMapped   )) ;
	HANDLE_ERROR( cudaHostGetDevicePointer((void **)&g_DownBlade_MC_Surface_Cuda,(void *)g_DownBlade_MC_Surface_Cpu,0));
	HANDLE_ERROR( cudaMemset( (void*)g_DownBlade_MC_Surface_Cuda,	0,g_nDownBladeSurfaceSize * 3 * sizeof(MC_Surface_Cuda))) ;
	memcpy((void *)g_DownBlade_MC_Surface_Cpu	,DownBlade_MC_Surface_Cuda	,g_nDownBladeSurfaceSize * sizeof(MC_Surface_Cuda));

	/*HANDLE_ERROR( cudaMalloc( (void**)&g_UpBlade_MC_Vertex_Cuda	,g_nUpBladeVertexSize * 2 * sizeof(MC_Vertex_Cuda))) ;
	HANDLE_ERROR( cudaMemset( (void*)g_UpBlade_MC_Vertex_Cuda,0, g_nUpBladeVertexSize * 2 * sizeof(MC_Vertex_Cuda))) ;
	HANDLE_ERROR( cudaMemcpy( (void *)g_UpBlade_MC_Vertex_Cuda	,UpBlade_MC_Vertex_Cuda	,g_nUpBladeVertexSize * sizeof(MC_Vertex_Cuda),cudaMemcpyHostToDevice ) );
	
	HANDLE_ERROR( cudaMalloc( (void**)&g_DownBlade_MC_Vertex_Cuda	,g_nDownBladeVertexSize * 2 * sizeof(MC_Vertex_Cuda))) ;
	HANDLE_ERROR( cudaMemset( (void*)g_DownBlade_MC_Vertex_Cuda,0, g_nDownBladeVertexSize * 2 * sizeof(MC_Vertex_Cuda))) ;
	HANDLE_ERROR( cudaMemcpy( (void *)g_DownBlade_MC_Vertex_Cuda	,DownBlade_MC_Vertex_Cuda	,g_nDownBladeVertexSize * sizeof(MC_Vertex_Cuda),cudaMemcpyHostToDevice ) );*/
	
	/*HANDLE_ERROR( cudaMalloc( (void**)&g_UpBlade_MC_Edge_Cuda	,g_nUpBladeEdgeSize * 2 * sizeof(MC_Edge_Cuda))) ;
	HANDLE_ERROR( cudaMemset( (void*)g_UpBlade_MC_Edge_Cuda,0, g_nUpBladeEdgeSize * 2 * sizeof(MC_Edge_Cuda))) ;
	HANDLE_ERROR( cudaMemcpy( (void *)g_UpBlade_MC_Edge_Cuda	,UpBlade_MC_Edge_Cuda	,g_nUpBladeEdgeSize * sizeof(MC_Edge_Cuda),cudaMemcpyHostToDevice ) );*/
	
	/*HANDLE_ERROR( cudaMalloc( (void**)&g_DownBlade_MC_Edge_Cuda	,g_nDownBladeEdgeSize * 2 * sizeof(MC_Edge_Cuda))) ;
	HANDLE_ERROR( cudaMemset( (void*)g_DownBlade_MC_Edge_Cuda,0, g_nDownBladeEdgeSize * 2 * sizeof(MC_Edge_Cuda))) ;
	HANDLE_ERROR( cudaMemcpy( (void *)g_DownBlade_MC_Edge_Cuda	,DownBlade_MC_Edge_Cuda	,g_nDownBladeEdgeSize * sizeof(MC_Edge_Cuda),cudaMemcpyHostToDevice ) );*/
	
	/*HANDLE_ERROR( cudaMalloc( (void**)&g_UpBlade_MC_Surface_Cuda	,g_nUpBladeSurfaceSize * 2 * sizeof(MC_Surface_Cuda))) ;
	HANDLE_ERROR( cudaMemset( (void*)g_UpBlade_MC_Surface_Cuda,0, g_nUpBladeSurfaceSize * 2 * sizeof(MC_Surface_Cuda))) ;
	HANDLE_ERROR( cudaMemcpy( (void *)g_UpBlade_MC_Surface_Cuda	,UpBlade_MC_Surface_Cuda	,g_nUpBladeSurfaceSize * sizeof(MC_Surface_Cuda),cudaMemcpyHostToDevice ) );*/
	
	/*HANDLE_ERROR( cudaMalloc( (void**)&g_DownBlade_MC_Surface_Cuda	,g_nDownBladeSurfaceSize * 2 * sizeof(MC_Surface_Cuda))) ;
	HANDLE_ERROR( cudaMemset( (void*)g_DownBlade_MC_Surface_Cuda,0, g_nDownBladeSurfaceSize * 2 * sizeof(MC_Surface_Cuda))) ;
	HANDLE_ERROR( cudaMemcpy( (void *)g_DownBlade_MC_Surface_Cuda	,DownBlade_MC_Surface_Cuda	,g_nDownBladeSurfaceSize * sizeof(MC_Surface_Cuda),cudaMemcpyHostToDevice ) );*/
	
	HANDLE_ERROR( cudaMalloc( (void**)&g_BladeEdgeVSSpliteFace	,g_nUpBladeEdgeSize * sizeof(MC_CuttingEdge_Cuda))) ;
	HANDLE_ERROR( cudaMemset( (void*)g_BladeEdgeVSSpliteFace,0  ,g_nUpBladeEdgeSize * sizeof(MC_CuttingEdge_Cuda))) ;	
}

int *g_CuttedEdgeFlagOnCpu = 0;
int *g_SplittedFaceFlagOnCpu = 0;
//float * g_CuttingFaceUp_X=0;//size is max meshcutting vertex
//float * g_CuttingFaceDown_X=0;//size is max meshcutting vertex
//float * g_CuttingFaceUp_Y=0;//size is max meshcutting vertex
//float * g_CuttingFaceDown_Y=0;//size is max meshcutting vertex
//float * g_CuttingFaceUp_Z=0;//size is max meshcutting vertex
//float * g_CuttingFaceDown_Z=0;//size is max meshcutting vertex
//int * g_CuttingFaceUpFlagCpu=0;
//int * g_CuttingFaceDownFlagCpu=0;

void initMeshCuttingStructure(const int nVertexSize,MC_Vertex_Cuda* MCVertexCuda,
							  const int nEdgeSize,  MC_Edge_Cuda*	 MCEdgeCuda,
							  const int nSurfaceSize,MC_Surface_Cuda*	MCSurfaceCuda,
							  const int nVertexNormal,float * elementVertexNormal)
{
	g_nMCVertexSize = nVertexSize;
	g_nMaxMCVertexSize = 2*nVertexSize;
	g_nMCEdgeSize = nEdgeSize;
	g_nMaxMCEdgeSize = 2*nEdgeSize;
	g_nMCSurfaceSize = nSurfaceSize;
	g_nMaxMCSurfaceSize = 2*nSurfaceSize;
	g_nVertexNormalSize = nVertexNormal;

	printf("g_nMCVertexSize(%d) g_nMaxMCVertexSize(%d) g_nMCEdgeSize(%d) g_nMaxMCEdgeSize(%d) g_nMCSurfaceSize(%d) g_nMaxMCSurfaceSize(%d)\n",
		  g_nMCVertexSize,g_nMaxMCVertexSize,g_nMCEdgeSize,g_nMaxMCEdgeSize,g_nMCSurfaceSize,g_nMaxMCSurfaceSize);

	HANDLE_ERROR( cudaMalloc( (void**)&g_MC_Vertex_Cuda, g_nMaxMCVertexSize * sizeof(MC_Vertex_Cuda))) ;
	HANDLE_ERROR( cudaMemset( (void*)g_MC_Vertex_Cuda,0, g_nMaxMCVertexSize * sizeof(MC_Vertex_Cuda))) ;
	HANDLE_ERROR( cudaMemcpy( (void*)g_MC_Vertex_Cuda, MCVertexCuda, g_nMCVertexSize * sizeof(MC_Vertex_Cuda), cudaMemcpyHostToDevice));

	HANDLE_ERROR( cudaMalloc( (void**)&g_MC_Edge_Cuda, g_nMaxMCEdgeSize * sizeof(MC_Edge_Cuda))) ;
	HANDLE_ERROR( cudaMemset( (void*)g_MC_Edge_Cuda,0, g_nMaxMCEdgeSize * sizeof(MC_Edge_Cuda))) ;
	HANDLE_ERROR( cudaMemcpy( (void*)g_MC_Edge_Cuda, MCEdgeCuda, g_nMCEdgeSize * sizeof(MC_Edge_Cuda), cudaMemcpyHostToDevice));

	HANDLE_ERROR( cudaMalloc( (void**)&g_MC_Surface_Cuda, g_nMaxMCSurfaceSize * sizeof(MC_Surface_Cuda))) ;
	HANDLE_ERROR( cudaMemset( (void*)g_MC_Surface_Cuda,0, g_nMaxMCSurfaceSize * sizeof(MC_Surface_Cuda))) ;
	HANDLE_ERROR( cudaMemcpy( (void*)g_MC_Surface_Cuda, MCSurfaceCuda, g_nMCSurfaceSize * sizeof(MC_Surface_Cuda), cudaMemcpyHostToDevice));

	HANDLE_ERROR( cudaHostAlloc( (void**)&g_CuttedEdgeFlagOnCpu, g_nMaxMCEdgeSize * sizeof(IndexType),cudaHostAllocMapped   )) ;
	HANDLE_ERROR( cudaHostAlloc( (void**)&g_SplittedFaceFlagOnCpu, g_nMaxMCSurfaceSize * sizeof(IndexType),cudaHostAllocMapped   )) ;

	HANDLE_ERROR( cudaMalloc( (void**)&g_elementVertexNormal, g_nVertexNormalSize * 3 * sizeof(float))) ;
	HANDLE_ERROR( cudaMemcpy( (void*)g_elementVertexNormal, elementVertexNormal, g_nVertexNormalSize * 3 * sizeof(float), cudaMemcpyHostToDevice));

	/*HANDLE_ERROR( cudaHostAlloc( (void**)&g_CuttingFaceUp_X, g_nMaxMCVertexSize * sizeof(float),cudaHostAllocMapped   )) ;
	HANDLE_ERROR( cudaHostAlloc( (void**)&g_CuttingFaceUp_Y, g_nMaxMCVertexSize * sizeof(float),cudaHostAllocMapped   )) ;
	HANDLE_ERROR( cudaHostAlloc( (void**)&g_CuttingFaceUp_Z, g_nMaxMCVertexSize * sizeof(float),cudaHostAllocMapped   )) ;

	HANDLE_ERROR( cudaHostAlloc( (void**)&g_CuttingFaceDown_X, g_nMaxMCVertexSize * sizeof(float),cudaHostAllocMapped   )) ;
	HANDLE_ERROR( cudaHostAlloc( (void**)&g_CuttingFaceDown_Y, g_nMaxMCVertexSize * sizeof(float),cudaHostAllocMapped   )) ;
	HANDLE_ERROR( cudaHostAlloc( (void**)&g_CuttingFaceDown_Z, g_nMaxMCVertexSize * sizeof(float),cudaHostAllocMapped   )) ;*/

	/*HANDLE_ERROR( cudaHostAlloc( (void**)&g_CuttingFaceUpFlagCpu, g_nMaxMCVertexSize * sizeof(int),cudaHostAllocMapped   )) ;
	HANDLE_ERROR( cudaHostAlloc( (void**)&g_CuttingFaceDownFlagCpu, g_nMaxMCVertexSize * sizeof(int),cudaHostAllocMapped   )) ;*/
}
//checkLineTriWithIntersect
//__device__ bool checkLineTriWithIntersect(float3 L1,float3 L2,float3 PV1,float3 PV2,float3 PV3,float* HitP )
__global__ void cuda_makeVertex2BladeRelationShip(const int nLineSize,MC_Edge_Cuda* curLineSet,
												  const int nVertexSize,MC_Vertex_Cuda* curVertexSet,
												  ValueTypePtr vecBlade,int nBladeBase,
												  int* CuttedEdgeFlagOnCuda)
{
	int curLineIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (curLineIdx < nLineSize)
	{
		MC_Edge_Cuda& refEdge = curLineSet[curLineIdx];
		if (refEdge.m_isValid)
		{	
			MC_Vertex_Cuda& p0 = curVertexSet[refEdge.m_Vertex[0]];
			MC_Vertex_Cuda& p1 = curVertexSet[refEdge.m_Vertex[1]];

			refEdge.m_isCut = checkLineTriWithIntersect(make_float3(p0.m_VertexPos[0],p0.m_VertexPos[1],p0.m_VertexPos[2]),
										   make_float3(p1.m_VertexPos[0],p1.m_VertexPos[1],p1.m_VertexPos[2]),
										   make_float3(vecBlade[nBladeBase+0],vecBlade[nBladeBase+1],vecBlade[nBladeBase+2]),
										   make_float3(vecBlade[nBladeBase+3],vecBlade[nBladeBase+4],vecBlade[nBladeBase+5]),
										   make_float3(vecBlade[nBladeBase+6],vecBlade[nBladeBase+7],vecBlade[nBladeBase+8]),&refEdge.m_intersectPos[0]);
			
			CuttedEdgeFlagOnCuda[curLineIdx] = (refEdge.m_isCut ? 2 : 0);
			//CuttedEdgeFlagOnCuda[curLineIdx] = 2;
		}
		else
		{
			//CuttedEdgeFlagOnCuda[curLineIdx] = 2;
		}
	}
}

__device__ void makeScalarPoint(float * src,float * dst, float * clonePos)
{
	clonePos[0] = (dst[0] - src[0]) * 0.9123f;
	clonePos[1] = (dst[1] - src[1]) * 0.9123f;
	clonePos[2] = (dst[2] - src[2]) * 0.9123f;
}

__global__ void cuda_spliteLine(const int nLineSize,MC_Edge_Cuda* curLineSet,
								const int nVertexSize,MC_Vertex_Cuda* curVertexSet,
								int* CuttedEdgeFlagOnCuda,int nLastVertexBase,int nLastEdgeBase)
{
	int curLineIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (curLineIdx < nLineSize)
	{
		MC_Edge_Cuda& refEdge = curLineSet[curLineIdx];
		if (refEdge.m_isValid && refEdge.m_isCut)
		{
			refEdge.m_isValid = false;
			int localCloneVertexIdx_0 = nLastVertexBase + CuttedEdgeFlagOnCuda[curLineIdx];
			int localCloneVertexIdx_1 = localCloneVertexIdx_0 + 1;
			int localCloneEdgeIdx_0 = nLastEdgeBase + CuttedEdgeFlagOnCuda[curLineIdx];
			int localCloneEdgeIdx_1 = localCloneEdgeIdx_0 + 1;

			//clone vertex 0
			MC_Vertex_Cuda& clonePoint0 = curVertexSet[localCloneVertexIdx_0];
			MC_Vertex_Cuda& srcPoint0 = curVertexSet[refEdge.m_Vertex[0]];

			clonePoint0.m_isValid = true;//有效
			clonePoint0.m_isJoint = true;//不参与本次分裂
			clonePoint0.m_isSplit = false;
			clonePoint0.m_MeshVertex2CellId = srcPoint0.m_MeshVertex2CellId;

			{//makeScalarPoint
				clonePoint0.m_VertexPos[0] = srcPoint0.m_VertexPos[0] + (refEdge.m_intersectPos[0] - srcPoint0.m_VertexPos[0]) * CloneScalarFactor;
				clonePoint0.m_VertexPos[1] = srcPoint0.m_VertexPos[1] + (refEdge.m_intersectPos[1] - srcPoint0.m_VertexPos[1]) * CloneScalarFactor;
				clonePoint0.m_VertexPos[2] = srcPoint0.m_VertexPos[2] + (refEdge.m_intersectPos[2] - srcPoint0.m_VertexPos[2]) * CloneScalarFactor;
			}

			clonePoint0.m_state = srcPoint0.m_state;
			clonePoint0.m_CloneVertexIdx[0] = clonePoint0.m_CloneVertexIdx[1] = InValidIdx;
			clonePoint0.m_nVertexId = localCloneVertexIdx_0;
			refEdge.m_CloneIntersectVertexIdx[0] = localCloneVertexIdx_0;

			//clone line 0
			MC_Edge_Cuda& cloneLine0 = curLineSet[localCloneEdgeIdx_0];
			cloneLine0.m_isValid = true;cloneLine0.m_isJoint = true;cloneLine0.m_isCut = false;cloneLine0.m_hasClone = false;
			cloneLine0.m_CloneIntersectVertexIdx[0] = cloneLine0.m_CloneIntersectVertexIdx[1] = InValidIdx;
			cloneLine0.m_CloneEdgeIdx[0] = cloneLine0.m_CloneEdgeIdx[1] = InValidIdx;
			cloneLine0.m_Vertex[0] = refEdge.m_Vertex[0];
			cloneLine0.m_Vertex[1] = clonePoint0.m_nVertexId;
			cloneLine0.m_state = 1;
			//printf("%d,%d\n",m_vec_MCEdge[l].m_Vertex[0],curLine.m_Vertex[1]);
			{//for (unsigned q=0;q<MaxLineShareTri;++q)
				{
					cloneLine0.m_belongToTri[0] = refEdge.m_belongToTri[0];
					cloneLine0.m_belongToTriVertexIdx[0][0] = refEdge.m_belongToTriVertexIdx[0][0];
					cloneLine0.m_belongToTriVertexIdx[0][1] = refEdge.m_belongToTriVertexIdx[0][1];
					//printf("%d,%d\n",curLine.m_Vertex[0],curLine.m_Vertex[1]);
					cloneLine0.m_belongToTri[1] = refEdge.m_belongToTri[1];
					cloneLine0.m_belongToTriVertexIdx[1][0] = refEdge.m_belongToTriVertexIdx[1][0];
					cloneLine0.m_belongToTriVertexIdx[1][1] = refEdge.m_belongToTriVertexIdx[1][1];
					//printf("%d,%d\n",curLine.m_Vertex[0],curLine.m_Vertex[1]);
					cloneLine0.m_belongToTri[2] = refEdge.m_belongToTri[2];
					cloneLine0.m_belongToTriVertexIdx[2][0] = refEdge.m_belongToTriVertexIdx[2][0];
					cloneLine0.m_belongToTriVertexIdx[2][1] = refEdge.m_belongToTriVertexIdx[2][1];
					//printf("%d,%d\n",curLine.m_Vertex[0],curLine.m_Vertex[1]);
				}
			}
			
				
			cloneLine0.m_hasClone = false;
			cloneLine0.m_nLineId = localCloneEdgeIdx_0;
			//curLineSet[cloneLine0.m_nLineId] = cloneLine0;	
			refEdge.m_CloneEdgeIdx[0] = cloneLine0.m_nLineId;

			//clone vertex 1
			MC_Vertex_Cuda& clonePoint1 = curVertexSet[localCloneVertexIdx_1];
			MC_Vertex_Cuda& srcPoint1 = curVertexSet[refEdge.m_Vertex[1]];

			clonePoint1.m_isValid = true;//有效
			clonePoint1.m_isJoint = true;//不参与本次分裂
			clonePoint1.m_isSplit = false;
			clonePoint1.m_MeshVertex2CellId = srcPoint1.m_MeshVertex2CellId;
			
			{//makeScalarPoint
				clonePoint1.m_VertexPos[0] = srcPoint1.m_VertexPos[0] + (refEdge.m_intersectPos[0] - srcPoint1.m_VertexPos[0]) * CloneScalarFactor;
				clonePoint1.m_VertexPos[1] = srcPoint1.m_VertexPos[1] + (refEdge.m_intersectPos[1] - srcPoint1.m_VertexPos[1]) * CloneScalarFactor;
				clonePoint1.m_VertexPos[2] = srcPoint1.m_VertexPos[2] + (refEdge.m_intersectPos[2] - srcPoint1.m_VertexPos[2]) * CloneScalarFactor;
			}
			clonePoint1.m_state = srcPoint1.m_state;
			clonePoint1.m_CloneVertexIdx[0] = clonePoint1.m_CloneVertexIdx[1] = InValidIdx;			
			clonePoint1.m_nVertexId = localCloneVertexIdx_1;
			refEdge.m_CloneIntersectVertexIdx[1] = localCloneVertexIdx_1;

			//clone line 1
			MC_Edge_Cuda& cloneLine1 = curLineSet[localCloneEdgeIdx_1];
			cloneLine1.m_isValid = true;cloneLine1.m_isJoint = true;cloneLine1.m_isCut = false;cloneLine1.m_hasClone = false;
			cloneLine1.m_CloneIntersectVertexIdx[0] = cloneLine1.m_CloneIntersectVertexIdx[1] = InValidIdx;
			cloneLine1.m_CloneEdgeIdx[0] = cloneLine1.m_CloneEdgeIdx[1] = InValidIdx;
			cloneLine1.m_Vertex[0] = clonePoint1.m_nVertexId;
			cloneLine1.m_Vertex[1] = refEdge.m_Vertex[1];
			cloneLine1.m_state = -1;

			{//for (unsigned q=0;q<MaxLineShareTri;++q)
				{
					cloneLine1.m_belongToTri[0] = refEdge.m_belongToTri[0];
					cloneLine1.m_belongToTriVertexIdx[0][0] = refEdge.m_belongToTriVertexIdx[0][0];
					cloneLine1.m_belongToTriVertexIdx[0][1] = refEdge.m_belongToTriVertexIdx[0][1];
					//printf("%d,%d\n",curLine.m_Vertex[0],curLine.m_Vertex[1]);
					cloneLine1.m_belongToTri[1] = refEdge.m_belongToTri[1];
					cloneLine1.m_belongToTriVertexIdx[1][0] = refEdge.m_belongToTriVertexIdx[1][0];
					cloneLine1.m_belongToTriVertexIdx[1][1] = refEdge.m_belongToTriVertexIdx[1][1];
					//printf("%d,%d\n",curLine.m_Vertex[0],curLine.m_Vertex[1]);
					cloneLine1.m_belongToTri[2] = refEdge.m_belongToTri[2];
					cloneLine1.m_belongToTriVertexIdx[2][0] = refEdge.m_belongToTriVertexIdx[2][0];
					cloneLine1.m_belongToTriVertexIdx[2][1] = refEdge.m_belongToTriVertexIdx[2][1];
					//printf("%d,%d\n",curLine.m_Vertex[0],curLine.m_Vertex[1]);
				}
			}

			cloneLine1.m_hasClone = false;
			cloneLine1.m_nLineId = localCloneEdgeIdx_1;
			refEdge.m_CloneEdgeIdx[1] = cloneLine1.m_nLineId;

			refEdge.m_hasClone = true;

			//for mesh cutting 
			clonePoint1.m_brotherPoint = clonePoint0.m_nVertexId;
			clonePoint0.m_brotherPoint = clonePoint1.m_nVertexId;
		}
	}
}


/*
int g_nMCVertexSize=0,g_nMaxMCVertexSize=0,g_nLastVertexSize=0;
int g_nMCEdgeSize=0,g_nMaxMCEdgeSize=0,g_nLastEdgeSize=0;
int g_nMCSurfaceSize=0,g_nMaxMCSurfaceSize=0,g_nLastSurfaceSize=0
MC_Vertex_Cuda*		g_MC_Vertex_Cuda=0;
MC_Edge_Cuda*		g_MC_Edge_Cuda=0;
MC_Surface_Cuda*	g_MC_Surface_Cuda=0;
*/

__global__ void cuda_computeSurfaceSplitType(const int nLineSize,MC_Edge_Cuda* curLineSet,
											 const int nTriSize,MC_Surface_Cuda* curFaceSet,
											 int *splittedFaceFlagOnCuda)
{
	int curFaceIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (curFaceIdx < nTriSize)
	{
		MC_Surface_Cuda& curFace = curFaceSet[curFaceIdx];
		if (curFace.m_isValid)
		{
			curFace.m_state =   (char)(curLineSet[curFace.m_Lines[0]].m_isCut ? 1 : 0) + 
										   (curLineSet[curFace.m_Lines[1]].m_isCut ? 1 : 0) + 
										   (curLineSet[curFace.m_Lines[2]].m_isCut ? 1 : 0);
			if (1 == (int)curFace.m_state)
			{
				splittedFaceFlagOnCuda[curFaceIdx] = 2;
			}
			else if (2 == (int)curFace.m_state)
			{
				splittedFaceFlagOnCuda[curFaceIdx] = 3;
			}
		}
	}
}
#if 1
__global__ void cuda_SpliteSurface(const int nVertexSize,MC_Vertex_Cuda* curVertexSet,
								   const int nLineSize,MC_Edge_Cuda* curLineSet,
								   const int nTriSize,MC_Surface_Cuda* curFaceSet,
								   int *splittedFaceFlagOnCuda,int nVertexBase,int nEdgeBase,int nFaceBase)
{
	int curFaceIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (curFaceIdx < nTriSize)
	{
		MC_Surface_Cuda& curface = curFaceSet[curFaceIdx];
		if (curface.m_isValid && 1 == curface.m_state)
		{
			int localSpliteFaceIdx_0 = nFaceBase + splittedFaceFlagOnCuda[curFaceIdx];
			int localCloneEdgeIdx_0 = nEdgeBase + splittedFaceFlagOnCuda[curFaceIdx];
			int localSpliteFaceIdx_1 = localSpliteFaceIdx_0 + 1;
			int localCloneEdgeIdx_1 = localCloneEdgeIdx_0 + 1;

			curface.m_isValid = false;
			curface.m_isJoint = true;
			const unsigned nCuttingEdgeIdx = 0 * (curLineSet[curface.m_Lines[0]].m_isCut ? 1 : 0) + 
											 1 * (curLineSet[curface.m_Lines[1]].m_isCut ? 1 : 0) + 
											 2 * (curLineSet[curface.m_Lines[2]].m_isCut ? 1 : 0);
			MC_Edge_Cuda & cuttingEdge = curLineSet[curface.m_Lines[nCuttingEdgeIdx]];
			//Q_ASSERT(cuttingEdge.m_hasClone);
			{
				//clone surface 0
				const unsigned pt0 = nCuttingEdgeIdx;
				unsigned pt1 = cuttingEdge.m_belongToTriVertexIdx[nCuttingEdgeIdx][0];
				unsigned pt2 = 3 - pt0 - pt1;

				MC_Edge_Cuda& clone_line0 = curLineSet[cuttingEdge.m_CloneEdgeIdx[0]];
				MC_Vertex_Cuda & clonePt0 = curVertexSet[cuttingEdge.m_CloneIntersectVertexIdx[0]];

				MC_Surface_Cuda& newFace0 = curFaceSet[localSpliteFaceIdx_0];
				newFace0.m_isValid = true;
				newFace0.m_isJoint = true;
				newFace0.m_Vertex[pt0] = curface.m_Vertex[pt0];
				newFace0.m_Vertex[pt1] = curface.m_Vertex[pt1];
				newFace0.m_Vertex[pt2] = clonePt0.m_nVertexId;
				newFace0.m_Lines[pt0] = clone_line0.m_nLineId;
				newFace0.m_VertexNormal[0] = curface.m_VertexNormal[0];
				newFace0.m_VertexNormal[1] = curface.m_VertexNormal[1];
				newFace0.m_VertexNormal[2] = curface.m_VertexNormal[2];
				newFace0.m_state = 0;
				//newFace0.m_nParentId = curface.m_nSurfaceId;
				

				//printf("line id %d\n",curface.m_Lines[pt1]);
				//Q_ASSERT(m_vec_MCEdge[curface.m_Lines[pt1]].m_hasClone == false);
				MC_Edge_Cuda& cloneLine1 = curLineSet[localCloneEdgeIdx_0];
				cloneLine1.m_isValid = true;cloneLine1.m_isCut = false;cloneLine1.m_isJoint = true;cloneLine1.m_hasClone = false;
				cloneLine1.m_state = 0;cloneLine1.m_CloneIntersectVertexIdx[0] = cloneLine1.m_CloneIntersectVertexIdx[1] = InValidIdx;
				cloneLine1.m_Vertex[0] = newFace0.m_Vertex[pt2];cloneLine1.m_Vertex[1] = newFace0.m_Vertex[pt0];
				cloneLine1.m_belongToTriVertexIdx[pt1][0] = pt2;cloneLine1.m_belongToTriVertexIdx[pt1][1] = pt0;
				cloneLine1.m_nLineId = localCloneEdgeIdx_0;
				//m_vec_MCEdge[cloneLine1.m_nLineId] = cloneLine1;
				newFace0.m_Lines[pt1] = cloneLine1.m_nLineId;

				newFace0.m_Lines[pt2] = curface.m_Lines[pt2];

				newFace0.m_nSurfaceId = localSpliteFaceIdx_0;
				curLineSet[newFace0.m_Lines[pt0]].m_belongToTri[pt0] = newFace0.m_nSurfaceId;
				curLineSet[newFace0.m_Lines[pt1]].m_belongToTri[pt1] = newFace0.m_nSurfaceId;
				curLineSet[newFace0.m_Lines[pt2]].m_belongToTri[pt2] = newFace0.m_nSurfaceId;
				//curFaceSet[newFace0.m_nSurfaceId] = newFace0;

				if (clonePt0.m_distanceToBlade > 0)
				{
					curface.m_nCloneBladeLine[0] = localCloneEdgeIdx_0;
					curface.m_nCloneBladeLine[1] = localCloneEdgeIdx_1;
				}
				else
				{
					curface.m_nCloneBladeLine[0] = localCloneEdgeIdx_1;
					curface.m_nCloneBladeLine[1] = localCloneEdgeIdx_0;
				}
				newFace0.m_nParentId4MC = curFaceIdx;
				
			}
			//clone Surface 1
			{
				const unsigned pt0 = nCuttingEdgeIdx;
				unsigned pt2 = cuttingEdge.m_belongToTriVertexIdx[nCuttingEdgeIdx][1];
				unsigned pt1 = 3 - pt0 - pt2;

				MC_Edge_Cuda& clone_line1 = curLineSet[cuttingEdge.m_CloneEdgeIdx[1]];
				MC_Vertex_Cuda & clonePt1 = curVertexSet[cuttingEdge.m_CloneIntersectVertexIdx[1]];

				MC_Surface_Cuda& newFace1 = curFaceSet[localSpliteFaceIdx_1];
				newFace1.m_isValid = true;
				newFace1.m_isJoint = true;
				newFace1.m_Vertex[pt0] = curface.m_Vertex[pt0];
				newFace1.m_Vertex[pt2] = curface.m_Vertex[pt2];
				newFace1.m_Vertex[pt1] = clonePt1.m_nVertexId;
				newFace1.m_VertexNormal[0] = curface.m_VertexNormal[0];
				newFace1.m_VertexNormal[1] = curface.m_VertexNormal[1];
				newFace1.m_VertexNormal[2] = curface.m_VertexNormal[2];
				newFace1.m_state = 0;
				newFace1.m_Lines[pt0] = clone_line1.m_nLineId;
				//newFace1.m_nParentId = curface.m_nSurfaceId;
				//printf("line id %d,face %d\n",curface.m_Lines[pt2],t);
				//Q_ASSERT(m_vec_MCEdge[curface.m_Lines[pt2]].m_hasClone == false);

				MC_Edge_Cuda& cloneLine1 = curLineSet[localCloneEdgeIdx_1];
				cloneLine1.m_isValid = true;cloneLine1.m_isCut = false;cloneLine1.m_isJoint = true;cloneLine1.m_hasClone = false;
				cloneLine1.m_state = 0;cloneLine1.m_CloneIntersectVertexIdx[0] = cloneLine1.m_CloneIntersectVertexIdx[1] = InValidIdx;
				cloneLine1.m_Vertex[0] = newFace1.m_Vertex[pt0];cloneLine1.m_Vertex[1] = newFace1.m_Vertex[pt1];
				cloneLine1.m_belongToTriVertexIdx[pt2][0] = pt0;cloneLine1.m_belongToTriVertexIdx[pt2][1] = pt1;
				cloneLine1.m_nLineId = localCloneEdgeIdx_1;
				//m_vec_MCEdge[cloneLine1.m_nLineId] = cloneLine1;
				newFace1.m_Lines[pt2] = cloneLine1.m_nLineId;
				newFace1.m_Lines[pt1] = curface.m_Lines[pt1];

				newFace1.m_nSurfaceId = localSpliteFaceIdx_1;
				curLineSet[newFace1.m_Lines[pt0]].m_belongToTri[pt0] = newFace1.m_nSurfaceId;
				curLineSet[newFace1.m_Lines[pt1]].m_belongToTri[pt1] = newFace1.m_nSurfaceId;
				curLineSet[newFace1.m_Lines[pt2]].m_belongToTri[pt2] = newFace1.m_nSurfaceId;
				//m_vec_MCSurface[newFace1.m_nSurfaceId] = newFace1;

				//Mesh Cutting
				newFace1.m_nParentId4MC = InValidIdx;
			}
		}
		else if (curface.m_isValid && 2 == curface.m_state)
		{
			int localSpliteFaceIdx_0 = nFaceBase + splittedFaceFlagOnCuda[curFaceIdx];
			int localCloneEdgeIdx_0 = nEdgeBase + splittedFaceFlagOnCuda[curFaceIdx];
			int localSpliteFaceIdx_1 = localSpliteFaceIdx_0 + 1;
			int localCloneEdgeIdx_1 = localCloneEdgeIdx_0 + 1;
			int localSpliteFaceIdx_2 = localSpliteFaceIdx_1 + 1;
			int localCloneEdgeIdx_2 = localCloneEdgeIdx_1 + 1;

			curface.m_isValid = false;
			curface.m_isJoint = true;
			const unsigned nNoCuttingEdgeIdx =  0 * (curLineSet[curface.m_Lines[0]].m_isCut ? 0 : 1) + 
												1 * (curLineSet[curface.m_Lines[1]].m_isCut ? 0 : 1) + 
												2 * (curLineSet[curface.m_Lines[2]].m_isCut ? 0 : 1);
			
			MC_Edge_Cuda & noCuttingEdge = curLineSet[curface.m_Lines[nNoCuttingEdgeIdx]];
			const unsigned nCuttingEdgeIdx0 = noCuttingEdge.m_belongToTriVertexIdx[nNoCuttingEdgeIdx][0];
			const unsigned nCuttingEdgeIdx1 = noCuttingEdge.m_belongToTriVertexIdx[nNoCuttingEdgeIdx][1];

			unsigned idxUp[2];
			unsigned idxDown[2];
			const unsigned pt0 = nNoCuttingEdgeIdx;
			unsigned pt1 = nCuttingEdgeIdx1;
			unsigned pt2 = nCuttingEdgeIdx0;


			MC_Edge_Cuda & cuttingEdge2 = curLineSet[curface.m_Lines[pt2]];
			MC_Edge_Cuda & cuttingEdge1 = curLineSet[curface.m_Lines[pt1]];
			if (nNoCuttingEdgeIdx == cuttingEdge2.m_belongToTriVertexIdx[pt2][0])
			{
				idxUp[0] = 0;
				idxDown[0] = 1;
			}
			else if (nNoCuttingEdgeIdx == cuttingEdge2.m_belongToTriVertexIdx[pt2][1])
			{
				idxUp[0] = 1;
				idxDown[0] = 0;
			}
			else
			{
				CUPRINTF("ERROR 3145\n");do{}while(true);
			}

			if (nNoCuttingEdgeIdx == cuttingEdge1.m_belongToTriVertexIdx[pt1][0])
			{
				idxUp[1] = 0;
				idxDown[1] = 1;
			}
			else if (nNoCuttingEdgeIdx == cuttingEdge1.m_belongToTriVertexIdx[pt1][1])
			{
				idxUp[1] = 1;
				idxDown[1] = 0;
			}
			else
			{
				CUPRINTF("ERROR 3160\n");do{}while(true);
			}
			{
				//clone surface 0
			
				MC_Vertex_Cuda & clonePt1 = curVertexSet[cuttingEdge2.m_CloneIntersectVertexIdx[idxUp[0]]];
				MC_Vertex_Cuda & clonePt2 = curVertexSet[cuttingEdge1.m_CloneIntersectVertexIdx[idxUp[1]]];

				MC_Surface_Cuda& newFace0 = curFaceSet[localSpliteFaceIdx_0];
				newFace0.m_isValid = true;
				newFace0.m_isJoint = true;
				newFace0.m_Vertex[pt0] = curface.m_Vertex[pt0];
				newFace0.m_Vertex[pt1] = clonePt1.m_nVertexId;
				newFace0.m_Vertex[pt2] = clonePt2.m_nVertexId;
				newFace0.m_VertexNormal[0] = curface.m_VertexNormal[0];
				newFace0.m_VertexNormal[1] = curface.m_VertexNormal[1];
				newFace0.m_VertexNormal[2] = curface.m_VertexNormal[2];
				newFace0.m_state = 0;
				
				if (clonePt1.m_distanceToBlade > 0)
				{
					curface.m_nCloneBladeLine[0] = localCloneEdgeIdx_0;
					curface.m_nCloneBladeLine[1] = localCloneEdgeIdx_2;
				}
				else
				{
					curface.m_nCloneBladeLine[0] = localCloneEdgeIdx_2;
					curface.m_nCloneBladeLine[1] = localCloneEdgeIdx_0;
				}

				MC_Edge_Cuda& cloneLine0 = curLineSet[localCloneEdgeIdx_0];
				cloneLine0.m_isValid = true;cloneLine0.m_isCut = false;cloneLine0.m_isJoint = true;cloneLine0.m_hasClone = false;
				cloneLine0.m_state = 0;cloneLine0.m_CloneIntersectVertexIdx[0] = cloneLine0.m_CloneIntersectVertexIdx[1] = InValidIdx;
				cloneLine0.m_Vertex[0] = newFace0.m_Vertex[pt2];cloneLine0.m_Vertex[1] = newFace0.m_Vertex[pt1];
				//cloneLine0.m_belongToTri[pt0] = xxx;
				cloneLine0.m_belongToTriVertexIdx[pt0][0] = pt2;cloneLine0.m_belongToTriVertexIdx[pt0][1] = pt1;
				cloneLine0.m_nLineId = localCloneEdgeIdx_0;
				//m_vec_MCEdge[cloneLine0.m_nLineId] = cloneLine0;
				newFace0.m_Lines[pt0] = cloneLine0.m_nLineId;

				MC_Edge_Cuda& clone_line2 = curLineSet[cuttingEdge2.m_CloneEdgeIdx[idxUp[0]]];
				MC_Edge_Cuda& clone_line1 = curLineSet[cuttingEdge1.m_CloneEdgeIdx[idxUp[1]]];
				newFace0.m_Lines[pt1] = clone_line1.m_nLineId;

				newFace0.m_Lines[pt2] = clone_line2.m_nLineId;

				newFace0.m_nSurfaceId = localSpliteFaceIdx_0;
				curLineSet[newFace0.m_Lines[pt0]].m_belongToTri[pt0] = newFace0.m_nSurfaceId;
				curLineSet[newFace0.m_Lines[pt1]].m_belongToTri[pt1] = newFace0.m_nSurfaceId;
				curLineSet[newFace0.m_Lines[pt2]].m_belongToTri[pt2] = newFace0.m_nSurfaceId;
				//m_vec_MCSurface[newFace0.m_nSurfaceId] = newFace0;

				newFace0.m_nParentId4MC = curFaceIdx;
			}
			{
				//clone surface 1
				MC_Vertex_Cuda & clonePt1 = curVertexSet[cuttingEdge2.m_CloneIntersectVertexIdx[idxDown[0]]];
				MC_Vertex_Cuda & clonePt2 = curVertexSet[cuttingEdge1.m_CloneIntersectVertexIdx[idxDown[1]]];
				unsigned clone1ID;
				{
					MC_Surface_Cuda& newFace0 = curFaceSet[localSpliteFaceIdx_1];
					newFace0.m_isValid = true;
					newFace0.m_isJoint = true;
					newFace0.m_Vertex[pt0] = clonePt1.m_nVertexId;
					newFace0.m_Vertex[pt1] = curface.m_Vertex[pt1];
					newFace0.m_Vertex[pt2] = curface.m_Vertex[pt2];
					newFace0.m_VertexNormal[0] = curface.m_VertexNormal[0];
					newFace0.m_VertexNormal[1] = curface.m_VertexNormal[1];
					newFace0.m_VertexNormal[2] = curface.m_VertexNormal[2];
					newFace0.m_state = 0;
					

					MC_Edge_Cuda& clone_line2 = curLineSet[cuttingEdge2.m_CloneEdgeIdx[idxDown[0]]];
					//MC_Edge& clone_line1 = m_vec_MCEdge[cuttingEdge1.m_CloneEdgeIdx[idxDown[1]]];
					newFace0.m_Lines[pt2] = clone_line2.m_nLineId;

					MC_Edge_Cuda& cloneLine1 = curLineSet[localCloneEdgeIdx_1];
					cloneLine1.m_isValid = true;cloneLine1.m_isCut = false;cloneLine1.m_isJoint = true;cloneLine1.m_hasClone = false;
					cloneLine1.m_state = 0;cloneLine1.m_CloneIntersectVertexIdx[0] = cloneLine1.m_CloneIntersectVertexIdx[1] = InValidIdx;
					cloneLine1.m_Vertex[0] = newFace0.m_Vertex[pt0];cloneLine1.m_Vertex[1] = newFace0.m_Vertex[pt2];
					cloneLine1.m_belongToTriVertexIdx[pt1][0] = pt0;cloneLine1.m_belongToTriVertexIdx[pt1][1] = pt2;
					cloneLine1.m_nLineId = localCloneEdgeIdx_1;
					//m_vec_MCEdge[cloneLine1.m_nLineId] = cloneLine1;
					newFace0.m_Lines[pt1] = cloneLine1.m_nLineId;
					clone1ID = cloneLine1.m_nLineId;

					newFace0.m_Lines[pt0] = curface.m_Lines[pt0];

					newFace0.m_nSurfaceId = localSpliteFaceIdx_1;
					curLineSet[newFace0.m_Lines[pt0]].m_belongToTri[pt0] = newFace0.m_nSurfaceId;
					curLineSet[newFace0.m_Lines[pt1]].m_belongToTri[pt1] = newFace0.m_nSurfaceId;
					curLineSet[newFace0.m_Lines[pt2]].m_belongToTri[pt2] = newFace0.m_nSurfaceId;
					//m_vec_MCSurface[newFace0.m_nSurfaceId] = newFace0;

					newFace0.m_nParentId4MC = InValidIdx;
				}

				//clone surface 2
				{	
					MC_Surface_Cuda& newFace0 = curFaceSet[localSpliteFaceIdx_2];
					newFace0.m_isValid = true;
					newFace0.m_isJoint = true;
					newFace0.m_Vertex[pt0] = clonePt2.m_nVertexId;
					newFace0.m_Vertex[pt1] = clonePt1.m_nVertexId;
					newFace0.m_Vertex[pt2] = curface.m_Vertex[pt2];
					newFace0.m_VertexNormal[0] = curface.m_VertexNormal[0];
					newFace0.m_VertexNormal[1] = curface.m_VertexNormal[1];
					newFace0.m_VertexNormal[2] = curface.m_VertexNormal[2];
					newFace0.m_state = 0;
					

					MC_Edge_Cuda& cloneLine2 = curLineSet[localCloneEdgeIdx_2];
					cloneLine2.m_isValid = true;cloneLine2.m_isCut = false;cloneLine2.m_isJoint = true;cloneLine2.m_hasClone = false;
					cloneLine2.m_state = 0;cloneLine2.m_CloneIntersectVertexIdx[0] = cloneLine2.m_CloneIntersectVertexIdx[1] = InValidIdx;
					cloneLine2.m_Vertex[0] = newFace0.m_Vertex[pt1];cloneLine2.m_Vertex[1] = newFace0.m_Vertex[pt0];
					cloneLine2.m_belongToTriVertexIdx[pt2][0] = pt1;cloneLine2.m_belongToTriVertexIdx[pt2][1] = pt0;
					cloneLine2.m_nLineId = localCloneEdgeIdx_2;
					//m_vec_MCEdge[cloneLine2.m_nLineId] = cloneLine2;
					newFace0.m_Lines[pt2] = cloneLine2.m_nLineId;

					MC_Edge_Cuda& clone_line1 = curLineSet[cuttingEdge1.m_CloneEdgeIdx[idxDown[1]]];
					newFace0.m_Lines[pt1] = clone_line1.m_nLineId;

					newFace0.m_Lines[pt0] = clone1ID;
					MC_Edge_Cuda& line0 = curLineSet[clone1ID];
					line0.m_belongToTriVertexIdx[pt0][0] = pt2;line0.m_belongToTriVertexIdx[pt0][1] = pt1;

					newFace0.m_nSurfaceId = localSpliteFaceIdx_2;
					curLineSet[newFace0.m_Lines[pt0]].m_belongToTri[pt0] = newFace0.m_nSurfaceId;
					curLineSet[newFace0.m_Lines[pt1]].m_belongToTri[pt1] = newFace0.m_nSurfaceId;
					curLineSet[newFace0.m_Lines[pt2]].m_belongToTri[pt2] = newFace0.m_nSurfaceId;
					line0.m_belongToTri[pt0] = newFace0.m_nSurfaceId;
					//m_vec_MCSurface[newFace0.m_nSurfaceId] = newFace0;

					newFace0.m_nParentId4MC = InValidIdx;
				}
			}
		}
	}
}
#endif

__global__ void cuda_computeVertexUpDown(const int nVertexSize,MC_Vertex_Cuda* curVertexSet,ValueTypePtr vecBlade,int nBladeBase, ValueTypePtr bladeNormal)
{
	int curVertexIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (curVertexIdx < nVertexSize)
	{
		MC_Vertex_Cuda& curVertex = curVertexSet[curVertexIdx];
		float3 PV1 = make_float3(vecBlade[nBladeBase+0],vecBlade[nBladeBase+1],vecBlade[nBladeBase+2]);
		float3 PV2 = make_float3(vecBlade[nBladeBase+3],vecBlade[nBladeBase+4],vecBlade[nBladeBase+5]);
		float3 PV3 = make_float3(vecBlade[nBladeBase+6],vecBlade[nBladeBase+7],vecBlade[nBladeBase+8]);

		float3 normal = cross(make_float3(PV2.x - PV1.x,PV2.y - PV1.y,PV2.z - PV1.z),make_float3(PV3.x - PV1.x,PV3.y - PV1.y,PV3.z - PV1.z));
		float3 tmpVec = make_float3(curVertex.m_VertexPos[0]-vecBlade[nBladeBase+0],curVertex.m_VertexPos[1]-vecBlade[nBladeBase+1],curVertex.m_VertexPos[2]-vecBlade[nBladeBase+2]);
		
		curVertex.m_distanceToBlade = normal.x * tmpVec.x + normal.y * tmpVec.y + normal.z * tmpVec.z;

		/*curVertex.m_distanceToBlade = checkPointPlaneOnCuda(
								curVertex.m_VertexPos[0], curVertex.m_VertexPos[1], curVertex.m_VertexPos[2],
								vecBlade[nBladeBase+0], vecBlade[nBladeBase+1], vecBlade[nBladeBase+2],
								bladeNormal[0], bladeNormal[1], bladeNormal[2]);*/
	}
}

__global__ void cuda_distributeNewCloneVertexToCloneCell(
	    const int /*nLastVertexSize*/, const int nMCVertexSize,MC_Vertex_Cuda* curVertexSet,const int nCellSize,EFGCellOnCuda *CellOnCudaPtr,const int nVertexSize,VertexOnCuda* VertexOnCudaPtr)
{
	//g_nLastVertexSize, g_nMCVertexSize,g_MC_Vertex_Cuda,EFG_CellOnCudaElementCount,EFG_CellOnCudaPtr);
	int curVertexIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if ( curVertexIdx <  nMCVertexSize)
	{
		MC_Vertex_Cuda& curVertex = curVertexSet[curVertexIdx];
		
		
		if (CellOnCudaPtr[curVertex.m_MeshVertex2CellId].m_bTopLevelOctreeNodeList)
		{
			if (curVertex.m_distanceToBlade > 0)
			{
				EFGCellOnCuda& srcCell = CellOnCudaPtr[curVertex.m_MeshVertex2CellId];
				for (int v=0;v<8;++v)
				{
					curVertex.m_elemVertexRelatedDofs[3*v+0] = VertexOnCudaPtr[srcCell.vertexId[v]].m_nDof[0];
					curVertex.m_elemVertexRelatedDofs[3*v+1] = VertexOnCudaPtr[srcCell.vertexId[v]].m_nDof[1];
					curVertex.m_elemVertexRelatedDofs[3*v+2] = VertexOnCudaPtr[srcCell.vertexId[v]].m_nDof[2];
				}
				float * p0 = &VertexOnCudaPtr[srcCell.vertexId[0]].local[0];
				float * p7 = &VertexOnCudaPtr[srcCell.vertexId[7]].local[0];
			
				float detaX = (p7[0] - curVertex.m_VertexPos[0]) / (p7[0] - p0[0]);
				float detaY = (p7[1] - curVertex.m_VertexPos[1]) / (p7[1] - p0[1]);
				float detaZ = (p7[2] - curVertex.m_VertexPos[2]) / (p7[2] - p0[2]);

				curVertex.m_TriLinearWeight[0] = 1.f*detaX * detaY * detaZ;
				curVertex.m_TriLinearWeight[1] = 1.f*(1-detaX) * detaY * detaZ;
				curVertex.m_TriLinearWeight[2] = 1.f*detaX * (1-detaY) * detaZ;
				curVertex.m_TriLinearWeight[3] = 1.f*(1-detaX) * (1-detaY) * detaZ;
				curVertex.m_TriLinearWeight[4] = 1.f*detaX * detaY * (1-detaZ);
				curVertex.m_TriLinearWeight[5] = 1.f*(1-detaX) * detaY * (1-detaZ);
				curVertex.m_TriLinearWeight[6] = 1.f*detaX * (1-detaY) * (1-detaZ);
				curVertex.m_TriLinearWeight[7] = 1.f*(1-detaX) * (1-detaY) * (1-detaZ);
				
				//if (curVertex.m_TriLinearWeight[0] < 0.001f && curVertex.m_TriLinearWeight[0] > -0.001f)
				//{
				//	CUPRINTF("Weight(%f,%f,%f,%f,%f,%f,%f,%f)\n",curVertex.m_TriLinearWeight[0],curVertex.m_TriLinearWeight[1],curVertex.m_TriLinearWeight[2],curVertex.m_TriLinearWeight[3],curVertex.m_TriLinearWeight[4],curVertex.m_TriLinearWeight[5],curVertex.m_TriLinearWeight[6],curVertex.m_TriLinearWeight[7]);
				//	//CUPRINTF("p0[%f,%f,%f] p7[%f,%f,%f] deta[%f,%f,%f]\n",p0[0],p0[1],p0[2],p7[0],p7[1],p7[2],detaX,detaY,detaZ);
				//}
			}
			else
			{
				EFGCellOnCuda& srcCell_tmp = CellOnCudaPtr[curVertex.m_MeshVertex2CellId];
				EFGCellOnCuda& cloneCell = CellOnCudaPtr[srcCell_tmp.m_nCloneCellIdx];
				curVertex.m_MeshVertex2CellId = srcCell_tmp.m_nCloneCellIdx;

				for (int v=0;v<8;++v)
				{
					curVertex.m_elemVertexRelatedDofs[3*v+0] = VertexOnCudaPtr[cloneCell.vertexId[v]].m_nDof[0];
					curVertex.m_elemVertexRelatedDofs[3*v+1] = VertexOnCudaPtr[cloneCell.vertexId[v]].m_nDof[1];
					curVertex.m_elemVertexRelatedDofs[3*v+2] = VertexOnCudaPtr[cloneCell.vertexId[v]].m_nDof[2];
				}
				float * p0 = &VertexOnCudaPtr[cloneCell.vertexId[0]].local[0];
				float * p7 = &VertexOnCudaPtr[cloneCell.vertexId[7]].local[0];
			
				float detaX = (p7[0] - curVertex.m_VertexPos[0]) / (p7[0] - p0[0]);
				float detaY = (p7[1] - curVertex.m_VertexPos[1]) / (p7[1] - p0[1]);
				float detaZ = (p7[2] - curVertex.m_VertexPos[2]) / (p7[2] - p0[2]);

				curVertex.m_TriLinearWeight[0] = 1.f*detaX * detaY * detaZ;
				curVertex.m_TriLinearWeight[1] = 1.f*(1-detaX) * detaY * detaZ;
				curVertex.m_TriLinearWeight[2] = 1.f*detaX * (1-detaY) * detaZ;
				curVertex.m_TriLinearWeight[3] = 1.f*(1-detaX) * (1-detaY) * detaZ;
				curVertex.m_TriLinearWeight[4] = 1.f*detaX * detaY * (1-detaZ);
				curVertex.m_TriLinearWeight[5] = 1.f*(1-detaX) * detaY * (1-detaZ);
				curVertex.m_TriLinearWeight[6] = 1.f*detaX * (1-detaY) * (1-detaZ);
				curVertex.m_TriLinearWeight[7] = 1.f*(1-detaX) * (1-detaY) * (1-detaZ);

				//if (curVertex.m_TriLinearWeight[0] < 0.001f && curVertex.m_TriLinearWeight[0] > -0.001f)
				//{
				//	CUPRINTF("Weight(%f,%f,%f,%f,%f,%f,%f,%f)\n",curVertex.m_TriLinearWeight[0],curVertex.m_TriLinearWeight[1],curVertex.m_TriLinearWeight[2],curVertex.m_TriLinearWeight[3],curVertex.m_TriLinearWeight[4],curVertex.m_TriLinearWeight[5],curVertex.m_TriLinearWeight[6],curVertex.m_TriLinearWeight[7]);
				//	//CUPRINTF("p0[%f,%f,%f] p7[%f,%f,%f] deta[%f,%f,%f]\n",p0[0],p0[1],p0[2],p7[0],p7[1],p7[2],detaX,detaY,detaZ);
				//}
			}
		}
		else
		{
			EFGCellOnCuda& srcCell = CellOnCudaPtr[curVertex.m_MeshVertex2CellId];
			for (int v=0;v<8;++v)
			{
				curVertex.m_elemVertexRelatedDofs[3*v+0] = VertexOnCudaPtr[srcCell.vertexId[v]].m_nDof[0];
				curVertex.m_elemVertexRelatedDofs[3*v+1] = VertexOnCudaPtr[srcCell.vertexId[v]].m_nDof[1];
				curVertex.m_elemVertexRelatedDofs[3*v+2] = VertexOnCudaPtr[srcCell.vertexId[v]].m_nDof[2];
			}
			float * p0 = &VertexOnCudaPtr[srcCell.vertexId[0]].local[0];
			float * p7 = &VertexOnCudaPtr[srcCell.vertexId[7]].local[0];
			
			float detaX = (p7[0] - curVertex.m_VertexPos[0]) / (p7[0] - p0[0]);
			float detaY = (p7[1] - curVertex.m_VertexPos[1]) / (p7[1] - p0[1]);
			float detaZ = (p7[2] - curVertex.m_VertexPos[2]) / (p7[2] - p0[2]);

			curVertex.m_TriLinearWeight[0] = 1.f*detaX * detaY * detaZ;
			curVertex.m_TriLinearWeight[1] = 1.f*(1-detaX) * detaY * detaZ;
			curVertex.m_TriLinearWeight[2] = 1.f*detaX * (1-detaY) * detaZ;
			curVertex.m_TriLinearWeight[3] = 1.f*(1-detaX) * (1-detaY) * detaZ;
			curVertex.m_TriLinearWeight[4] = 1.f*detaX * detaY * (1-detaZ);
			curVertex.m_TriLinearWeight[5] = 1.f*(1-detaX) * detaY * (1-detaZ);
			curVertex.m_TriLinearWeight[6] = 1.f*detaX * (1-detaY) * (1-detaZ);
			curVertex.m_TriLinearWeight[7] = 1.f*(1-detaX) * (1-detaY) * (1-detaZ);
		}
	}
}

__global__ void cuda_TestSurface(const int nVertexSize,MC_Vertex_Cuda* curVertexSet,
								   const int nTriSize,MC_Surface_Cuda* curFaceSet)
{
	int curTriIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (curTriIdx < nTriSize)
	{
		MC_Surface_Cuda& curFace = curFaceSet[curTriIdx];
		MC_Vertex_Cuda& p0 = curVertexSet[curFace.m_Vertex[0]];
		MC_Vertex_Cuda& p1 = curVertexSet[curFace.m_Vertex[1]];
		MC_Vertex_Cuda& p2 = curVertexSet[curFace.m_Vertex[2]];

		if (curFace.m_isValid)
		{
			if ( (p0.m_distanceToBlade > 0 && p1.m_distanceToBlade > 0 && p2.m_distanceToBlade > 0 ) || (p0.m_distanceToBlade < 0 && p1.m_distanceToBlade < 0 && p2.m_distanceToBlade < 0 ))
			{}
			else
			{
				CUPRINTF("cuda_TestSurface (curTriIdx[%d])\n",curTriIdx);
			}
		}

	}
}

__global__ void cuda_ComputeCuttingMeshMassCenter(const int nLastVertexSize,const int nVertexSize,MC_Vertex_Cuda* curVertexSet,
												  float * CuttingFaceUp_X_Cuda,float * CuttingFaceUp_Y_Cuda,float * CuttingFaceUp_Z_Cuda,
												  float * CuttingFaceDown_X_Cuda,float * CuttingFaceDown_Y_Cuda,float * CuttingFaceDown_Z_Cuda,
												  int * CuttingFaceUpFlagCuda, int * CuttingFaceDownFlagCuda)
{
	const int curVertexIdx = threadIdx.x + blockIdx.x * blockDim.x + nLastVertexSize;
	if (curVertexIdx < nVertexSize )
	{
		MC_Vertex_Cuda& curVertex = curVertexSet[curVertexIdx];
		if (curVertex.m_distanceToBlade < 0)
		{
			CuttingFaceDown_X_Cuda[curVertexIdx] = curVertex.m_VertexPos[0];
			CuttingFaceDown_Y_Cuda[curVertexIdx] = curVertex.m_VertexPos[1];
			CuttingFaceDown_Z_Cuda[curVertexIdx] = curVertex.m_VertexPos[2];
			CuttingFaceUpFlagCuda[curVertexIdx] = 1;
		}
		else
		{
			CuttingFaceUp_X_Cuda[curVertexIdx] = curVertex.m_VertexPos[0];
			CuttingFaceUp_Y_Cuda[curVertexIdx] = curVertex.m_VertexPos[1];
			CuttingFaceUp_Z_Cuda[curVertexIdx] = curVertex.m_VertexPos[2];
			CuttingFaceDownFlagCuda[curVertexIdx] = 1;
		}
	}
}

__global__ void cuda_makeCuttingFaceCenter2Cell(const int nCellSize,EFGCellOnCuda *CellOnCudaPtr,const int nVertexSize,VertexOnCuda* VertexOnCudaPtr,
												float UpX,float UpY,float UpZ,float DownX,float DownY,float DownZ,
												const int nMeshVertexSize,MC_Vertex_Cuda* curVertexSet)
{
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	const int nCurCellIdx = tid / 2;
	const int nCurUpDown = tid % 2;//0 : Down; 1 Up
	if (nCurCellIdx < nCellSize )
	{
		float FaceCenter[2][3] = {{DownX,DownY,DownZ},{UpX,UpY,UpZ}};
		EFGCellOnCuda& curCell = CellOnCudaPtr[nCurCellIdx];
		int nDstCellIdx = -1;
		if (curCell.m_bTopLevelOctreeNodeList)
		{
			if (0 == nCurUpDown)
			{
				//down 
				nDstCellIdx = curCell.m_nCloneCellIdx;
			}
			else
			{
				//up
				nDstCellIdx = nCurCellIdx;
			}

			EFGCellOnCuda& dstCell = CellOnCudaPtr[nDstCellIdx];

			float * p0 = &VertexOnCudaPtr[dstCell.vertexId[0]].local[0];
			float * p7 = &VertexOnCudaPtr[dstCell.vertexId[7]].local[0];
			float cellRadius = (p7[0] - p0[0]) / 2.f;
			if ( my_abs( (p7[0]+p0[0])/2.f- FaceCenter[nCurUpDown][0]) < cellRadius &&
				 my_abs( (p7[1]+p0[1])/2.f- FaceCenter[nCurUpDown][1]) < cellRadius &&
				 my_abs( (p7[2]+p0[2])/2.f- FaceCenter[nCurUpDown][2]) < cellRadius) 
			{
				MC_Vertex_Cuda& curVertex = curVertexSet[nMeshVertexSize+nCurUpDown];
				curVertex.m_isValid = true;//有效
				curVertex.m_isJoint = true;//不参与本次分裂
				curVertex.m_isSplit = false;
				curVertex.m_MeshVertex2CellId = nDstCellIdx;
				curVertex.m_VertexPos[0] = FaceCenter[nCurUpDown][0];
				curVertex.m_VertexPos[1] = FaceCenter[nCurUpDown][1];
				curVertex.m_VertexPos[2] = FaceCenter[nCurUpDown][2];
				curVertex.m_state = 0;
				curVertex.m_CloneVertexIdx[0] = curVertex.m_CloneVertexIdx[1] = InValidIdx;
				curVertex.m_nVertexId = nMeshVertexSize+nCurUpDown;
				curVertex.m_distanceToBlade = -0.5f + nCurUpDown;
	
				
				for (int v=0;v<8;++v)
				{
					curVertex.m_elemVertexRelatedDofs[3*v+0] = VertexOnCudaPtr[dstCell.vertexId[v]].m_nDof[0];
					curVertex.m_elemVertexRelatedDofs[3*v+1] = VertexOnCudaPtr[dstCell.vertexId[v]].m_nDof[1];
					curVertex.m_elemVertexRelatedDofs[3*v+2] = VertexOnCudaPtr[dstCell.vertexId[v]].m_nDof[2];
				}
				float * p0 = &VertexOnCudaPtr[dstCell.vertexId[0]].local[0];
				float * p7 = &VertexOnCudaPtr[dstCell.vertexId[7]].local[0];
			
				float detaX = (p7[0] - curVertex.m_VertexPos[0]) / (p7[0] - p0[0]);
				float detaY = (p7[1] - curVertex.m_VertexPos[1]) / (p7[1] - p0[1]);
				float detaZ = (p7[2] - curVertex.m_VertexPos[2]) / (p7[2] - p0[2]);

				curVertex.m_TriLinearWeight[0] = 1.f*detaX * detaY * detaZ;
				curVertex.m_TriLinearWeight[1] = 1.f*(1-detaX) * detaY * detaZ;
				curVertex.m_TriLinearWeight[2] = 1.f*detaX * (1-detaY) * detaZ;
				curVertex.m_TriLinearWeight[3] = 1.f*(1-detaX) * (1-detaY) * detaZ;
				curVertex.m_TriLinearWeight[4] = 1.f*detaX * detaY * (1-detaZ);
				curVertex.m_TriLinearWeight[5] = 1.f*(1-detaX) * detaY * (1-detaZ);
				curVertex.m_TriLinearWeight[6] = 1.f*detaX * (1-detaY) * (1-detaZ);
				curVertex.m_TriLinearWeight[7] = 1.f*(1-detaX) * (1-detaY) * (1-detaZ);

				curVertex.m_VertexNormal[0] = curVertex.m_VertexNormal[1] = curVertex.m_VertexNormal[2] = 0;
    
			}
		}
	}
}

/*
int g_nUpBladeVertexSize=0,g_nDownBladeVertexSize=0;
int g_nUpBladeEdgeSize=0,g_nDownBladeEdgeSize=0;
int g_nUpBladeSurfaceSize=0,g_nDownBladeSurfaceSize=0;
MC_Vertex_Cuda*		g_UpBlade_MC_Vertex_Cuda=0;
MC_Edge_Cuda*		g_UpBlade_MC_Edge_Cuda=0;
MC_Surface_Cuda*	g_UpBlade_MC_Surface_Cuda=0;

MC_Vertex_Cuda*		g_DownBlade_MC_Vertex_Cuda=0;
MC_Edge_Cuda*		g_DownBlade_MC_Edge_Cuda=0;
MC_Surface_Cuda*	g_DownBlade_MC_Surface_Cuda=0;
*/
__global__ void cuda_MakeBladeVertex2CellRelationShip(const int nCellSize,EFGCellOnCuda *CellOnCudaPtr,
														const int nVertexSize,VertexOnCuda* VertexOnCudaPtr,
														const int nUpVertexSize,MC_Vertex_Cuda* UpVertex,
														const int nDownVertexSize,MC_Vertex_Cuda* DownVertex)
{
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	const int nCellIdx = tid / BladeVertexSize;
	const int nBladeVertexIdx = tid % BladeVertexSize;
	
	if (nCellIdx < nCellSize && nBladeVertexIdx < nUpVertexSize)
	{
		
		if (/*CellOnCudaPtr[nCellIdx].m_bLeaf &&*/ CellOnCudaPtr[nCellIdx].m_bTopLevelOctreeNodeList)
		{
			
			EFGCellOnCuda& curCell = CellOnCudaPtr[nCellIdx];
			const int nCloneCellId = curCell.m_nCloneCellIdx;
			if (-1 == nCloneCellId) CUPRINTF("-1 == nCloneCellId\n");
			EFGCellOnCuda& curCloneCell = CellOnCudaPtr[nCloneCellId];
			float* pv0 = &VertexOnCudaPtr[curCell.vertexId[0]].local[0];
			float* pv7 = &VertexOnCudaPtr[curCell.vertexId[7]].local[0];
			
			MC_Vertex_Cuda& UpPt = UpVertex[nBladeVertexIdx];
			MC_Vertex_Cuda& DownPt = DownVertex[nBladeVertexIdx];

			//if (UpPt.m_VertexPos[2] > 0.5f) CUPRINTF("UpPt.m_VertexPos[2] %f\n",UpPt.m_VertexPos[2]);
			if ( (UpPt.m_VertexPos[0] > pv0[0] && UpPt.m_VertexPos[0] < pv7[0]) &&
				 (UpPt.m_VertexPos[1] > pv0[1] && UpPt.m_VertexPos[1] < pv7[1]) && 
				 (UpPt.m_VertexPos[2] > pv0[2] && UpPt.m_VertexPos[2] < pv7[2]) )
			{
				UpPt.m_isValid = true;
				DownPt.m_isValid = true;

				//CuttingFaceUpFlagCuda[nBladeVertexIdx] = 1;

				UpPt.m_MeshVertex2CellId = nCellIdx;//nCellIdx;
				DownPt.m_MeshVertex2CellId = nCloneCellId;//nCloneCellId;

				/*CuttingFaceUp_X_Cuda[nBladeVertexIdx] = UpPt.m_VertexPos[0];
				CuttingFaceUp_Y_Cuda[nBladeVertexIdx] = UpPt.m_VertexPos[1];
				CuttingFaceUp_Z_Cuda[nBladeVertexIdx] = UpPt.m_VertexPos[2];*/

				{
					for (int v=0;v<8;++v)
					{

						UpPt.m_elemVertexRelatedDofs[3*v+0] = VertexOnCudaPtr[curCell.vertexId[v]].m_nDof[0];
						UpPt.m_elemVertexRelatedDofs[3*v+1] = VertexOnCudaPtr[curCell.vertexId[v]].m_nDof[1];
						UpPt.m_elemVertexRelatedDofs[3*v+2] = VertexOnCudaPtr[curCell.vertexId[v]].m_nDof[2];

						DownPt.m_elemVertexRelatedDofs[3*v+0] = VertexOnCudaPtr[curCloneCell.vertexId[v]].m_nDof[0];
						DownPt.m_elemVertexRelatedDofs[3*v+1] = VertexOnCudaPtr[curCloneCell.vertexId[v]].m_nDof[1];
						DownPt.m_elemVertexRelatedDofs[3*v+2] = VertexOnCudaPtr[curCloneCell.vertexId[v]].m_nDof[2];
					}
			
					float detaX = (pv7[0] - UpPt.m_VertexPos[0]) / (pv7[0] - pv0[0]);
					float detaY = (pv7[1] - UpPt.m_VertexPos[1]) / (pv7[1] - pv0[1]);
					float detaZ = (pv7[2] - UpPt.m_VertexPos[2]) / (pv7[2] - pv0[2]);

					UpPt.m_TriLinearWeight[0] = detaX * detaY * detaZ;
					UpPt.m_TriLinearWeight[1] = (1-detaX) * detaY * detaZ;
					UpPt.m_TriLinearWeight[2] = detaX * (1-detaY) * detaZ;
					UpPt.m_TriLinearWeight[3] = (1-detaX) * (1-detaY) * detaZ;
					UpPt.m_TriLinearWeight[4] = detaX * detaY * (1-detaZ);
					UpPt.m_TriLinearWeight[5] = (1-detaX) * detaY * (1-detaZ);
					UpPt.m_TriLinearWeight[6] = detaX * (1-detaY) * (1-detaZ);
					UpPt.m_TriLinearWeight[7] = (1-detaX) * (1-detaY) * (1-detaZ);

					DownPt.m_TriLinearWeight[0] = detaX * detaY * detaZ;
					DownPt.m_TriLinearWeight[1] = (1-detaX) * detaY * detaZ;
					DownPt.m_TriLinearWeight[2] = detaX * (1-detaY) * detaZ;
					DownPt.m_TriLinearWeight[3] = (1-detaX) * (1-detaY) * detaZ;
					DownPt.m_TriLinearWeight[4] = detaX * detaY * (1-detaZ);
					DownPt.m_TriLinearWeight[5] = (1-detaX) * detaY * (1-detaZ);
					DownPt.m_TriLinearWeight[6] = detaX * (1-detaY) * (1-detaZ);
					DownPt.m_TriLinearWeight[7] = (1-detaX) * (1-detaY) * (1-detaZ);
				}

			}
		}
	}
}

__global__ void cuda_MakeBladeTriangle2CellRelationShip(const int nUpVertexSize,MC_Vertex_Cuda* UpVertex,
														const int nUpTriSize,MC_Surface_Cuda*	UpTri,
														const int nDownVertexSize,MC_Vertex_Cuda* DownVertex,
														const int nDownTriSize,MC_Surface_Cuda*	DownTri)
{
	const int nTriIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (nTriIdx < nUpTriSize)
	{
		//CUPRINTF("cuda_MakeBladeTriangle2CellRelationShip\n");
		MC_Surface_Cuda& curUpTri = UpTri[nTriIdx];
		MC_Surface_Cuda& curDownTri = DownTri[nTriIdx];
		MC_Vertex_Cuda& pt0 = UpVertex[curUpTri.m_Vertex[0]];
		MC_Vertex_Cuda& pt1 = UpVertex[curUpTri.m_Vertex[1]];
		MC_Vertex_Cuda& pt2 = UpVertex[curUpTri.m_Vertex[2]];

		MC_Vertex_Cuda& pt0_down = DownVertex[curDownTri.m_Vertex[0]];
		MC_Vertex_Cuda& pt1_down = DownVertex[curDownTri.m_Vertex[1]];
		MC_Vertex_Cuda& pt2_down = DownVertex[curDownTri.m_Vertex[2]];

		curUpTri.m_isValid = false;//(pt0.m_isValid && pt1.m_isValid && pt2.m_isValid);
		curDownTri.m_isValid = curUpTri.m_isValid;

		pt0.m_isValid = curUpTri.m_isValid;
		pt1.m_isValid = curUpTri.m_isValid;
		pt2.m_isValid = curUpTri.m_isValid;

		pt0_down.m_isValid = curDownTri.m_isValid;
		pt1_down.m_isValid = curDownTri.m_isValid;
		pt2_down.m_isValid = curDownTri.m_isValid;

		//if (curUpTri.m_isValid) CuttingFaceUpFlagCuda[nTriIdx] = 1;
	}
}
//#define BladeTriangleSize (1024)
//#define BladeEdgeSize (BladeTriangleSize * 3)
//#define BladeVertexSize (BladeTriangleSize*3)
__global__ void cuda_SplittingFace2BladeEdge(const int nLastVertexSize, const int nSplitVertexSize,MC_Vertex_Cuda* vecSplitVertex,
											 const int nLastEdgeSize, const int nSplitEdgeSize,MC_Edge_Cuda* vecSplitEdge,
											 const int nLastTriSize,const int nSplitTriSize,MC_Surface_Cuda* vecSplitTri,
											 const int nUpVertexSize,MC_Vertex_Cuda* UpVertex,
											 const int nUpEdgeSize,MC_Edge_Cuda* UpEdge,
											 const int nUpTriSize,MC_Surface_Cuda*	UpTri,
											 MC_CuttingEdge_Cuda* BladeEdgeVSSpliteFace/*size is nUpEdgeSize*/,
											 const float  massCenterX, const float massCenterY, const float massCenterZ,
											 int *CuttedBladeEdgeFlagOnCuda)
{
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	const int beCutTriangleSize = nSplitTriSize - nLastTriSize;
	const int nBladeEdgeId = tid / beCutTriangleSize;
	const int nSplitTriId = tid % beCutTriangleSize;
	bool bFlag = false;
	if (nSplitTriId < beCutTriangleSize && nBladeEdgeId < nUpEdgeSize)
	{
		if (InValidIdx != vecSplitTri[nLastTriSize+nSplitTriId].m_nParentId4MC)
		{
			
			MC_Surface_Cuda& curSpliteFace = vecSplitTri[ vecSplitTri[nLastTriSize+nSplitTriId].m_nParentId4MC ];
			MC_Edge_Cuda& curBladeEdge = UpEdge[nBladeEdgeId];
			MC_Vertex_Cuda& bladePoint0 = UpVertex[curBladeEdge.m_Vertex[0]];
			MC_Vertex_Cuda& bladePoint1 = UpVertex[curBladeEdge.m_Vertex[1]];

			MC_Vertex_Cuda& splitePoint0 = vecSplitVertex[curSpliteFace.m_Vertex[0]];
			MC_Vertex_Cuda& splitePoint1 = vecSplitVertex[curSpliteFace.m_Vertex[1]];
			MC_Vertex_Cuda& splitePoint2 = vecSplitVertex[curSpliteFace.m_Vertex[2]];

			bFlag = checkLineTri(make_float3(bladePoint0.m_VertexPos[0],bladePoint0.m_VertexPos[1],bladePoint0.m_VertexPos[2]),
								make_float3(bladePoint1.m_VertexPos[0],bladePoint1.m_VertexPos[1],bladePoint1.m_VertexPos[2]),
								make_float3(splitePoint0.m_VertexPos[0],splitePoint0.m_VertexPos[1],splitePoint0.m_VertexPos[2]),
								make_float3(splitePoint1.m_VertexPos[0],splitePoint1.m_VertexPos[1],splitePoint1.m_VertexPos[2]),
								make_float3(splitePoint2.m_VertexPos[0],splitePoint2.m_VertexPos[1],splitePoint2.m_VertexPos[2]));
			/*CUPRINTF("(%f,%f,%f)(%f,%f,%f)(%f,%f,%f)\n",splitePoint0.m_VertexPos[0],splitePoint0.m_VertexPos[1],splitePoint0.m_VertexPos[2],
														splitePoint1.m_VertexPos[0],splitePoint1.m_VertexPos[1],splitePoint1.m_VertexPos[2],
														splitePoint2.m_VertexPos[0],splitePoint2.m_VertexPos[1],splitePoint2.m_VertexPos[2]);*/
			if (bFlag)
			{
				//CUPRINTF("cuda_SplittingFace2BladeEdge\n");
				BladeEdgeVSSpliteFace[nBladeEdgeId].bCut = true;
				BladeEdgeVSSpliteFace[nBladeEdgeId].m_MeshSurfaceParentId = curSpliteFace.m_nSurfaceId;
				/*const float len0 = (bladePoint0.m_VertexPos[0] - massCenterX)*(bladePoint0.m_VertexPos[0] - massCenterX) + 
								   (bladePoint0.m_VertexPos[1] - massCenterY)*(bladePoint0.m_VertexPos[1] - massCenterY) + 
								   (bladePoint0.m_VertexPos[2] - massCenterZ)*(bladePoint0.m_VertexPos[2] - massCenterZ); 
				   
				const float len1 = (bladePoint1.m_VertexPos[0] - massCenterX)*(bladePoint1.m_VertexPos[0] - massCenterX) + 
								   (bladePoint1.m_VertexPos[1] - massCenterY)*(bladePoint1.m_VertexPos[1] - massCenterY) + 
								   (bladePoint1.m_VertexPos[2] - massCenterZ)*(bladePoint1.m_VertexPos[2] - massCenterZ);*/
				if (bladePoint0.m_isValid && !bladePoint1.m_isValid)
				{
					BladeEdgeVSSpliteFace[nBladeEdgeId].validPort = 0;
				}
				else if (!bladePoint0.m_isValid && bladePoint1.m_isValid)
				{
					BladeEdgeVSSpliteFace[nBladeEdgeId].validPort = 1;
				}
				else
				{
					CUPRINTF("cuda_SplittingFace2BladeEdge error \n");
				}

				CuttedBladeEdgeFlagOnCuda[nBladeEdgeId] = 1;
			}
		}
	}
}


__global__ void cuda_MakeSplitFaceVsBladeEdgeTriangleUpDown(const int nSplitVertexSize,MC_Vertex_Cuda* vecSplitVertex,
															const int nSplitEdgeSize,MC_Edge_Cuda* vecSplitEdge,
															const int nSplitTriSize,MC_Surface_Cuda* vecSplitTri,
															const int nUpVertexSize,MC_Vertex_Cuda* UpVertex,
															const int nUpEdgeSize,MC_Edge_Cuda* UpEdge,
															const int nUpTriSize,MC_Surface_Cuda*	UpTri,
															const int nDownVertexSize,MC_Vertex_Cuda* DownVertex,
															const int nDownEdgeSize,MC_Edge_Cuda* DownEdge,
															const int nDownTriSize,MC_Surface_Cuda*	DownTri,
															MC_CuttingEdge_Cuda* BladeEdgeVSSpliteFace/*size is nUpEdgeSize*/,
															int *CuttedBladeEdgeFlagOnCuda)
{
	const int nBladeEdgeIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (nBladeEdgeIdx < nUpEdgeSize)
	{
		MC_CuttingEdge_Cuda& curEdgeInfo = BladeEdgeVSSpliteFace[nBladeEdgeIdx];
		if (curEdgeInfo.bCut)
		{
			MC_Surface_Cuda& curSpliteSurface = vecSplitTri[curEdgeInfo.m_MeshSurfaceParentId];
			
			//make Up Blade Triangle
			{
				const int newTriangleId = nUpTriSize + CuttedBladeEdgeFlagOnCuda[nBladeEdgeIdx];
				const int newEdgeId_0 = nUpEdgeSize + CuttedBladeEdgeFlagOnCuda[nBladeEdgeIdx]*3 + 0;
				const int newEdgeId_1 = nUpEdgeSize + CuttedBladeEdgeFlagOnCuda[nBladeEdgeIdx]*3 + 1;
				const int newEdgeId_2 = nUpEdgeSize + CuttedBladeEdgeFlagOnCuda[nBladeEdgeIdx]*3 + 2;
				//pt0 is created
				const int newVertexId_1 = nUpVertexSize + CuttedBladeEdgeFlagOnCuda[nBladeEdgeIdx]*2 + 0;
				const int newVertexId_2 = nUpVertexSize + CuttedBladeEdgeFlagOnCuda[nBladeEdgeIdx]*2 + 1;
				//blade mesh :
				//new triangle = nNewBladeTriangle * 1;
				//new edge = nNewBladeTriangle * 3;
				//new point = nNewBladeTriangle * 2; with modify TriLinearWeight

				MC_Edge_Cuda& cloneUpEdge = vecSplitEdge[curSpliteSurface.m_nCloneBladeLine[0]];
				MC_Vertex_Cuda& clonePt1_Up = vecSplitVertex[cloneUpEdge.m_Vertex[0]];
				MC_Vertex_Cuda& clonePt2_Up = vecSplitVertex[cloneUpEdge.m_Vertex[1]];
				MC_Edge_Cuda& curUpBladeEdge = UpEdge[nBladeEdgeIdx];
				MC_Vertex_Cuda& pt0_Up = UpVertex[curUpBladeEdge.m_Vertex[(int)curEdgeInfo.validPort]];
				MC_Vertex_Cuda& pt1_Up = UpVertex[newVertexId_1];
				MC_Vertex_Cuda& pt2_Up = UpVertex[newVertexId_2];
				MC_Edge_Cuda& line0_Up = UpEdge[newEdgeId_0];
				MC_Edge_Cuda& line1_Up = UpEdge[newEdgeId_1];
				MC_Edge_Cuda& line2_Up = UpEdge[newEdgeId_2];
				MC_Surface_Cuda& face_Up = UpTri[newTriangleId];
		
				//clone vertex 1 & 2
				pt1_Up.m_isValid = true;pt1_Up.m_isJoint = true;pt1_Up.m_isSplit = false;pt1_Up.m_state = 0;
				pt1_Up.m_CloneVertexIdx[0] = pt1_Up.m_CloneVertexIdx[1] = InValidIdx;pt1_Up.m_brotherPoint = InValidIdx;
				pt1_Up.m_nVertexId = newVertexId_1;
				pt1_Up.m_MeshVertex2CellId = clonePt1_Up.m_MeshVertex2CellId;
				pt1_Up.m_distanceToBlade = clonePt1_Up.m_distanceToBlade;
				pt1_Up.m_VertexPos[0] = clonePt1_Up.m_VertexPos[0];pt1_Up.m_VertexPos[1] = clonePt1_Up.m_VertexPos[1];pt1_Up.m_VertexPos[2] = clonePt1_Up.m_VertexPos[2];
				pt1_Up.m_VertexNormal[0] = clonePt1_Up.m_VertexNormal[0];pt1_Up.m_VertexNormal[1] = clonePt1_Up.m_VertexNormal[1];pt1_Up.m_VertexNormal[2] = clonePt1_Up.m_VertexNormal[2];
				
				pt2_Up.m_isValid = true;pt2_Up.m_isJoint = true;pt2_Up.m_isSplit = false;pt2_Up.m_state = 0;
				pt2_Up.m_CloneVertexIdx[0] = pt2_Up.m_CloneVertexIdx[1] = InValidIdx;pt2_Up.m_brotherPoint = InValidIdx;
				pt2_Up.m_nVertexId = newVertexId_2;
				pt2_Up.m_MeshVertex2CellId = clonePt2_Up.m_MeshVertex2CellId;
				pt2_Up.m_distanceToBlade = clonePt2_Up.m_distanceToBlade;
				pt2_Up.m_VertexPos[0] = clonePt2_Up.m_VertexPos[0];pt2_Up.m_VertexPos[1] = clonePt2_Up.m_VertexPos[1];pt2_Up.m_VertexPos[2] = clonePt2_Up.m_VertexPos[2];
				pt2_Up.m_VertexNormal[0] = clonePt2_Up.m_VertexNormal[0];pt2_Up.m_VertexNormal[1] = clonePt2_Up.m_VertexNormal[1];pt2_Up.m_VertexNormal[2] = clonePt2_Up.m_VertexNormal[2];

				for (int v=0;v<8;++v)
				{
					pt1_Up.m_TriLinearWeight[v] = clonePt1_Up.m_TriLinearWeight[v];
					pt2_Up.m_TriLinearWeight[v] = clonePt2_Up.m_TriLinearWeight[v];

					pt1_Up.m_elemVertexRelatedDofs[3*v+0] = clonePt1_Up.m_elemVertexRelatedDofs[3*v+0];
					pt1_Up.m_elemVertexRelatedDofs[3*v+1] = clonePt1_Up.m_elemVertexRelatedDofs[3*v+1];
					pt1_Up.m_elemVertexRelatedDofs[3*v+2] = clonePt1_Up.m_elemVertexRelatedDofs[3*v+2];

					pt2_Up.m_elemVertexRelatedDofs[3*v+0] = clonePt2_Up.m_elemVertexRelatedDofs[3*v+0];
					pt2_Up.m_elemVertexRelatedDofs[3*v+1] = clonePt2_Up.m_elemVertexRelatedDofs[3*v+1];
					pt2_Up.m_elemVertexRelatedDofs[3*v+2] = clonePt2_Up.m_elemVertexRelatedDofs[3*v+2];
				}

				//clone line 0 & 1 & 2
				line0_Up.m_hasClone = false;line0_Up.m_isValid = true;line0_Up.m_isJoint = true;line0_Up.m_isCut = false;line0_Up.m_state = 0;
				line0_Up.m_CloneIntersectVertexIdx[0] = line0_Up.m_CloneIntersectVertexIdx[1] = InValidIdx;
				line0_Up.m_CloneEdgeIdx[0] = line0_Up.m_CloneEdgeIdx[1] = InValidIdx;
				line0_Up.m_belongToTri[0] = line0_Up.m_belongToTri[1] = line0_Up.m_belongToTri[2] = InValidIdx;
				line0_Up.m_belongToTriVertexIdx[0][0] = line0_Up.m_belongToTriVertexIdx[0][1] = InValidIdx;
				line0_Up.m_belongToTriVertexIdx[1][0] = line0_Up.m_belongToTriVertexIdx[1][1] = InValidIdx;
				line0_Up.m_belongToTriVertexIdx[2][0] = line0_Up.m_belongToTriVertexIdx[2][1] = InValidIdx;

				line0_Up.m_nLineId = newEdgeId_0;
				line0_Up.m_Vertex[0] = pt1_Up.m_nVertexId;line0_Up.m_Vertex[1] = pt2_Up.m_nVertexId;

				line1_Up.m_hasClone = false;line1_Up.m_isValid = true;line1_Up.m_isJoint = true;line1_Up.m_isCut = false;line1_Up.m_state = 0;
				line1_Up.m_CloneIntersectVertexIdx[0] = line1_Up.m_CloneIntersectVertexIdx[1] = InValidIdx;
				line1_Up.m_CloneEdgeIdx[0] = line1_Up.m_CloneEdgeIdx[1] = InValidIdx;
				line1_Up.m_belongToTri[0] = line1_Up.m_belongToTri[1] = line1_Up.m_belongToTri[2] = InValidIdx;
				line1_Up.m_belongToTriVertexIdx[0][0] = line1_Up.m_belongToTriVertexIdx[0][1] = InValidIdx;
				line1_Up.m_belongToTriVertexIdx[1][0] = line1_Up.m_belongToTriVertexIdx[1][1] = InValidIdx;
				line1_Up.m_belongToTriVertexIdx[2][0] = line1_Up.m_belongToTriVertexIdx[2][1] = InValidIdx;

				line1_Up.m_nLineId = newEdgeId_1;
				line1_Up.m_Vertex[0] = pt2_Up.m_nVertexId;line1_Up.m_Vertex[1] = pt0_Up.m_nVertexId;

				line2_Up.m_hasClone = false;line2_Up.m_isValid = true;line2_Up.m_isJoint = true;line2_Up.m_isCut = false;line2_Up.m_state = 0;
				line2_Up.m_CloneIntersectVertexIdx[0] = line2_Up.m_CloneIntersectVertexIdx[1] = InValidIdx;
				line2_Up.m_CloneEdgeIdx[0] = line2_Up.m_CloneEdgeIdx[1] = InValidIdx;
				line2_Up.m_belongToTri[0] = line2_Up.m_belongToTri[1] = line2_Up.m_belongToTri[2] = InValidIdx;
				line2_Up.m_belongToTriVertexIdx[0][0] = line2_Up.m_belongToTriVertexIdx[0][1] = InValidIdx;
				line2_Up.m_belongToTriVertexIdx[1][0] = line2_Up.m_belongToTriVertexIdx[1][1] = InValidIdx;
				line2_Up.m_belongToTriVertexIdx[2][0] = line2_Up.m_belongToTriVertexIdx[2][1] = InValidIdx;

				line2_Up.m_nLineId = newEdgeId_2;
				line2_Up.m_Vertex[0] = pt0_Up.m_nVertexId;line2_Up.m_Vertex[1] = pt1_Up.m_nVertexId;

				line0_Up.m_belongToTriVertexIdx[0][0] = 1;line0_Up.m_belongToTriVertexIdx[1][1] = 2;
				line1_Up.m_belongToTriVertexIdx[1][0] = 2;line1_Up.m_belongToTriVertexIdx[1][1] = 0;
				line2_Up.m_belongToTriVertexIdx[2][0] = 0;line2_Up.m_belongToTriVertexIdx[2][1] = 1;

				face_Up.m_isValid = true;face_Up.m_isJoint = true;face_Up.m_state = 0;
				face_Up.m_VertexNormal[0] = face_Up.m_VertexNormal[1] = face_Up.m_VertexNormal[2] = 0;face_Up.m_nParentId4MC = InValidIdx;
				face_Up.m_nCloneBladeLine[0] = face_Up.m_nCloneBladeLine[1] = InValidIdx;
				face_Up.m_nSurfaceId = newTriangleId;
				face_Up.m_Vertex[0] = pt0_Up.m_nVertexId;face_Up.m_Vertex[1] = pt1_Up.m_nVertexId;face_Up.m_Vertex[2] = pt2_Up.m_nVertexId;
				face_Up.m_Lines[0] = line0_Up.m_nLineId;face_Up.m_Lines[1] = line1_Up.m_nLineId;face_Up.m_Lines[2] = line2_Up.m_nLineId;

				line0_Up.m_belongToTri[0] = face_Up.m_nSurfaceId;
				line1_Up.m_belongToTri[1] = face_Up.m_nSurfaceId;
				line2_Up.m_belongToTri[2] = face_Up.m_nSurfaceId;

			}
			//make Down Blade Triangle
			{
				const int newTriangleId = nDownTriSize + CuttedBladeEdgeFlagOnCuda[nBladeEdgeIdx];
				const int newEdgeId_0 = nDownEdgeSize + CuttedBladeEdgeFlagOnCuda[nBladeEdgeIdx]*3 + 0;
				const int newEdgeId_1 = nDownEdgeSize + CuttedBladeEdgeFlagOnCuda[nBladeEdgeIdx]*3 + 1;
				const int newEdgeId_2 = nDownEdgeSize + CuttedBladeEdgeFlagOnCuda[nBladeEdgeIdx]*3 + 2;
				//pt0 is created
				const int newVertexId_1 = nDownVertexSize + CuttedBladeEdgeFlagOnCuda[nBladeEdgeIdx]*2 + 0;
				const int newVertexId_2 = nDownVertexSize + CuttedBladeEdgeFlagOnCuda[nBladeEdgeIdx]*2 + 1;
				//blade mesh :
				//new triangle = nNewBladeTriangle * 1;
				//new edge = nNewBladeTriangle * 3;
				//new point = nNewBladeTriangle * 2; with modify TriLinearWeight

				MC_Edge_Cuda& cloneDownEdge = vecSplitEdge[curSpliteSurface.m_nCloneBladeLine[1]];
				MC_Vertex_Cuda& clonePt1_Down = vecSplitVertex[cloneDownEdge.m_Vertex[0]];
				MC_Vertex_Cuda& clonePt2_Down = vecSplitVertex[cloneDownEdge.m_Vertex[1]];
				MC_Edge_Cuda& curDownBladeEdge = DownEdge[nBladeEdgeIdx];
				MC_Vertex_Cuda& pt0_Down = DownVertex[curDownBladeEdge.m_Vertex[(int)curEdgeInfo.validPort]];
				MC_Vertex_Cuda& pt1_Down = DownVertex[newVertexId_1];
				MC_Vertex_Cuda& pt2_Down = DownVertex[newVertexId_2];
				MC_Edge_Cuda& line0_Down = DownEdge[newEdgeId_0];
				MC_Edge_Cuda& line1_Down = DownEdge[newEdgeId_1];
				MC_Edge_Cuda& line2_Down = DownEdge[newEdgeId_2];
				MC_Surface_Cuda& face_Down = DownTri[newTriangleId];

				//clone vertex 1 & 2
				pt1_Down.m_isValid = true;pt1_Down.m_isJoint = true;pt1_Down.m_isSplit = false;pt1_Down.m_state = 0;
				pt1_Down.m_CloneVertexIdx[0] = pt1_Down.m_CloneVertexIdx[1] = InValidIdx;pt1_Down.m_brotherPoint = InValidIdx;
				pt1_Down.m_nVertexId = newVertexId_1;
				pt1_Down.m_MeshVertex2CellId = clonePt1_Down.m_MeshVertex2CellId;
				pt1_Down.m_distanceToBlade = clonePt1_Down.m_distanceToBlade;
				pt1_Down.m_VertexPos[0] = clonePt1_Down.m_VertexPos[0];pt1_Down.m_VertexPos[1] = clonePt1_Down.m_VertexPos[1];pt1_Down.m_VertexPos[2] = clonePt1_Down.m_VertexPos[2];
				pt1_Down.m_VertexNormal[0] = clonePt1_Down.m_VertexNormal[0];pt1_Down.m_VertexNormal[1] = clonePt1_Down.m_VertexNormal[1];pt1_Down.m_VertexNormal[2] = clonePt1_Down.m_VertexNormal[2];
	
				pt2_Down.m_isValid = true;pt2_Down.m_isJoint = true;pt2_Down.m_isSplit = false;pt2_Down.m_state = 0;
				pt2_Down.m_CloneVertexIdx[0] = pt2_Down.m_CloneVertexIdx[1] = InValidIdx;pt2_Down.m_brotherPoint = InValidIdx;
				pt2_Down.m_nVertexId = newVertexId_2;
				pt2_Down.m_MeshVertex2CellId = clonePt2_Down.m_MeshVertex2CellId;
				pt2_Down.m_distanceToBlade = clonePt2_Down.m_distanceToBlade;
				pt2_Down.m_VertexPos[0] = clonePt2_Down.m_VertexPos[0];pt2_Down.m_VertexPos[1] = clonePt2_Down.m_VertexPos[1];pt2_Down.m_VertexPos[2] = clonePt2_Down.m_VertexPos[2];
				pt2_Down.m_VertexNormal[0] = clonePt2_Down.m_VertexNormal[0];pt2_Down.m_VertexNormal[1] = clonePt2_Down.m_VertexNormal[1];pt2_Down.m_VertexNormal[2] = clonePt2_Down.m_VertexNormal[2];

				for (int v=0;v<8;++v)
				{
					pt1_Down.m_TriLinearWeight[v] = clonePt1_Down.m_TriLinearWeight[v];
					pt2_Down.m_TriLinearWeight[v] = clonePt2_Down.m_TriLinearWeight[v];

					pt1_Down.m_elemVertexRelatedDofs[3*v+0] = clonePt1_Down.m_elemVertexRelatedDofs[3*v+0];
					pt1_Down.m_elemVertexRelatedDofs[3*v+1] = clonePt1_Down.m_elemVertexRelatedDofs[3*v+1];
					pt1_Down.m_elemVertexRelatedDofs[3*v+2] = clonePt1_Down.m_elemVertexRelatedDofs[3*v+2];

					pt2_Down.m_elemVertexRelatedDofs[3*v+0] = clonePt2_Down.m_elemVertexRelatedDofs[3*v+0];
					pt2_Down.m_elemVertexRelatedDofs[3*v+1] = clonePt2_Down.m_elemVertexRelatedDofs[3*v+1];
					pt2_Down.m_elemVertexRelatedDofs[3*v+2] = clonePt2_Down.m_elemVertexRelatedDofs[3*v+2];
				}

				//clone line 0 & 1 & 2
				line0_Down.m_hasClone = false;line0_Down.m_isValid = true;line0_Down.m_isJoint = true;line0_Down.m_isCut = false;line0_Down.m_state = 0;
				line0_Down.m_CloneIntersectVertexIdx[0] = line0_Down.m_CloneIntersectVertexIdx[1] = InValidIdx;
				line0_Down.m_CloneEdgeIdx[0] = line0_Down.m_CloneEdgeIdx[1] = InValidIdx;
				line0_Down.m_belongToTri[0] = line0_Down.m_belongToTri[1] = line0_Down.m_belongToTri[2] = InValidIdx;
				line0_Down.m_belongToTriVertexIdx[0][0] = line0_Down.m_belongToTriVertexIdx[0][1] = InValidIdx;
				line0_Down.m_belongToTriVertexIdx[1][0] = line0_Down.m_belongToTriVertexIdx[1][1] = InValidIdx;
				line0_Down.m_belongToTriVertexIdx[2][0] = line0_Down.m_belongToTriVertexIdx[2][1] = InValidIdx;

				line0_Down.m_nLineId = newEdgeId_0;
				line0_Down.m_Vertex[0] = pt1_Down.m_nVertexId;line0_Down.m_Vertex[1] = pt2_Down.m_nVertexId;

				line1_Down.m_hasClone = false;line1_Down.m_isValid = true;line1_Down.m_isJoint = true;line1_Down.m_isCut = false;line1_Down.m_state = 0;
				line1_Down.m_CloneIntersectVertexIdx[0] = line1_Down.m_CloneIntersectVertexIdx[1] = InValidIdx;
				line1_Down.m_CloneEdgeIdx[0] = line1_Down.m_CloneEdgeIdx[1] = InValidIdx;
				line1_Down.m_belongToTri[0] = line1_Down.m_belongToTri[1] = line1_Down.m_belongToTri[2] = InValidIdx;
				line1_Down.m_belongToTriVertexIdx[0][0] = line1_Down.m_belongToTriVertexIdx[0][1] = InValidIdx;
				line1_Down.m_belongToTriVertexIdx[1][0] = line1_Down.m_belongToTriVertexIdx[1][1] = InValidIdx;
				line1_Down.m_belongToTriVertexIdx[2][0] = line1_Down.m_belongToTriVertexIdx[2][1] = InValidIdx;

				line1_Down.m_nLineId = newEdgeId_1;
				line1_Down.m_Vertex[0] = pt2_Down.m_nVertexId;line1_Down.m_Vertex[1] = pt0_Down.m_nVertexId;

				line2_Down.m_hasClone = false;line2_Down.m_isValid = true;line2_Down.m_isJoint = true;line2_Down.m_isCut = false;line2_Down.m_state = 0;
				line2_Down.m_CloneIntersectVertexIdx[0] = line2_Down.m_CloneIntersectVertexIdx[1] = InValidIdx;
				line2_Down.m_CloneEdgeIdx[0] = line2_Down.m_CloneEdgeIdx[1] = InValidIdx;
				line2_Down.m_belongToTri[0] = line2_Down.m_belongToTri[1] = line2_Down.m_belongToTri[2] = InValidIdx;
				line2_Down.m_belongToTriVertexIdx[0][0] = line2_Down.m_belongToTriVertexIdx[0][1] = InValidIdx;
				line2_Down.m_belongToTriVertexIdx[1][0] = line2_Down.m_belongToTriVertexIdx[1][1] = InValidIdx;
				line2_Down.m_belongToTriVertexIdx[2][0] = line2_Down.m_belongToTriVertexIdx[2][1] = InValidIdx;

				line2_Down.m_nLineId = newEdgeId_2;
				line2_Down.m_Vertex[0] = pt0_Down.m_nVertexId;line2_Down.m_Vertex[1] = pt1_Down.m_nVertexId;

				line0_Down.m_belongToTriVertexIdx[0][0] = 1;line0_Down.m_belongToTriVertexIdx[1][1] = 2;
				line1_Down.m_belongToTriVertexIdx[1][0] = 2;line1_Down.m_belongToTriVertexIdx[1][1] = 0;
				line2_Down.m_belongToTriVertexIdx[2][0] = 0;line2_Down.m_belongToTriVertexIdx[2][1] = 1;

				face_Down.m_isValid = true;face_Down.m_isJoint = true;face_Down.m_state = 0;
				face_Down.m_VertexNormal[0] = face_Down.m_VertexNormal[1] = face_Down.m_VertexNormal[2] = 0;face_Down.m_nParentId4MC = InValidIdx;
				face_Down.m_nCloneBladeLine[0] = face_Down.m_nCloneBladeLine[1] = InValidIdx;
				face_Down.m_nSurfaceId = newTriangleId;
				face_Down.m_Vertex[0] = pt0_Down.m_nVertexId;face_Down.m_Vertex[1] = pt1_Down.m_nVertexId;face_Down.m_Vertex[2] = pt2_Down.m_nVertexId;
				face_Down.m_Lines[0] = line0_Down.m_nLineId;face_Down.m_Lines[1] = line1_Down.m_nLineId;face_Down.m_Lines[2] = line2_Down.m_nLineId;

				line0_Down.m_belongToTri[0] = face_Down.m_nSurfaceId;
				line1_Down.m_belongToTri[1] = face_Down.m_nSurfaceId;
				line2_Down.m_belongToTri[2] = face_Down.m_nSurfaceId;

			}
		}
	}
}

__global__ void cuda_disableBlade(/*const int nUpVertexSize,MC_Vertex_Cuda* UpVertex,
								const int nUpEdgeSize,MC_Edge_Cuda* UpEdge,*/
								const int nUpTriSize,MC_Surface_Cuda*	UpTri,
								/*const int nDownVertexSize,MC_Vertex_Cuda* DownVertex,
								const int nDownEdgeSize,MC_Edge_Cuda* DownEdge,*/
								const int nDownTriSize,MC_Surface_Cuda*	DownTri,
								MC_CuttingEdge_Cuda* BladeEdgeVSSpliteFace)
{
	const int curFaceIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (curFaceIdx < nUpTriSize)
	{
		if (UpTri[curFaceIdx].m_isValid)
		{
			const int line0Id = UpTri[curFaceIdx].m_Lines[0];
			const int line1Id = UpTri[curFaceIdx].m_Lines[1];
			const int line2Id = UpTri[curFaceIdx].m_Lines[2];

			UpTri[curFaceIdx].m_isValid = !(BladeEdgeVSSpliteFace[line0Id].bCut || BladeEdgeVSSpliteFace[line1Id].bCut || BladeEdgeVSSpliteFace[line2Id].bCut);
			DownTri[curFaceIdx].m_isValid = UpTri[curFaceIdx].m_isValid;

		}
	}
}
#if 0
__global__ void cuda_computeBladeFaceSplitType(const int nUpVertexSize,MC_Vertex_Cuda* UpVertex,
											   const int nUpEdgeSize,MC_Edge_Cuda* UpEdge,
											   const int nUpTriSize,MC_Surface_Cuda*	UpTri,
											   float massX,float massY,float massZ,
											   int *splittedFaceFlagOnCuda)
{
	const int curFaceIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (curFaceIdx < nUpTriSize)
	{
		MC_Surface_Cuda& curFace = UpTri[curFaceIdx];
		if (!curFace.m_isValid)//blade valid is false！！！！！！！
		{
			curFace.m_state =   (char)(UpEdge[curFace.m_Lines[0]].m_isCut ? 1 : 0) + 
									  (UpEdge[curFace.m_Lines[1]].m_isCut ? 1 : 0) + 
									  (UpEdge[curFace.m_Lines[2]].m_isCut ? 1 : 0);
			if (0 != curFace.m_state)
			{
				float* pt0 = &UpVertex[curFace.m_Vertex[0]].m_VertexPos[0];
				float* pt1 = &UpVertex[curFace.m_Vertex[1]].m_VertexPos[0];
				float* pt2 = &UpVertex[curFace.m_Vertex[2]].m_VertexPos[0];
				float masslen = (curFace.m_center[0] - massX)*(curFace.m_center[0] - massX) + 
								(curFace.m_center[1] - massY)*(curFace.m_center[1] - massY) +
								(curFace.m_center[2] - massZ)*(curFace.m_center[2] - massZ);

				float len0 = (pt0[0] - massX)*(pt0[0] - massX) + (pt0[1] - massY)*(pt0[1] - massY) + (pt0[2] - massZ)*(pt0[2] - massZ);
				float len1 = (pt1[0] - massX)*(pt1[0] - massX) + (pt1[1] - massY)*(pt1[1] - massY) + (pt1[2] - massZ)*(pt1[2] - massZ);
				float len2 = (pt2[0] - massX)*(pt2[0] - massX) + (pt2[1] - massY)*(pt2[1] - massY) + (pt2[2] - massZ)*(pt2[2] - massZ);

				curFace.m_state = (len0 < masslen ? 1 : 0) + (len1 < masslen ? 1 : 0) + (len2 < masslen ? 1 : 0);
				splittedFaceFlagOnCuda[curFaceIdx] = curFace.m_state;
			}
		}
	}
}
#endif
void test();
extern void makeDelaunay(int& nVertexSize,MC_Vertex_Cuda* vec_CudaMCVertex,
							int& nEdgeSize,MC_Edge_Cuda* vec_CudaMCEdge, 
							int& nTriSize, MC_Surface_Cuda* vec_CudaMCSurface,
							int  nNewVertexSize,MC_Vertex_Cuda* vec_New,float signs);
void makeVertex2BladeRelationShipOnCuda()
{
	g_nLastVertexSize = g_nMCVertexSize;
	g_nLastEdgeSize = g_nMCEdgeSize;
	g_nLastSurfaceSize = g_nMCSurfaceSize;

	IndexTypePtr g_CuttedEdgeFlagOnCuda;
	HANDLE_ERROR(cudaHostGetDevicePointer((void **)&g_CuttedEdgeFlagOnCuda,(void *)g_CuttedEdgeFlagOnCpu,0));
	HANDLE_ERROR( cudaMemset( (void*)g_CuttedEdgeFlagOnCuda,	0,g_nMaxMCEdgeSize * sizeof(IndexType))) ;

	cuda_makeVertex2BladeRelationShip<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(g_nMCEdgeSize,g_MC_Edge_Cuda,g_nMCVertexSize,g_MC_Vertex_Cuda,g_BladeElementOnCuda,nBladeBase,g_CuttedEdgeFlagOnCuda);

	cudaDeviceSynchronize();
	
	thrust::exclusive_scan(g_CuttedEdgeFlagOnCpu, g_CuttedEdgeFlagOnCpu+g_nMCEdgeSize+1, g_CuttedEdgeFlagOnCpu); //generate ghost cell count
	int nNewClonePointCount = g_CuttedEdgeFlagOnCpu[g_nMCEdgeSize];
	int nNewCloneEdgeCount = nNewClonePointCount;
	//splite edge
	cuda_spliteLine<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(g_nMCEdgeSize,g_MC_Edge_Cuda,
																					g_nMCVertexSize,g_MC_Vertex_Cuda,
																					g_CuttedEdgeFlagOnCuda,g_nLastVertexSize,g_nLastEdgeSize);
	cudaDeviceSynchronize();
	g_nMCVertexSize += nNewClonePointCount;
	g_nMCEdgeSize += nNewCloneEdgeCount;

	cuda_computeVertexUpDown<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(g_nMCVertexSize,g_MC_Vertex_Cuda,g_BladeElementOnCuda,nBladeBase,g_BladeNormalOnCuda);

	//return;

	IndexTypePtr g_SplittedFaceFlagOnCuda;
	HANDLE_ERROR(cudaHostGetDevicePointer((void **)&g_SplittedFaceFlagOnCuda,(void *)g_SplittedFaceFlagOnCpu,0));
	HANDLE_ERROR( cudaMemset( (void*)g_SplittedFaceFlagOnCuda,	0,g_nMaxMCSurfaceSize * sizeof(IndexType))) ;

	cuda_computeSurfaceSplitType<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(g_nMCEdgeSize,g_MC_Edge_Cuda,
																								 g_nMCSurfaceSize,g_MC_Surface_Cuda,
																								 g_SplittedFaceFlagOnCuda);
	cudaDeviceSynchronize();
	thrust::exclusive_scan(g_SplittedFaceFlagOnCpu, g_SplittedFaceFlagOnCpu+g_nMCSurfaceSize+1, g_SplittedFaceFlagOnCpu); //generate ghost cell count
	int nNewSplitFaceCount = g_SplittedFaceFlagOnCpu[g_nMCSurfaceSize];
	nNewCloneEdgeCount = nNewSplitFaceCount;

	cuda_SpliteSurface<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(g_nMCVertexSize,g_MC_Vertex_Cuda,
																					   g_nMCEdgeSize,g_MC_Edge_Cuda,
																					   g_nMCSurfaceSize,g_MC_Surface_Cuda,
																					   g_SplittedFaceFlagOnCuda,g_nMCVertexSize,g_nMCEdgeSize,g_nMCSurfaceSize);
	cudaDeviceSynchronize();
	g_nMCEdgeSize += nNewCloneEdgeCount;
	g_nMCSurfaceSize += nNewSplitFaceCount;

	/*cuda_TestSurface<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(g_nMCVertexSize,g_MC_Vertex_Cuda,g_nMCSurfaceSize,g_MC_Surface_Cuda);
	cudaDeviceSynchronize();*/

	cuda_distributeNewCloneVertexToCloneCell<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(
		g_nLastVertexSize, g_nMCVertexSize,g_MC_Vertex_Cuda,EFG_CellOnCudaElementCount,EFG_CellOnCudaPtr,g_nVertexOnCudaCount,g_VertexOnCudaPtr);
	

	//float * CuttingFaceUp_X_Cuda;//size is max meshcutting vertex
	//float * CuttingFaceUp_Y_Cuda;//size is max meshcutting vertex
	//float * CuttingFaceUp_Z_Cuda;//size is max meshcutting vertex
	//int *	CuttingFaceUpFlagCuda;

	/*HANDLE_ERROR(cudaHostGetDevicePointer((void **)&CuttingFaceUp_X_Cuda,(void *)g_CuttingFaceUp_X,0));
	HANDLE_ERROR( cudaMemset( (void*)CuttingFaceUp_X_Cuda,	0,g_nMaxMCVertexSize * sizeof(float))) ;
	HANDLE_ERROR(cudaHostGetDevicePointer((void **)&CuttingFaceUp_Y_Cuda,(void *)g_CuttingFaceUp_Y,0));
	HANDLE_ERROR( cudaMemset( (void*)CuttingFaceUp_Y_Cuda,	0,g_nMaxMCVertexSize * sizeof(float))) ;
	HANDLE_ERROR(cudaHostGetDevicePointer((void **)&CuttingFaceUp_Z_Cuda,(void *)g_CuttingFaceUp_Z,0));
	HANDLE_ERROR( cudaMemset( (void*)CuttingFaceUp_Z_Cuda,	0,g_nMaxMCVertexSize * sizeof(float))) ;*/
	/*HANDLE_ERROR(cudaHostGetDevicePointer((void **)&CuttingFaceUpFlagCuda,(void *)g_CuttingFaceUpFlagCpu,0));
	HANDLE_ERROR( cudaMemset( (void*)CuttingFaceUpFlagCuda,	0,g_nMaxMCVertexSize * sizeof(int))) ;*/

	cuda_MakeBladeVertex2CellRelationShip<<< KERNEL_COUNT_TMP,576 >>>(EFG_CellOnCudaElementCount_Last,EFG_CellOnCudaPtr,
																					g_nVertexOnCudaCount,g_VertexOnCudaPtr,
																					g_nUpBladeVertexSize,g_UpBlade_MC_Vertex_Cuda,
																					g_nDownBladeVertexSize,g_DownBlade_MC_Vertex_Cuda);
	
	cudaDeviceSynchronize();
	//return ;
	/*int nUpCloneVertexCount = thrust::reduce(g_CuttingFaceUpFlagCpu,g_CuttingFaceUpFlagCpu+g_nUpBladeVertexSize+1);
	float dbUpX = thrust::reduce(g_CuttingFaceUp_X,g_CuttingFaceUp_X+g_nUpBladeVertexSize+1) / nUpCloneVertexCount;
	float dbUpY = thrust::reduce(g_CuttingFaceUp_Y,g_CuttingFaceUp_Y+g_nUpBladeVertexSize+1) / nUpCloneVertexCount;
	float dbUpZ = thrust::reduce(g_CuttingFaceUp_Z,g_CuttingFaceUp_Z+g_nUpBladeVertexSize+1) / nUpCloneVertexCount;
	printf("nUpCloneVertexCount(%d) (%f,%f,%f)\n",nUpCloneVertexCount,dbUpX,dbUpY,dbUpZ);*/
	
	//cudaDeviceSynchronize();
	//HANDLE_ERROR( cudaMemset( (void*)CuttingFaceUpFlagCuda,	0,g_nMaxMCVertexSize * sizeof(int))) ;

	
	cuda_MakeBladeTriangle2CellRelationShip<<<(KERNEL_COUNT + BLOCK_COUNT - 1) / BLOCK_COUNT,BLOCK_COUNT>>>(g_nUpBladeVertexSize,g_UpBlade_MC_Vertex_Cuda,
																					g_nUpBladeSurfaceSize,g_UpBlade_MC_Surface_Cuda,
																					g_nDownBladeVertexSize,g_DownBlade_MC_Vertex_Cuda,
																					g_nDownBladeSurfaceSize,g_DownBlade_MC_Surface_Cuda/*,CuttingFaceUpFlagCuda*/);
	cudaDeviceSynchronize();

	MC_Vertex_Cuda* vec_New = (MC_Vertex_Cuda*)malloc(g_nMCVertexSize * sizeof(MC_Vertex_Cuda));
	HANDLE_ERROR( cudaMemcpy( (void *)vec_New,	g_MC_Vertex_Cuda,	(g_nMCVertexSize ) * sizeof(MC_Vertex_Cuda),	cudaMemcpyDeviceToHost ) );
	
	/*
	void makeDelaunayMySelf(int& nVertexSize,int nMaxVertexSize,MC_Vertex_Cuda* vec_CudaMCVertex,
							int& nEdgeSize,int nMaxEdgeSize,MC_Edge_Cuda* vec_CudaMCEdge, 
							int& nTriSize,int nMaxTriSize, MC_Surface_Cuda* vec_CudaMCSurface,
							int  nNewVertexSize,MC_Vertex_Cuda* vec_New);
	*/

	makeDelaunay(g_nDownBladeVertexSize,g_DownBlade_MC_Vertex_Cpu,
						g_nDownBladeEdgeSize,g_DownBlade_MC_Edge_Cpu,
						g_nDownBladeSurfaceSize,g_DownBlade_MC_Surface_Cpu,
						g_nMCVertexSize - g_nLastVertexSize,&vec_New[g_nLastVertexSize],-1);

	makeDelaunay(g_nUpBladeVertexSize,g_UpBlade_MC_Vertex_Cpu,
				g_nUpBladeEdgeSize,g_UpBlade_MC_Edge_Cpu,
				g_nUpBladeSurfaceSize,g_UpBlade_MC_Surface_Cpu,
				g_nMCVertexSize - g_nLastVertexSize,&vec_New[g_nLastVertexSize],1);
	free(vec_New);
	
	
	//test();
	return ;
}

void test()
{
	
	MC_Vertex_Cuda* DownVertex = (MC_Vertex_Cuda*)malloc(g_nDownBladeVertexSize * sizeof(MC_Vertex_Cuda));
	MC_Edge_Cuda* DownEdge = (MC_Edge_Cuda*)malloc(g_nDownBladeEdgeSize * sizeof(MC_Edge_Cuda));
	MC_Surface_Cuda* DownTri = (MC_Surface_Cuda*)malloc(g_nDownBladeSurfaceSize * sizeof(MC_Surface_Cuda));

	HANDLE_ERROR( cudaMemcpy( (void *)DownVertex,	g_DownBlade_MC_Vertex_Cuda,	(g_nDownBladeVertexSize) * sizeof(MC_Vertex_Cuda),	cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( (void *)DownEdge,	g_DownBlade_MC_Edge_Cuda,	(g_nDownBladeEdgeSize) * sizeof(MC_Edge_Cuda),	cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( (void *)DownTri,	g_DownBlade_MC_Surface_Cuda,	(g_nDownBladeSurfaceSize) * sizeof(MC_Surface_Cuda),	cudaMemcpyDeviceToHost ) );

	for (unsigned t=0;t<g_nDownBladeSurfaceSize;++t)
	{
		MC_Surface_Cuda& curTri = DownTri[t];
		if (curTri.m_isValid)
		{
			MC_Vertex_Cuda& pt0 = DownVertex[curTri.m_Vertex[0]];
			MC_Vertex_Cuda& pt1 = DownVertex[curTri.m_Vertex[1]];
			MC_Vertex_Cuda& pt2 = DownVertex[curTri.m_Vertex[2]];
			printf("{%f,%f,%f,%f,%f,%f,%f,%f,%f},",pt0.m_VertexPos[0],pt0.m_VertexPos[1],pt0.m_VertexPos[2],
													pt1.m_VertexPos[0],pt1.m_VertexPos[1],pt1.m_VertexPos[2],
													pt2.m_VertexPos[0],pt2.m_VertexPos[1],pt2.m_VertexPos[2]);
			printf("dof{%d,%d,%d}\n",pt0.m_elemVertexRelatedDofs[0],pt0.m_elemVertexRelatedDofs[1],pt0.m_elemVertexRelatedDofs[2]);
			printf("weight{%f,%f,%f}\n",pt0.m_TriLinearWeight[0],pt0.m_TriLinearWeight[1],pt0.m_TriLinearWeight[2]);

		}
	}
	free(DownVertex);
	free(DownEdge);
	free(DownTri);
	printf("\n");
#if 0 //for Up
	MC_Vertex_Cuda* UpVertex = (MC_Vertex_Cuda*)malloc(g_nUpBladeVertexSize * sizeof(MC_Vertex_Cuda));
	MC_Edge_Cuda* UpEdge = (MC_Edge_Cuda*)malloc(g_nUpBladeEdgeSize * sizeof(MC_Edge_Cuda));
	MC_Surface_Cuda* UpTri = (MC_Surface_Cuda*)malloc(g_nUpBladeSurfaceSize * sizeof(MC_Surface_Cuda));

	HANDLE_ERROR( cudaMemcpy( (void *)UpVertex,	g_UpBlade_MC_Vertex_Cuda,	(g_nUpBladeVertexSize) * sizeof(MC_Vertex_Cuda),	cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( (void *)UpEdge,	g_UpBlade_MC_Edge_Cuda,	(g_nUpBladeEdgeSize) * sizeof(MC_Edge_Cuda),	cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( (void *)UpTri,	g_UpBlade_MC_Surface_Cuda,	(g_nUpBladeSurfaceSize) * sizeof(MC_Surface_Cuda),	cudaMemcpyDeviceToHost ) );

	for (unsigned t=0;t<g_nUpBladeSurfaceSize;++t)
	{
		MC_Surface_Cuda& curTri = UpTri[t];
		if (curTri.m_isValid)
		{
			MC_Vertex_Cuda& pt0 = UpVertex[curTri.m_Vertex[0]];
			MC_Vertex_Cuda& pt1 = UpVertex[curTri.m_Vertex[1]];
			MC_Vertex_Cuda& pt2 = UpVertex[curTri.m_Vertex[2]];
			printf("{%f,%f,%f,%f,%f,%f,%f,%f,%f},",pt0.m_VertexPos[0],pt0.m_VertexPos[1],pt0.m_VertexPos[2],
													pt1.m_VertexPos[0],pt1.m_VertexPos[1],pt1.m_VertexPos[2],
													pt2.m_VertexPos[0],pt2.m_VertexPos[1],pt2.m_VertexPos[2]);

		}
	}
	free(UpVertex);
	free(UpEdge);
	free(UpTri);
	printf("\n");
#endif
}

void getMCStructureSize(int & nVertexSize,int & nEdgeSize, int& nFaceSize)
{
	nVertexSize = g_nMCVertexSize;
	nEdgeSize = g_nMCEdgeSize;
	nFaceSize = g_nMCSurfaceSize;
}

void getMCStructureData( int nVertexSize,MC_Vertex_Cuda* curVertexSet,
						 int nLineSize,MC_Edge_Cuda* curLineSet,
						 int nTriSize,MC_Surface_Cuda* curFaceSet)
{
	printf("nVertexSize(%d) nLineSize(%d) nTriSize(%d)\n",nVertexSize,nLineSize,nTriSize);
	HANDLE_ERROR( cudaMemcpy( (void *)curVertexSet,	g_MC_Vertex_Cuda,	(nVertexSize) * sizeof(MC_Vertex_Cuda),	cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( (void *)curLineSet,	g_MC_Edge_Cuda,		(nLineSize) * sizeof(MC_Edge_Cuda),		cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( (void *)curFaceSet,	g_MC_Surface_Cuda,	(nTriSize) * sizeof(MC_Surface_Cuda),		cudaMemcpyDeviceToHost ) );
}

#endif
