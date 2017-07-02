#include "VR_MACRO.h"
#include "VR_GPU_Utils.h"//for profiler
#include "MyFunctionCall.h"
#include "MyGLM.h"

/************************************************************************/
/*      For krylov cg  begin                                            */
/************************************************************************/
#include <cusp/array1d.h>
#include <cusp/blas.h>
#include <cusp/multiply.h>
#include <cusp/monitor.h>
#include <cusp/linear_operator.h>
#include <cusp/print.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <thrust/system/cuda/detail/detail/launch_closure.h>
#include <thrust/system/cuda/detail/reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
/************************************************************************/
/*      For krylov cg  end                                              */
/************************************************************************/

#include <helper_math.h>
//#include "VR_CUDA_GlobalDefine.cuh"
#include "cuda_runtime.h"//cudaMalloc
#include "driver_types.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#define USE_DEBUG (1)

#define USE_CUPRINTF (1)
#if USE_CUPRINTF
#include "./CUDA/cuPrintf.cu"

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

#define CONST_8X9 (72)

//The macro CUPRINTF is defined for architectures
//with different compute capabilities.
#if __CUDA_ARCH__ < 200     //Compute capability 1.x architectures
#define CUPRINTF cuPrintf
#else                       //Compute capability 2.x architectures
#define CUPRINTF(fmt, ...) printf("[%d, %d]:\t" fmt, \
	blockIdx.y*gridDim.x+blockIdx.x,\
	threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
	__VA_ARGS__)
#endif

__global__ void testKernel(int val)
{
	CUPRINTF("\tValue is:%d\n", val);
}

extern "C" int initCuPrintf(int argc, char **argv)
{
	int devID;
	cudaDeviceProp props;

	// This will pick the best possible CUDA capable device
	devID = findCudaDevice(argc, (const char **)argv);

	//Get GPU information
	checkCudaErrors(cudaGetDevice(&devID));
	checkCudaErrors(cudaGetDeviceProperties(&props, devID));
	printf("Device %d: \"%s\" with Compute %d.%d capability\n",
		devID, props.name, props.major, props.minor);

	//Architectures with compute capability 1.x, function
	//cuPrintf() is used. Otherwise, function printf() is called.
	bool use_cuPrintf = (props.major < 2);

	if (use_cuPrintf)
	{
		//Initializaton, allocate buffers on both host
		//and device for data to be printed.
		cudaPrintfInit();

		printf("cuPrintf() is called. Output:\n\n");
	}
	//Architecture with compute capability 2.x, function
	//printf() is called.
	else
	{
		printf("printf() is called. Output:\n\n");
	}

	//Kernel configuration, where a two-dimensional grid and
	//three-dimensional blocks are configured.
	dim3 dimGrid(2, 2);
	dim3 dimBlock(2, 2, 2);
	testKernel<<<dimGrid, dimBlock>>>(10);
	cudaDeviceSynchronize();

	if (use_cuPrintf)
	{
		//Dump current contents of output buffer to standard
		//output, and origin (block id and thread id) of each line
		//of output is enabled(true).
		cudaPrintfDisplay(stdout, true);

		//Free allocated buffers by cudaPrintfInit().
		cudaPrintfEnd();
	}

	cudaDeviceReset();

	return EXIT_SUCCESS;
}
#endif

#define WRAP_SIZE (32)
#define MAX_BLOCK_SIZE (8192)
#define MAX_KERNEL_PER_BLOCK (1024)

#include "VR_GPU_Physic_StructInfo.h"
#include "VR_Geometry_TriangleMeshStruct.h"


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
#define GRIDCOUNT(THREAD_COUNT, THREAD_COUNT_PER_BLOCK) ((THREAD_COUNT + THREAD_COUNT_PER_BLOCK - 1) / THREAD_COUNT_PER_BLOCK)


#include "VR_Physic_StructOnCUDA.h"


#define Definition_Host_Device_Buffer(cpuPtr,gpuPtr,DataType,DataSize)\
	HANDLE_ERROR( cudaHostAlloc( (void**)&cpuPtr, DataSize * sizeof(DataType),cudaHostAllocMapped   )) ;\
	HANDLE_ERROR(cudaHostGetDevicePointer((void **)&gpuPtr,(void *)(cpuPtr),0));\
	HANDLE_ERROR( cudaMemset( (void*)gpuPtr,	0, DataSize * sizeof(DataType)));


#define Definition_Device_Buffer_With_Data(ptrOnCuda,dataType,dataSize,externalFactor,ptrInitData)\
	HANDLE_ERROR( cudaMalloc( (void**)&ptrOnCuda, externalFactor * dataSize * sizeof(dataType)));\
	HANDLE_ERROR( cudaMemset( (void*)ptrOnCuda,	'\0', externalFactor * dataSize * sizeof(dataType)));\
	HANDLE_ERROR( cudaMemcpy( (void *)ptrOnCuda,ptrInitData,dataSize * sizeof(dataType), cudaMemcpyHostToDevice));

#define Definition_Device_Buffer_With_Zero(ptrOnCuda,dataType,dataSize)\
	HANDLE_ERROR( cudaMalloc( (void**)&ptrOnCuda, dataSize * sizeof(dataType)));\
	HANDLE_ERROR( cudaMemset( (void*)ptrOnCuda,	0, dataSize * sizeof(dataType)));

#define Mem_Zero(ptrOnCuda,dataType,dataSize)\
	HANDLE_ERROR( cudaMemset( (void*)ptrOnCuda,0,dataSize * sizeof(dataType))) ;

#define CUDA_SKINNING (1)
#if CUDA_SKINNING
int g_cuda_triangleMeshVertex_Count = MyZero;
int g_cuda_CellCenterPoint_Count = MyZero;

MeshVertex2CellInfo * g_cuda_triangleMeshVertex = MyNull;
MeshVertex2CellInfo * g_cpu_triangleMeshVertex = MyNull;

Cell2MeshVertexInfo * g_cuda_CellCenterPoint = MyNull;
Cell2MeshVertexInfo * g_cpu_CellCenterPoint = MyNull;

YC::Geometry::TriangleMeshNode* g_cuda_TriangleMeshNode = MyNull;
YC::Geometry::TriangleMeshNode* g_cpu_TriangleMeshNode = MyNull;

int initCudaStructForSkinningTriLinearWeight(const int nVtxSize, const int nCellSize)
{
	g_cuda_triangleMeshVertex_Count = nVtxSize;
	g_cuda_CellCenterPoint_Count = nCellSize;;

	HANDLE_ERROR( cudaHostAlloc( (void**)&g_cpu_triangleMeshVertex, nVtxSize * sizeof(MeshVertex2CellInfo),cudaHostAllocMapped   )) ;
	HANDLE_ERROR(cudaHostGetDevicePointer((void **)&g_cuda_triangleMeshVertex,(void *)(g_cpu_triangleMeshVertex),0));
	HANDLE_ERROR( cudaMemset( (void*)g_cuda_triangleMeshVertex,	0, nVtxSize * sizeof(MeshVertex2CellInfo))) ;

	HANDLE_ERROR( cudaHostAlloc( (void**)&g_cpu_CellCenterPoint, nCellSize * sizeof(Cell2MeshVertexInfo),cudaHostAllocMapped   )) ;
	HANDLE_ERROR(cudaHostGetDevicePointer((void **)&g_cuda_CellCenterPoint,(void *)(g_cpu_CellCenterPoint),0));
	HANDLE_ERROR( cudaMemset( (void*)g_cuda_CellCenterPoint,	0, nCellSize * sizeof(Cell2MeshVertexInfo))) ;

	HANDLE_ERROR( cudaHostAlloc( (void**)&g_cpu_TriangleMeshNode, nVtxSize * sizeof(YC::Geometry::TriangleMeshNode),cudaHostAllocMapped   )) ;
	HANDLE_ERROR(cudaHostGetDevicePointer((void **)&g_cuda_TriangleMeshNode,(void *)(g_cpu_TriangleMeshNode),0));
	HANDLE_ERROR( cudaMemset( (void*)g_cuda_TriangleMeshNode,	0, nVtxSize * sizeof(YC::Geometry::TriangleMeshNode))) ;
	return 0;
}

void freeCudaStructForSkinningTriLinearWeight()
{
	cudaFreeHost(g_cpu_triangleMeshVertex);
	cudaFreeHost(g_cpu_CellCenterPoint);
	cudaFreeHost(g_cpu_TriangleMeshNode);
}

__global__ void cuda_ComputeTrilinearWeight(const int nVtxSize, MeshVertex2CellInfo * triangleMeshVertex, YC::Geometry::TriangleMeshNode* triangleMeshNode,
	const int nCellSize,Cell2MeshVertexInfo * cellCenterPoint)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;	
	//const int vtxId = tid / MAX_KERNEL_PER_BLOCK;


	if (tid < nVtxSize)
	{
		YC::Geometry::TriangleMeshNode& refNode = triangleMeshNode[tid];
		Cell2MeshVertexInfo & refCellInfo = cellCenterPoint[triangleMeshVertex[tid].m_cellIdBelong];

		refNode.nBelongCellId = triangleMeshVertex[tid].m_cellIdBelong;

		const float3 step0 = make_float3(-1.f,-1.f,-1.f);
		const float3 step7 = make_float3(1.f,1.f,1.f);
		float3 p0 = refCellInfo.m_radius * step0 + refCellInfo.m_centerPos;
		float3 p7 = refCellInfo.m_radius * step7 + refCellInfo.m_centerPos;

		float3 tmpUp = p7 - triangleMeshVertex[tid].m_vertexPos;
		float3 tmpDown = p7 - p0;


		float detaX = tmpUp.x / tmpDown.x;
		float detaY = tmpUp.y / tmpDown.y;
		float detaZ = tmpUp.z / tmpDown.z;

		refNode.m_TriLinearWeight[0] = detaX * detaY * detaZ;
		refNode.m_TriLinearWeight[1] = (1-detaX) * detaY * detaZ;
		refNode.m_TriLinearWeight[2] = detaX * (1-detaY) * detaZ;
		refNode.m_TriLinearWeight[3] = (1-detaX) * (1-detaY) * detaZ;
		refNode.m_TriLinearWeight[4] = detaX * detaY * (1-detaZ);
		refNode.m_TriLinearWeight[5] = (1-detaX) * detaY * (1-detaZ);
		refNode.m_TriLinearWeight[6] = detaX * (1-detaY) * (1-detaZ);
		refNode.m_TriLinearWeight[7] = (1-detaX) * (1-detaY) * (1-detaZ);

		for (int vv = 0; vv < 24; ++vv)
		{
			refNode.m_VertexDofs[vv] = refCellInfo.m_nDofs[vv];
		}

		/*if (0 == tid)
		{
		CUPRINTF("tmpUp(%f,%f,%f) tmpDown(%d,%d,%d)\n",refNode.m_TriLinearWeight[0],refNode.m_TriLinearWeight[1],refNode.m_TriLinearWeight[2],
		refNode.m_VertexDofs[0],refNode.m_VertexDofs[1],refNode.m_VertexDofs[2]);
		}*/
	}
}

__global__ void cuda_ParserMeshVertex2CellCenterPoint(const int nVtxSize, MeshVertex2CellInfo * triangleMeshVertex,
	const int nCellSize,Cell2MeshVertexInfo * cellCenterPoint,
	const int nVtxBegin, const int nVtxEnd,
	const int nCellBegin, const int nCellEnd)
{
	__shared__ float cellDistance [MAX_KERNEL_PER_BLOCK];
	__shared__ int cellId [MAX_KERNEL_PER_BLOCK];

	const int nVtxId = blockIdx.x + nVtxBegin;	
	//const int nFaceBase = threadIdx.x;
	const int cellIdx = threadIdx.x;
	int nCellId = nCellBegin + threadIdx.x;
	const int nCellStep = blockDim.x;

	cellDistance[cellIdx] = FLT_MAX;//否则出现乱ID
	cellId[cellIdx] = Invalid_Id;

	if ( (nVtxId < nVtxSize) && (nCellId < nCellSize)  )
	{
		MeshVertex2CellInfo& CurrentVtxInfo = triangleMeshVertex[nVtxId];
		float3& currentCellCenter = cellCenterPoint[nCellId].m_centerPos;

		float3 v = CurrentVtxInfo.m_vertexPos - currentCellCenter;
		float dist = dot(v,v);

		cellDistance[cellIdx] = dist;
		cellId[cellIdx] = nCellId;

		__syncthreads();
		if (cellIdx < WRAP_SIZE)
		{
			for (int lane = cellIdx+WRAP_SIZE;lane < nCellStep; lane+= WRAP_SIZE)
			{
				if (cellDistance[cellIdx] > cellDistance[lane])
				{
					cellDistance[cellIdx] = cellDistance[lane];
					cellId[cellIdx] = cellId[lane];
				}
			}
			if ( cellIdx < 16)
			{
				if (cellDistance[cellIdx] > cellDistance[cellIdx+16])
				{
					cellDistance[cellIdx] = cellDistance[cellIdx+16];
					cellId[cellIdx] = cellId[cellIdx+16];
				}
			}
			if ( cellIdx < 8)
			{
				if (cellDistance[cellIdx] > cellDistance[cellIdx+8])
				{
					cellDistance[cellIdx] = cellDistance[cellIdx+8];
					cellId[cellIdx] = cellId[cellIdx+8];
				}
			}
			if ( cellIdx < 4)
			{
				if (cellDistance[cellIdx] > cellDistance[cellIdx+4])
				{
					cellDistance[cellIdx] = cellDistance[cellIdx+4];
					cellId[cellIdx] = cellId[cellIdx+4];
				}
			}
			if ( cellIdx < 2)
			{
				if (cellDistance[cellIdx] > cellDistance[cellIdx+2])
				{
					cellDistance[cellIdx] = cellDistance[cellIdx+2];
					cellId[cellIdx] = cellId[cellIdx+2];
				}
			}
			if ( cellIdx < 1)
			{
				if (cellDistance[cellIdx] > cellDistance[cellIdx+1])
				{
					cellDistance[cellIdx] = cellDistance[cellIdx+1];
					cellId[cellIdx] = cellId[cellIdx+1];
				}
				if (CurrentVtxInfo.m_dist > cellDistance[0])
				{
					CurrentVtxInfo.m_dist = cellDistance[0];
					CurrentVtxInfo.m_cellIdBelong = cellId[0];
				}
			}
			// first thread writes the result			
		}
	}
}

void createVertex2CellForSkinning()
{
#define Vertex2Cell_Mesh_BlockSize (8192)
#define Vertex2Cell_Mesh_ThreadPerBlock (1024)
	LogInfo("TriangleVertexCount %d, CellCount %d\n",g_cuda_triangleMeshVertex_Count,g_cuda_CellCenterPoint_Count);
	for (int nVtxBegin=0;nVtxBegin < g_cuda_triangleMeshVertex_Count;nVtxBegin += Vertex2Cell_Mesh_BlockSize)
	{
		const int nVtxEnd = ((nVtxBegin+Vertex2Cell_Mesh_BlockSize) < g_cuda_triangleMeshVertex_Count) ? (nVtxBegin+Vertex2Cell_Mesh_BlockSize) : (g_cuda_triangleMeshVertex_Count);
		for (int nCellBegin=0;nCellBegin < g_cuda_CellCenterPoint_Count;nCellBegin += Vertex2Cell_Mesh_ThreadPerBlock)
		{			
			const int nCellEnd = ((nCellBegin+Vertex2Cell_Mesh_ThreadPerBlock) < g_cuda_CellCenterPoint_Count) ? (nCellBegin+Vertex2Cell_Mesh_ThreadPerBlock) : (g_cuda_CellCenterPoint_Count);
			LogInfo("MeshVertex[%d,%d] Cell[%d,%d]\n",nVtxBegin,nVtxEnd,nCellBegin,nCellEnd);
			//MyPause;
			cuda_ParserMeshVertex2CellCenterPoint<<<Vertex2Cell_Mesh_BlockSize ,Vertex2Cell_Mesh_ThreadPerBlock>>>
				(g_cuda_triangleMeshVertex_Count, g_cuda_triangleMeshVertex,
				g_cuda_CellCenterPoint_Count, g_cuda_CellCenterPoint,
				nVtxBegin,nVtxEnd,
				nCellBegin,nCellEnd);
			cudaDeviceSynchronize();

		}
	}

	cuda_ComputeTrilinearWeight<<<GRIDCOUNT(g_cuda_triangleMeshVertex_Count,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK>>>
		(g_cuda_triangleMeshVertex_Count, g_cuda_triangleMeshVertex, g_cuda_TriangleMeshNode,
		g_cuda_CellCenterPoint_Count, g_cuda_CellCenterPoint);
	cudaDeviceSynchronize();
	LogInfo("createVertex2CellForSkinning Finish!\n");

}
#endif

#define USE_CUDA_SIMULATION (1)
#if USE_CUDA_SIMULATION
extern int g_nNativeSurfaceVertexCount;
extern int cuda_bcMaxCount;
extern int cuda_bcMinCount;
extern float cuda_scriptForceFactor;

PhysicsContext FEM_State_Ctx;

namespace YC
{
	namespace GlobalVariable
	{
		extern float g_bladeForce; 
		extern int g_isApplyBladeForceCurrentFrame;
	}
}

namespace CUDA_SIMULATION
{
	namespace CUDA_SKNNING_CUTTING
	{
		extern VBOStructForDraw g_VBO_Struct_Node;
	}

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

	namespace CUDA_CUTTING_GRID
	{
		typedef glm::vec3 Vector3;
		
		int nouseRHSSize=0;
		float nouseRHSVec[16083 * 2];
		float nouseCusp_Array_Rhs[16083 * 2];
		float nouseCusp_Array_Old_Acceleration[16083 * 2];
		float nouseCusp_Array_Old_Displacement[16083 * 2];
		float nouseCusp_Array_R_rhs_Corotaion[16083 * 2];
		bool g_needCheckCutting = false;

		void debug_ShowDofs()
		{
			const int nVtxSize = FEM_State_Ctx.g_nVertexOnCudaCount;
			VertexOnCuda*  vtxCpu = FEM_State_Ctx.g_VertexOnCudaPtrOnCPU;
			printf("dofs : ");
			for (int i=0;i<nVtxSize;++i)
			{
				const VertexOnCuda& ref = vtxCpu[i];
				printf("{%d,%d,%d},",ref.m_nGlobalDof[0],ref.m_nGlobalDof[1],ref.m_nGlobalDof[2]);
			}
			printf("\n");
		}

		 __host__ __device__ float cuda_Area(const glm::vec3& p0,const glm::vec3& p1,const glm::vec3& p2 )
		{
			return glm::length(glm::cross(p0-p1,p1-p2)) / 2.f;
			/*const float3 tmp = cross(p0-p1,p1-p2);
			return sqrtf(tmp.x*tmp.x+tmp.y*tmp.y+tmp.z*tmp.z) / 2.f;*/
		}

		void setCurrentBlade(const int nIdx,const Vector3& lastHandle, const Vector3& lastTip,const Vector3& currentHandle, const Vector3& currentTip)
		{
			Q_ASSERT(nIdx < (MaxCuttingBladeStructCount));
			CUDA_SKNNING_CUTTING::g_VBO_Struct_Node.g_nCuttingBladeCount = nIdx+1;
			CuttingBladeStruct& refNode = CUDA_SKNNING_CUTTING::g_VBO_Struct_Node.g_CuttingBladeStructOnCpu[nIdx];
			refNode.m_lastRayHandle = lastHandle;
			refNode.m_lastRayTip = lastTip;
			refNode.m_currentRayHandle = currentHandle;
			refNode.m_currentRayTip = currentTip;
			refNode.m_bladeNormal_lh_lt_ct = glm::normalize(glm::cross( (lastTip-lastHandle),(currentTip-lastTip) ));
			refNode.m_bladeNormal_ct_ch_lh = glm::normalize(glm::cross( (currentHandle-currentTip),(lastHandle-currentHandle) ));
			refNode.m_cuttingArea_lh_lt_ct = cuda_Area(lastHandle,lastTip,currentTip);
			refNode.m_cuttingArea_ct_ch_lh = cuda_Area(currentTip,currentHandle,lastHandle);
			
		}

		void resetCuttingFlagElement(bool needCheckCutting)
		{
			cudaDeviceSynchronize();
			g_needCheckCutting = needCheckCutting;
			PhysicsContext& currentCtx = FEM_State_Ctx;
			HANDLE_ERROR( cudaMemset( (void*)currentCtx.beCuttingLinesFlagElement,	0, currentCtx.g_nCuttingLineSetCount * sizeof(char)));
			HANDLE_ERROR( cudaMemset( (void*)currentCtx.topLevelCellInMain_OnCuda,	0, currentCtx.nCellOnCudaCount * sizeof(IndexType)));
			HANDLE_ERROR( cudaMemset( (void*)currentCtx.g_CellCollisionFlag_onCuda,	0, currentCtx.nCellOnCudaCount * sizeof(IndexType)));
		}

		__device__ bool checkLineTri(const glm::vec3& L1,const glm::vec3&  L2,const glm::vec3& PV1,const glm::vec3& PV2,const glm::vec3& PV3 )
		{
			glm::vec3 VIntersect;
			glm::vec3 VNorm,tmpVec3;

			tmpVec3 = glm::cross( (PV2 - PV1),(PV3-PV1) );//tmpVec3 = cross(make_float3(PV2.x - PV1.x,PV2.y - PV1.y,PV2.z - PV1.z),make_float3(PV3.x - PV1.x,PV3.y - PV1.y,PV3.z - PV1.z));
			VNorm = glm::normalize(tmpVec3);//VNorm = normalize(tmpVec3);
			float fDst1 = glm::dot( (L1 - PV1),(VNorm) );//float fDst1 = dot( make_float3(L1.x - PV1.x,L1.y - PV1.y,L1.z - PV1.z),VNorm );
			float fDst2 = glm::dot( (L2 - PV1),(VNorm) );//dot( make_float3(L2.x - PV1.x,L2.y - PV1.y,L2.z - PV1.z),VNorm );
			if ( (fDst1 * fDst2) >= 0.0f) return false;  // line doesn't cross the triangle.
			if ( fDst1 == fDst2) {return false;} // line and plane are parallel
			// Find point on the line that intersects with the plane
			VIntersect = L1 + (L2-L1) * ( -fDst1/(fDst2-fDst1) );
			//float3 VTest;
			glm::vec3 VTest = glm::cross( VNorm,(PV2-PV1) );//cross(VNorm,PV2-PV1);
			if ( glm::dot(VTest,VIntersect-PV1) < 0.0f ) return false;
			VTest = glm::cross(VNorm,PV3-PV2);
			if ( glm::dot(VTest,VIntersect-PV2) < 0.0f ) return false;
			VTest = glm::cross(VNorm, PV1-PV3);
			if ( glm::dot(VTest,VIntersect-PV1) < 0.0f ) return false;
			return true;
		}

		__device__ float checkPointPlaneOnCuda(const glm::vec3& destPoint, const glm::vec3& bladePoint, const glm::vec3& bladeNormal)
		{
			return glm::dot( bladeNormal, (bladePoint-destPoint) );
		}
#if 1
		
		__global__ void collisionDetection_onMain(const int nLocalDomainId,const int nDofCount,
			const int nCellCount,CommonCellOnCuda * cellOnCudaPointer,VertexOnCuda * VertexOnCudaPtr,
			CuttingLinePair * cuttingLineSetPair,/*int * CellCollisionFlag,*/
			CuttingBladeStruct * cuttingBladeStruct, char* beCuttingLinesFlagElement,
			IndexTypePtr topLevelCellInMain_OnCuda)
		{
			const int currentCellIdx = threadIdx.x + blockIdx.x * blockDim.x;
			if (currentCellIdx < nCellCount && cellOnCudaPointer[currentCellIdx].m_bLeaf && cellOnCudaPointer[currentCellIdx].m_needBeCutting)
			{
				CommonCellOnCuda& currentCellRef = cellOnCudaPointer[currentCellIdx];

				int curVertexId;
				topLevelCellInMain_OnCuda[currentCellIdx] = 0;
				currentCellRef.m_bTopLevelOctreeNodeList = false;

				currentCellRef.m_bladeForceDirct[0] = 0.f;
				currentCellRef.m_bladeForceDirct[1] = 0.f;
				currentCellRef.m_bladeForceDirct[2] = 0.f;
				currentCellRef.m_bladeForceDirectFlag = 0;//initialize		
								
				CuttingLinePair * lineBase = cuttingLineSetPair + currentCellRef.m_nLinesBaseIdx;

				const glm::vec3& lastRayHandle = cuttingBladeStruct->m_lastRayHandle;
				const glm::vec3& lastRayTip = cuttingBladeStruct->m_lastRayTip;
				const glm::vec3& currentRayHandle = cuttingBladeStruct->m_currentRayHandle;
				const glm::vec3& currentRayTip = cuttingBladeStruct->m_currentRayTip;
				const float bladeArea_ct_ch_lh = cuttingBladeStruct->m_cuttingArea_ct_ch_lh;
				const float bladeArea_lh_lt_ct = cuttingBladeStruct->m_cuttingArea_lh_lt_ct;
				const glm::vec3& bladeNormal_lh_lt_ct = cuttingBladeStruct->m_bladeNormal_lh_lt_ct;
				const glm::vec3& bladeNormal_ct_ch_lh = cuttingBladeStruct->m_bladeNormal_ct_ch_lh;

				for (int v=0; v<currentCellRef.m_nLinesCount;++v)
				{
					if (0 == beCuttingLinesFlagElement[currentCellRef.m_nLinesBaseIdx+v])
					{
						const glm::vec3& cuttingBladeHandle = lineBase[v].first;
						const glm::vec3& cuttingBladeTip = lineBase[v].second;

						if (bladeArea_lh_lt_ct > MinCuttingBladeArea &&
							true == checkLineTri( cuttingBladeHandle, cuttingBladeTip,  lastRayHandle,lastRayTip,currentRayTip) )
						{
							if (0 == currentCellRef.m_nGhostCellCount)
							{
								topLevelCellInMain_OnCuda[currentCellIdx] = 1;
								currentCellRef.m_bTopLevelOctreeNodeList = true;
								for (int p=0;p<8;++p)
								{
									curVertexId = currentCellRef.vertexId[p];
									currentCellRef.nPointPlane[p] = checkPointPlaneOnCuda(VertexOnCudaPtr[ curVertexId ].local,lastRayTip,bladeNormal_lh_lt_ct);
								}
							}
							else
							{
								CUPRINTF("currentCellRef.m_nGhostCellCount>0\n");
								while (true) ;
							}
							currentCellRef.m_bladeForceDirct = bladeNormal_lh_lt_ct;
							/*if (0 == currentCellIdx)
							{
								CUPRINTF("1{%f,%f,%f},{%f,%f,%f}\n",
									cuttingBladeHandle[0],cuttingBladeHandle[1],cuttingBladeHandle[2],
									cuttingBladeTip[0],cuttingBladeTip[1],cuttingBladeTip[2]);
								CUPRINTF("1{%f,%f,%f},{%f,%f,%f},{%f,%f,%f}\n",
									lastRayHandle[0],lastRayHandle[1],lastRayHandle[2],
									lastRayTip[0],lastRayTip[1],lastRayTip[2],
									currentRayTip[0],currentRayTip[1],currentRayTip[2]);
							}*/
							break;
						}

						if (bladeArea_ct_ch_lh > MinCuttingBladeArea &&
							true == checkLineTri( cuttingBladeHandle, cuttingBladeTip,  currentRayTip,currentRayHandle,lastRayHandle) )
						{
							if (0 == currentCellRef.m_nGhostCellCount)
							{
								topLevelCellInMain_OnCuda[currentCellIdx] = 1;
								currentCellRef.m_bTopLevelOctreeNodeList = true;
								for (int p=0;p<8;++p)
								{
									curVertexId = currentCellRef.vertexId[p];
									currentCellRef.nPointPlane[p] = checkPointPlaneOnCuda(VertexOnCudaPtr[ curVertexId ].local,currentRayHandle,bladeNormal_ct_ch_lh);
								}
							}
							else
							{
								CUPRINTF("currentCellRef.m_nGhostCellCount>0\n");
								while (true) ;
							}
							currentCellRef.m_bladeForceDirct = bladeNormal_ct_ch_lh;
							/*if (0 == currentCellIdx)
							{
								CUPRINTF("2{%f,%f,%f},{%f,%f,%f}\n",
									cuttingBladeHandle[0],cuttingBladeHandle[1],cuttingBladeHandle[2],
									cuttingBladeTip[0],cuttingBladeTip[1],cuttingBladeTip[2]);
								CUPRINTF("2{%f,%f,%f},{%f,%f,%f},{%f,%f,%f}\n",
									currentRayTip[0],currentRayTip[1],currentRayTip[2],
									currentRayHandle[0],currentRayHandle[1],currentRayHandle[2],
									lastRayHandle[0],lastRayHandle[1],lastRayHandle[2]);
							}*/
							break;
						}
					}//if (0 == beCuttingLinesFlagElement[currentCellRef.m_nLinesBaseIdx+v])
				}

			}//if (currentCellIdx < nCellCount && cellOnCudaPointer[currentCellIdx].m_bLeaf && cellOnCudaPointer[currentCellIdx].m_needBeCutting)
		}

		__global__ void ReInitCellOnCuda(const int nLocalDomainId,
			int nCellCount,CommonCellOnCuda * cellOnCuda, VertexOnCuda* VertexOnCudaPtr, 
			IndexTypePtr beClonedObjectFlag, IndexTypePtr beCloneVertexFlag )
		{
			int currentCellIdx = threadIdx.x + blockIdx.x * blockDim.x;
			if (currentCellIdx < nCellCount && true == cellOnCuda[currentCellIdx].m_bLeaf )
			{
				if (cellOnCuda[currentCellIdx].m_bTopLevelOctreeNodeList)
				{
					int * localVtxId = &cellOnCuda[currentCellIdx].vertexId[0];
					beClonedObjectFlag[currentCellIdx] = 1;
					beCloneVertexFlag[ localVtxId[0] ] = 1;
					beCloneVertexFlag[ localVtxId[1] ] = 1;
					beCloneVertexFlag[ localVtxId[2] ] = 1;
					beCloneVertexFlag[ localVtxId[3] ] = 1;
					beCloneVertexFlag[ localVtxId[4] ] = 1;
					beCloneVertexFlag[ localVtxId[5] ] = 1;
					beCloneVertexFlag[ localVtxId[6] ] = 1;
					beCloneVertexFlag[ localVtxId[7] ] = 1;
				}
				else
				{
					beClonedObjectFlag[currentCellIdx] = 0;
				}
			}
		}

		__global__ void cloneVertexOnCuda(const int nLocalDomainId,int nVertexOnCudaCount, VertexOnCuda* VertexOnCudaPtr, IndexTypePtr beCloneVertexFlag, int nDofBase, int createTimeStamp)
		{
			int currentVertexIdx = threadIdx.x + blockIdx.x * blockDim.x;
			if (currentVertexIdx < nVertexOnCudaCount && (beCloneVertexFlag[currentVertexIdx+1] - beCloneVertexFlag[currentVertexIdx])>0 )
			{
				int cloneVertexIdx = nVertexOnCudaCount + beCloneVertexFlag[currentVertexIdx];
				int cloneDofBase = nDofBase + beCloneVertexFlag[currentVertexIdx] * 3;

				VertexOnCudaPtr[cloneVertexIdx] = VertexOnCudaPtr[currentVertexIdx];
				VertexOnCudaPtr[cloneVertexIdx].local = VertexOnCudaPtr[currentVertexIdx].local;
				/*VertexOnCudaPtr[cloneVertexIdx].local[0] = VertexOnCudaPtr[currentVertexIdx].local[0];
				VertexOnCudaPtr[cloneVertexIdx].local[1] = VertexOnCudaPtr[currentVertexIdx].local[1];
				VertexOnCudaPtr[cloneVertexIdx].local[2] = VertexOnCudaPtr[currentVertexIdx].local[2];*/
				VertexOnCudaPtr[cloneVertexIdx].m_createTimeStamp = createTimeStamp;
				VertexOnCudaPtr[cloneVertexIdx].m_nGlobalDof[0] = cloneDofBase;
				VertexOnCudaPtr[cloneVertexIdx].m_nGlobalDof[1] = cloneDofBase+1;
				VertexOnCudaPtr[cloneVertexIdx].m_nGlobalDof[2] = cloneDofBase+2;
				VertexOnCudaPtr[cloneVertexIdx].m_nId = cloneVertexIdx;
				VertexOnCudaPtr[currentVertexIdx].m_nCloneId = cloneVertexIdx;
			}
		}

		__global__ void cloneCellOnCuda(const int nLocalDomainId,int nCellOnCudaCount, CommonCellOnCuda * cellOnCuda, VertexOnCuda* VertexOnCudaPtr, IndexTypePtr beCloneCellFlag)
		{
			int currentCellIdx = threadIdx.x + blockIdx.x * blockDim.x;
			if (currentCellIdx < nCellOnCudaCount && (beCloneCellFlag[currentCellIdx+1] - beCloneCellFlag[currentCellIdx])>0)
			{
				int cloneCellIdx = nCellOnCudaCount + beCloneCellFlag[currentCellIdx];

				//CUPRINTF("currentCellIdx[%d] cloneCellIdx[%d]\n",currentCellIdx,cloneCellIdx);
				CommonCellOnCuda& srcCellRef =  cellOnCuda[currentCellIdx];
				CommonCellOnCuda& cloneCellRef = cellOnCuda[cloneCellIdx];

				cellOnCuda[cloneCellIdx] = cellOnCuda[currentCellIdx];

				cloneCellRef.m_bLeaf = true;//cellOnCuda[currentCellIdx].m_bLeaf;
				srcCellRef.m_nCloneCellIdx = cloneCellIdx;
				cloneCellRef.m_nCloneCellIdx = -1;
				

				int nativeVertexId,cloneVertexId;

				cloneCellRef.m_bladeForceDirct *= -1.f;
				/*cloneCellRef.m_bladeForceDirct[0] *= -1.f;
				cloneCellRef.m_bladeForceDirct[1] *= -1.f;
				cloneCellRef.m_bladeForceDirct[2] *= -1.f;*/

				srcCellRef.m_bladeForceDirectFlag = 1;
				cloneCellRef.m_bladeForceDirectFlag = -1;
//				return ;
				for (int v=0;v<8;++v)
				{
					nativeVertexId = srcCellRef.vertexId[v];
					cloneVertexId = VertexOnCudaPtr[ nativeVertexId ].m_nCloneId;

#if 0
					srcCellRef.vertexId[v] = nativeVertexId;
					cloneCellRef.vertexId[v]   = cloneVertexId;
#else
					//CUPRINTF("nativeVertexId[%d] cloneVertexId[%d]\n",nativeVertexId,cloneVertexId);			

					if (srcCellRef.nPointPlane[v] < 0)
					{
						srcCellRef.vertexId[v] = nativeVertexId;
						cloneCellRef.vertexId[v]   = cloneVertexId;

						/*srcCellRef.m_cellInflunceVertexList[v] = nativeVertexId;
						cloneCellRef.m_cellInflunceVertexList[v]   = cloneVertexId;*/

					} 
					else
					{
						srcCellRef.vertexId[v] = cloneVertexId;
						cloneCellRef.vertexId[v]   = nativeVertexId;

						/*srcCellRef.m_cellInflunceVertexList[v] = cloneVertexId;
						cloneCellRef.m_cellInflunceVertexList[v]   = nativeVertexId;*/
					}
#endif

				}
			}
		}

		__global__ void cuda_BladeForceApplyToGrid(const int nLocalDomainId,
			int nCellCount,int nDofCount,CommonCellOnCuda * cellOnCudaPointer,VertexOnCuda * VertexOnCudaPtr,
			FEMShapeValue* FEMShapeValueOnCuda, ValueTypePtr globalExternalForceOnCudax8, ValueType force)
		{
			int tid = threadIdx.x + blockIdx.x * blockDim.x;
			const int nCellId = tid / 24;
			const int result0 = tid % 24;
			const int vtxId = result0/3;/*0-7*/
			const int nDim = result0%3;/*0-2*/

			if (nCellId < nCellCount && cellOnCudaPointer[nCellId].m_bLeaf/* && cellOnCudaPointer[nCellId].m_bTopLevelOctreeNodeList*/)
			{
				CommonCellOnCuda& currentCellRef = cellOnCudaPointer[nCellId];
				const float f = currentCellRef.m_bladeForceDirct[nDim] * force;		
				const float FxJxW = currentCellRef.m_nJxW * f;

				FEMShapeValue& curFEMShapeValue = FEMShapeValueOnCuda[currentCellRef.m_nFEMShapeIdx];
				
				const int currentVtxId = currentCellRef.vertexId[vtxId];		
				const int nCurrentDof = VertexOnCudaPtr[currentVtxId].m_nGlobalDof[nDim];

				float &Cvalue = globalExternalForceOnCudax8[nCurrentDof*8+vtxId];
				Cvalue = 0.f;

				for (int i=0;i<8;++i)
				{
					Cvalue += FxJxW*curFEMShapeValue.shapeFunctionValue_8_8[i][vtxId];
				}
			}
		}

		__global__ void assembleRhsValue_on_cuda_bladeForce(int nDofCount, ValueTypePtr systemRhsPtr,ValueTypePtr tmp_blockShareRhs)
		{
			const int currentDof = threadIdx.x + blockIdx.x * blockDim.x;
			if (currentDof < nDofCount)
			{
				ValueTypePtr blockShareRhs = tmp_blockShareRhs + currentDof*8;
				systemRhsPtr[currentDof] = blockShareRhs[0] + blockShareRhs[1] + blockShareRhs[2] + blockShareRhs[3] + 
					blockShareRhs[4] + blockShareRhs[5] + blockShareRhs[6] + blockShareRhs[7];
			}
		}

		void cuttingBladeCheckVolumnGrid(const int nCurrentBladeIdx)
		{
			MyFunctionCall;
			PhysicsContext& currentCtx = FEM_State_Ctx;
			
			collisionDetection_onMain<<<GRIDCOUNT(currentCtx.nCellOnCudaCount,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK  ,0, streamForSkin>>>
				(0,currentCtx.g_nDofs,
				currentCtx.nCellOnCudaCount,currentCtx.CellOnCudaPtr,currentCtx.g_VertexOnCudaPtr,
				currentCtx.g_CuttingLineSet,/*currentCtx.g_CellCollisionFlag_onCuda,*/
				&CUDA_SKNNING_CUTTING::g_VBO_Struct_Node.g_CuttingBladeStructOnCuda[nCurrentBladeIdx],currentCtx.beCuttingLinesFlagElement,
				currentCtx.topLevelCellInMain_OnCuda);

			cudaDeviceSynchronize();

			int nNativeCellCount = currentCtx.nCellOnCudaCount;

			//must be do it !
			Mem_Zero(currentCtx.beClonedVertexFlag_OnCuda,int,currentCtx.g_nVertexOnCudaCount);

			ReInitCellOnCuda<<<GRIDCOUNT(currentCtx.nCellOnCudaCount,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK  ,0, streamForSkin>>>
				(0,currentCtx.nCellOnCudaCount,currentCtx.CellOnCudaPtr,currentCtx.g_VertexOnCudaPtr, 
				currentCtx.beClonedObjectFlag_OnCuda, currentCtx.beClonedVertexFlag_OnCuda);
			cudaDeviceSynchronize();

			
			thrust::exclusive_scan(currentCtx.beClonedObjectFlag_OnHost, currentCtx.beClonedObjectFlag_OnHost+currentCtx.nCellOnCudaCount+1	, currentCtx.beClonedObjectFlag_OnHost); //generate ghost cell count

			thrust::exclusive_scan(currentCtx.beClonedVertexFlag_OnHost, currentCtx.beClonedVertexFlag_OnHost+currentCtx.g_nVertexOnCudaCount+1	, currentCtx.beClonedVertexFlag_OnHost); //generate ghost cell count

			int nNewCellInMain = currentCtx.beClonedObjectFlag_OnHost[currentCtx.nCellOnCudaCount];//be clone cell size
			int nNewVertexCount = currentCtx.beClonedVertexFlag_OnHost[currentCtx.g_nVertexOnCudaCount];//be clone vertex size
			printf("nNewCellInMain(%d) nNewVertexCount(%d)\n",nNewCellInMain,nNewVertexCount);
			
			int nNewDofCount = nNewVertexCount * 3;

			cloneVertexOnCuda<<<GRIDCOUNT(currentCtx.g_nVertexOnCudaCount,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK  ,0, streamForSkin>>>
				(0,currentCtx.g_nVertexOnCudaCount, currentCtx.g_VertexOnCudaPtr, currentCtx.beClonedVertexFlag_OnCuda, currentCtx.g_nDofs, nCurrentBladeIdx);
			currentCtx.g_nVertexOnCudaCount += nNewVertexCount;
			
			currentCtx.g_nDofs += nNewDofCount;

			cloneCellOnCuda<<<GRIDCOUNT(currentCtx.nCellOnCudaCount,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK  ,0, streamForSkin>>>
				(0,currentCtx.nCellOnCudaCount, currentCtx.CellOnCudaPtr, currentCtx.g_VertexOnCudaPtr, currentCtx.beClonedObjectFlag_OnCuda);
			cudaDeviceSynchronize();
			printf("cellsize %d %d, g_nVertexOnCudaCount[%d]\n",currentCtx.nCellOnCudaCount,currentCtx.nCellOnCudaCount+nNewCellInMain,currentCtx.g_nVertexOnCudaCount);
			currentCtx.nCellOnCudaCount += nNewCellInMain;
			//return;
			Mem_Zero(currentCtx.myOptimize_BladeForce,ValueType,currentCtx.g_nDofs);
			Mem_Zero(currentCtx.myOptimize_BladeForce_In8_MF,ValueType,8*(currentCtx.g_nDofs));
			
			cuda_BladeForceApplyToGrid<<<GRIDCOUNT(currentCtx.nCellOnCudaCount*24,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK  ,0, streamForSkin>>>
				(0,	currentCtx.nCellOnCudaCount,currentCtx.g_nDofs,currentCtx.CellOnCudaPtr,currentCtx.g_VertexOnCudaPtr,
				currentCtx.FEMShapeValueOnCuda, currentCtx.myOptimize_BladeForce_In8_MF,YC::GlobalVariable::g_bladeForce);

			assembleRhsValue_on_cuda_bladeForce<<<GRIDCOUNT(currentCtx.g_nDofs,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK  ,0, streamForSkin>>>
				(currentCtx.g_nDofs,currentCtx.myOptimize_BladeForce,currentCtx.myOptimize_BladeForce_In8_MF);

			/*CuspVec tmpBladeForce;
			currentCtx.g_globalBladeForce.resize(currentCtx.g_nDofs);		
			setCuspVector_deviceMemory(tmpBladeForce,currentCtx.g_nDofs,currentCtx.myOptimize_BladeForce);		
			cusp::blas::axpby(tmpBladeForce,currentCtx.g_globalBladeForce,	currentCtx.g_globalBladeForce,	ValueType(1),ValueType(1));		
			printf("assembleRhsValue_on_cuda_bladeForce !\n");*/
			currentCtx.g_isApplyBladeForceCurrentFrame = YC::GlobalVariable::g_isApplyBladeForceCurrentFrame;
		}
#endif

		

	}//namespace CUDA_CUTTING_GRID

	inline __device__ float3 myCross(const float3& a, const float3& b)
	{
		return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
	}

	inline __device__ float computeNorm(float3 a, float3 b)
	{
		return sqrtf((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)+(a.z-b.z)*(a.z-b.z));
	}
	
	CuspVec cusp_Array_NewMarkConstant;	
	ValueTypePtr element_Array_NewMarkConstant = MyNull;

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

#if USE_CO_RATION
	namespace CUDA_COROTATION
	{
		ValueTypePtr Signs_on_cuda = 0;

		__device__ float determinant(float* src_element)
		{
			return  src_element[0*MyMatrixRows+0]*src_element[1*MyMatrixRows+1]*src_element[2*MyMatrixRows+2]+
				src_element[0*MyMatrixRows+1]*src_element[1*MyMatrixRows+2]*src_element[2*MyMatrixRows+0]+
				src_element[0*MyMatrixRows+2]*src_element[1*MyMatrixRows+0]*src_element[2*MyMatrixRows+1]-
				src_element[2*MyMatrixRows+0]*src_element[1*MyMatrixRows+1]*src_element[0*MyMatrixRows+2]-
				src_element[1*MyMatrixRows+0]*src_element[0*MyMatrixRows+1]*src_element[2*MyMatrixRows+2]-
				src_element[0*MyMatrixRows+0]*src_element[2*MyMatrixRows+1]*src_element[1*MyMatrixRows+2];
		}
#define COROTATION_CLASSICAL (0)
#if COROTATION_CLASSICAL

		__global__ void update_u_v_a_init_corotaion(const int nCellCount, CommonCellOnCuda * cellOnCudaPointer, int /*nDofCount*/, ValueTypePtr global_incremental_displacement,ValueTypePtr para_Signs_on_cuda, VertexOnCuda* _VertexOnCudaPtr, int _VertexOnCudaCount)
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
					const int vtxId = cellOnCudaPointer[currentCellIdx].vertexId[i];

					const int currentDof = _VertexOnCudaPtr[vtxId].m_nGlobalDof[component_row];

					float u = weight * global_incremental_displacement[currentDof] * para_Signs_on_cuda[i*9+matrixIdx3x3];
					cellOnCudaPointer[currentCellIdx].RotationMatrix[matrixIdx3x3] += u;

				}
			}
		}


		__global__ void update_u_v_a_iterator_corotaion(const int nCellCount, CommonCellOnCuda * cellOnCudaPointer)
		{
			const int currentCellIdx = threadIdx.x + blockIdx.x * blockDim.x;
			if(currentCellIdx < nCellCount  && true == cellOnCudaPointer[currentCellIdx].m_bLeaf)
			{
				const int inverseIdx[3*3][4] = {{1*MyMatrixRows+1,2*MyMatrixRows+2,2*MyMatrixRows+1,1*MyMatrixRows+2},
				{0*MyMatrixRows+1,2*MyMatrixRows+2,2*MyMatrixRows+1,0*MyMatrixRows+2},
				{0*MyMatrixRows+1,1*MyMatrixRows+2,1*MyMatrixRows+1,0*MyMatrixRows+2},
				{1*MyMatrixRows+2,2*MyMatrixRows+0,2*MyMatrixRows+2,1*MyMatrixRows+0},
				{0*MyMatrixRows+2,2*MyMatrixRows+0,0*MyMatrixRows+0,2*MyMatrixRows+2},
				{0*MyMatrixRows+2,1*MyMatrixRows+0,0*MyMatrixRows+0,1*MyMatrixRows+2},
				{1*MyMatrixRows+0,2*MyMatrixRows+1,1*MyMatrixRows+1,2*MyMatrixRows+0},
				{0*MyMatrixRows+0,2*MyMatrixRows+1,0*MyMatrixRows+1,2*MyMatrixRows+0},
				{0*MyMatrixRows+0,1*MyMatrixRows+1,1*MyMatrixRows+0,0*MyMatrixRows+1}};
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

		__global__ void update_u_v_a_corotaion_compute_R_Rt(const int nCellCount, CommonCellOnCuda * cellOnCudaPointer, ValueTypePtr localStiffnessMatrixPtr/*localStiffnessMatrixOnCuda*/)
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

		__global__ void update_u_v_a_corotaion_compute_RK(const int nCellCount, CommonCellOnCuda * cellOnCudaPointer, ValueTypePtr localStiffnessMatrixPtr/*localStiffnessMatrixOnCuda*/)
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
					/*if (100 == currentCellIdx && 0 == localRowIdx )
					{
					CUPRINTF("R_24x24[%d]=%f Cvalue=%f\n", localRowIdx * Geometry_dofs_per_cell + e,
					R_24x24[localRowIdx * Geometry_dofs_per_cell + e],Cvalue);
					}*/
				}

				RK_24X24[localRowIdx * Geometry_dofs_per_cell + localColIdx] = Cvalue;

				/*if (100 == currentCellIdx)
				{
				CUPRINTF("RK_24X24[%d][%d]=%f K[%d][%d]=%f \n",localRowIdx,localColIdx,RK_24X24[localRowIdx*24+localColIdx],localRowIdx,localColIdx,K_24X24[localRowIdx*24+localColIdx]);
				}*/
			}
		}

		__global__ void update_u_v_a_corotaion_compute_RtPj(const int nCellCount, CommonCellOnCuda * cellOnCudaPointer)
		{
			const int currentCellIdx = blockIdx.x;
			const int localRowIdx = threadIdx.x;/*0-23*/
			//const int localColIdx = threadIdx.y;/*0-23*/

			/*if (0 == currentCellIdx)
			{
			CUPRINTF("update_u_v_a_corotaion_compute_RtPj\n");
			}*/

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

				/*if (0 == currentCellIdx)
				{
				CUPRINTF("Pj_24[%d]=%f  Rhs_24[%d]=%f Pj[%d]=%f \n",localRowIdx,Pj_24[localRowIdx],localRowIdx,Rhs_24[localRowIdx],localRowIdx,Pj_24[localRowIdx]);
				}*/
			}
		}

		__global__ void update_u_v_a_corotaion_compute_RKR_RPj(const int nCellCount, CommonCellOnCuda * cellOnCudaPointer)
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
					/*if (100 == currentCellIdx)
					{
					CUPRINTF("RKR_24X24[%d][%d]=%f  \n",localRowIdx,localColIdx,RKR_24X24[localRowIdx*24+localColIdx]);
					}*/
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
						/*if (352 == cellOnCudaPointer[currentCellIdx].globalDofs[localRowIdx])
						{
						CUPRINTF("### RK[%e][%f]*RtPj[%d][%f]=Cvalue[%f]\n",e,RK_24X24[localRowIdx * Geometry_dofs_per_cell + e],e,RtPj[e],Cvalue);
						}*/
					}

					RKRtPj[localRowIdx] = Cvalue;

					/*if (352 == cellOnCudaPointer[currentCellIdx].globalDofs[localRowIdx])
					{
					CUPRINTF("RKRtPj###[352][%d]=%f  \n",localRowIdx,RKRtPj[localRowIdx]);
					}*/
				}
			}
		}

		void computeRotationMatrix()
		{
			PhysicsContext& currentCtx = FEM_State_Ctx;
			//return ;
			update_u_v_a_init_corotaion<<< KERNEL_COUNT_TMP,3*3 >>>(currentCtx.nCellOnCudaCount,currentCtx.CellOnCudaPtr,currentCtx.g_nDofs,currentCtx.displacementOnCuda,Signs_on_cuda,currentCtx.g_VertexOnCudaPtr,currentCtx.g_nVertexOnCudaCount);
			cudaDeviceSynchronize();
			update_u_v_a_iterator_corotaion<<<KERNEL_COUNT_TMP,128>>>(currentCtx.nCellOnCudaCount,currentCtx.CellOnCudaPtr);
			cudaDeviceSynchronize();
			//	system("pause");

			dim3 threads4RK(24,24);
			update_u_v_a_corotaion_compute_R_Rt<<<KERNEL_COUNT_TMP,threads4RK>>>(currentCtx.nCellOnCudaCount,currentCtx.CellOnCudaPtr,currentCtx.localStiffnessMatrixOnCuda);
			cudaDeviceSynchronize();
			update_u_v_a_corotaion_compute_RK<<<KERNEL_COUNT_TMP,threads4RK>>>(currentCtx.nCellOnCudaCount,currentCtx.CellOnCudaPtr,currentCtx.localStiffnessMatrixOnCuda);
			cudaDeviceSynchronize();
			//	system("pause");
			update_u_v_a_corotaion_compute_RtPj<<<KERNEL_COUNT_TMP,24>>>(currentCtx.nCellOnCudaCount,currentCtx.CellOnCudaPtr);
			cudaDeviceSynchronize();
			//	system("pause");
			//exit(66);
			dim3 threads4RKR(24,25);
			//	printf("update_u_v_a_corotaion_compute_RKR_RPj\n");
			update_u_v_a_corotaion_compute_RKR_RPj<<<KERNEL_COUNT_TMP,threads4RKR>>>(currentCtx.nCellOnCudaCount,currentCtx.CellOnCudaPtr);
			cudaDeviceSynchronize();
		}
#endif//#if COROTATION_CLASSICAL

#define COROTATION_SPEEDUP (1)
#if COROTATION_SPEEDUP
		int * g_corotation_inverseIdx = MyNull;
		float * g_corotation_inverseFlag = MyNull;
		int * g_corotation_transposeIdx = MyNull;
		void corotation_Init_Const()
		{
			
			const int inverseIdx[3*3][4] = {{1*MyMatrixRows+1,2*MyMatrixRows+2,2*MyMatrixRows+1,1*MyMatrixRows+2},
			{0*MyMatrixRows+1,2*MyMatrixRows+2,2*MyMatrixRows+1,0*MyMatrixRows+2},
			{0*MyMatrixRows+1,1*MyMatrixRows+2,1*MyMatrixRows+1,0*MyMatrixRows+2},
			{1*MyMatrixRows+2,2*MyMatrixRows+0,2*MyMatrixRows+2,1*MyMatrixRows+0},
			{0*MyMatrixRows+2,2*MyMatrixRows+0,0*MyMatrixRows+0,2*MyMatrixRows+2},
			{0*MyMatrixRows+2,1*MyMatrixRows+0,0*MyMatrixRows+0,1*MyMatrixRows+2},
			{1*MyMatrixRows+0,2*MyMatrixRows+1,1*MyMatrixRows+1,2*MyMatrixRows+0},
			{0*MyMatrixRows+0,2*MyMatrixRows+1,0*MyMatrixRows+1,2*MyMatrixRows+0},
			{0*MyMatrixRows+0,1*MyMatrixRows+1,1*MyMatrixRows+0,0*MyMatrixRows+1}};
			const float inverseFlag[3*3] = {1.f,-1.f,1.f,1.f,-1.f,1.f,1.f,-1.f,1.f};
			const int transposeIdx[3*3] = {0,3,6,1,4,7,2,5,8};
			
			Definition_Device_Buffer_With_Data(g_corotation_inverseIdx,int,36,1,&inverseIdx[0][0]);
			Definition_Device_Buffer_With_Data(g_corotation_inverseFlag,float,9,1,&inverseFlag[0]);
			Definition_Device_Buffer_With_Data(g_corotation_transposeIdx,int,9,1,&transposeIdx[0]);
		}
		__global__ void update_u_v_a_init_corotaion_SpeedUp(const int nCellCount, CommonCellOnCuda * cellOnCudaPointer, int /*nDofCount*/, ValueTypePtr global_incremental_displacement,ValueTypePtr para_Signs_on_cuda, VertexOnCuda* _VertexOnCudaPtr, int _VertexOnCudaCount)
		{
			const int threadId = threadIdx.x + blockIdx.x * blockDim.x;	
			const int currentCellIdx = threadId / (CONST_8X9);	

			if(currentCellIdx < nCellCount  && true == cellOnCudaPointer[currentCellIdx].m_bLeaf)
			{
				const int result0 = MyMod(threadId , CONST_8X9);
				const int _currentI = result0 / 9;/*0-7*/
				const int matrixIdx3x3 = MyMod(result0 , 9);/*0-8*/
				const int component_row = matrixIdx3x3 / 3;
				const int component_col = MyMod(matrixIdx3x3 , 3);	

				float * RotationMatrix_InitSpeedUp_3x3 = &cellOnCudaPointer[currentCellIdx].RotationMatrix_InitSpeedUp[_currentI][0];
				const float& weight = cellOnCudaPointer[currentCellIdx].weight4speedup;//0.25f * (1.f / cellOnCudaPointer[currentCellIdx].radiusx2);

				const int vtxId = cellOnCudaPointer[currentCellIdx].vertexId[_currentI];
				const int currentDof = _VertexOnCudaPtr[vtxId].m_nGlobalDof[component_row];

				RotationMatrix_InitSpeedUp_3x3[matrixIdx3x3] = weight * global_incremental_displacement[currentDof] * para_Signs_on_cuda[_currentI*9+matrixIdx3x3];

				__syncthreads();
				if (0 == _currentI)
				{
					if (component_row == component_col)
					{
						cellOnCudaPointer[currentCellIdx].RotationMatrix[matrixIdx3x3] = 1;
					}
					else
					{
						cellOnCudaPointer[currentCellIdx].RotationMatrix[matrixIdx3x3] = 0;
					}
					for (int i=0;i<8;++i)
					{
						cellOnCudaPointer[currentCellIdx].RotationMatrix[matrixIdx3x3] += cellOnCudaPointer[currentCellIdx].RotationMatrix_InitSpeedUp[i][matrixIdx3x3];
					}

				}
			}
		}

		__global__ void update_u_v_a_iterator_corotaion_SpeedUp(const int nCellCount, CommonCellOnCuda * cellOnCudaPointer, const int inverseIdx[9][4],const float inverseFlag[9],const int transposeIdx[9])
		{
			const int currentCellIdx = threadIdx.x + blockIdx.x * blockDim.x;
			if(currentCellIdx < nCellCount  && true == cellOnCudaPointer[currentCellIdx].m_bLeaf)
			{
				/*const int inverseIdx[3*3][4] = {{1*MyMatrixRows+1,2*MyMatrixRows+2,2*MyMatrixRows+1,1*MyMatrixRows+2},
				{0*MyMatrixRows+1,2*MyMatrixRows+2,2*MyMatrixRows+1,0*MyMatrixRows+2},
				{0*MyMatrixRows+1,1*MyMatrixRows+2,1*MyMatrixRows+1,0*MyMatrixRows+2},
				{1*MyMatrixRows+2,2*MyMatrixRows+0,2*MyMatrixRows+2,1*MyMatrixRows+0},
				{0*MyMatrixRows+2,2*MyMatrixRows+0,0*MyMatrixRows+0,2*MyMatrixRows+2},
				{0*MyMatrixRows+2,1*MyMatrixRows+0,0*MyMatrixRows+0,1*MyMatrixRows+2},
				{1*MyMatrixRows+0,2*MyMatrixRows+1,1*MyMatrixRows+1,2*MyMatrixRows+0},
				{0*MyMatrixRows+0,2*MyMatrixRows+1,0*MyMatrixRows+1,2*MyMatrixRows+0},
				{0*MyMatrixRows+0,1*MyMatrixRows+1,1*MyMatrixRows+0,0*MyMatrixRows+1}};
				const float inverseFlag[3*3] = {1.f,-1.f,1.f,1.f,-1.f,1.f,1.f,-1.f,1.f};
				const int transposeIdx[3*3] = {0,3,6,1,4,7,2,5,8};*/

				float* R = &cellOnCudaPointer[currentCellIdx].RotationMatrix[0];
				float* Rt = &cellOnCudaPointer[currentCellIdx].RotationMatrixTranspose[0];
				float* RIT = &cellOnCudaPointer[currentCellIdx].RotationMatrix4Inverse[0];
				for (unsigned i0=1;i0<5;++i0)
				{
					const float det = 1.0f/determinant(R);

					for (unsigned i=0;i<(3*3);++i)
					{
						RIT[transposeIdx[i]] = (det) * inverseFlag[i] * (R[inverseIdx[i][0]]*R[inverseIdx[i][1]]-
							R[inverseIdx[i][2]]*R[inverseIdx[i][3]]);

						Rt[ transposeIdx[i] ] = R[i] = 0.5f*(R[i] + RIT[i]);
					}

					/*for (unsigned i=0;i<(3*3);++i)
					{

						Rt[ transposeIdx[i] ] = R[i] = 0.5f*(R[i] + RIT[i]);

					}*/
				}


			}
		}
#if 0
		__global__ void update_u_v_a_corotaion_compute_RK_SpeedUp(const int nCellCount, CommonCellOnCuda * cellOnCudaPointer, ValueTypePtr localStiffnessMatrixPtr/*localStiffnessMatrixOnCuda*/)
		{			
			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			const int currentCellIdx = tid / CONST_24x24;
			const int kernel2Dim = MyMod(tid,CONST_24x24);
			

			if(currentCellIdx < nCellCount && true == cellOnCudaPointer[currentCellIdx].m_bLeaf )
			{
				const int localRowIdx = kernel2Dim / CONST_24;;/*0-23*/
				const int localColIdx = MyMod(kernel2Dim,CONST_24);/*0-23*/

				/*float* R_24x24 = &cellOnCudaPointer[currentCellIdx].R[0];
				float* Rt_24x24 = &cellOnCudaPointer[currentCellIdx].Rt[0];*/
				float* R_3x3  = &cellOnCudaPointer[currentCellIdx].RotationMatrix[0];
				float* K_24X24 = &localStiffnessMatrixPtr[cellOnCudaPointer[currentCellIdx].m_nStiffnessMatrixIdx * Geometry_dofs_per_cell_squarte];
				float* RK_24X24 = &cellOnCudaPointer[currentCellIdx].RxK[0];

				const int R33_Row_x3 = (MyMod(localRowIdx , 3))*3;
				const int K24X24_Row_x24 = ((localRowIdx / 3) * 3) * Geometry_dofs_per_cell;

				const int index = localRowIdx * Geometry_dofs_per_cell + localColIdx;
				RK_24X24[index] = R_3x3[R33_Row_x3+0] * K_24X24[K24X24_Row_x24 + localColIdx];
				RK_24X24[index] += R_3x3[R33_Row_x3+1] * K_24X24[K24X24_Row_x24 + localColIdx + Geometry_dofs_per_cell];
				RK_24X24[index] += R_3x3[R33_Row_x3+2] * K_24X24[K24X24_Row_x24 + localColIdx + Geometry_dofs_per_cell + Geometry_dofs_per_cell];
			}
		}
#else
		__global__ void update_u_v_a_corotaion_compute_RK_SpeedUp(const int nCellCount,const int nCellCount_half, CommonCellOnCuda * cellOnCudaPointer, ValueTypePtr localStiffnessMatrixPtr/*localStiffnessMatrixOnCuda*/)
		{			
			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			int currentCellIdx = tid / CONST_24x24;
			const int kernel2Dim = MyMod(tid,CONST_24x24);
			

			while(currentCellIdx < nCellCount && true == cellOnCudaPointer[currentCellIdx].m_bLeaf )
			{
				const int localRowIdx = kernel2Dim / CONST_24;;/*0-23*/
				const int localColIdx = MyMod(kernel2Dim,CONST_24);/*0-23*/

				/*float* R_24x24 = &cellOnCudaPointer[currentCellIdx].R[0];
				float* Rt_24x24 = &cellOnCudaPointer[currentCellIdx].Rt[0];*/
				float* R_3x3  = &cellOnCudaPointer[currentCellIdx].RotationMatrix[0];
				float* K_24X24 = &localStiffnessMatrixPtr[cellOnCudaPointer[currentCellIdx].m_nStiffnessMatrixIdx * Geometry_dofs_per_cell_squarte];
				float* RK_24X24 = &cellOnCudaPointer[currentCellIdx].RxK[0];

				const int R33_Row_x3 = (MyMod(localRowIdx , 3))*3;
				const int K24X24_Row_x24 = ((localRowIdx / 3) * 3) * Geometry_dofs_per_cell;

				const int index = localRowIdx * Geometry_dofs_per_cell + localColIdx;
				RK_24X24[index] = R_3x3[R33_Row_x3+0] * K_24X24[K24X24_Row_x24 + localColIdx];
				RK_24X24[index] += R_3x3[R33_Row_x3+1] * K_24X24[K24X24_Row_x24 + localColIdx + Geometry_dofs_per_cell];
				RK_24X24[index] += R_3x3[R33_Row_x3+2] * K_24X24[K24X24_Row_x24 + localColIdx + Geometry_dofs_per_cell + Geometry_dofs_per_cell];

				currentCellIdx += nCellCount_half;
			}
		}
#endif

		__global__ void update_u_v_a_corotaion_compute_RtPj_SpeedUp(const int nCellCount, CommonCellOnCuda * cellOnCudaPointer)
		{
			//const int currentCellIdx = blockIdx.x;
			//const int localRowIdx = threadIdx.x;/*0-23*/

			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			const int currentCellIdx = tid / CONST_24;
			const int localRowIdx = MyMod(tid , CONST_24);

			if(currentCellIdx < nCellCount  && true == cellOnCudaPointer[currentCellIdx].m_bLeaf)
			{
				//float* Rt_24x24 = &cellOnCudaPointer[currentCellIdx].Rt[0];
				float* Rt_3x3 = &cellOnCudaPointer[currentCellIdx].RotationMatrixTranspose[0];
				float* Pj_24 = &cellOnCudaPointer[currentCellIdx].Pj[0];
				float* Rhs_24 = &cellOnCudaPointer[currentCellIdx].CorotaionRhs[0];

				const int Rt33_Row_x3 = (MyMod(localRowIdx , 3))*3;
				const int Pj_Row_x3 = (localRowIdx / 3)*3;

				float Cvalue = Rt_3x3[Rt33_Row_x3] * Pj_24[Pj_Row_x3];
				Cvalue      += Rt_3x3[Rt33_Row_x3+1] * Pj_24[Pj_Row_x3+1];
				Cvalue      += Rt_3x3[Rt33_Row_x3+2] * Pj_24[Pj_Row_x3+2];
				Rhs_24[localRowIdx] =  Pj_24[localRowIdx] - Cvalue;
				/*Rhs_24[localRowIdx] = Rt_3x3[Rt33_Row_x3] * Pj_24[Pj_Row_x3];
				Rhs_24[localRowIdx] += Rt_3x3[Rt33_Row_x3+1] * Pj_24[Pj_Row_x3+1];
				Rhs_24[localRowIdx] += Rt_3x3[Rt33_Row_x3+2] * Pj_24[Pj_Row_x3+1];*/
			}
		}

		__global__ void update_u_v_a_corotaion_compute_RKR_RPj_SpeedUp_1(const int nCellCount, const int nStep, CommonCellOnCuda * cellOnCudaPointer)
		{
			//const int currentCellIdx = blockIdx.x;
			//const int localRowIdx = threadIdx.x;/*0-23*/
			//const int localColIdx = threadIdx.y;/*0-23*/

			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			int currentCellIdx = tid / CONST_24x24;
			const int kernel2Dim = MyMod(tid,CONST_24x24);
			const int localRowIdx = kernel2Dim / CONST_24;;/*0-23*/
			const int localColIdx = MyMod(kernel2Dim,CONST_24);/*0-23*/

			while(currentCellIdx < nCellCount  && true == cellOnCudaPointer[currentCellIdx].m_bLeaf)
			{
				float* RK_24X24 = &cellOnCudaPointer[currentCellIdx].RxK[0];
				float* RKR_24X24 = &cellOnCudaPointer[currentCellIdx].RKR[0];
				float* R_3x3  = &cellOnCudaPointer[currentCellIdx].RotationMatrix[0];

				const int index = localRowIdx * Geometry_dofs_per_cell + localColIdx;
				const int R33_Col_x3  = (MyMod(localColIdx , 3)) * 3;
				const int K24X24_Col_24 = (localColIdx / 3) * 3 + localRowIdx * Geometry_dofs_per_cell;

				RKR_24X24[index]  = RK_24X24[K24X24_Col_24]		* R_3x3[R33_Col_x3];
				RKR_24X24[index] += RK_24X24[K24X24_Col_24+1]	* R_3x3[R33_Col_x3+1];
				RKR_24X24[index] += RK_24X24[K24X24_Col_24+2]	* R_3x3[R33_Col_x3+2];

				currentCellIdx += nStep;
			}
		}

		__global__ void update_u_v_a_corotaion_compute_RKR_RPj_SpeedUp_2(const int nCellCount, CommonCellOnCuda * cellOnCudaPointer)
		{
			__shared__ float vals [Geometry_dofs_per_cell][32];
			const int currentCellIdx = blockIdx.x;
			const int localRowIdx = threadIdx.x;/*0-23*/
			const int localColIdx = threadIdx.y;/*0-23*/

			if(currentCellIdx < nCellCount  && true == cellOnCudaPointer[currentCellIdx].m_bLeaf)
			{
				float* RK_24X24 = &cellOnCudaPointer[currentCellIdx].RxK[0];
				float* RtPj = &cellOnCudaPointer[currentCellIdx].CorotaionRhs[0];
				float* RKRtPj = &cellOnCudaPointer[currentCellIdx].RKRtPj[0];
				float* tmpRt = &cellOnCudaPointer[currentCellIdx].Rt[0];

				vals[localRowIdx][localColIdx] = RK_24X24[localRowIdx * Geometry_dofs_per_cell + localColIdx]	* RtPj[localColIdx];

				__syncthreads();

				if (localColIdx < 12)
				{
					vals[localRowIdx][localColIdx] += vals[localRowIdx][localColIdx+12];
				}
				__syncthreads();
				if (localColIdx < 6)
				{
					vals[localRowIdx][localColIdx] += vals[localRowIdx][localColIdx+6];
				}
				__syncthreads();
				if (localColIdx < 3)
				{
					vals[localRowIdx][localColIdx] += vals[localRowIdx][localColIdx+3];
				}
				__syncthreads();
				if (0 == localColIdx )
				{
					RKRtPj[localRowIdx] = vals[localRowIdx][0]+vals[localRowIdx][1]+vals[localRowIdx][2];

					/*if (500 == currentCellIdx && 0 == localRowIdx)
					{
					CUPRINTF(" %f\n",RKRtPj[localRowIdx]);
					}*/
				}

				/*__syncthreads();

				if (localColIdx < 16)
				{
				vals[localRowIdx][localColIdx] += vals[localRowIdx][localColIdx+16];
				}
				__syncthreads();
				if (localColIdx < 8)
				{
				vals[localRowIdx][localColIdx] += vals[localRowIdx][localColIdx+8];
				}
				__syncthreads();
				if (localColIdx < 4)
				{
				vals[localRowIdx][localColIdx] += vals[localRowIdx][localColIdx+4];
				}

				__syncthreads();
				if (localColIdx < 2)
				{
				vals[localRowIdx][localColIdx] += vals[localRowIdx][localColIdx+2];
				}

				__syncthreads();
				if (localColIdx < 1)
				{
				vals[localRowIdx][localColIdx] += vals[localRowIdx][localColIdx+1];
				}
				__syncthreads();
				if (0 == localColIdx )
				{
				RKRtPj[localRowIdx] = vals[localRowIdx][0];
				}*/
			}
		}

		__global__ void update_u_v_a_corotaion_compute_RKR_RPj_SpeedUp_3(const int nCellCount, CommonCellOnCuda * cellOnCudaPointer)
		{
			__shared__ float vals [Geometry_dofs_per_cell];
			const int currentCellIdx = blockIdx.x;
			const int localRowIdx = blockIdx.y;/*0-23*/
			const int localColIdx = threadIdx.x;
			//const int localRowIdx = tid/24;/*0-23*/
			//const int localColIdx = tid%24;/*0-23*/

			if(currentCellIdx < nCellCount  && true == cellOnCudaPointer[currentCellIdx].m_bLeaf)
			{
				float* RK_24X24 = &cellOnCudaPointer[currentCellIdx].RxK[0];
				float* RtPj = &cellOnCudaPointer[currentCellIdx].CorotaionRhs[0];
				float* RKRtPj = &cellOnCudaPointer[currentCellIdx].RKRtPj[0];
				float* tmpRt = &cellOnCudaPointer[currentCellIdx].Rt[0];

				//vals[localColIdx] = tmpRt[localRowIdx * Geometry_dofs_per_cell + localColIdx];//RK_24X24[localRowIdx * Geometry_dofs_per_cell + localColIdx]	* RtPj[localColIdx];
				vals[localColIdx] = RK_24X24[localRowIdx * Geometry_dofs_per_cell + localColIdx]	* RtPj[localColIdx];
				__syncthreads();

				if (localColIdx < 12) vals[localColIdx] += vals[localColIdx+12];
				if (localColIdx < 6) vals[localColIdx] += vals[localColIdx+6];
				if (localColIdx < 3) vals[localColIdx] += vals[localColIdx+3];
				if (localColIdx < 1) vals[localColIdx] += vals[localColIdx+1];
				if (0 == localColIdx ) 
				{
					RKRtPj[localRowIdx] = vals[0]/*+vals[1]*/+vals[2];
					/*if (500 == currentCellIdx && 0 == localRowIdx)
					{
					CUPRINTF("speed up %f\n",RKRtPj[localRowIdx]);
					}*/
				}
			}
		}

		__global__ void update_u_v_a_corotaion_compute_RKR_RPj_SpeedUp_4(const int nCellCount, CommonCellOnCuda * cellOnCudaPointer)
		{
			__shared__ float vals [Geometry_dofs_per_cell][Geometry_dofs_per_cell];
			const int currentCellIdx = blockIdx.x;


			if(currentCellIdx < nCellCount  && true == cellOnCudaPointer[currentCellIdx].m_bLeaf)
			{
				const int tid = threadIdx.x;
				const int localRowIdx = threadIdx.x /24;/*0-23*/
				const int localColIdx = MyMod(threadIdx.x ,24) ;/*0-23*/

				float* RK_24X24 = &cellOnCudaPointer[currentCellIdx].RxK[0];
				float* RtPj = &cellOnCudaPointer[currentCellIdx].CorotaionRhs[0];
				float* RKRtPj = &cellOnCudaPointer[currentCellIdx].RKRtPj[0];
				//float* tmpRt = &cellOnCudaPointer[currentCellIdx].Rt[0];

				vals[localRowIdx][localColIdx] = RK_24X24[localRowIdx * Geometry_dofs_per_cell + localColIdx]	* RtPj[localColIdx];

				__syncthreads();

				if (tid < 24)
				{
					RKRtPj[tid] = vals[tid][0];
					for (int i=1;i<24;++i)
					{
						RKRtPj[tid] += vals[tid][i];
					}


					/*if (localColIdx < 12)
					{
					vals[tid][localColIdx] += vals[tid][localColIdx+12];
					}
					if (localColIdx < 6)
					{
					vals[tid][localColIdx] += vals[tid][localColIdx+6];
					}
					if (localColIdx < 3)
					{
					vals[tid][localColIdx] += vals[tid][localColIdx+3];
					}
					if (0 == localColIdx )
					{
					RKRtPj[tid] = vals[tid][0]+vals[tid][1]+vals[tid][2];
					}*/
				}
			}
		}

#endif//#if COROTATION_SPEEDUP
	}//namespace CUDA_COROATION
#endif//#if USE_CO_RATION
	namespace MyKrylov
	{
		void initMyKrylovContext();
		template <typename Array>
		typename Array::value_type nrm2(const Array& x);
	}
	void initFEMStateContext(PhysicsContext* curCtx)
	{
		curCtx->nBcCount = MyZero;
		curCtx->cusp_boundaryCondition = MyZero;
		curCtx->nForceCount = MyZero;
		curCtx->cuda_forceCondition = MyZero;

		//curCtx->diagnosticValue = MyZero;

		curCtx->g_nDofs = MyZero;
		curCtx->g_nDofsLast = MyZero;
		curCtx->displacementOnCuda = MyZero;
		curCtx->rhsOnCuda = MyZero;
		curCtx->cuda_boundaryCondition = MyZero;
		curCtx->cuda_diagnosticValue = MyZero;

		curCtx->funcMatrixCount = MyZero;
		curCtx->localStiffnessMatrixOnCuda = MyZero;
//		curCtx->localMassMatrixOnCuda = MyZero;
//		curCtx->localRhsVectorOnCuda = MyZero;
		curCtx->nFEMShapeValueCount = MyZero;
		curCtx->FEMShapeValueOnCuda = MyZero;

		curCtx->SystemMatrix.setNeedDiag(true);
		curCtx->MassMatrix.setNeedDiag(false);
		curCtx->DampingMatrix.setNeedDiag(false);


#if USE_CUTTING
		curCtx->topLevelCellInMain_OnHost = MyNull;
		curCtx->topLevelCellInMain_OnCuda = MyNull;
		curCtx->g_CellCollisionFlag_onCuda = MyNull;
		curCtx->g_CellCollisionFlag = MyNull;
		curCtx->beClonedObjectFlag_OnHost = MyNull;
		curCtx->beClonedObjectFlag_OnCuda = MyNull;
		curCtx->beCuttingLinesFlagElement = MyNull;
#endif

		curCtx->CellOnCudaPtr = MyZero;
		curCtx->cuda_invalidCellFlagPtr = MyZero;
		curCtx->nCellOnCudaCount = MyZero;
		curCtx->nCellOnCudaCountLast = MyZero;

		curCtx->g_nCuttingLineSetCount = MyZero;
		curCtx->g_CuttingLineSet = MyNull;
		curCtx->beCuttingLinesFlagElement = MyNull;
		//curCtx->g_CellCollisionFlag = MyZero;

		curCtx->g_globalDof_MF = MyNull;
		curCtx->g_globalValue_MF = MyNull;
		curCtx->g_globalDof_Mass_MF = MyNull;
		curCtx->g_globalValue_Mass_MF = MyNull;
		curCtx->g_globalDof_Damping_MF = MyNull;
		curCtx->g_globalValue_Damping_MF = MyNull;
		curCtx->g_globalDof_System_MF = MyNull;
		curCtx->g_globalValue_System_MF = MyNull;
		curCtx->g_systemRhsPtr_MF = MyNull;
		curCtx->g_systemRhsPtr_In8_MF = MyNull;

		curCtx->materialParams= MyNull;
		curCtx->materialIndex = MyNull;
		curCtx->materialValue = MyNull;
		curCtx->g_externForce = MyNull;
		curCtx->g_lai = MyNull;
		curCtx->g_Density = MyNull;

#if USE_CO_RATION
		curCtx->cuda_RKR = MyNull;
		curCtx->cuda_RKRtPj = MyNull;
#endif

		curCtx->myOptimize_BladeForce = MyNull;
		curCtx->myOptimize_BladeForce_In8_MF = MyNull;
		curCtx->g_isApplyExternalForceAtCurrentFrame = MyZero;

		curCtx->g_VertexOnCudaPtr = MyNull;
		curCtx->g_nVertexOnCudaCount = MyZero;

#if USE_CUDA_STREAM
		cudaStreamCreate(&streamForCalc);
		cudaStreamCreate(&streamForSkin);
#endif
		
	}

	void VR_Physics_FEM_Simulation_InitializeOnCUDA()
	{
		using namespace CUDA_COROTATION;
		initFEMStateContext(&FEM_State_Ctx);

		float Signs_on_cpu[8*9] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,
			1,-1,-1,1,-1,-1,1,-1,-1,
			-1,1,-1,-1,1,-1,-1,1,-1,
			1,1,-1,1,1,-1,1,1,-1,					  
			-1,-1,1,-1,-1,1,-1,-1,1,					  
			1,-1,1, 1,-1,1,1,-1,1,					  
			-1,1,1,-1,1,1,-1,1,1,					  
			1,1,1,1,1,1,1,1,1};

		HANDLE_ERROR( cudaMalloc( (void**)&Signs_on_cuda	,8*9 * sizeof(ValueType)));
		HANDLE_ERROR( cudaMemcpy( (void *)Signs_on_cuda		, Signs_on_cpu,8*9 * sizeof(ValueType)	,cudaMemcpyHostToDevice ));

		CUDA_COROTATION::corotation_Init_Const();
	}

	void VR_Physics_FEM_Simulation_InitLocalCellMatrixOnCuda(const int nCount,ValueType * localStiffnessMatrixOnCpu,
		ValueType * localMassMatrixOnCpu,ValueType * localRhsVectorOnCpu,
		FEMShapeValue* femShapeValuePtr)
	{
		
		VR_Physics_FEM_Simulation_InitializeOnCUDA();
		cudaDeviceSynchronize();
		PhysicsContext & currentCtx = FEM_State_Ctx;
		currentCtx.funcMatrixCount = nCount;
		currentCtx.nFEMShapeValueCount = nCount;
		HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.localStiffnessMatrixOnCuda	,nCount * Geometry_dofs_per_cell_squarte * sizeof(ValueType))) ;
		HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.localMassMatrixOnCuda		,nCount * Geometry_dofs_per_cell_squarte * sizeof(ValueType))) ;
//		HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.localRhsVectorOnCuda			,nCount * Geometry_dofs_per_cell		 * sizeof(ValueType))) ;
		HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.FEMShapeValueOnCuda	,nCount * sizeof(FEMShapeValue))) ;

		HANDLE_ERROR( cudaMemcpy( (void *)currentCtx.localStiffnessMatrixOnCuda	,localStiffnessMatrixOnCpu	,nCount * Geometry_dofs_per_cell_squarte * sizeof(ValueType),cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( (void *)currentCtx.localMassMatrixOnCuda			,localMassMatrixOnCpu		,nCount * Geometry_dofs_per_cell_squarte * sizeof(ValueType),cudaMemcpyHostToDevice ) );
//		HANDLE_ERROR( cudaMemcpy( (void *)currentCtx.localRhsVectorOnCuda			,localRhsVectorOnCpu		,nCount * Geometry_dofs_per_cell		 * sizeof(ValueType),cudaMemcpyHostToDevice ) );	
		HANDLE_ERROR( cudaMemcpy( (void *)currentCtx.FEMShapeValueOnCuda	,femShapeValuePtr	,nCount * sizeof(FEMShapeValue),cudaMemcpyHostToDevice ) );
	}

	void  VR_Physics_FEM_Simulation_InitPhysicsCellAndVertex(const int vertexSize, VertexOnCuda** retVertexPtr, const int cellSize, CommonCellOnCuda** retCellPtr)
	{
		cudaDeviceSynchronize();

		PhysicsContext& currentCtx = FEM_State_Ctx;

		currentCtx.g_nVertexOnCudaCount = vertexSize;
		currentCtx.nCellOnCudaCount = cellSize;
		currentCtx.nCellOnCudaCountLast = cellSize;
#if USE_HOST_MEMORY
		Definition_Host_Device_Buffer(currentCtx.g_VertexOnCudaPtrOnCPU,currentCtx.g_VertexOnCudaPtr,VertexOnCuda,currentCtx.g_nVertexOnCudaCount*_nExternalMemory);
		Definition_Host_Device_Buffer(currentCtx.CellOnCudaPtrOnCPU,currentCtx.CellOnCudaPtr,CommonCellOnCuda,currentCtx.nCellOnCudaCount*_nExternalMemory);

		*retVertexPtr = currentCtx.g_VertexOnCudaPtrOnCPU;
		*retCellPtr = currentCtx.CellOnCudaPtrOnCPU;
#else
		Definition_Device_Buffer_With_Data(currentCtx.g_VertexOnCudaPtr,VertexOnCuda,currentCtx.g_nVertexOnCudaCount,_nExternalMemory,(*retVertexPtr));
		Definition_Device_Buffer_With_Data(currentCtx.CellOnCudaPtr,CommonCellOnCuda,currentCtx.nCellOnCudaCount,_nExternalMemory,(*retCellPtr));
#endif
	}

	void  VR_Physics_FEM_Simulation_InitCellCuttingLineSetOnCuda(const int linesCount,CuttingLinePair * linesPtrOnCpu)
	{
		cudaDeviceSynchronize();

		PhysicsContext& currentCtx = FEM_State_Ctx;
		currentCtx.g_nCuttingLineSetCount = linesCount;
		Definition_Device_Buffer_With_Data(currentCtx.g_CuttingLineSet,CuttingLinePair,linesCount,_nExternalMemory,linesPtrOnCpu);

		Definition_Device_Buffer_With_Zero(currentCtx.beCuttingLinesFlagElement,char,_nExternalMemory * linesCount);
	}

	void VR_Physics_FEM_Simulation_InitLinePair()//done
	{
		cudaDeviceSynchronize();
		//int linePair[12][2] = {{0,2},{4,6},{0,4},{2,6},{1,3},{5,7},{1,5},{3,7},{0,1},{4,5},{2,3},{6,7}};
		IndexType linePair[24] = {0,2,4,6,0,4,2,6,1,3,5,7,1,5,3,7,0,1,4,5,2,3,6,7};
		PhysicsContext& currentCtx = FEM_State_Ctx;
		Definition_Device_Buffer_With_Data(currentCtx.g_linePairOnCuda,IndexType,24,1,&linePair[0]);
	}

	void VR_Physics_FEM_Simulation_InitialLocalDomainOnCuda(const IndexType nDofs,
		EigenValueType dbNewMarkConstant[8],
		IndexTypePtr boundaryconditionPtr,
		IndexType bcCount,
		IndexTypePtr boundaryconditionDisplacementPtr,
		IndexType bcCountDisplace
		)//done
	{
		cudaDeviceSynchronize();
#if 1
		{
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, 0); 
			if (!prop.canMapHostMemory ) 
			{
				printf("prop.canMapHostMemory can't support. \n");
				exit(0); 
			}
			if (prop.maxThreadsPerBlock < MAX_KERNEL_PER_BLOCK )
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
		PhysicsContext& currentCtx = FEM_State_Ctx;
		currentCtx.g_nDofs = nDofs;
		currentCtx.g_nDofsLast = nDofs;
		currentCtx.nBcCount = bcCount;

		
		printf("9  nBcCount = %d  nBcCount * sizeof(IndexType) is %d\n",currentCtx.nBcCount,currentCtx.nBcCount * sizeof(IndexType));	
		Q_ASSERT(currentCtx.nBcCount > 50);

		currentCtx.cusp_boundaryCondition = (IndexTypePtr)malloc( currentCtx.nBcCount * sizeof(IndexType) );
		memcpy(currentCtx.cusp_boundaryCondition,boundaryconditionPtr,currentCtx.nBcCount * sizeof(IndexType));
		//currentCtx.diagnosticValue = (ValueTypePtr)malloc( nDofs * sizeof(ValueType) );

		Definition_Device_Buffer_With_Zero(currentCtx.displacementOnCuda,ValueType,_nExternalMemory * nDofs);
		Definition_Device_Buffer_With_Zero(currentCtx.displacementOnCuda4SpeedUp,ValueType,_nExternalMemory * nDofs);
		Definition_Device_Buffer_With_Zero(currentCtx.rhsOnCuda,ValueType,_nExternalMemory * nDofs);
		Definition_Device_Buffer_With_Zero(currentCtx.cuda_diagnosticValue,ValueType,_nExternalMemory*nDofs);
		//HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.displacementOnCuda,_nExternalMemory * nDofs * sizeof(ValueType))) ;
		//HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.rhsOnCuda,_nExternalMemory * nDofs * sizeof(ValueType))) ;

		//HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.cuda_boundaryCondition,currentCtx.nBcCount * sizeof(IndexType))) ;
		//HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.cuda_diagnosticValue,_nExternalMemory*nDofs * sizeof(ValueType))) ;

		currentCtx.nForceCount = bcCountDisplace;
		Definition_Device_Buffer_With_Data(currentCtx.cuda_forceCondition,IndexType,currentCtx.nForceCount,1,boundaryconditionDisplacementPtr);
		currentCtx.nBcCount = bcCount;
		Definition_Device_Buffer_With_Data(currentCtx.cuda_boundaryCondition,IndexType,currentCtx.nBcCount,1,boundaryconditionPtr);
		//HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.cuda_forceCondition,currentCtx.nForceCount * sizeof(IndexType))) ;
		//HANDLE_ERROR( cudaMemcpy( (void *)currentCtx.cuda_forceCondition,boundaryconditionDisplacementPtr,currentCtx.nForceCount * sizeof(IndexType),cudaMemcpyHostToDevice ) );


		//HANDLE_ERROR( cudaMemcpy( (void *)currentCtx.cuda_boundaryCondition,boundaryconditionPtr,currentCtx.nBcCount * sizeof(IndexType),cudaMemcpyHostToDevice ) );



		printf("5* \n");
		thrust::device_ptr<ValueType> wrapped_displacement(currentCtx.displacementOnCuda);
		currentCtx.cusp_Array_Incremental_displace = MyCuspVecView(wrapped_displacement,wrapped_displacement+_nExternalMemory * nDofs);
		currentCtx.cusp_Array_Incremental_displace.resize(nDofs);

		thrust::device_ptr<ValueType> wrapped_rhs(currentCtx.rhsOnCuda);
		currentCtx.cusp_Array_R_rhs = MyCuspVecView(wrapped_rhs,wrapped_rhs+_nExternalMemory * nDofs);
		currentCtx.cusp_Array_R_rhs.resize(nDofs);

		
		Definition_Device_Buffer_With_Zero(currentCtx.myOptimize_Array_Displacement,ValueType,_nExternalMemory *(nDofs));
		{
			thrust::device_ptr<ValueType> wrapped_myOptimize_Array_Displacement(currentCtx.myOptimize_Array_Displacement);
			currentCtx.cusp_Array_Displacement = MyCuspVecView(wrapped_myOptimize_Array_Displacement,wrapped_myOptimize_Array_Displacement+_nExternalMemory * nDofs);
			currentCtx.cusp_Array_Displacement.resize(nDofs);
		}

		Definition_Device_Buffer_With_Zero(currentCtx.myOptimize_Array_Mass_rhs,ValueType,_nExternalMemory *(nDofs));
		{
			thrust::device_ptr<ValueType> wrapped_myOptimize_Array_Mass_rhs(currentCtx.myOptimize_Array_Mass_rhs);
			currentCtx.cusp_Array_Mass_rhs = MyCuspVecView(wrapped_myOptimize_Array_Mass_rhs,wrapped_myOptimize_Array_Mass_rhs+_nExternalMemory * nDofs);
			currentCtx.cusp_Array_Mass_rhs.resize(nDofs);
		}
		
		Definition_Device_Buffer_With_Zero(currentCtx.myOptimize_Array_Damping_rhs,ValueType,_nExternalMemory *(nDofs));
		{
			thrust::device_ptr<ValueType> wrapped_myOptimize_Array_Damping_rhs(currentCtx.myOptimize_Array_Damping_rhs);
			currentCtx.cusp_Array_Damping_rhs = MyCuspVecView(wrapped_myOptimize_Array_Damping_rhs,wrapped_myOptimize_Array_Damping_rhs+_nExternalMemory * nDofs);
			currentCtx.cusp_Array_Damping_rhs.resize(nDofs);
		}

		Definition_Device_Buffer_With_Zero(currentCtx.myOptimize_Array_Velocity,ValueType,_nExternalMemory *(nDofs));
		{
			thrust::device_ptr<ValueType> wrapped_myOptimize_Array_Velocity(currentCtx.myOptimize_Array_Velocity);
			currentCtx.cusp_Array_Velocity = MyCuspVecView(wrapped_myOptimize_Array_Velocity,wrapped_myOptimize_Array_Velocity+_nExternalMemory * nDofs);
			currentCtx.cusp_Array_Velocity.resize(nDofs);
		}

		Definition_Device_Buffer_With_Zero(currentCtx.myOptimize_Array_Acceleration,ValueType,_nExternalMemory *(nDofs));
		{
			thrust::device_ptr<ValueType> wrapped_myOptimize_Array_Acceleration(currentCtx.myOptimize_Array_Acceleration);
			currentCtx.cusp_Array_Acceleration = MyCuspVecView(wrapped_myOptimize_Array_Acceleration,wrapped_myOptimize_Array_Acceleration+_nExternalMemory * nDofs);
			currentCtx.cusp_Array_Acceleration.resize(nDofs);
		}
		
		//cusp_Array_Rhs.resize(nDofs,0);
		Definition_Device_Buffer_With_Zero(currentCtx.myOptimize_Old_Acceleration,ValueType,_nExternalMemory *(nDofs));
		{
			thrust::device_ptr<ValueType> wrapped_myOptimize_Array_Old_Acceleration(currentCtx.myOptimize_Old_Acceleration);
			currentCtx.cusp_Array_Old_Acceleration = MyCuspVecView(wrapped_myOptimize_Array_Old_Acceleration,wrapped_myOptimize_Array_Old_Acceleration+_nExternalMemory * nDofs);
			currentCtx.cusp_Array_Old_Acceleration.resize(nDofs);
		}
		
		
		Definition_Device_Buffer_With_Zero(currentCtx.myOptimize_Old_Displacement,ValueType,_nExternalMemory *(nDofs));
		{
			thrust::device_ptr<ValueType> wrapped_myOptimize_Array_Old_Displacement(currentCtx.myOptimize_Old_Displacement);
			currentCtx.cusp_Array_Old_Displacement = MyCuspVecView(wrapped_myOptimize_Array_Old_Displacement,wrapped_myOptimize_Array_Old_Displacement+_nExternalMemory * nDofs);
			currentCtx.cusp_Array_Old_Displacement.resize(nDofs);
		}
		

#if USE_CO_RATION
		
		Definition_Device_Buffer_With_Zero(currentCtx.myOptimize_Array_R_rhs_tmp4Corotaion,ValueType,_nExternalMemory *(nDofs));
		{
			thrust::device_ptr<ValueType> wrapped_Array_R_rhs_tmp4Corotaion(currentCtx.myOptimize_Array_R_rhs_tmp4Corotaion);
			currentCtx.cusp_Array_R_rhs_tmp4Corotaion = MyCuspVecView(wrapped_Array_R_rhs_tmp4Corotaion,wrapped_Array_R_rhs_tmp4Corotaion+_nExternalMemory * nDofs);
			currentCtx.cusp_Array_R_rhs_tmp4Corotaion.resize(nDofs);

		}
		
		//currentCtx.cusp_Array_R_rhs_Corotaion.resize(nDofs);
#endif

		currentCtx.g_globalExternalForce.resize(nDofs,0);
		//currentCtx.g_globalBladeForce.resize(nDofs,0);

		cusp_Array_NewMarkConstant.resize(8);
		for (int v=0;v < 8;++v)
		{
			cusp_Array_NewMarkConstant[v] = (ValueType)dbNewMarkConstant[v];
		}
		Definition_Device_Buffer_With_Data(element_Array_NewMarkConstant,ValueType,8,1,&dbNewMarkConstant[0]);
		

		if (fabsf(NewMarkConstant_0 - dbNewMarkConstant[0]) > 0.000001f)
		{
			printf("NewMarkConstant[0] [%f][%f]\n",NewMarkConstant_0,dbNewMarkConstant[0]);
			MyError("");
		}

		if (fabsf(NewMarkConstant_1 - dbNewMarkConstant[1]) > 0.000001f)
		{
			printf("NewMarkConstant[1] [%f][%f]\n",NewMarkConstant_1,dbNewMarkConstant[1]);
			MyError("");
		}
		printf("5 2 \n");

		Definition_Device_Buffer_With_Zero(currentCtx.g_globalDof_MF,IndexType,_nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM);
		//HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.g_globalDof_MF	,_nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM * sizeof(IndexType))) ;
		//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalDof_MF,  0, _nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM * sizeof(IndexType)));

		Definition_Device_Buffer_With_Zero(currentCtx.g_globalValue_MF,ValueType,_nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM);
		//HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.g_globalValue_MF	,_nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM * sizeof(ValueType))) ;
		//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalValue_MF,  0, _nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM * sizeof(ValueType)));

		Definition_Device_Buffer_With_Zero(currentCtx.g_globalDof_Mass_MF,IndexType,_nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM);
		//HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.g_globalDof_Mass_MF	,_nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM * sizeof(IndexType))) ;
		//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalDof_Mass_MF,  0, _nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM * sizeof(IndexType)));

		Definition_Device_Buffer_With_Zero(currentCtx.g_globalValue_Mass_MF,ValueType,_nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM);
		//HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.g_globalValue_Mass_MF	,_nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM * sizeof(ValueType))) ;
		//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalValue_Mass_MF,  0, _nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM * sizeof(ValueType)));

		Definition_Device_Buffer_With_Zero(currentCtx.g_globalDof_Damping_MF,IndexType,_nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM);
		//HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.g_globalDof_Damping_MF	,_nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM * sizeof(IndexType))) ;
		//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalDof_Damping_MF,  0, _nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM * sizeof(IndexType)));

		Definition_Device_Buffer_With_Zero(currentCtx.g_globalValue_Damping_MF,ValueType,_nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM);
		//HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.g_globalValue_Damping_MF	,_nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM * sizeof(ValueType))) ;
		//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalValue_Damping_MF,  0, _nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM * sizeof(ValueType)));

		Definition_Device_Buffer_With_Zero(currentCtx.g_globalDof_System_MF,IndexType,_nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM);
		//HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.g_globalDof_System_MF	,_nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM * sizeof(IndexType))) ;
		//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalDof_System_MF,  0, _nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM * sizeof(IndexType)));

		Definition_Device_Buffer_With_Zero(currentCtx.g_globalValue_System_MF,ValueType,_nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM);
		//HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.g_globalValue_System_MF	,_nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM * sizeof(ValueType))) ;
		//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalValue_System_MF,  0, _nExternalMemory *(nDofs)*nMaxNonZeroSizeInFEM * sizeof(ValueType)));

		Definition_Device_Buffer_With_Zero(currentCtx.g_systemRhsPtr_MF,ValueType,_nExternalMemory *(nDofs));
		{
			thrust::device_ptr<ValueType> wrapped_systemRhsPtr_MF(currentCtx.g_systemRhsPtr_MF);
			currentCtx.cusp_Array_Rhs = MyCuspVecView(wrapped_systemRhsPtr_MF,wrapped_systemRhsPtr_MF+_nExternalMemory * nDofs);
			currentCtx.cusp_Array_Rhs.resize(nDofs);
		}
		//HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.g_systemRhsPtr_MF	,_nExternalMemory *(nDofs) * sizeof(ValueType))) ;
		//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_systemRhsPtr_MF,  0, _nExternalMemory *(nDofs) * sizeof(ValueType)));

		Definition_Device_Buffer_With_Zero(currentCtx.g_systemRhsPtr_In8_MF,ValueType,_nExternalMemory *(nDofs)*VertxMaxInflunceCellCount);
		//HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.g_systemRhsPtr_In8_MF	,_nExternalMemory *(nDofs) * VertxMaxInflunceCellCount * sizeof(ValueType))) ;
		//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_systemRhsPtr_In8_MF,  0, _nExternalMemory *(nDofs) * VertxMaxInflunceCellCount * sizeof(ValueType)));

#if USE_CO_RATION
		Definition_Device_Buffer_With_Zero(currentCtx.g_systemRhsPtr_MF_Rotation,ValueType,_nExternalMemory *(nDofs));
		{
			thrust::device_ptr<ValueType> wrapped_systemRhsPtr_MF_Rotation(currentCtx.g_systemRhsPtr_MF_Rotation);
			currentCtx.cusp_Array_R_rhs_Corotaion = MyCuspVecView(wrapped_systemRhsPtr_MF_Rotation,wrapped_systemRhsPtr_MF_Rotation+_nExternalMemory * nDofs);
			currentCtx.cusp_Array_R_rhs_Corotaion.resize(nDofs);
			
		}

		Definition_Device_Buffer_With_Zero(currentCtx.g_systemRhsPtr_In8_MF_Rotation,ValueType,_nExternalMemory *(nDofs)* VertxMaxInflunceCellCount);
		//HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.g_systemRhsPtr_In8_MF_Rotation	,_nExternalMemory *(nDofs) * VertxMaxInflunceCellCount * sizeof(ValueType))) ;
		//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_systemRhsPtr_In8_MF_Rotation,  0, _nExternalMemory *(nDofs) * VertxMaxInflunceCellCount * sizeof(ValueType)));
#endif
		//
		printf("5*5 \n");
		Definition_Device_Buffer_With_Zero(currentCtx.myOptimize_BladeForce,ValueType,_nExternalMemory *(nDofs));
		{
			thrust::device_ptr<ValueType> wrapped_myOptimize_BladeForce(currentCtx.myOptimize_BladeForce);
			currentCtx.cusp_Array_BladeForce = MyCuspVecView(wrapped_myOptimize_BladeForce,wrapped_myOptimize_BladeForce+_nExternalMemory * nDofs);
			currentCtx.cusp_Array_BladeForce.resize(nDofs);
		}
		Definition_Device_Buffer_With_Zero(currentCtx.myOptimize_BladeForce_In8_MF,ValueType,_nExternalMemory *(nDofs)*8);
		//HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.g_globalExternalOnCuda	,_nExternalMemory *(nDofs) * sizeof(ValueType))) ;
		//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalExternalOnCuda,  0, _nExternalMemory *(nDofs) * sizeof(ValueType)));

		Definition_Device_Buffer_With_Zero(currentCtx.g_globalExternalOnCuda_x8,ValueType,_nExternalMemory *(nDofs)*8);
		//HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.g_globalExternalOnCuda_x8	,_nExternalMemory*8 *(nDofs) * sizeof(ValueType))) ;
		//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalExternalOnCuda_x8,  0, _nExternalMemory*8 *(nDofs) * sizeof(ValueType)));

#if USE_CUTTING
		Definition_Host_Device_Buffer(currentCtx.topLevelCellInMain_OnHost,currentCtx.topLevelCellInMain_OnCuda,IndexType,_nExternalMemory * currentCtx.nCellOnCudaCount);
		Definition_Host_Device_Buffer(currentCtx.g_CellCollisionFlag,currentCtx.g_CellCollisionFlag_onCuda,IndexType,_nExternalMemory * currentCtx.nCellOnCudaCount);
		Definition_Host_Device_Buffer(currentCtx.beClonedObjectFlag_OnHost,currentCtx.beClonedObjectFlag_OnCuda,IndexType,_nExternalMemory * currentCtx.nCellOnCudaCount);
		Definition_Host_Device_Buffer(currentCtx.beClonedVertexFlag_OnHost,currentCtx.beClonedVertexFlag_OnCuda,IndexType,_nExternalMemory * currentCtx.g_nVertexOnCudaCount);
#endif

		MyKrylov::initMyKrylovContext();
	}

#define bbcoeff(r,_local,_dim) (materialParam[r*3+_dim] * D1N_[_local*3+materialIndex[r*3+_dim]] )

	void makeGlobalIndexPara( float YoungModulus, float PossionRatio, float Density, float *externForce)
	{
		cudaDeviceSynchronize();
		PhysicsContext& currentCtx = FEM_State_Ctx;
		float params[6*3] =				 {1,0,0,  0,1,0,  0,0,1,  1,1,0,  1,0,1,  0,1,1};
		int   index[6*3] =			{0,0,0,  0,1,0,  0,0,2,  1,0,0,  2,0,0,  0,2,1};
		float E = YoungModulus;
		float mu = PossionRatio;
		float G = E/(2*(1+mu));
		float lai = mu*E/((1+mu)*(1-2*mu));
		float material[6] = {lai+2*G,lai+2*G,lai+2*G,G,G,G};

		currentCtx.g_lai = lai;
		currentCtx.g_Density = Density;


		HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.materialParams,  6*3* sizeof(float) ) );
		HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.materialIndex,   6*3* sizeof(int) ) );
		HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.materialValue,   6*   sizeof(float) ) );
		HANDLE_ERROR( cudaMalloc( (void**)&currentCtx.g_externForce,   3*   sizeof(float) ) );

		HANDLE_ERROR( cudaMemcpy( (void *)currentCtx.materialParams,&params[0],6*3* sizeof(float),cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( (void *)currentCtx.materialIndex ,&index[0] ,6*3* sizeof(int)  ,cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( (void *)currentCtx.materialValue ,&material[0],6* sizeof(float),cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy( (void *)currentCtx.g_externForce ,&externForce[0],3* sizeof(float),cudaMemcpyHostToDevice ) );
	}

	__device__ bool isEqual(float var1,float var2)
	{
#define EPSINON_ (0.000001f)
		const float positive = fabs(EPSINON_);
		const float negative = -1.0f * positive;

		if ( (var1 - var2) < positive && (var1 - var2) > negative )
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	
	__global__ void assemble_matrix_free_on_cuda_4_Corotation(
		const int nCellCount,  int nDofCount,CommonCellOnCuda * cellOnCudaPointer,VertexOnCuda *	VertexOnCudaPtr,
		IndexTypePtr systemOuterIdxPtr,ValueTypePtr systemRhsPtr,
		IndexTypePtr g_globalDof,ValueTypePtr g_globalValue,
		IndexTypePtr globalDof_Mass,ValueTypePtr globalValue_Mass,
		IndexTypePtr globalDof_Damping,ValueTypePtr globalValue_Damping,
		IndexTypePtr globalDof_System,ValueTypePtr globalValue_System,
		ValueTypePtr tmp_blockShareRhs_Rotation,ValueTypePtr tmp_blockShareRhs,
		ValueType * localStiffnessMatrixPtr,ValueType * /*localMassMatrixPtr*/,ValueType * /*localRhsVectorPtr*/,
		ValueTypePtr /*para_RKRtPj*/
		)
	{
		//const int currentCellIdx = blockIdx.x;
		//const int localDofIdx = threadIdx.x;/*0-23*/
		//const int localColIdx = threadIdx.y;/*0-23*/

		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		int currentCellIdx = tid / CONST_24x24;
		const int localDofIdx = (MyMod(tid , CONST_24x24)) / CONST_24;/*0-23*/
		const int localColIdx = MyMod((MyMod(tid , CONST_24x24)) , CONST_24);/*0-23*/

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
		float *blockShareRhs_Rotation;

		const int r_local = localDofIdx / Dim_Count;//[0-7]
		const int r_dim   = MyMod(localDofIdx , Dim_Count);//[0-2]
		const int c_local = localColIdx / Dim_Count;//[0-7]
		const int c_dim   = MyMod(localColIdx , Dim_Count);//[0-2]	

		if(currentCellIdx < nCellCount && true == cellOnCudaPointer[currentCellIdx].m_bLeaf )
		{			
			CommonCellOnCuda& curCommonCell = cellOnCudaPointer[currentCellIdx];

			float * localRhsVectorPtr = &curCommonCell.localRhsVectorOnCuda[0];
			float * localMassMatrixPtrPerCell = &curCommonCell.localMassMatrixOnCuda[0];
			/*livedCellFlag[currentCellIdx] = 1;*/

			const int global_row = VertexOnCudaPtr[ curCommonCell.vertexId[r_local] ].m_nGlobalDof[r_dim];
			const int global_col = VertexOnCudaPtr[ curCommonCell.vertexId[c_local] ].m_nGlobalDof[c_dim];

			const int currentDof = global_row;

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
			blockShareRhs_Rotation = tmp_blockShareRhs_Rotation + currentDof*8;
			//blockShareRhs_compare = tmp_blockShareRhs_compare + currentDof*8;

			const int loc_row = localDofIdx;//cellOnCudaPointer[currentCellIdx].localDofs[localDofIdx];
			//const int global_col = cellOnCudaPointer[currentCellIdx].globalDofs[localColIdx];
			const int loc_col = localColIdx;//cellOnCudaPointer[currentCellIdx].localDofs[localColIdx];//localColIdx;
			const int idx_in_8 = loc_row / 3;

			//const float col_val_mass		 =  localMassMatrixPtr       [ loc_row * Geometry_dofs_per_cell + loc_col]*MASS_MATRIX_COEFF_2;
			const float col_val_mass =  localMassMatrixPtrPerCell[ loc_row * Geometry_dofs_per_cell + loc_col]*MASS_MATRIX_COEFF_2;
			
#if 1
			const float col_val_stiffness = cellOnCudaPointer[currentCellIdx].RKR[loc_row * Geometry_dofs_per_cell + loc_col];//cellOnCudaPointer[currentCellIdx]localStiffnessMatrixPtr[cellOnCudaPointer[currentCellIdx].m_nStiffnessMatrixIdx * Geometry_dofs_per_cell_squarte + loc_row * Geometry_dofs_per_cell + loc_col];

#else
			const float col_val_stiffness = localStiffnessMatrixPtr[currentCellIdx*Geometry_dofs_per_cell_squarte+loc_row * Geometry_dofs_per_cell + loc_col];
#endif
			const int index = idx_in_8 * LocalMaxDofCount_YC + loc_col;
			local_globalDof_mass[index] = global_col;
			value_mass[index] = col_val_mass;
			local_globalDof[index] = global_col;
			value[index] = col_val_stiffness;
			local_globalDof_damping[index] = global_col;
			value_damping[index] = Material_Damping_Alpha/*0.183f*/ * col_val_mass + Material_Damping_Beta/*0.00128f*/ * col_val_stiffness;
			local_globalDof_system[index] = global_col;
			value_system[index] = NewMarkConstant_0/*16384*/ * col_val_mass + NewMarkConstant_1/*128*/ * value_damping[index] + col_val_stiffness;

			if (0 == localColIdx)
			{	
#if 1
				blockShareRhs_Rotation[idx_in_8] = cellOnCudaPointer[currentCellIdx].RKRtPj[loc_row];
				blockShareRhs[idx_in_8] = localRhsVectorPtr[/*curCommonCell.m_nRhsIdx * Geometry_dofs_per_cell +*/ loc_row];
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

	__global__ void assemble_matrix_free_on_cuda(
		int nCellCount,int /*funcMatrixCount*/,int nDofCount,CommonCellOnCuda * cellOnCudaPointer,VertexOnCuda *	VertexOnCudaPtr,
		IndexTypePtr /*systemOuterIdxPtr*/,ValueTypePtr systemRhsPtr,
		IndexTypePtr g_globalDof,ValueTypePtr g_globalValue,
		IndexTypePtr globalDof_Mass,ValueTypePtr globalValue_Mass,
		IndexTypePtr globalDof_Damping,ValueTypePtr globalValue_Damping,
		IndexTypePtr globalDof_System,ValueTypePtr globalValue_System,
		ValueTypePtr tmp_blockShareRhs,
		float* materialParam, int* materialIndex, float * materialValue,
		float lai, float Density, float *externForce,
		ValueType * localStiffnessMatrixPtr,ValueType * localMassMatrixPtr,ValueType * /*localRhsVectorPtr*/
		)
	{
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;

		/*const int currentCellIdx = tid / CONST_24x24;
		const int localDofIdx = (tid % CONST_24x24) / CONST_24;
		const int localColIdx = (tid % CONST_24x24) % CONST_24;*/

		const int currentCellIdx = tid / CONST_24x24;
		const int localDofIdx = (MyMod(tid , CONST_24x24)) / CONST_24;/*0-23*/
		const int localColIdx = MyMod((MyMod(tid , CONST_24x24)) , CONST_24);/*0-23*/
		

		
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

			CommonCellOnCuda& curCommonCell = cellOnCudaPointer[currentCellIdx];

			float * localRhsVectorPtr = &curCommonCell.localRhsVectorOnCuda[0];
			//float * localMassMatrixPtr = &curCommonCell.localMassMatrixOnCuda[0];

			if (_CellTypeFEM == curCommonCell.cellType)
			{
				
				//FEM
#if 1
				const int r_local = localDofIdx / Dim_Count;//[0-7]
				const int r_dim   = MyMod(localDofIdx , Dim_Count);//[0-2]
				const int c_local = localColIdx / Dim_Count;//[0-7]
				const int c_dim   = MyMod(localColIdx , Dim_Count);//[0-2]	

				const int global_row = VertexOnCudaPtr[ curCommonCell.vertexId[r_local] ].m_nGlobalDof[r_dim];
				const int global_col = VertexOnCudaPtr[ curCommonCell.vertexId[c_local] ].m_nGlobalDof[c_dim];

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

				const float col_val_mass =  localMassMatrixPtr[curCommonCell.m_nMassMatrixIdx * Geometry_dofs_per_cell_squarte + loc_row * Geometry_dofs_per_cell + loc_col]*MASS_MATRIX_COEFF_2;
				const float col_val_stiffness = localStiffnessMatrixPtr[curCommonCell.m_nStiffnessMatrixIdx * Geometry_dofs_per_cell_squarte + loc_row * Geometry_dofs_per_cell + loc_col];
				
				const int index = idx_in_8 * LocalMaxDofCount_YC + loc_col;

				local_globalDof_mass[index] = global_col;
				value_mass[index] = col_val_mass;
				local_globalDof[index] = global_col;
				value[index] = col_val_stiffness;
				local_globalDof_damping[index] = global_col;
				value_damping[index] = Material_Damping_Alpha * col_val_mass + Material_Damping_Beta * col_val_stiffness;
				local_globalDof_system[index] = global_col;
				value_system[index] = NewMarkConstant_0 * value_mass[index] + NewMarkConstant_1 * value_damping[index] + value[index];

				if (0 == localColIdx)
				{	
					blockShareRhs[idx_in_8] = localRhsVectorPtr[/*curCommonCell.m_nRhsIdx * Geometry_dofs_per_cell +*/ loc_row];
					//CUPRINTF("rhs_value[%d](%f) col_val_mass(%f) col_val_stiffness(%f)\n",global_row*8+idx_in_8,blockShareRhs[idx_in_8],col_val_mass,col_val_stiffness);
				}
#endif
			}
			else //if (1 == curEFGCell.cellType)
			{
				CUPRINTF("CellTypeFEM != curCommonCell.cellType \n");
				//EFG
			}		
		}
		else
		{
			//CUPRINTF("CUDA ERROR\n");
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
			//CUPRINTF("value(%f){%f,%f,%f,%f,%f,%f,%f,%f}\n",systemRhsPtr[currentDof],blockShareRhs[0] , blockShareRhs[1] , blockShareRhs[2] , blockShareRhs[3] , blockShareRhs[4] , blockShareRhs[5] , blockShareRhs[6] , blockShareRhs[7]);
		}
	}

	__global__ void assembleRhsValue_on_cuda_Corotation(int nDofCount, ValueTypePtr systemRhsPtr,ValueTypePtr tmp_blockShareRhs, 
		ValueTypePtr systemRhsPtrRotation, ValueTypePtr tmp_blockShareRhsRotation)
	{
		const int currentDof = threadIdx.x + blockIdx.x * blockDim.x;
		if (currentDof < nDofCount)
		{
			ValueTypePtr blockShareRhs = tmp_blockShareRhs + currentDof*8;
			systemRhsPtr[currentDof] = blockShareRhs[0] + blockShareRhs[1] + blockShareRhs[2] + blockShareRhs[3] + 
				blockShareRhs[4] + blockShareRhs[5] + blockShareRhs[6] + blockShareRhs[7];

			ValueTypePtr blockShareRhsRotation = tmp_blockShareRhsRotation + currentDof*8;
			systemRhsPtrRotation[currentDof] = blockShareRhsRotation[0] + blockShareRhsRotation[1] + blockShareRhsRotation[2] + blockShareRhsRotation[3] + 
				blockShareRhsRotation[4] + blockShareRhsRotation[5] + blockShareRhsRotation[6] + blockShareRhsRotation[7];

		}
	}

	

	dim3 threads_24_24(MaxCellDof4FEM,MaxCellDof4FEM);

	void computeRotationMatrix_SpeedUp();

	void assembleSystemOnCuda_FEM_RealTime_ForCorotation()//done
	{
		MY_RANGE("assembleSystemOnCuda_FEM_RealTime_ForCorotation",6);
#if DEBUG_TIME
		nLastTick = GetTickCount()/*,nnCount = 0*/;
#endif
		PhysicsContext& currentCtx = FEM_State_Ctx;
		int nCellCount = currentCtx.nCellOnCudaCount;

		int nouse=0;
		const int nDofCount = currentCtx.g_nDofs;
		int nonZeroCount/*,nonZeroCountMass*/;

		//printf("nCellCount[%d] nDofCount[%d] g_lai[%f] g_Density[%f]\n",nCellCount,nDofCount,g_lai,g_Density);

		CommonCellOnCuda * cellOnCudaPointer = currentCtx.CellOnCudaPtr;
		VertexOnCuda*   VertexOnCudaPtr = currentCtx.g_VertexOnCudaPtr;
		if (currentCtx.g_nDofsLast < nDofCount)
		{
			const int dataSize = (nDofCount)*nMaxNonZeroSizeInFEM;
			Mem_Zero(currentCtx.g_globalDof_MF,IndexType,dataSize);//	HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalDof_MF,  0, (nDofCount)*nMaxNonZeroSizeInFEM * sizeof(IndexType)));
			Mem_Zero(currentCtx.g_globalValue_MF,ValueType,dataSize);//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalValue_MF,  0, (nDofCount)*nMaxNonZeroSizeInFEM * sizeof(ValueType)));
			Mem_Zero(currentCtx.g_globalDof_Mass_MF,IndexType,dataSize);//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalDof_Mass_MF,  0, (nDofCount)*nMaxNonZeroSizeInFEM * sizeof(IndexType)));
			Mem_Zero(currentCtx.g_globalValue_Mass_MF,ValueType,dataSize);//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalValue_Mass_MF,  0, (nDofCount)*nMaxNonZeroSizeInFEM * sizeof(ValueType)));
			Mem_Zero(currentCtx.g_globalDof_Damping_MF,IndexType,dataSize);//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalDof_Damping_MF,  0, (nDofCount)*nMaxNonZeroSizeInFEM * sizeof(IndexType)));
			Mem_Zero(currentCtx.g_globalValue_Damping_MF,ValueType,dataSize);//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalValue_Damping_MF,  0, (nDofCount)*nMaxNonZeroSizeInFEM * sizeof(ValueType)));
			Mem_Zero(currentCtx.g_globalDof_System_MF,IndexType,dataSize);//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalDof_System_MF,  0, (nDofCount)*nMaxNonZeroSizeInFEM * sizeof(IndexType)));
			Mem_Zero(currentCtx.g_globalValue_System_MF,ValueType,dataSize);//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalValue_System_MF,  0, (nDofCount)*nMaxNonZeroSizeInFEM * sizeof(ValueType)));
			Mem_Zero(currentCtx.g_systemRhsPtr_MF,ValueType,nDofCount);//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_systemRhsPtr_MF,  0, (nDofCount) * sizeof(ValueType)));
			Mem_Zero(currentCtx.g_systemRhsPtr_In8_MF,ValueType,(nDofCount)* VertxMaxInflunceCellCount);//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_systemRhsPtr_In8_MF,  0, (nDofCount)* VertxMaxInflunceCellCount * sizeof(ValueType)));

#if USE_CO_RATION
			Mem_Zero(currentCtx.myOptimize_Array_R_rhs_tmp4Corotaion,ValueType,nDofCount);
			Mem_Zero(currentCtx.g_systemRhsPtr_MF_Rotation,ValueType,nDofCount);//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_systemRhsPtr_MF_Rotation,  0, (nDofCount) * sizeof(ValueType)));
			Mem_Zero(currentCtx.g_systemRhsPtr_In8_MF_Rotation,ValueType,(nDofCount)* VertxMaxInflunceCellCount);//HANDLE_ERROR( cudaMemset((void *)currentCtx.g_systemRhsPtr_In8_MF_Rotation,  0, (nDofCount)* VertxMaxInflunceCellCount * sizeof(ValueType)));
#endif

		}
#define SpeedUpFactor0 (24)

		//computeRotationMatrix_SpeedUp();
		
#if USE_CUDA_STREAM
		assemble_matrix_free_on_cuda_4_Corotation<<< GRIDCOUNT(nCellCount*CONST_24x24 ,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK ,0, streamForSkin>>>
#else
		assemble_matrix_free_on_cuda_4_Corotation<<< GRIDCOUNT(nCellCount*CONST_24x24,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK >>>
#endif
		
		(	nCellCount,  nDofCount, cellOnCudaPointer, VertexOnCudaPtr,
			/*g_systemOuterIdxPtr*/(IndexTypePtr)0 , currentCtx.g_systemRhsPtr_MF,
			currentCtx.g_globalDof_MF,	currentCtx.g_globalValue_MF,
			currentCtx.g_globalDof_Mass_MF,	currentCtx.g_globalValue_Mass_MF,
			currentCtx.g_globalDof_Damping_MF,	currentCtx.g_globalValue_Damping_MF,
			currentCtx.g_globalDof_System_MF,	currentCtx.g_globalValue_System_MF,	
			currentCtx.g_systemRhsPtr_In8_MF_Rotation,currentCtx.g_systemRhsPtr_In8_MF,
			/*currentCtx.materialParams,	currentCtx.materialIndex,	currentCtx.materialValue,
			currentCtx.g_lai,	currentCtx.g_Density,	currentCtx.g_externForce,*/
			currentCtx.localStiffnessMatrixOnCuda, /*MyNull*/currentCtx.localMassMatrixOnCuda, MyNull/*currentCtx.localRhsVectorOnCuda*/,currentCtx.cuda_RKRtPj);

		//computeRotationMatrix_SpeedUp();
#if USE_CUDA_STREAM
		assembleRhsValue_on_cuda_Corotation<<<GRIDCOUNT(nDofCount,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK ,0, streamForSkin>>>
#else
		assembleRhsValue_on_cuda_Corotation<<<GRIDCOUNT(nDofCount,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK>>>
#endif
		
			(nDofCount,currentCtx.g_systemRhsPtr_MF,currentCtx.g_systemRhsPtr_In8_MF,currentCtx.g_systemRhsPtr_MF_Rotation,currentCtx.g_systemRhsPtr_In8_MF_Rotation);

#if USE_CO_RATION
		//setCuspVector_deviceMemory(currentCtx.cusp_Array_R_rhs_Corotaion/*cusp_Array_Rhs*/,nDofCount,currentCtx.g_systemRhsPtr_MF_Rotation);
#endif

		if (currentCtx.g_nDofsLast < nDofCount)
		{
			printf("[####]currentCtx.g_nDofsLast < nDofCount \n");
			currentCtx.cusp_Array_Rhs.resize(nDofCount);
			currentCtx.cusp_Array_Incremental_displace.resize(nDofCount);
			currentCtx.cusp_Array_Displacement.resize(nDofCount);
			currentCtx.cusp_Array_R_rhs.resize(nDofCount);
			//currentCtx.tmpRhs.resize(nDofCount,0);
			currentCtx.cusp_Array_Mass_rhs.resize(nDofCount);
			currentCtx.cusp_Array_Damping_rhs.resize(nDofCount);
			currentCtx.cusp_Array_Velocity.resize(nDofCount);
			currentCtx.cusp_Array_Acceleration.resize(nDofCount);
			currentCtx.cusp_Array_Old_Acceleration.resize(nDofCount);
			currentCtx.cusp_Array_Old_Displacement.resize(nDofCount);
			currentCtx.cusp_Array_R_rhs_tmp4Corotaion.resize(nDofCount);
			currentCtx.cusp_Array_R_rhs_Corotaion.resize(nDofCount);

			currentCtx.cg_y.resize(nDofCount);;
			currentCtx.cg_z.resize(nDofCount);;
			currentCtx.cg_r.resize(nDofCount);;
			currentCtx.cg_p.resize(nDofCount);;
			currentCtx.g_nDofsLast = nDofCount;

			currentCtx.MassMatrix.initialize(currentCtx.g_nDofs,/*g_systemOuterIdxPtr*/(IndexTypePtr)0, currentCtx.g_globalDof_Mass_MF, currentCtx.g_globalValue_Mass_MF);
			currentCtx.DampingMatrix.initialize(currentCtx.g_nDofs,/*g_systemOuterIdxPtr*/(IndexTypePtr)0, currentCtx.g_globalDof_Damping_MF, currentCtx.g_globalValue_Damping_MF);
			currentCtx.SystemMatrix.initialize(currentCtx.g_nDofs,/*g_systemOuterIdxPtr*/(IndexTypePtr)0, currentCtx.g_globalDof_System_MF, currentCtx.g_globalValue_System_MF);
		}
		else
		{
			currentCtx.SystemMatrix.initialize(currentCtx.g_nDofs,/*g_systemOuterIdxPtr*/(IndexTypePtr)0, currentCtx.g_globalDof_System_MF, currentCtx.g_globalValue_System_MF);
		}
		
		//return ;
		
#if DEBUG_TIME
		nCurrentTick = GetTickCount();
		printf("assembleSystemOnCuda %d  currentCtx.g_nDofsLast(%d) nDofCount(%d)\n",nCurrentTick - nLastTick,currentCtx.g_nDofsLast,nDofCount);
#endif
		return ;
	}
	void assembleSystemOnCuda_FEM_RealTime()//done
	{
#if DEBUG_TIME
		nLastTick = GetTickCount();
#endif
		PhysicsContext& currentCtx = FEM_State_Ctx;
		const int nCellCount = currentCtx.nCellOnCudaCount;

		int nouse=0;
		const int nDofCount = currentCtx.g_nDofs;
		int nonZeroCount/*,nonZeroCountMass*/;

		//printf("nCellCount[%d] nDofCount[%d] g_lai[%f] g_Density[%f]\n",nCellCount,nDofCount,g_lai,g_Density);

		CommonCellOnCuda * cellOnCudaPointer = currentCtx.CellOnCudaPtr;
		VertexOnCuda*   VertexOnCudaPtr = currentCtx.g_VertexOnCudaPtr;
		
		{
			HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalDof_MF,  0, (nDofCount)*nMaxNonZeroSizeInFEM * sizeof(IndexType)));
			HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalValue_MF,  0, (nDofCount)*nMaxNonZeroSizeInFEM * sizeof(ValueType)));
			HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalDof_Mass_MF,  0, (nDofCount)*nMaxNonZeroSizeInFEM * sizeof(IndexType)));
			HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalValue_Mass_MF,  0, (nDofCount)*nMaxNonZeroSizeInFEM * sizeof(ValueType)));
			HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalDof_Damping_MF,  0, (nDofCount)*nMaxNonZeroSizeInFEM * sizeof(IndexType)));
			HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalValue_Damping_MF,  0, (nDofCount)*nMaxNonZeroSizeInFEM * sizeof(ValueType)));
			HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalDof_System_MF,  0, (nDofCount)*nMaxNonZeroSizeInFEM * sizeof(IndexType)));
			HANDLE_ERROR( cudaMemset((void *)currentCtx.g_globalValue_System_MF,  0, (nDofCount)*nMaxNonZeroSizeInFEM * sizeof(ValueType)));
			HANDLE_ERROR( cudaMemset((void *)currentCtx.g_systemRhsPtr_MF,  0, (nDofCount) * sizeof(ValueType)));
			HANDLE_ERROR( cudaMemset((void *)currentCtx.g_systemRhsPtr_In8_MF,  0, (nDofCount)* VertxMaxInflunceCellCount * sizeof(ValueType)));

		}
		
		
		//dim3 threads_24_24(24,24);
		assemble_matrix_free_on_cuda<<< GRIDCOUNT(nCellCount*CONST_24x24,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK >>>(
			nCellCount, nouse, nDofCount, cellOnCudaPointer, VertexOnCudaPtr,
			/*g_systemOuterIdxPtr*/(IndexTypePtr)0 , currentCtx.g_systemRhsPtr_MF,
			currentCtx.g_globalDof_MF,	currentCtx.g_globalValue_MF,
			currentCtx.g_globalDof_Mass_MF,	currentCtx.g_globalValue_Mass_MF,
			currentCtx.g_globalDof_Damping_MF,	currentCtx.g_globalValue_Damping_MF,
			currentCtx.g_globalDof_System_MF,	currentCtx.g_globalValue_System_MF,	currentCtx.g_systemRhsPtr_In8_MF,
			currentCtx.materialParams,	currentCtx.materialIndex,	currentCtx.materialValue,
			currentCtx.g_lai,	currentCtx.g_Density,	currentCtx.g_externForce,
			currentCtx.localStiffnessMatrixOnCuda, /*MyNull*/currentCtx.localMassMatrixOnCuda, MyNull/*currentCtx.localRhsVectorOnCuda*/);
		
		assembleRhsValue_on_cuda<<<GRIDCOUNT(nDofCount,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK>>>(nDofCount,currentCtx.g_systemRhsPtr_MF,currentCtx.g_systemRhsPtr_In8_MF);
		
		//setCuspVector_deviceMemory(currentCtx.cusp_Array_Rhs,nDofCount,currentCtx.g_systemRhsPtr_MF);
	
		if (currentCtx.g_nDofsLast < nDofCount)
		{
			printf("[XXXXX] currentCtx.g_nDofsLast < nDofCount \n");
			currentCtx.cusp_Array_Rhs.resize(nDofCount);
			currentCtx.cusp_Array_Incremental_displace.resize(nDofCount);
			currentCtx.cusp_Array_Displacement.resize(nDofCount);
			currentCtx.cusp_Array_R_rhs.resize(nDofCount);
			//currentCtx.tmpRhs.resize(nDofCount,0);
			currentCtx.cusp_Array_Mass_rhs.resize(nDofCount);
			currentCtx.cusp_Array_Damping_rhs.resize(nDofCount);
			currentCtx.cusp_Array_Velocity.resize(nDofCount);
			currentCtx.cusp_Array_Acceleration.resize(nDofCount);
			currentCtx.cusp_Array_Old_Acceleration.resize(nDofCount);
			currentCtx.cusp_Array_Old_Displacement.resize(nDofCount);
			currentCtx.cusp_Array_R_rhs_tmp4Corotaion.resize(nDofCount);
			currentCtx.cusp_Array_R_rhs_Corotaion.resize(nDofCount);
			currentCtx.g_nDofsLast = nDofCount;
		}

		currentCtx.MassMatrix.initialize(currentCtx.g_nDofs,(IndexTypePtr)0, currentCtx.g_globalDof_Mass_MF, currentCtx.g_globalValue_Mass_MF);
		currentCtx.DampingMatrix.initialize(currentCtx.g_nDofs,(IndexTypePtr)0, currentCtx.g_globalDof_Damping_MF, currentCtx.g_globalValue_Damping_MF);
		currentCtx.SystemMatrix.initialize(currentCtx.g_nDofs,(IndexTypePtr)0, currentCtx.g_globalDof_System_MF, currentCtx.g_globalValue_System_MF);

#if DEBUG_TIME
		nCurrentTick = GetTickCount();
		printf("assembleSystemOnCuda %d  currentCtx.g_nDofsLast(%d) nDofCount(%d)\n",nCurrentTick - nLastTick,currentCtx.g_nDofsLast,nDofCount);
#endif
		return ;
	}

	void assembleSystemOnCuda_FEM_RealTime_MatrixInitDiag()
	{
		PhysicsContext& currentCtx = FEM_State_Ctx;
		currentCtx.MassMatrix.initialize(currentCtx.g_nDofs,(IndexTypePtr)0, currentCtx.g_globalDof_Mass_MF, currentCtx.g_globalValue_Mass_MF);
		currentCtx.DampingMatrix.initialize(currentCtx.g_nDofs,(IndexTypePtr)0, currentCtx.g_globalDof_Damping_MF, currentCtx.g_globalValue_Damping_MF);
		currentCtx.SystemMatrix.initialize(currentCtx.g_nDofs,(IndexTypePtr)0, currentCtx.g_globalDof_System_MF, currentCtx.g_globalValue_System_MF);

		//cusp::blas::nrm2(currentCtx.cusp_Array_Rhs);
		//MyKrylov::nrm2(currentCtx.cusp_Array_Rhs);
	}

	__global__ void cuda_apply_boundary_values(IndexType nbcCount,IndexTypePtr param_boundaryCondition,ValueTypePtr param_diagnosticValue/*,ValueTypePtr para_boundaryConditionVal*/,ValueTypePtr para_Array_R_rhs,ValueTypePtr para_Array_Incremental_displace)
	{
		int v = threadIdx.x + blockIdx.x * blockDim.x;
		if (v < nbcCount)
		{
			const int dof_number = param_boundaryCondition[v];
			const  ValueType EPSINON = 0.000001;
			para_Array_R_rhs[dof_number] = 0.f;//para_boundaryConditionVal[v] * param_diagnosticValue[dof_number];
			para_Array_Incremental_displace[dof_number] = 0.f;//para_boundaryConditionVal[v];
			//v += blockDim.x * gridDim.x;
		}
	}
	void apply_boundary_values_Init( int *dofsList,int nListCount)
	{
		PhysicsContext& currentCtx = FEM_State_Ctx;
		for (int v=0;v < nListCount;++v)
		{
			const int dof_number = dofsList[v];
			currentCtx.SystemMatrix.setBoundaryCondition(dof_number);
		}
		return ;
	}
	void apply_boundary_values( )
	{
		MY_RANGE("apply_boundary_values",1);
		//cudaDeviceSynchronize();
		PhysicsContext& currentCtx = FEM_State_Ctx;
		
		{
#if USE_CUDA_STREAM
			cuda_apply_boundary_values<<<GRIDCOUNT(currentCtx.nBcCount,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK,0,streamForSkin>>>
#else
			cuda_apply_boundary_values<<<GRIDCOUNT(currentCtx.nBcCount,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK>>>
#endif
			(currentCtx.nBcCount,currentCtx.cuda_boundaryCondition,currentCtx.cuda_diagnosticValue/*,cuda_boundaryConditionVal*/,currentCtx.rhsOnCuda,currentCtx.displacementOnCuda);
		}
	}

	void initBoundaryCondition()//done
	{
		LogInfo("FEM_State_Ctx.nBcCount %d\n",FEM_State_Ctx.nBcCount);
		apply_boundary_values_Init(FEM_State_Ctx.cusp_boundaryCondition,FEM_State_Ctx.nBcCount);
	}

	__global__ void cudaApplyExternalForce(IndexType nBCCount, IndexTypePtr displaceBoundaryConditionDofs, ValueTypePtr rhs_on_cuda ,ValueType scaled)
	{
		int nCurrentIdx = threadIdx.x + blockIdx.x * blockDim.x;
		if (nCurrentIdx < nBCCount)
		{
			rhs_on_cuda[ displaceBoundaryConditionDofs[nCurrentIdx] ] += scaled;
		}
	}

	__global__ void cuda_Update_Rhs(const int nDofs, 
		ValueTypePtr cusp_Array_Displacement, ValueTypePtr cusp_Array_Velocity, ValueTypePtr cusp_Array_Acceleration, 
		ValueTypePtr cusp_Array_Mass_rhs, ValueTypePtr cusp_Array_Damping_rhs, ValueTypePtr NewMarkConstant)
	{
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < nDofs)
		{
			cusp_Array_Mass_rhs[tid] = cusp_Array_Displacement[tid]*NewMarkConstant[0] + cusp_Array_Velocity[tid]*NewMarkConstant[2] + cusp_Array_Acceleration[3];
			cusp_Array_Damping_rhs[tid] = cusp_Array_Displacement[tid]*NewMarkConstant[1] + cusp_Array_Velocity[tid]*NewMarkConstant[4] + cusp_Array_Acceleration[5];
		}
	}

	__global__ void cuda_Update_Rhs_Plus(const int nDofs, 
		ValueTypePtr g_systemRhsPtr_MF, ValueTypePtr myOptimize_Old_Acceleration, ValueTypePtr myOptimize_Old_Displacement, ValueTypePtr g_systemRhsPtr_MF_Rotation,
		ValueTypePtr rhsOnCuda)
	{
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < nDofs)
		{
			rhsOnCuda[tid] = g_systemRhsPtr_MF[tid] + myOptimize_Old_Acceleration[tid] + myOptimize_Old_Displacement[tid] + g_systemRhsPtr_MF_Rotation[tid];
		}		
	}

	__global__ void cuda_Update_Rhs_ApplyBladeFoce(const int nDofs, ValueTypePtr rhsOnCuda, ValueTypePtr bladeForceOnCuda)
	{
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < nDofs)
		{
			rhsOnCuda[tid] += bladeForceOnCuda[tid];
		}		
	}

	void update_rhs(const int nCount)
	{
		MY_RANGE("update_rhs",0);
		
		PhysicsContext& currentCtx = FEM_State_Ctx;
#if 1
		//;
#if USE_CUDA_STREAM
		cuda_Update_Rhs<<< GRIDCOUNT(currentCtx.g_nDofs,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK ,0,streamForSkin>>>
#else
		cuda_Update_Rhs<<< GRIDCOUNT(currentCtx.g_nDofs,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK >>>
#endif
			(currentCtx.g_nDofs, 
			currentCtx.myOptimize_Array_Displacement, currentCtx.myOptimize_Array_Velocity, currentCtx.myOptimize_Array_Acceleration, 
			currentCtx.myOptimize_Array_Mass_rhs, currentCtx.myOptimize_Array_Damping_rhs, element_Array_NewMarkConstant);
#else
		cusp::blas::axpbypcz(currentCtx.cusp_Array_Displacement,currentCtx.cusp_Array_Velocity,currentCtx.cusp_Array_Acceleration,
			currentCtx.cusp_Array_Mass_rhs,
			cusp_Array_NewMarkConstant[0],cusp_Array_NewMarkConstant[2],cusp_Array_NewMarkConstant[3]);


		cusp::blas::axpbypcz(currentCtx.cusp_Array_Displacement,currentCtx.cusp_Array_Velocity,currentCtx.cusp_Array_Acceleration,
			currentCtx.cusp_Array_Damping_rhs,
			cusp_Array_NewMarkConstant[1],cusp_Array_NewMarkConstant[4],cusp_Array_NewMarkConstant[5]);
#endif				
		cusp::multiply(currentCtx.MassMatrix, currentCtx.cusp_Array_Mass_rhs, currentCtx.cusp_Array_Old_Acceleration);		
		cusp::multiply(currentCtx.DampingMatrix,currentCtx.cusp_Array_Damping_rhs,currentCtx.cusp_Array_Old_Displacement);
		
#if 1
		if ( nCount >= cuda_bcMinCount && nCount < cuda_bcMaxCount)
		{
#if USE_CUDA_STREAM
			cudaApplyExternalForce<<<GRIDCOUNT(currentCtx.nForceCount,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK ,0,streamForSkin>>>
#else
			cudaApplyExternalForce<<<GRIDCOUNT(currentCtx.nForceCount,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK>>>
#endif
				(currentCtx.nForceCount, currentCtx.cuda_forceCondition, currentCtx.g_systemRhsPtr_MF,cuda_scriptForceFactor* 346.0f );
		}
		
#if USE_CUDA_STREAM
		cuda_Update_Rhs_Plus<<< GRIDCOUNT(currentCtx.g_nDofs,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK ,0,streamForSkin>>>
#else
		cuda_Update_Rhs_Plus<<< GRIDCOUNT(currentCtx.g_nDofs,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK >>>
#endif
			(currentCtx.g_nDofs, 
			 currentCtx.g_systemRhsPtr_MF, currentCtx.myOptimize_Old_Acceleration, currentCtx.myOptimize_Old_Displacement, currentCtx.g_systemRhsPtr_MF_Rotation,
			 currentCtx.rhsOnCuda);

		//return ;

		if (currentCtx.g_isApplyExternalForceAtCurrentFrame > 0)
		{
			MyError("currentCtx.g_isApplyExternalForceAtCurrentFrame > 0");
			currentCtx.g_isApplyExternalForceAtCurrentFrame--;
			cusp::blas::axpbypcz(currentCtx.cusp_Array_R_rhs_tmp4Corotaion,currentCtx.cusp_Array_R_rhs_Corotaion,currentCtx.g_globalExternalForce,
				currentCtx.cusp_Array_R_rhs,
				ValueType(1),ValueType(1),ValueType(1));
		}
		else
		{
			if (currentCtx.g_isApplyBladeForceCurrentFrame > 0)
			{
				printf("blade force\n");
				//MyError("currentCtx.g_isApplyBladeForceCurrentFrame > 0");
				currentCtx.g_isApplyBladeForceCurrentFrame--;
#if USE_CUDA_STREAM
				cuda_Update_Rhs_ApplyBladeFoce<<< GRIDCOUNT(currentCtx.g_nDofs,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK ,0,streamForSkin>>>
#else
				cuda_Update_Rhs_ApplyBladeFoce<<< GRIDCOUNT(currentCtx.g_nDofs,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK >>>
#endif
				(currentCtx.g_nDofs, currentCtx.rhsOnCuda, currentCtx.myOptimize_BladeForce);
			}
			else
			{
				printf("zero force\n");
				/*cusp::blas::axpby(currentCtx.cusp_Array_R_rhs_tmp4Corotaion,currentCtx.cusp_Array_R_rhs_Corotaion,
					currentCtx.cusp_Array_R_rhs,
					ValueType(1),ValueType(1));*/
			}

		}

#else
		if ( nCount >= cuda_bcMinCount && nCount < cuda_bcMaxCount)
		{	
			cudaApplyExternalForce<<<GRIDCOUNT(currentCtx.g_nDofs,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK>>>
				(currentCtx.nForceCount, currentCtx.cuda_forceCondition, currentCtx.g_systemRhsPtr_MF,cuda_scriptForceFactor* 346.0f );
		}

		if (currentCtx.g_isApplyExternalForceAtCurrentFrame > 0)
		{
			currentCtx.g_isApplyExternalForceAtCurrentFrame--;
			cusp::blas::axpby(currentCtx.cusp_Array_Rhs,currentCtx.cusp_Array_Old_Acceleration,
				currentCtx.cusp_Array_R_rhs_Corotaion,
				ValueType(1),ValueType(1));

			cusp::blas::axpbypcz(currentCtx.cusp_Array_R_rhs_Corotaion,currentCtx.cusp_Array_Old_Displacement,currentCtx.g_globalExternalForce,
				currentCtx.cusp_Array_R_rhs,
				ValueType(1),ValueType(1),ValueType(1));
		}
		else
		{
			
			cusp::blas::axpbypcz(currentCtx.cusp_Array_Rhs,currentCtx.cusp_Array_Old_Acceleration,currentCtx.cusp_Array_Old_Displacement,
				currentCtx.cusp_Array_R_rhs,
				ValueType(1),ValueType(1),ValueType(1));
		}
#endif
	}

	namespace MyKrylov
	{	

		__global__ void cuda_xmy(const int nDofs, 
			ValueTypePtr x, ValueTypePtr m, ValueTypePtr y)
		{
			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			y[tid] = x[tid] * m[tid];
		}

		template <typename ValueType, typename MemorySpace>
		class my_diagonal : public cusp::linear_operator<ValueType, MemorySpace>
		{       
			typedef cusp::linear_operator<ValueType, MemorySpace> Parent;
			MyCuspVecView diagonal_reciprocals;

		public:
			/*! construct a \p diagonal preconditioner
			 *
			 * \param A matrix to precondition
			 * \tparam MatrixType matrix
			 */
			template<typename MatrixType>
			my_diagonal(const MatrixType& A);
        
			/*! apply the preconditioner to vector \p x and store the result in \p y
			 *
			 * \param x input vector
			 * \param y ouput vector
			 * \tparam VectorType1 vector
			 * \tparam VectorType2 vector
			 */
			template <typename VectorType1, typename VectorType2>
			void operator()(const VectorType1& x, VectorType2& y) const;
		};

		template <typename ValueType, typename MemorySpace>
		template<typename MatrixType>
		my_diagonal<ValueType,MemorySpace>
			::my_diagonal(const MatrixType& A)
			: cusp::linear_operator<ValueType,MemorySpace>(A.num_rows, A.num_cols, A.num_rows)
		{
			// extract the main diagonal
			diagonal_reciprocals = A.diagonalVec;
			//my_extract_diagonal(A, diagonal_reciprocals,cusp::unknown_format());

			// invert the entries
			/*thrust::transform(diagonal_reciprocals.begin(), diagonal_reciprocals.end(),
				diagonal_reciprocals.begin(), detail::reciprocal<ValueType>());*/
		}

		template <typename ValueType, typename MemorySpace>
		template <typename VectorType1, typename VectorType2>
		void my_diagonal<ValueType, MemorySpace>
			::operator()(const VectorType1& x, VectorType2& y) const
		{
			float * x_ptr = thrust::raw_pointer_cast(&x[0]);
			float * y_ptr = thrust::raw_pointer_cast(&y[0]);
			float * m_ptr = thrust::raw_pointer_cast(&diagonal_reciprocals[0]);
			//cusp::blas::xmy(diagonal_reciprocals, x, y);
#if USE_CUDA_STREAM
			cuda_xmy<<<GRIDCOUNT(num_rows,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK ,0,streamForSkin>>>
#else
			cuda_xmy<<<GRIDCOUNT(currentCtx.nForceCount,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK>>>
#endif
				(num_rows,x_ptr,m_ptr,y_ptr);
			
		}
		//typedef thrust::system::cuda::detail::reduce_detail::unordered_reduce_closure<InputIterator,difference_type,OutputType,OutputIterator,BinaryFunction,Context> Closure;

		template<typename Closure>
		__global__ __launch_bounds__(Closure::context_type::ThreadsPerBlock::value, Closure::context_type::BlocksPerMultiprocessor::value)
			void launch_closure_by_value_0(Closure f)
		{
			f();
		}

		template<typename Closure, typename Size1, typename Size2, typename Size3>
		static void launch_0(Closure f, Size1 num_blocks, Size2 block_size, Size3 smem_size)
		{
#if 1
			if(num_blocks > 0)
			{
#if USE_CUDA_STREAM	
				launch_closure_by_value_0<<<(unsigned int) num_blocks, (unsigned int) block_size, (unsigned int) smem_size, streamForSkin>>>(f);
#else
				launch_closure_by_value_0<<<(unsigned int) num_blocks, (unsigned int) block_size, (unsigned int) smem_size>>>(f);
#endif
				thrust::system::cuda::detail::synchronize_if_enabled("launch_closure_by_value");
			}
#endif // THRUST_DEVICE_COMPILER_NVCC
		}

		template<typename Closure, typename Size1, typename Size2, typename Size3>
		void launch_closure_0(Closure f, Size1 num_blocks, Size2 block_size, Size3 smem_size)
		{
			launch_0(f, num_blocks, block_size, smem_size);
		} // end launch_closure()

		template<typename DerivedPolicy,
			typename InputIterator,
			typename OutputType,
			typename BinaryFunction>
			OutputType reduce_1(thrust::execution_policy<DerivedPolicy> &exec,
			InputIterator first,
			InputIterator last,
			OutputType init,
			BinaryFunction binary_op)
		{
			// we're attempting to launch a kernel, assert we're compiling with nvcc
			// ========================================================================
			// X Note to the user: If you've found this line due to a compiler error, X
			// X you need to compile your code using nvcc, rather than g++ or cl.exe  X
			// ========================================================================
			THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<InputIterator, THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC>::value) );

			typedef typename thrust::iterator_difference<InputIterator>::type difference_type;

			difference_type n = thrust::distance(first,last);

			if (n == 0)
				return init;

			typedef thrust::detail::temporary_array<OutputType, DerivedPolicy> OutputArray;
			typedef typename OutputArray::iterator OutputIterator;

			typedef thrust::system::cuda::detail::detail::blocked_thread_array Context;
			
			typedef thrust::system::cuda::detail::reduce_detail::unordered_reduce_closure<InputIterator,difference_type,OutputType,OutputIterator,BinaryFunction,Context> Closure;

			thrust::system::cuda::detail::function_attributes_t attributes = thrust::system::cuda::detail::detail::closure_attributes<Closure>();

			// TODO chose this in a more principled manner
			size_t threshold = thrust::max<size_t>(2 * attributes.maxThreadsPerBlock, 1024);

			thrust::system::cuda::detail::device_properties_t properties = thrust::system::cuda::detail::device_properties();

			// launch configuration
			size_t num_blocks; 
			size_t block_size; 
			size_t array_size; 
			size_t smem_bytes; 

			// first level reduction
			if (static_cast<size_t>(n) < threshold)
			{
				num_blocks = 1;
				block_size = thrust::min(static_cast<size_t>(n), static_cast<size_t>(attributes.maxThreadsPerBlock));
				array_size = thrust::min(block_size, (properties.sharedMemPerBlock - attributes.sharedSizeBytes) / sizeof(OutputType));
				smem_bytes = sizeof(OutputType) * array_size;
			}
			else
			{
				thrust::system::cuda::detail::detail::launch_calculator<Closure> calculator;

				thrust::tuple<size_t,size_t,size_t> config = calculator.with_variable_block_size_available_smem();

				num_blocks = thrust::min(thrust::get<0>(config), static_cast<size_t>(n) / thrust::get<1>(config));
				block_size = thrust::get<1>(config);
				array_size = thrust::min(block_size, thrust::get<2>(config) / sizeof(OutputType));
				smem_bytes = sizeof(OutputType) * array_size;
			}

			// TODO assert(n <= num_blocks * block_size);
			// TODO if (shared_array_size < 1) throw cuda exception "insufficient shared memory"

			OutputArray output(exec, num_blocks);

			Closure closure(first, n, init, output.begin(), binary_op, array_size);

			//std::cout << "Launching " << num_blocks << " blocks of kernel with " << block_size << " threads and " << smem_bytes << " shared memory per block " << std::endl;

			launch_closure_0(closure, num_blocks, block_size, smem_bytes);

			// second level reduction
			if (num_blocks > 1)
			{
				typedef thrust::system::cuda::detail::detail::blocked_thread_array Context;
				typedef thrust::system::cuda::detail::reduce_detail::unordered_reduce_closure<OutputIterator,difference_type,OutputType,OutputIterator,BinaryFunction,Context> Closure;

				thrust::system::cuda::detail::function_attributes_t attributes = thrust::system::cuda::detail::detail::closure_attributes<Closure>();

				num_blocks = 1;
				block_size = thrust::min(output.size(), static_cast<size_t>(attributes.maxThreadsPerBlock));
				array_size = thrust::min(block_size, (properties.sharedMemPerBlock - attributes.sharedSizeBytes) / sizeof(OutputType));
				smem_bytes = sizeof(OutputType) * array_size;

				// TODO if (shared_array_size < 1) throw cuda exception "insufficient shared memory"

				Closure closure(output.begin(), output.size(), init, output.begin(), binary_op, array_size);

				//std::cout << "Launching " << num_blocks << " blocks of kernel with " << block_size << " threads and " << smem_bytes << " shared memory per block " << std::endl;

				launch_closure_0(closure, num_blocks, block_size, smem_bytes);
			}

			return output[0];
		} // end reduce

		template<typename DerivedPolicy,
			typename InputIterator,
			typename OutputType,
			typename BinaryFunction>
			OutputType reduce_2(thrust::execution_policy<DerivedPolicy> &exec,
			InputIterator first,
			InputIterator last,
			OutputType init,
			BinaryFunction binary_op)
		{
			return reduce_1(exec, first, last, init, binary_op);
		} // end reduce()

		template<typename DerivedPolicy,
			typename InputIterator,
			typename T,
			typename BinaryFunction>
			T reduce_0(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
			InputIterator first,
			InputIterator last,
			T init,
			BinaryFunction binary_op)
		{
			return reduce_2(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, init, binary_op);
		} // end reduce()

		template<typename DerivedPolicy,
			typename InputIterator, 
			typename UnaryFunction, 
			typename OutputType,
			typename BinaryFunction>
			OutputType transform_reduce_1(thrust::execution_policy<DerivedPolicy> &exec,
			InputIterator first,
			InputIterator last,
			UnaryFunction unary_op,
			OutputType init,
			BinaryFunction binary_op)
		{
			thrust::transform_iterator<UnaryFunction, InputIterator, OutputType> xfrm_first(first, unary_op);
			thrust::transform_iterator<UnaryFunction, InputIterator, OutputType> xfrm_last(last, unary_op);

			return reduce_0(exec, xfrm_first, xfrm_last, init, binary_op);
		} // end transform_reduce()

		template<typename InputIterator, 
			typename UnaryFunction, 
			typename OutputType,
			typename BinaryFunction>
			OutputType transform_reduce_0(InputIterator first,
			InputIterator last,
			UnaryFunction unary_op,
			OutputType init,
			BinaryFunction binary_op)
		{
			using thrust::system::detail::generic::select_system;

			typedef typename thrust::iterator_system<InputIterator>::type System;

			System system;

			return transform_reduce_1(select_system(system), first, last, unary_op, init, binary_op);
		} // end transform_reduce()

		template <typename InputIterator>
		typename thrust::iterator_value<InputIterator>::type nrm2(InputIterator first,	InputIterator last)
		{
			typedef typename thrust::iterator_value<InputIterator>::type ValueType;

			cusp::blas::detail::norm_squared<ValueType> unary_op;
			thrust::plus<ValueType>   binary_op;

			ValueType init = 0;

			return std::sqrt( abs(transform_reduce_0(first, last, unary_op, init, binary_op)) );
		}

		template <typename Array>
		typename Array::value_type nrm2(const Array& x)
		{
			
			CUSP_PROFILE_SCOPED();
			return nrm2(x.begin(), x.end());
		}

		template <typename Real>
		class MyMonitor
		{
		public:
			/*! Construct a \p default_monitor for a given right-hand-side \p b
			*
			*  The \p default_monitor terminates iteration when the residual norm
			*  satisfies the condition
			*       ||b - A x|| <= absolute_tolerance + relative_tolerance * ||b||
			*  or when the iteration limit is reached.
			*
			*  \param b right-hand-side of the linear system A x = b
			*  \param iteration_limit maximum number of solver iterations to allow
			*  \param relative_tolerance determines convergence criteria
			*  \param absolute_tolerance determines convergence criteria
			*
			*  \tparam VectorType vector
			*/
			//template <typename Vector>
			MyMonitor(const Real& bNorm, size_t iteration_limit = 500, Real relative_tolerance = 1e-5, Real absolute_tolerance = 0)
				: b_norm(bNorm),
				r_norm(std::numeric_limits<Real>::max()),
				iteration_limit_(iteration_limit),
				iteration_count_(0),
				relative_tolerance_(relative_tolerance),
				absolute_tolerance_(absolute_tolerance)
			{}

			/*! increment the iteration count
			*/
			void operator++(void) {  ++iteration_count_; } // prefix increment

			/*! applies convergence criteria to determine whether iteration is finished
			*
			*  \param r residual vector of the linear system (r = b - A x)
			*  \tparam Vector vector
			*/
			//template <typename Vector>
			bool finished(const Real& r)
			{
				r_norm = r;//cusp::blas::nrm2(r);

				if (iteration_count() >= iteration_limit())
				{
					printf("[#####]iteration_count() >= iteration_limit();\n");
					return true;
				}
				return converged() || iteration_count() >= iteration_limit();
			}

			/*! whether the last tested residual satifies the convergence tolerance
			*/
			bool converged() const
			{
				return residual_norm() <= tolerance();
			}

			/*! Euclidean norm of last residual
			*/
			Real residual_norm() const { return r_norm; }

			/*! number of iterations
			*/
			size_t iteration_count() const { return iteration_count_; }

			/*! maximum number of iterations
			*/
			size_t iteration_limit() const { return iteration_limit_; }

			/*! relative tolerance
			*/
			Real relative_tolerance() const { return relative_tolerance_; }

			/*! absolute tolerance
			*/
			Real absolute_tolerance() const { return absolute_tolerance_; }

			/*! tolerance
			*
			*  Equal to absolute_tolerance() + relative_tolerance() * ||b||
			*
			*/ 
			Real tolerance() const { return absolute_tolerance() + relative_tolerance() * b_norm; }

		protected:

			Real r_norm;
			Real b_norm;
			Real relative_tolerance_;
			Real absolute_tolerance_;

			size_t iteration_limit_;
			size_t iteration_count_;
		};

		void initMyKrylovContext()
		{
			PhysicsContext& currentCtx = FEM_State_Ctx;
			const int & nDofs = currentCtx.g_nDofs;

			Definition_Device_Buffer_With_Zero(currentCtx.cg_ptr_y,ValueType,_nExternalMemory * nDofs);
			thrust::device_ptr<ValueType> wrapped_y(currentCtx.cg_ptr_y);
			currentCtx.cg_y = MyCuspVecView(wrapped_y,wrapped_y+_nExternalMemory * nDofs);
			currentCtx.cg_y.resize(nDofs);

			Definition_Device_Buffer_With_Zero(currentCtx.cg_ptr_z,ValueType,_nExternalMemory * nDofs);
			thrust::device_ptr<ValueType> wrapped_z(currentCtx.cg_ptr_z);
			currentCtx.cg_z = MyCuspVecView(wrapped_z,wrapped_z+_nExternalMemory * nDofs);
			currentCtx.cg_z.resize(nDofs);

			Definition_Device_Buffer_With_Zero(currentCtx.cg_ptr_r,ValueType,_nExternalMemory * nDofs);
			thrust::device_ptr<ValueType> wrapped_r(currentCtx.cg_ptr_r);
			currentCtx.cg_r = MyCuspVecView(wrapped_r,wrapped_r+_nExternalMemory * nDofs);
			currentCtx.cg_r.resize(nDofs);

			Definition_Device_Buffer_With_Zero(currentCtx.cg_ptr_p,ValueType,_nExternalMemory * nDofs);
			thrust::device_ptr<ValueType> wrapped_p(currentCtx.cg_ptr_p);
			currentCtx.cg_p = MyCuspVecView(wrapped_p,wrapped_p+_nExternalMemory * nDofs);
			currentCtx.cg_p.resize(nDofs);
		}

		__global__ void cuda_axpby(const int nDofs,	ValueTypePtr x, ValueTypePtr y, ValueTypePtr output, ValueType alpha, ValueType beta)
		{
			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			output[tid] = (alpha * x[tid]) + (beta * y[tid]);
		}



		template <typename Array1,
			typename Array2,
			typename Array3,
			typename ScalarType1,
			typename ScalarType2>
			void axpby(const Array1& x,
			const Array2& y,
			const Array3& output,
			ScalarType1 alpha,
			ScalarType2 beta)
		{
#if USE_CUDA_STREAM
			cuda_axpby<<<GRIDCOUNT(output.size(),MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK ,0,streamForSkin>>>
#else
			cuda_axpby<<<GRIDCOUNT(currentCtx.nForceCount,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK>>>
#endif
				(output.size(),thrust::raw_pointer_cast(&x[0]),thrust::raw_pointer_cast(&y[0]),thrust::raw_pointer_cast(&output[0]),alpha,beta);
		}

		__global__ void cuda_axpy_axpy(const int nDofs,	ValueTypePtr x1, ValueTypePtr y1,ValueType alpha1, ValueTypePtr x2, ValueTypePtr y2,ValueType alpha2)
		{
			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			y1[tid] += alpha1 * x1[tid]; 
			y2[tid] += alpha2 * x2[tid];
		}

		template <typename Array1,
			typename Array2,
			typename Array3,
			typename Array4,
			typename ScalarType1,
			typename ScalarType2>
			void axpy_axpy(const Array1& x1, Array2& y1, ScalarType1 alpha1,const Array3& x2,	Array4& y2, ScalarType2 alpha2)
		{
			// y = y + alpha * x
			//blas::axpy(x, y, alpha);
#if USE_CUDA_STREAM
			cuda_axpy_axpy<<<GRIDCOUNT(x1.size(),MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK ,0,streamForSkin>>>
#else
			cuda_axpy_axpy<<<GRIDCOUNT(currentCtx.nForceCount,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK>>>
#endif
				(x1.size(),
				 thrust::raw_pointer_cast(&x1[0]),thrust::raw_pointer_cast(&y1[0]),alpha1,
				 thrust::raw_pointer_cast(&x2[0]),thrust::raw_pointer_cast(&y2[0]),alpha2);
		}
		

		void copyAsync(ValueTypePtr src, ValueTypePtr dst, const int nSize)
		{
			cudaMemcpyAsync(dst,src,nSize*sizeof(ValueType),cudaMemcpyDeviceToDevice,streamForSkin);
		}

		template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputType, typename BinaryFunction1, typename BinaryFunction2>
		OutputType inner_product_4(thrust::execution_policy<DerivedPolicy> &exec,
			InputIterator1 first1,
			InputIterator1 last1,
			InputIterator2 first2,
			OutputType init, 
			BinaryFunction1 binary_op1,
			BinaryFunction2 binary_op2)
		{
			typedef thrust::zip_iterator<thrust::tuple<InputIterator1,InputIterator2> > ZipIter;

			ZipIter first = thrust::make_zip_iterator(thrust::make_tuple(first1,first2));

			// only the first iterator in the tuple is relevant for the purposes of last
			ZipIter last  = thrust::make_zip_iterator(thrust::make_tuple(last1, first2));

			return transform_reduce_1(exec, first, last, thrust::detail::zipped_binary_op<OutputType,BinaryFunction2>(binary_op2), init, binary_op1);
		} // end inner_product()

		template<typename DerivedPolicy,
			typename InputIterator1,
			typename InputIterator2,
			typename OutputType,
			typename BinaryFunction1,
			typename BinaryFunction2>
			OutputType inner_product_3(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
			InputIterator1 first1,
			InputIterator1 last1,
			InputIterator2 first2,
			OutputType init, 
			BinaryFunction1 binary_op1,
			BinaryFunction2 binary_op2)
		{
			using thrust::system::detail::generic::inner_product;
			return inner_product_4(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, init, binary_op1, binary_op2);
		} // end inner_product()

		template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputType>
		OutputType inner_product_2(thrust::execution_policy<DerivedPolicy> &exec,
			InputIterator1 first1,
			InputIterator1 last1,
			InputIterator2 first2,
			OutputType init)
		{
			thrust::plus<OutputType>       binary_op1;
			thrust::multiplies<OutputType> binary_op2;
			return inner_product_3(exec, first1, last1, first2, init, binary_op1, binary_op2);
		} // end inner_product()

		template<typename DerivedPolicy,
			typename InputIterator1,
			typename InputIterator2,
			typename OutputType>
			OutputType inner_product_1(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
			InputIterator1 first1,
			InputIterator1 last1,
			InputIterator2 first2,
			OutputType init)
		{
			using thrust::system::detail::generic::inner_product;
			return inner_product_2(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, init);
		} // end inner_product()

		template <typename InputIterator1, typename InputIterator2, typename OutputType>
		OutputType 
			inner_product_0(InputIterator1 first1, InputIterator1 last1,
			InputIterator2 first2, OutputType init)
		{
			using thrust::system::detail::generic::select_system;

			typedef typename thrust::iterator_system<InputIterator1>::type System1;
			typedef typename thrust::iterator_system<InputIterator2>::type System2;

			System1 system1;
			System2 system2;

			return inner_product_1(select_system(system1,system2), first1, last1, first2, init);
		} // end inner_product()

		template <typename InputIterator1,
			typename InputIterator2>
			typename thrust::iterator_value<InputIterator1>::type
			dotc_0(InputIterator1 first1,
			InputIterator1 last1,
			InputIterator2 first2)
		{
			typedef typename thrust::iterator_value<InputIterator1>::type OutputType;
			return inner_product_0(thrust::make_transform_iterator(first1, cusp::blas::detail::conjugate<OutputType>()),
										thrust::make_transform_iterator(last1,  cusp::blas::detail::conjugate<OutputType>()),
										first2,
										OutputType(0));
		}

		template <typename Array1,
			typename Array2>
			typename Array1::value_type
			mydotc(const Array1& x,
			const Array2& y)
		{
			CUSP_PROFILE_SCOPED();
			cusp::blas::detail::assert_same_dimensions(x, y);
			return dotc_0(x.begin(), x.end(), y.begin());
		}

		template <class LinearOperator,
		class Vector,
		class Monitor,
		class Preconditioner>
			void cg(LinearOperator& A,
			Vector& x,
			Vector& b,
			Monitor& monitor,
			Preconditioner& M)
		{
			CUSP_PROFILE_SCOPED();

			typedef typename LinearOperator::value_type   ValueType;
			typedef typename LinearOperator::memory_space MemorySpace;

			assert(A.num_rows == A.num_cols);        // sanity check

			const size_t N = A.num_rows;

			// allocate workspace
			/*cusp::array1d<ValueType,MemorySpace> y(N);
			cusp::array1d<ValueType,MemorySpace> z(N);
			cusp::array1d<ValueType,MemorySpace> r(N);
			cusp::array1d<ValueType,MemorySpace> p(N);*/

			PhysicsContext& currentCtx = FEM_State_Ctx;
			MyCuspVecView& y = currentCtx.cg_y; 
			MyCuspVecView& z = currentCtx.cg_z;
			MyCuspVecView& r = currentCtx.cg_r;
			MyCuspVecView& p = currentCtx.cg_p;
			y.resize(N);
			z.resize(N);
			r.resize(N);
			p.resize(N);

			

			// y <- Ax
			cusp::multiply(A, x, y);

			// r <- b - A*x
			axpby(b, y, r, ValueType(1), ValueType(-1));

			// z <- M*r
			cusp::multiply(M, r, z);

			// p <- z
			copyAsync(thrust::raw_pointer_cast(&z[0]),thrust::raw_pointer_cast(&p[0]),N);
			//blas::copy(z, p);

			// rz = <r^H, z>
			ValueType rz = mydotc(r, z);

			ValueType r_nrm2 = nrm2(r);

			while (!monitor.finished(r_nrm2))
			{
				// y <- Ap
				cusp::multiply(A, p, y);

				// alpha <- <r,z>/<y,p>
				ValueType alpha =  rz / mydotc(y, p);

				// x <- x + alpha * p
				//blas::axpy(p, x, alpha);

				// r <- r - alpha * y		
				//blas::axpy(y, r, -alpha);

				axpy_axpy(p,x,alpha,y,r,-alpha);

				// z <- M*r
				cusp::multiply(M, r, z);

				ValueType rz_old = rz;

				// rz = <r^H, z>
				rz = mydotc(r, z);

				// beta <- <r_{i+1},r_{i+1}>/<r,r> 
				ValueType beta = rz / rz_old;

				// p <- r + beta*p
				axpby(z, p, p, ValueType(1), beta);

				++monitor;
				
				r_nrm2 = nrm2(r);
			}
		} 
	}

	void solve_cusp_cg_inner()
	{
		MY_RANGE("solve_cusp_cg_inner",2);
		PhysicsContext& currentCtx = FEM_State_Ctx;
#if 1
		{

			static float r_nrm2;
			r_nrm2 = MyKrylov::nrm2(currentCtx.cusp_Array_R_rhs);
			//cusp::blas::nrm2(currentCtx.cusp_Array_R_rhs);
			/*LogInfo("MyKrylov::nrm2(%f)  cusp::blas::nrm2(%f)\n",MyKrylov::nrm2(currentCtx.cusp_Array_R_rhs),cusp::blas::nrm2(currentCtx.cusp_Array_R_rhs));
			MyPause;*/
			MyKrylov::MyMonitor<float> monitor(r_nrm2/*currentCtx.cusp_Array_R_rhs*/, 500, 0.001f);

			//cusp::verbose_monitor<float> monitor(currentCtx.cusp_Array_R_rhs, 500, 0.005f);

			//cusp::precond::diagonal<ValueType, MemorySpace> M(currentCtx.SystemMatrix);
			MyKrylov::my_diagonal<ValueType, MemorySpace> M(currentCtx.SystemMatrix);
			
			MyKrylov::cg(currentCtx.SystemMatrix, currentCtx.cusp_Array_Incremental_displace, currentCtx.cusp_Array_R_rhs, monitor,M);
			
			/*thrust::device_ptr<ValueType> wrapped_displacement(currentCtx.displacementOnCuda);
			cusp::array1d_view<thrust::device_ptr<ValueType>> cusp_dX (wrapped_displacement, wrapped_displacement + currentCtx.g_nDofs);*/			
		}
#else
		cudaMemset(currentCtx.displacementOnCuda,'\0',currentCtx.g_nDofs*sizeof(float));
#endif
	}

	dim3 threads4RK(24,24);
	dim3 threads4RKR(24,25);
	dim3 grid_30000_24(KERNEL_COUNT_TMP,24);

	void computeRotationMatrix_SpeedUp()
	{
#define SpeedUpFactor (24)
		MY_RANGE("computeRotationMatrix_SpeedUp",5);
		using namespace CUDA_COROTATION;
		PhysicsContext& currentCtx = FEM_State_Ctx;
#if USE_CUDA_STREAM
		#define MY_STREAM_FLAGS ,0,streamForCalc
#else
		#define MY_STREAM_FLAGS
#endif
		/*1.5%  [77,1024]*/update_u_v_a_init_corotaion_SpeedUp<<<GRIDCOUNT(currentCtx.nCellOnCudaCount*CONST_8X9,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK MY_STREAM_FLAGS>>>(currentCtx.nCellOnCudaCount,currentCtx.CellOnCudaPtr,currentCtx.g_nDofs,currentCtx.displacementOnCuda,Signs_on_cuda,currentCtx.g_VertexOnCudaPtr,currentCtx.g_nVertexOnCudaCount);
		
		
		/*14.8%  [2,1024]*/update_u_v_a_iterator_corotaion_SpeedUp<<<GRIDCOUNT(currentCtx.nCellOnCudaCount,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK MY_STREAM_FLAGS>>>
			(currentCtx.nCellOnCudaCount,currentCtx.CellOnCudaPtr,(int(*)[4])g_corotation_inverseIdx,g_corotation_inverseFlag,g_corotation_transposeIdx);
		
		/*18.1% [609,1024]*/update_u_v_a_corotaion_compute_RK_SpeedUp<<<GRIDCOUNT(currentCtx.nCellOnCudaCount*CONST_24x24 / SpeedUpFactor,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK MY_STREAM_FLAGS>>>(currentCtx.nCellOnCudaCount,currentCtx.nCellOnCudaCount/SpeedUpFactor,currentCtx.CellOnCudaPtr,currentCtx.localStiffnessMatrixOnCuda);
		
		

		/*0.6%  [26,1024]*/update_u_v_a_corotaion_compute_RtPj_SpeedUp<<<GRIDCOUNT(currentCtx.nCellOnCudaCount*CONST_24,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK MY_STREAM_FLAGS>>>(currentCtx.nCellOnCudaCount,currentCtx.CellOnCudaPtr);
		

		/*16.6% [609,1024]*/update_u_v_a_corotaion_compute_RKR_RPj_SpeedUp_1<<<GRIDCOUNT(currentCtx.nCellOnCudaCount*CONST_24x24 / SpeedUpFactor,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK MY_STREAM_FLAGS>>>(currentCtx.nCellOnCudaCount,currentCtx.nCellOnCudaCount/SpeedUpFactor,currentCtx.CellOnCudaPtr);
		
		/*7.2%  [1081,576]*/update_u_v_a_corotaion_compute_RKR_RPj_SpeedUp_4<<<currentCtx.nCellOnCudaCount,24*24 MY_STREAM_FLAGS>>>(currentCtx.nCellOnCudaCount,currentCtx.CellOnCudaPtr);
		
	}

	__global__ void cuda_Update_u_v_a(const int nDofs, ValueTypePtr solu, ValueTypePtr disp_vec, ValueTypePtr vel_vec,
		ValueTypePtr acc_vec, ValueTypePtr old_acc, ValueTypePtr old_solu, ValueTypePtr NewMarkConstant)
	{
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid < nDofs)
		{
			old_solu[tid] = disp_vec[tid];
			disp_vec[tid] = solu[tid];
			old_acc[tid] = acc_vec[tid];


			/*acc_vec *= (-1 * m_db_NewMarkConstant[3]);*/
			acc_vec[tid] *= (-1 * NewMarkConstant[3]);

			/*acc_vec += (disp_vec * m_db_NewMarkConstant[0]); */
			acc_vec[tid] += disp_vec[tid] * NewMarkConstant[0];

			/*acc_vec += (old_solu * (-1 * m_db_NewMarkConstant[0]));*/
			acc_vec[tid] += (old_solu[tid] * (-1 * NewMarkConstant[0]));

			/*acc_vec += (vel_vec * (-1 * m_db_NewMarkConstant[2]));*/
			acc_vec[tid] += (vel_vec[tid] * (-1 * NewMarkConstant[2]));

			/*vel_vec += (old_acc * m_db_NewMarkConstant[6]);*/
			vel_vec[tid] += (old_acc[tid] * NewMarkConstant[6]);

			/*vel_vec += (acc_vec * m_db_NewMarkConstant[7]);*/
			vel_vec[tid] += (acc_vec[tid] * NewMarkConstant[7]);
		}
	}

	void update_u_v_a(const int nCount)
	{
		MY_RANGE("update_u_v_a",3);
		PhysicsContext& currentCtx = FEM_State_Ctx;

#if 1

#if USE_CUDA_STREAM
		cuda_Update_u_v_a<<< GRIDCOUNT(currentCtx.g_nDofs,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK ,0,streamForSkin>>>
#else
		cuda_Update_u_v_a<<< GRIDCOUNT(currentCtx.g_nDofs,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK >>>
#endif
		(
			currentCtx.g_nDofs, currentCtx.displacementOnCuda, currentCtx.myOptimize_Array_Displacement,
			currentCtx.myOptimize_Array_Velocity, currentCtx.myOptimize_Array_Acceleration,
			currentCtx.myOptimize_Old_Acceleration, currentCtx.myOptimize_Old_Displacement, element_Array_NewMarkConstant);
#else
		const MyCuspVecView& solu = currentCtx.cusp_Array_Incremental_displace;
		MyCuspVecView& disp_vec = currentCtx.cusp_Array_Displacement;
		MyCuspVecView& vel_vec = currentCtx.cusp_Array_Velocity;
		MyCuspVecView& acc_vec = currentCtx.cusp_Array_Acceleration;
		MyCuspVecView& old_acc = currentCtx.cusp_Array_Old_Acceleration;
		MyCuspVecView& old_solu = currentCtx.cusp_Array_Old_Displacement;

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

#if USE_CO_RATION

#if DEBUG_TIME
		nLastTick = GetTickCount()/*,nnCount = 0*/;
#endif


		//computeRotationMatrix(nDomainId);
#if DEBUG_TIME
		nCurrentTick = GetTickCount();
		printf("computeRotationMatrix %d  \n",nCurrentTick - nLastTick);
#endif
#endif
#endif
		computeRotationMatrix_SpeedUp();
	}

#define SKINNING_CUTTING (1)
#if SKINNING_CUTTING
	namespace CUDA_SKNNING_CUTTING
	{
		VBOStructForDraw g_VBO_Struct_Node;

		void initVBOStructContext()
		{
			g_VBO_Struct_Node.vbo_line_vertex_pair = MyNull;
			g_VBO_Struct_Node.vbo_lineCount = MyZero;

			g_VBO_Struct_Node.g_nMCVertexSize=0;
			g_VBO_Struct_Node.g_nMaxMCVertexSize=0;
			g_VBO_Struct_Node.g_nLastVertexSize=0;
			g_VBO_Struct_Node.g_nMCEdgeSize=0;
			g_VBO_Struct_Node.g_nMaxMCEdgeSize=0;
			g_VBO_Struct_Node.g_nLastEdgeSize=0;
			g_VBO_Struct_Node.g_nMCSurfaceSize=0;
			g_VBO_Struct_Node.g_nMaxMCSurfaceSize=0;
			g_VBO_Struct_Node.g_nLastSurfaceSize=0;
			g_VBO_Struct_Node.g_nVertexNormalSize = 0;
			g_VBO_Struct_Node.g_MC_Vertex_Cuda=MyNull;
			g_VBO_Struct_Node.g_MC_Edge_Cuda=MyNull;
			g_VBO_Struct_Node.g_MC_Surface_Cuda=MyNull;
			g_VBO_Struct_Node.g_elementVertexNormal = MyNull;//g_nVertexNormalSize * 3


			/*g_VBO_Struct_Node.g_CuttingRayOnCpuSize = MyZero;
			g_VBO_Struct_Node.g_Nouse = MyZero;
			g_VBO_Struct_Node.g_RefineCuttingRayOnCpuSizeDebug = MyZero;
			g_VBO_Struct_Node.g_g_CuttingRayStructOnCpu = MyNull;
			g_VBO_Struct_Node.g_g_CuttingRayStructOnCuda = MyNull;*/

			g_VBO_Struct_Node.g_ExternalForceRaySize = MyZero;
			g_VBO_Struct_Node.g_ExternalForceRayOnCpu = MyNull;
			g_VBO_Struct_Node.g_ExternalForceRayOnCuda = MyNull;
		}


		void initExternalForceRayOnCpuStruct()
		{
			g_VBO_Struct_Node.g_ExternalForceRaySize = MyZero;
			Definition_Host_Device_Buffer(g_VBO_Struct_Node.g_ExternalForceRayOnCpu,g_VBO_Struct_Node.g_ExternalForceRayOnCuda,ExternalForceRayStruct,MaxCuttingRayCount);
			/*HANDLE_ERROR( cudaHostAlloc( (void**)&g_VBO_Struct_Node.g_ExternalForceRayOnCpu, MaxCuttingRayCount * sizeof(ExternalForceRayStruct),cudaHostAllocMapped   )) ;
			HANDLE_ERROR(cudaHostGetDevicePointer((void **)&g_VBO_Struct_Node.g_ExternalForceRayOnCuda,(void *)g_VBO_Struct_Node.g_ExternalForceRayOnCpu,0));*/
		}

		void initVBODataStruct_LineSet(IndexTypePtr line_vertex_pair,IndexType lineCount)
		{
			cudaDeviceSynchronize();
			g_VBO_Struct_Node.vbo_lineCount = lineCount;

			printf("initVBODataStruct lineCount is %d  sizeof(%d)\n",lineCount,lineCount * VBO_LineSetSize * sizeof(IndexType));
			Definition_Device_Buffer_With_Data(g_VBO_Struct_Node.vbo_line_vertex_pair,IndexType,lineCount * VBO_LineSetSize,1,line_vertex_pair);
			/*HANDLE_ERROR( cudaMalloc( (void**)&g_VBO_Struct_Node.vbo_line_vertex_pair,lineCount * VBO_LineSetSize * sizeof(IndexType))) ;
			HANDLE_ERROR( cudaMemcpy( (void *)g_VBO_Struct_Node.vbo_line_vertex_pair,line_vertex_pair,lineCount * VBO_LineSetSize * sizeof(IndexType),cudaMemcpyHostToDevice ) );*/
		}

		void initMeshCuttingStructure(const int nVertexSize,MC_Vertex_Cuda* MCVertexCuda,
			const int nEdgeSize,  MC_Edge_Cuda*	 MCEdgeCuda,
			const int nSurfaceSize,MC_Surface_Cuda*	MCSurfaceCuda,
			const int nVertexNormal,float * elementVertexNormal)
		{
			cudaDeviceSynchronize();
			g_VBO_Struct_Node.g_nMCVertexSize = nVertexSize;
			g_VBO_Struct_Node.g_nLastVertexSize = nVertexSize;
			g_VBO_Struct_Node.g_nMaxMCVertexSize = _nExternalMemory*nVertexSize;
			g_VBO_Struct_Node.g_nMCEdgeSize = nEdgeSize;
			g_VBO_Struct_Node.g_nLastEdgeSize = nEdgeSize;
			g_VBO_Struct_Node.g_nMaxMCEdgeSize = _nExternalMemory*nEdgeSize;
			g_VBO_Struct_Node.g_nMCSurfaceSize = nSurfaceSize;
			g_VBO_Struct_Node.g_nLastSurfaceSize = nSurfaceSize;
			g_VBO_Struct_Node.g_nMaxMCSurfaceSize = _nExternalMemory*nSurfaceSize;
			g_VBO_Struct_Node.g_nVertexNormalSize = nVertexNormal;

			printf("g_nMCVertexSize(%d) g_nMaxMCVertexSize(%d) g_nMCEdgeSize(%d) g_nMaxMCEdgeSize(%d) g_nMCSurfaceSize(%d) g_nMaxMCSurfaceSize(%d)\n",
				g_VBO_Struct_Node.g_nMCVertexSize,g_VBO_Struct_Node.g_nMaxMCVertexSize,g_VBO_Struct_Node.g_nMCEdgeSize,g_VBO_Struct_Node.g_nMaxMCEdgeSize,g_VBO_Struct_Node.g_nMCSurfaceSize,g_VBO_Struct_Node.g_nMaxMCSurfaceSize);
			//system("pause");

			HANDLE_ERROR( cudaMalloc( (void**)&g_VBO_Struct_Node.g_MC_Vertex_Cuda, g_VBO_Struct_Node.g_nMaxMCVertexSize * sizeof(MC_Vertex_Cuda))) ;
			HANDLE_ERROR( cudaMemset( (void*)g_VBO_Struct_Node.g_MC_Vertex_Cuda,0, g_VBO_Struct_Node.g_nMaxMCVertexSize * sizeof(MC_Vertex_Cuda))) ;
			HANDLE_ERROR( cudaMemcpy( (void*)g_VBO_Struct_Node.g_MC_Vertex_Cuda, MCVertexCuda, g_VBO_Struct_Node.g_nMCVertexSize * sizeof(MC_Vertex_Cuda), cudaMemcpyHostToDevice));

			HANDLE_ERROR( cudaMalloc( (void**)&g_VBO_Struct_Node.g_MC_Edge_Cuda, g_VBO_Struct_Node.g_nMaxMCEdgeSize * sizeof(MC_Edge_Cuda))) ;
			HANDLE_ERROR( cudaMemset( (void*)g_VBO_Struct_Node.g_MC_Edge_Cuda,0, g_VBO_Struct_Node.g_nMaxMCEdgeSize * sizeof(MC_Edge_Cuda))) ;
			HANDLE_ERROR( cudaMemcpy( (void*)g_VBO_Struct_Node.g_MC_Edge_Cuda, MCEdgeCuda, g_VBO_Struct_Node.g_nMCEdgeSize * sizeof(MC_Edge_Cuda), cudaMemcpyHostToDevice));

			HANDLE_ERROR( cudaMalloc( (void**)&g_VBO_Struct_Node.g_MC_Surface_Cuda, g_VBO_Struct_Node.g_nMaxMCSurfaceSize * sizeof(MC_Surface_Cuda))) ;
			HANDLE_ERROR( cudaMemset( (void*)g_VBO_Struct_Node.g_MC_Surface_Cuda,0, g_VBO_Struct_Node.g_nMaxMCSurfaceSize * sizeof(MC_Surface_Cuda))) ;
			HANDLE_ERROR( cudaMemcpy( (void*)g_VBO_Struct_Node.g_MC_Surface_Cuda, MCSurfaceCuda, g_VBO_Struct_Node.g_nMCSurfaceSize * sizeof(MC_Surface_Cuda), cudaMemcpyHostToDevice));

			HANDLE_ERROR( cudaMalloc( (void**)&g_VBO_Struct_Node.g_elementVertexNormal, g_VBO_Struct_Node.g_nMaxMCVertexSize*3 * sizeof(float))) ;
			HANDLE_ERROR( cudaMemset( (void*)g_VBO_Struct_Node.g_elementVertexNormal,0, g_VBO_Struct_Node.g_nMaxMCVertexSize*3 * sizeof(float))) ;
			HANDLE_ERROR( cudaMemcpy( (void*)g_VBO_Struct_Node.g_elementVertexNormal, elementVertexNormal, g_VBO_Struct_Node.g_nVertexNormalSize * 3 * sizeof(float), cudaMemcpyHostToDevice));

			HANDLE_ERROR( cudaHostAlloc( (void**)&g_VBO_Struct_Node.g_CuttedEdgeFlagOnCpu, g_VBO_Struct_Node.g_nMaxMCEdgeSize * sizeof(IndexType),cudaHostAllocMapped   )) ;
			HANDLE_ERROR(cudaHostGetDevicePointer((void **)&g_VBO_Struct_Node.g_CuttedEdgeFlagOnCuda,(void *)g_VBO_Struct_Node.g_CuttedEdgeFlagOnCpu,0));
			HANDLE_ERROR( cudaMemset( (void*)g_VBO_Struct_Node.g_CuttedEdgeFlagOnCuda,	0,g_VBO_Struct_Node.g_nMaxMCEdgeSize * sizeof(IndexType))) ;

			HANDLE_ERROR( cudaHostAlloc( (void**)&g_VBO_Struct_Node.g_SplittedFaceFlagOnCpu, g_VBO_Struct_Node.g_nMaxMCSurfaceSize * sizeof(IndexType),cudaHostAllocMapped   )) ;
			HANDLE_ERROR(cudaHostGetDevicePointer((void **)&g_VBO_Struct_Node.g_SplittedFaceFlagOnCuda,(void *)g_VBO_Struct_Node.g_SplittedFaceFlagOnCpu,0));
			HANDLE_ERROR( cudaMemset( (void*)g_VBO_Struct_Node.g_SplittedFaceFlagOnCuda,	0,g_VBO_Struct_Node.g_nMaxMCSurfaceSize * sizeof(IndexType))) ;

			g_VBO_Struct_Node.g_nCuttingBladeCount = MyZero;
			Definition_Host_Device_Buffer(g_VBO_Struct_Node.g_CuttingBladeStructOnCpu,g_VBO_Struct_Node.g_CuttingBladeStructOnCuda,CuttingBladeStruct,MaxCuttingBladeStructCount);
#if 0
			HANDLE_ERROR( cudaHostAlloc( (void**)&g_CuttedEdgeFlagOnCpu, g_nMaxMCEdgeSize * sizeof(IndexType),cudaHostAllocMapped   )) ;
			HANDLE_ERROR( cudaHostAlloc( (void**)&g_SplittedFaceFlagOnCpu, g_nMaxMCSurfaceSize * sizeof(IndexType),cudaHostAllocMapped   )) ;

			HANDLE_ERROR( cudaHostAlloc( (void**)&g_CuttingFaceUp_X, g_nMaxMCVertexSize * sizeof(float),cudaHostAllocMapped   )) ;
			HANDLE_ERROR( cudaHostAlloc( (void**)&g_CuttingFaceUp_Y, g_nMaxMCVertexSize * sizeof(float),cudaHostAllocMapped   )) ;
			HANDLE_ERROR( cudaHostAlloc( (void**)&g_CuttingFaceUp_Z, g_nMaxMCVertexSize * sizeof(float),cudaHostAllocMapped   )) ;

			HANDLE_ERROR( cudaHostAlloc( (void**)&g_CuttingFaceDown_X, g_nMaxMCVertexSize * sizeof(float),cudaHostAllocMapped   )) ;
			HANDLE_ERROR( cudaHostAlloc( (void**)&g_CuttingFaceDown_Y, g_nMaxMCVertexSize * sizeof(float),cudaHostAllocMapped   )) ;
			HANDLE_ERROR( cudaHostAlloc( (void**)&g_CuttingFaceDown_Z, g_nMaxMCVertexSize * sizeof(float),cudaHostAllocMapped   )) ;

			HANDLE_ERROR( cudaHostAlloc( (void**)&g_CuttingFaceUpFlagCpu, g_nMaxMCVertexSize * sizeof(int),cudaHostAllocMapped   )) ;
			HANDLE_ERROR( cudaHostAlloc( (void**)&g_CuttingFaceDownFlagCpu, g_nMaxMCVertexSize * sizeof(int),cudaHostAllocMapped   )) ;
#endif
		}
		bool cuttingBladeCheckSurfaceTriangle()
		{
			return false;
		}

#if USE_OUTPUT_RENDER_OBJ_MESH
		void getObjMeshInfoFromCUDA(int & nVertexCount,MC_Vertex_Cuda** curVertexSet,
			int& nTriSize,MC_Surface_Cuda** curFaceSet,
			int& nCellCount,CommonCellOnCuda** CellOnCudaPtr,
			float** displacement)
		{
			nVertexCount = g_VBO_Struct_Node.g_nMCVertexSize;
			nTriSize = g_VBO_Struct_Node.g_nMCSurfaceSize;
			nCellCount = FEM_State_Ctx.nCellOnCudaCount;

			(*curVertexSet) = new MC_Vertex_Cuda[nVertexCount];
			(*curFaceSet) = new MC_Surface_Cuda[nTriSize];
			(*CellOnCudaPtr) = new CommonCellOnCuda[nCellCount];
			(*displacement) = new float[FEM_State_Ctx.g_nDofs];
			memset((*curVertexSet),'\0',sizeof(MC_Vertex_Cuda)*nVertexCount);
			memset((*curFaceSet),'\0',sizeof(MC_Surface_Cuda)*nTriSize);
			memset((*CellOnCudaPtr),'\0',sizeof(CommonCellOnCuda)*nCellCount);
			memset((*displacement),'\0',sizeof(float)*FEM_State_Ctx.g_nDofs);

			HANDLE_ERROR( cudaMemcpy( (void *)(*curVertexSet), g_VBO_Struct_Node.g_MC_Vertex_Cuda,	nVertexCount * sizeof(MC_Vertex_Cuda),	cudaMemcpyDeviceToHost	));
			HANDLE_ERROR( cudaMemcpy( (void *)(*curFaceSet), g_VBO_Struct_Node.g_MC_Surface_Cuda,	nTriSize * sizeof(MC_Surface_Cuda),	cudaMemcpyDeviceToHost	));
			HANDLE_ERROR( cudaMemcpy( (void *)(*CellOnCudaPtr), FEM_State_Ctx.CellOnCudaPtr,	nCellCount * sizeof(CommonCellOnCuda),	cudaMemcpyDeviceToHost	));
			HANDLE_ERROR( cudaMemcpy( (void *)(*displacement), FEM_State_Ctx.displacementOnCuda,	FEM_State_Ctx.g_nDofs * sizeof(float),	cudaMemcpyDeviceToHost	));
		}

		void freeObjMeshInfoFromCUDA(int & nVertexCount,MC_Vertex_Cuda** curVertexSet,
			int& nTriSize,MC_Surface_Cuda** curFaceSet,
			int& nCellCount,CommonCellOnCuda** CellOnCudaPtr,
			float** displacement)
		{
			delete [] (*curVertexSet);
			(*curVertexSet) = MyNull;

			delete [] (*curFaceSet);
			(*curFaceSet) = MyNull;

			delete [] (*CellOnCudaPtr);
			(*CellOnCudaPtr) = MyNull;

			delete [] (*displacement);
			(*displacement) = MyNull;
		}
#endif

		//__global__ void cudaMoveMesh_TriangleMesh4MeshCuttingWithoutDisplacement(
		__global__ void cudaMoveMesh_TriangleMesh4MeshCuttingWithoutDisplacement(
			float3* pos,float3 * vn,
			const int nVertexSize,const int nLastVertexSize,MC_Vertex_Cuda* curVertexSet,
			const int nLineSize,const int nLastLineSize,MC_Edge_Cuda* curLineSet,
			const int nTriSize,const int nLastTriSize,MC_Surface_Cuda* curFaceSet)

		{
			int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid < nTriSize)
			{
				const int faceIdx = tid * 3;
				MC_Surface_Cuda& curface = curFaceSet[tid];


				if (!curface.m_isValid )
				{
					for (int i = 0; i < 3; ++i)
					{
						pos[faceIdx + i] = make_float3(1000.f, 1000.f, 1000.f);
						vn[faceIdx + i].x = 1.f;
						vn[faceIdx + i].y = 1.f;
						vn[faceIdx + i].z = 1.f;
					}
				}
				else
				{
					for (int i = 0; i < 3; ++i)//pt0,pt1,pt2
					{
						MC_Vertex_Cuda& curVertex = curVertexSet[curface.m_Vertex[i]];

						pos[faceIdx + i].x = curVertex.m_VertexPos[0] ;
						pos[faceIdx + i].y = curVertex.m_VertexPos[1] ;
						pos[faceIdx + i].z = curVertex.m_VertexPos[2] ;
					}

					vn[faceIdx] = normalize(myCross(pos[faceIdx+1] - pos[faceIdx+0],pos[faceIdx+2] - pos[faceIdx+1]));
					vn[faceIdx+1] = vn[faceIdx];
					vn[faceIdx+2] = vn[faceIdx];
				}
			}
		}

		int cuda_SkinningCutting_GetTriangle(float3** cpuVertexPtr, float3** cpuNormalsPtr)
		{
			PhysicsContext& currentCtx = FEM_State_Ctx;

			cudaMoveMesh_TriangleMesh4MeshCuttingWithoutDisplacement<<<GRIDCOUNT(g_VBO_Struct_Node.g_nMCSurfaceSize,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK>>>
				(*cpuVertexPtr, *cpuNormalsPtr,
				g_VBO_Struct_Node.g_nMCVertexSize,g_VBO_Struct_Node.g_nLastVertexSize,g_VBO_Struct_Node.g_MC_Vertex_Cuda,
				g_VBO_Struct_Node.g_nMCEdgeSize,g_VBO_Struct_Node.g_nLastEdgeSize,g_VBO_Struct_Node.g_MC_Edge_Cuda,
				g_VBO_Struct_Node.g_nMCSurfaceSize,g_VBO_Struct_Node.g_nLastSurfaceSize,g_VBO_Struct_Node.g_MC_Surface_Cuda);
			return g_VBO_Struct_Node.g_nMCSurfaceSize;
		}

		void cuda_SkinningCutting_SetBladeList(float3** ptrVertex, float3 * cpuVertex, int2 ** ptrIndex, int2 * cpuIndex, const int nLineSize)
		{
			cudaMemcpy((*ptrVertex),cpuVertex,2*nLineSize*sizeof(float3),cudaMemcpyHostToDevice);
			cudaMemcpy((*ptrIndex) ,cpuIndex ,nLineSize*sizeof(int2),cudaMemcpyHostToDevice);
		}

		void cuda_SkinningCutting_SetBladeTriangleList(float3** ptrVertex, float3 * cpuVertex, float3 ** ptrNormal, float3 * cpuNormal, const int nTriSize)
		{
			cudaMemcpy((*ptrVertex),cpuVertex,3*nTriSize*sizeof(float3),cudaMemcpyHostToDevice);
			cudaMemcpy((*ptrNormal),cpuNormal,3*nTriSize*sizeof(float3),cudaMemcpyHostToDevice);
		}
	}//namespace CUDA_SKNNING_CUTTING
#endif//#if SKINNING_CUTTING

	



	__global__ void cudaMoveMesh_TriangleMesh4MeshCutting4MultiDomain(float3* pos,float3 * vn,int3* index_Triangles,
			const int nVertexSize,const int nLastVertexSize,MC_Vertex_Cuda* curVertexSet,
			const int nLineSize,const int nLastLineSize,MC_Edge_Cuda* curLineSet,
			const int nTriSize,const int nLastTriSize,MC_Surface_Cuda* curFaceSet,
			const int nCellCount,CommonCellOnCuda* CellOnCudaPtr,
			const int nVertexNormalSize,float* vecVN,
			ValueTypePtr displacement/*,ValueTypePtr rhs,const int nLastTriSize*/)

		{
			int tid = threadIdx.x + blockIdx.x * blockDim.x;
			while (tid < nTriSize)
			{
				const int faceIdx = tid * 3;
				MC_Surface_Cuda& curface = curFaceSet[tid];


				if (!curface.m_isValid /*&& tid != 30000 && tid != 30003*/)
				{
					for (int i = 0; i < 3; ++i)
					{
						pos[faceIdx + i] = make_float3(1000.f, 1000.f, 1000.f);
						vn[faceIdx + i].x = vecVN[curface.m_VertexNormal[i] * 3 + 0];
						vn[faceIdx + i].y = vecVN[curface.m_VertexNormal[i] * 3 + 1];
						vn[faceIdx + i].z = vecVN[curface.m_VertexNormal[i] * 3 + 2];
					}
				}
				else
				{
					float p[2][2][2];
					float curDisplace[3];
					bool bNouse = false;
					for (int i = 0; i < 3; ++i)//pt0,pt1,pt2
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
						pos[faceIdx + i].x = curVertex.m_VertexPos[0] + curDisplace[0];
						pos[faceIdx + i].y = curVertex.m_VertexPos[1] + curDisplace[1];
						pos[faceIdx + i].z = curVertex.m_VertexPos[2] + curDisplace[2];
					}

					vn[faceIdx] = normalize(myCross(pos[faceIdx+1] - pos[faceIdx+0],pos[faceIdx+2] - pos[faceIdx+1]));
					vn[faceIdx+1] = vn[faceIdx];
					vn[faceIdx+2] = vn[faceIdx];
#if USE_DYNAMIC_VERTEX_NORMAL
					curface.m_vertexNormalValue[0] = vn[faceIdx+2].x;
					curface.m_vertexNormalValue[1] = vn[faceIdx+2].y;
					curface.m_vertexNormalValue[2] = vn[faceIdx+2].z;
#endif
					//index_Triangles[tid] = make_int3(tid,tid+1,tid+2);
				}

				tid += blockDim.x * gridDim.x;
			}
		}

	__global__ void cudaMoveMesh_TriangleMesh4MeshCutting4MultiDomain_PerVertex(float3* pos,float3 * vn,int3* index_Triangles,
		const int nVertexSize,const int nLastVertexSize,MC_Vertex_Cuda* curVertexSet,
		const int nLineSize,const int nLastLineSize,MC_Edge_Cuda* curLineSet,
		const int nTriSize,const int nLastTriSize,MC_Surface_Cuda* curFaceSet,
		const int nCellCount,CommonCellOnCuda* CellOnCudaPtr,
		const int nVertexNormalSize,float* vecVN,
		ValueTypePtr displacement/*,ValueTypePtr rhs,const int nLastTriSize*/)

	{
		const int nThreadTotalCount = threadIdx.x + blockIdx.x * blockDim.x;
		const int tid = nThreadTotalCount / 9;
		
		//if (tid < nTriSize)
		{
			const int nTmp = MyMod(nThreadTotalCount,9);
			const int nVtxIdx = nTmp/3;//MyMod(nThreadTotalCount,3);
			const int step = MyMod(nTmp,3);
			const int faceIdx = tid * 3;
			MC_Surface_Cuda& curface = curFaceSet[tid];


			if (!curface.m_isValid /*&& tid != 30000 && tid != 30003*/)
			{
				//for (int i = 0; i < 3; ++i)
				{
					*(((float*)&pos[faceIdx + nVtxIdx])+step) = 1000.f;
					*(((float*)&vn[faceIdx + nVtxIdx])+step) = vecVN[curface.m_VertexNormal[nVtxIdx] * 3 + step];
					//pos[faceIdx + nVtxIdx] = make_float3(1000.f, 1000.f, 1000.f);
					/*vn[faceIdx + nVtxIdx].x = vecVN[curface.m_VertexNormal[nVtxIdx] * 3 + 0];
					vn[faceIdx + nVtxIdx].y = vecVN[curface.m_VertexNormal[nVtxIdx] * 3 + 1];
					vn[faceIdx + nVtxIdx].z = vecVN[curface.m_VertexNormal[nVtxIdx] * 3 + 2];*/
				}
			}
			else
			{
				float p[2][2][2];
				float curDisplace[3];
				bool bNouse = false;

				//vn[faceIdx+nVtxIdx] = normalize(myCross(pos[faceIdx+1] - pos[faceIdx+0],pos[faceIdx+2] - pos[faceIdx+1]));
				//for (int i = 0; i < 3; ++i)//pt0,pt1,pt2
				{
					MC_Vertex_Cuda& curVertex = curVertexSet[curface.m_Vertex[nVtxIdx]];
					//for (int step = 0; step < 3; ++step)//x,y,z
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
					*(((float*)&pos[faceIdx + nVtxIdx])+step) = curVertex.m_VertexPos[step] + curDisplace[step];
					/*pos[faceIdx + nVtxIdx].x = curVertex.m_VertexPos[0] + curDisplace[0];
					pos[faceIdx + nVtxIdx].y = curVertex.m_VertexPos[1] + curDisplace[1];
					pos[faceIdx + nVtxIdx].z = curVertex.m_VertexPos[2] + curDisplace[2];*/
				}

				
				/*vn[faceIdx+1] = vn[faceIdx];
				vn[faceIdx+2] = vn[faceIdx];*/
#if USE_DYNAMIC_VERTEX_NORMAL
				curface.m_vertexNormalValue[0] = vn[faceIdx+2].x;
				curface.m_vertexNormalValue[1] = vn[faceIdx+2].y;
				curface.m_vertexNormalValue[2] = vn[faceIdx+2].z;
#endif
				//index_Triangles[tid] = make_int3(tid,tid+1,tid+2);
			}
		}
	}

#if USE_DYNAMIC_VERTEX_NORMAL
	__global__ void cudaMoveMesh_ComputeVertexNormal/*<<<(KERNEL_COUNT_VERTEX + BLOCK_COUNT_512 - 1) / BLOCK_COUNT_512,BLOCK_COUNT_512>>>*/
		(const int nNativeVertexSize,
		const int nVertexSize,const int nLastVertexSize,MC_Vertex_Cuda* curVertexSet,
		const int nLineSize,const int nLastLineSize,MC_Edge_Cuda* curLineSet,
		const int nTriSize,const int nLastTriSize,MC_Surface_Cuda* curFaceSet)
	{
		const int nVtxId = threadIdx.x + blockIdx.x * blockDim.x;

		if (nVtxId < nNativeVertexSize)
		{
			float * vertexNormal = &curVertexSet[nVtxId].m_VertexNormalValue[0];
			int  * shareTriangleId = &curVertexSet[nVtxId].m_eleShareTriangle[0];
			const int nShareTriangleCount = curVertexSet[nVtxId].m_nShareTriangleCount;
			if (nShareTriangleCount > 0)
			{
				vertexNormal[0] = vertexNormal[1] = vertexNormal[2] = 0.f;
				for (int i=0;i<curVertexSet[nVtxId].m_nShareTriangleCount;++i)
				{
					MC_Surface_Cuda& curFace = curFaceSet[shareTriangleId[i]];
					if (curFace.m_isValid)
					{
						vertexNormal[0] += curFace.m_vertexNormalValue[0];
						vertexNormal[1] += curFace.m_vertexNormalValue[1];
						vertexNormal[2] += curFace.m_vertexNormalValue[2];
					}
				}

				vertexNormal[0] /= (float)nShareTriangleCount;
				vertexNormal[1] /= (float)nShareTriangleCount;
				vertexNormal[2] /= (float)nShareTriangleCount;

				float normm = sqrtf(vertexNormal[0]*vertexNormal[0]+vertexNormal[1]*vertexNormal[1]+vertexNormal[2]*vertexNormal[2]);
				if (normm > 0)
				{
					vertexNormal[0] /= normm;
					vertexNormal[1] /= normm;
					vertexNormal[2] /= normm;
				}
			}
		}
		else if (nVtxId < nVertexSize)
		{
			curVertexSet[nVtxId].m_nShareTriangleCount = 0;
		}
	}

	__global__ void cudaMoveMesh_SettingVertexNormal(float3* pos,float3 * vn,
		const int nVertexSize,const int nLastVertexSize,MC_Vertex_Cuda* curVertexSet,
		const int nLineSize,const int nLastLineSize,MC_Edge_Cuda* curLineSet,
		const int nTriSize,const int nLastTriSize,MC_Surface_Cuda* curFaceSet)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < nTriSize)
		{
			const int faceIdx = tid * 3;
			MC_Surface_Cuda& curface = curFaceSet[tid];
			if (!curface.m_isValid /*&& tid != 30000 && tid != 30003*/)
			{
				for (int i = 0; i < 3; ++i)
				{
					vn[faceIdx + i].x = 0.f;
					vn[faceIdx + i].y = 0.f;
					vn[faceIdx + i].z = 0.f;
				}
			}
			else
			{
				for (int i = 0; i < 3; ++i)
				{
					MC_Vertex_Cuda& curVertex = curVertexSet[curface.m_Vertex[i]];
					if (0 == curVertex.m_nShareTriangleCount)
					{
						vn[faceIdx+i] = make_float3(curface.m_vertexNormalValue[0],curface.m_vertexNormalValue[1],curface.m_vertexNormalValue[2]);
					} 
					else
					{
						vn[faceIdx+i] = make_float3(curVertex.m_VertexNormalValue[0],curVertex.m_VertexNormalValue[1],curVertex.m_VertexNormalValue[2]);
					}
				}

			}
		}
	}
#endif
	void moveMeshForTriangle(float3** pos_Triangles, float3** vertexNormal, int3** index_Triangles)
	{
		MY_RANGE("moveMeshForTriangle",4);
		using namespace CUDA_SKNNING_CUTTING;
		PhysicsContext& currentCtx = FEM_State_Ctx;
#if USE_CUDA_STREAM
		cudaMoveMesh_TriangleMesh4MeshCutting4MultiDomain_PerVertex<<<GRIDCOUNT(g_VBO_Struct_Node.g_nMCSurfaceSize*9,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK,0,streamForSkin>>>
#else
		cudaMoveMesh_TriangleMesh4MeshCutting4MultiDomain_PerVertex<<<GRIDCOUNT(g_VBO_Struct_Node.g_nMCSurfaceSize*9,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK>>>
#endif
			(*pos_Triangles, *vertexNormal,*index_Triangles,
			g_VBO_Struct_Node.g_nMCVertexSize,g_VBO_Struct_Node.g_nLastVertexSize,g_VBO_Struct_Node.g_MC_Vertex_Cuda,
			g_VBO_Struct_Node.g_nMCEdgeSize,g_VBO_Struct_Node.g_nLastEdgeSize,g_VBO_Struct_Node.g_MC_Edge_Cuda,
			g_VBO_Struct_Node.g_nMCSurfaceSize,g_VBO_Struct_Node.g_nLastSurfaceSize,g_VBO_Struct_Node.g_MC_Surface_Cuda,
			currentCtx.nCellOnCudaCount MyNotice,currentCtx.CellOnCudaPtr MyNotice,
			g_VBO_Struct_Node.g_nVertexNormalSize,g_VBO_Struct_Node.g_elementVertexNormal,
			currentCtx.displacementOnCuda MyNotice);

#if USE_DYNAMIC_VERTEX_NORMAL
		cudaMoveMesh_ComputeVertexNormal<<<GRIDCOUNT(Max_Vertex_Count,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK>>>
			(g_nNativeSurfaceVertexCount,
			g_VBO_Struct_Node.g_nMCVertexSize,g_VBO_Struct_Node.g_nLastVertexSize,g_VBO_Struct_Node.g_MC_Vertex_Cuda,
			g_VBO_Struct_Node.g_nMCEdgeSize,g_VBO_Struct_Node.g_nLastEdgeSize,g_VBO_Struct_Node.g_MC_Edge_Cuda,
			g_VBO_Struct_Node.g_nMCSurfaceSize,g_VBO_Struct_Node.g_nLastSurfaceSize,g_VBO_Struct_Node.g_MC_Surface_Cuda);

		cudaMoveMesh_SettingVertexNormal<<<GRIDCOUNT(Max_Triangle_Count,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK>>>
			(*pos_Triangles, *vertexNormal,
			g_VBO_Struct_Node.g_nMCVertexSize,g_VBO_Struct_Node.g_nLastVertexSize,g_VBO_Struct_Node.g_MC_Vertex_Cuda,
			g_VBO_Struct_Node.g_nMCEdgeSize,g_VBO_Struct_Node.g_nLastEdgeSize,g_VBO_Struct_Node.g_MC_Edge_Cuda,
			g_VBO_Struct_Node.g_nMCSurfaceSize,g_VBO_Struct_Node.g_nLastSurfaceSize,g_VBO_Struct_Node.g_MC_Surface_Cuda);
#endif
		//KERNEL_COUNT_VERTEX
	}

	__global__ void cudaMoveMesh_LineSet_PerCell(float3* pos, int2* index_Lines,const int nCellCount,CommonCellOnCuda* CellOnCudaPtr,
		VertexOnCuda* _VertexOnCudaPtr,float * displaceOnCuda,IndexTypePtr linePairOnCuda)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < nCellCount)
		{
			const int lineBase = tid * 24;
			CommonCellOnCuda& currentCellRef = CellOnCudaPtr[tid];

			/*if (currentCellRef.m_bTopLevelOctreeNodeList)
			{
				for (int i=0;i<12;++i)
				{
					const int lineIdx = lineBase + 2*i;
					const int localStartVtxId = linePairOnCuda[i*2];
					const int localEndVtxId = linePairOnCuda[i*2+1];
					const int BeginVertexId = currentCellRef.vertexId[localStartVtxId];
					const int EndVertexId = currentCellRef.vertexId[localEndVtxId];

					int* beginVtxDofs = &_VertexOnCudaPtr[BeginVertexId].m_nGlobalDof[0];
					float* beginVtxRestPos = &_VertexOnCudaPtr[BeginVertexId].local[0];
					int* endVtxDofs = &_VertexOnCudaPtr[EndVertexId].m_nGlobalDof[0];
					float* endVtxRestPos = &_VertexOnCudaPtr[EndVertexId].local[0];

					pos[ lineIdx ] = make_float3(0.f,0.f,0.f);	
					pos[lineIdx+1 ] = make_float3(0.f,0.f,0.f);	

					index_Lines[tid*12+i] = make_int2(lineIdx,lineIdx+1);
				}
			}
			else*/
			if (currentCellRef.m_bLeaf)
			{
				for (int i=0;i<12;++i)
				{
					const int lineIdx = lineBase + 2*i;
					//linePairOnCuda
					const int localStartVtxId = linePairOnCuda[i*2];
					const int localEndVtxId = linePairOnCuda[i*2+1];
					const int BeginVertexId = currentCellRef.vertexId[localStartVtxId];
					const int EndVertexId = currentCellRef.vertexId[localEndVtxId];

					int* beginVtxDofs = &_VertexOnCudaPtr[BeginVertexId].m_nGlobalDof[0];
					float* beginVtxRestPos = &_VertexOnCudaPtr[BeginVertexId].local[0];
					int* endVtxDofs = &_VertexOnCudaPtr[EndVertexId].m_nGlobalDof[0];
					float* endVtxRestPos = &_VertexOnCudaPtr[EndVertexId].local[0];

					pos[ lineIdx ] = make_float3(beginVtxRestPos[0] + displaceOnCuda[beginVtxDofs[0]]/*-0.5f*/,
						beginVtxRestPos[1] + displaceOnCuda[beginVtxDofs[1]]/*-0.f*/,
						beginVtxRestPos[2] + displaceOnCuda[beginVtxDofs[2]]/*-0.5f*/);	
					pos[lineIdx+1 ] = make_float3(endVtxRestPos[0] + displaceOnCuda[endVtxDofs[0]]/*-0.5f*/,
						endVtxRestPos[1] + displaceOnCuda[endVtxDofs[1]]/*-0.f*/,
						endVtxRestPos[2] + displaceOnCuda[endVtxDofs[2]]/*-0.5f*/);

					index_Lines[tid*12+i] = make_int2(lineIdx,lineIdx+1);
				}
			}
			else
			{
				for (int i=0;i<12;++i)
				{
					const int lineIdx = lineBase + 2*i;
					pos[ lineIdx ] = make_float3(0.f,0.f,0.f);	
					pos[lineIdx+1 ] = make_float3(0.f,0.f,0.f);	
					index_Lines[tid*12+i] = make_int2(lineIdx,lineIdx+1);
				}
			}
		}
	}

	void moveMeshForLineSet(float3** pos, int2 **index_Lines)
	{
		using namespace CUDA_SKNNING_CUTTING;
		cudaMoveMesh_LineSet_PerCell<<<GRIDCOUNT(MaxCellCount,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK>>>
			(*pos,*index_Lines,FEM_State_Ctx.nCellOnCudaCount,FEM_State_Ctx.CellOnCudaPtr,
			 FEM_State_Ctx.g_VertexOnCudaPtr,FEM_State_Ctx.displacementOnCuda,FEM_State_Ctx.g_linePairOnCuda);
		g_VBO_Struct_Node.vbo_lineCount = FEM_State_Ctx.nCellOnCudaCount * 12;
	}

	void do_loop_single(const int nTimeStep,float3** pos_Lines, int2 ** index_Lines, float3** pos_Triangles, float3** vertexNormal, int3 ** index_Triangles, unsigned int& nTrianleSize, unsigned int& nLineSize)
	{
#if DEBUG_TIME
		nLastTick = GetTickCount();
#endif
		using namespace CUDA_SKNNING_CUTTING;

		update_rhs(nTimeStep);	
#if DEBUG_TIME
		//nCurrentTick = GetTickCount();
		printf("update_rhs %d \n",GetTickCount() - nLastTick);
		nLastTick = GetTickCount();
#endif	
		apply_boundary_values();
#if DEBUG_TIME
		//nCurrentTick = GetTickCount();
		printf("apply_boundary_values %d \n",GetTickCount() - nLastTick);
		nLastTick = GetTickCount();
#endif		
#if 1
		solve_cusp_cg_inner();
#if DEBUG_TIME
		//nCurrentTick = GetTickCount();
		printf("solve_cusp_cg_inner %d \n",GetTickCount() - nLastTick);
		nLastTick = GetTickCount();
#endif
#else		
		FEM_State_Ctx.cusp_Array_Incremental_displace.resize(0,0.f);
		FEM_State_Ctx.cusp_Array_Incremental_displace.resize(FEM_State_Ctx.g_nDofs,0.f);
#endif
		//moveMeshForLineSet(pos_Lines,index_Lines);
		//moveMeshForTriangle(pos_Triangles,vertexNormal,index_Triangles);
#if DEBUG_TIME
		//nCurrentTick = GetTickCount();
		printf("moveMeshForTriangle %d \n",GetTickCount() - nLastTick);
		nLastTick = GetTickCount();
#endif
		
		
		
		update_u_v_a(nTimeStep);	
#if DEBUG_TIME
		//nCurrentTick = GetTickCount();
		printf("update_u_v_a %d \n",GetTickCount() - nLastTick);
		nLastTick = GetTickCount();
#endif
		

#if DEBUG_TIME
		//nCurrentTick = GetTickCount();
		printf("do_loop_single %d \n",GetTickCount() - nLastTick);
		nLastTick = GetTickCount();
#endif

#if SHOWTRIMESH
		moveMeshForTriangle(pos_Triangles,vertexNormal,index_Triangles);
		nTrianleSize = g_VBO_Struct_Node.g_nMCSurfaceSize;
#else
		nTrianleSize = 0;
#endif
		

#if SHOWGRID
		moveMeshForLineSet(pos_Lines,index_Lines);
		nLineSize = g_VBO_Struct_Node.vbo_lineCount;
#else
		nLineSize = 0;
#endif
		//assembleSystemOnCuda_FEM_RealTime();
		//if (0 == MyMod(nTimeStep,4))
		/*CUDA_CUTTING_GRID::nouseRHSSize = FEM_State_Ctx.g_nDofs;
		cudaMemcpy(&CUDA_CUTTING_GRID::nouseRHSVec[0],FEM_State_Ctx.myOptimize_BladeForce,sizeof(float)*FEM_State_Ctx.g_nDofs,cudaMemcpyDeviceToHost);*/
		if (CUDA_CUTTING_GRID::g_needCheckCutting)
		{		
			/*CUDA_CUTTING_GRID::nouseRHSSize = FEM_State_Ctx.g_nDofs;
			memset(&CUDA_CUTTING_GRID::nouseRHSVec[0],0,sizeof(float)*16083 * 2);
			memset(&CUDA_CUTTING_GRID::nouseCusp_Array_Rhs[0],0,sizeof(float)*16083 * 2);
			memset(&CUDA_CUTTING_GRID::nouseCusp_Array_Old_Acceleration[0],0,sizeof(float)*16083 * 2);
			memset(&CUDA_CUTTING_GRID::nouseCusp_Array_Old_Displacement[0],0,sizeof(float)*16083 * 2);
			memset(&CUDA_CUTTING_GRID::nouseCusp_Array_R_rhs_Corotaion[0],0,sizeof(float)*16083 * 2);
			cudaMemcpy(&CUDA_CUTTING_GRID::nouseCusp_Array_Old_Displacement[0],FEM_State_Ctx.myOptimize_Old_Acceleration,sizeof(float)*FEM_State_Ctx.g_nDofs,cudaMemcpyDeviceToHost);
			*/
			CUDA_CUTTING_GRID::g_needCheckCutting = false;
			CUDA_CUTTING_GRID::cuttingBladeCheckVolumnGrid(0);	
			
			/*cudaMemcpy(&CUDA_CUTTING_GRID::nouseRHSVec[0],FEM_State_Ctx.rhsOnCuda,sizeof(float)*FEM_State_Ctx.g_nDofs,cudaMemcpyDeviceToHost);
			cudaMemcpy(&CUDA_CUTTING_GRID::nouseCusp_Array_Rhs[0],FEM_State_Ctx.g_systemRhsPtr_MF,sizeof(float)*FEM_State_Ctx.g_nDofs,cudaMemcpyDeviceToHost);
			cudaMemcpy(&CUDA_CUTTING_GRID::nouseCusp_Array_Old_Acceleration[0],FEM_State_Ctx.myOptimize_Old_Acceleration,sizeof(float)*FEM_State_Ctx.g_nDofs,cudaMemcpyDeviceToHost);
			
			cudaMemcpy(&CUDA_CUTTING_GRID::nouseCusp_Array_R_rhs_Corotaion[0],FEM_State_Ctx.g_systemRhsPtr_MF_Rotation,sizeof(float)*FEM_State_Ctx.g_nDofs,cudaMemcpyDeviceToHost);
*/
		}
		//assembleSystemOnCuda_FEM_RealTime();
		assembleSystemOnCuda_FEM_RealTime_ForCorotation();
	}

	namespace CUDA_DEBUG
	{
		void cuda_OuputObjMesh4Video(float3 ** cpu_vertex, float3 ** cpu_normal,float3 ** cuda_vertex, float3 ** cuda_normal,const int nVtxSize,MC_Vertex_Cuda** tmp_Vertex, MC_Surface_Cuda ** tmp_triangle)
		{
			using namespace CUDA_SKNNING_CUTTING;
			cudaMemcpy((*cpu_vertex),(*cuda_vertex),nVtxSize * sizeof(float3),cudaMemcpyDeviceToHost);
			cudaMemcpy((*cpu_normal),(*cuda_normal),nVtxSize * sizeof(float3),cudaMemcpyDeviceToHost);
			cudaMemcpy((*tmp_Vertex),(g_VBO_Struct_Node.g_MC_Vertex_Cuda),g_VBO_Struct_Node.g_nMCVertexSize * sizeof(MC_Vertex_Cuda),cudaMemcpyDeviceToHost);
			cudaMemcpy((*tmp_triangle),(g_VBO_Struct_Node.g_MC_Surface_Cuda),g_VBO_Struct_Node.g_nMCSurfaceSize * sizeof(MC_Surface_Cuda),cudaMemcpyDeviceToHost);
		}
		void cuda_Debug_Get_MatrixData(int & nDofs,
			int ** systemInnerIndexPtr, float ** systemValuePtr,
			int ** stiffInnerIndexPtr, float ** stiffValuePtr,
			int ** massInnerIndexPtr, float ** massValuePtr,
			float ** rhsValuePtr)
		{
			PhysicsContext& currentCtx = FEM_State_Ctx;
			nDofs = currentCtx.g_nDofs;
			
			(*systemInnerIndexPtr) = new int[nDofs*nMaxNonZeroSizeInFEM];
			(*systemValuePtr) = new float[nDofs*nMaxNonZeroSizeInFEM];
			(*stiffInnerIndexPtr) = new int[nDofs*nMaxNonZeroSizeInFEM];
			(*stiffValuePtr) = new float[nDofs*nMaxNonZeroSizeInFEM];
			(*massInnerIndexPtr) = new int[nDofs*nMaxNonZeroSizeInFEM];
			(*massValuePtr) = new float[nDofs*nMaxNonZeroSizeInFEM];
			(*rhsValuePtr) = new float[nDofs];

			cudaMemcpy((*systemInnerIndexPtr),currentCtx.g_globalDof_System_MF,nDofs*nMaxNonZeroSizeInFEM*sizeof(int),cudaMemcpyDeviceToHost);
			cudaMemcpy((*systemValuePtr),currentCtx.g_globalValue_System_MF,nDofs*nMaxNonZeroSizeInFEM*sizeof(float),cudaMemcpyDeviceToHost);

			cudaMemcpy((*stiffInnerIndexPtr),currentCtx.g_globalDof_MF,nDofs*nMaxNonZeroSizeInFEM*sizeof(int),cudaMemcpyDeviceToHost);
			cudaMemcpy((*stiffValuePtr),currentCtx.g_globalValue_MF,nDofs*nMaxNonZeroSizeInFEM*sizeof(float),cudaMemcpyDeviceToHost);

			cudaMemcpy((*massInnerIndexPtr),currentCtx.g_globalDof_Mass_MF,nDofs*nMaxNonZeroSizeInFEM*sizeof(int),cudaMemcpyDeviceToHost);
			cudaMemcpy((*massValuePtr),currentCtx.g_globalValue_Mass_MF,nDofs*nMaxNonZeroSizeInFEM*sizeof(float),cudaMemcpyDeviceToHost);

			float * tmpRhsValuePtr = *rhsValuePtr;
			cusp::array1d_view<float*> tmp_rhs (tmpRhsValuePtr, tmpRhsValuePtr + currentCtx.g_nDofs);
			CuspVec::view rhsView(currentCtx.cusp_Array_R_rhs.begin(),currentCtx.cusp_Array_R_rhs.end());
			cusp::copy(rhsView, tmp_rhs);
			//cudaMemcpy((*rhsValuePtr),currentCtx.g_globalValue_System_MF,nDofs*nMaxNonZeroSizeInFEM*sizeof(float),cudaMemcpyDeviceToHost);
		}

		void cuda_Debug_free_MatrixData(int ** systemInnerIndexPtr, float ** systemValuePtr,
										int ** stiffInnerIndexPtr, float ** stiffValuePtr,
										int ** massInnerIndexPtr, float ** massValuePtr,
										float ** rhsValuePtr)
		{
			delete [] *systemInnerIndexPtr;
			delete [] *systemValuePtr;
			delete [] *stiffInnerIndexPtr;
			delete [] *stiffValuePtr;
			delete [] *massInnerIndexPtr;
			delete [] *massValuePtr;
			delete [] *rhsValuePtr;
		}

		void func()
		{
			printf("call function begin.\n");
			//thrust::device_vector<float> tmp3(50000);

			CuspVec tmp1(50000);
			CuspVec tmp2(tmp1);
			tmp2.resize(10,0);

			dim3 dimGrid(2, 2);
			dim3 dimBlock(2, 2, 2);
			testKernel<<<dimGrid, dimBlock>>>(10);
			cudaDeviceSynchronize();
			//MyPause;
			printf("call function end.\n");
			MyPause;
		}

		int    g_cuda_faces_count = MyZero;
		int3 * g_cuda_faces = MyNull;
		int3 * g_cpu_faces = MyNull;

		int    g_cuda_vertexes_count = MyZero;
		float3 * g_cuda_vertexes = MyNull;
		float3 * g_cpu_vertexes = MyNull;

		int    g_cuda_normals_count = MyZero;
		float3 * g_cuda_normals = MyNull;
		float3 * g_cpu_normals = MyNull;


		int tmp_cuda_initSkinData(const int vertexSize, const int normalSize, const int faceSize)
		{

			g_cuda_vertexes_count = vertexSize;
			g_cuda_normals_count = normalSize;
			g_cuda_faces_count = faceSize;


			Definition_Host_Device_Buffer(g_cpu_vertexes,g_cuda_vertexes,float3,g_cuda_vertexes_count);
			Definition_Host_Device_Buffer(g_cpu_normals,g_cuda_normals,float3,g_cuda_normals_count);
			Definition_Host_Device_Buffer(g_cpu_faces,g_cuda_faces,int3,g_cuda_faces_count);

			return 0;
		}

		int tmp_cuda_freeSkinData()
		{
			cudaFreeHost(g_cpu_vertexes);
			cudaFreeHost(g_cpu_normals);
			cudaFreeHost(g_cpu_faces);
			return 0;
		}

		__global__ void cuda_SettingSkinData(const int nVtxSize, float3* targetBuffer_vertex, float3* gpuBuffer_vertex, 
			const int nNorSize, float3* targetBuffer_normal, float3* gpuBuffer_normal,
			const int nTriSize, int3* targetBuffer_faces, int3* gpuBuffer_faces)
		{
			const int tid = blockIdx.x * blockDim.x + threadIdx.x;	
			return;
			//const int vtxId = tid / MAX_KERNEL_PER_BLOCK;
			if (tid < nVtxSize)
			{
				targetBuffer_vertex[tid] = gpuBuffer_vertex[tid];
			}

			if (tid < nNorSize)
			{
				targetBuffer_normal[tid] = gpuBuffer_normal[tid];
			}

			if (tid < nTriSize)
			{
				targetBuffer_faces[tid] = gpuBuffer_faces[tid];
			}
		}

		int tmp_cuda_setSkinData(float3** cpuVertexPtr, float3** cpuNormalsPtr, int3** cpuFacesPtr)
		{
			int tmp = MAX(g_cuda_normals_count,g_cuda_vertexes_count);
			const int nSize = MAX(g_cuda_faces_count,tmp);

			thrust::device_vector<float> tmp2(50000);

			CuspVec tmp1(50000);

			CuspVec tmp3(tmp1);


			cuda_SettingSkinData<<<GRIDCOUNT(Max_Triangle_Count,MAX_KERNEL_PER_BLOCK),MAX_KERNEL_PER_BLOCK>>>(g_cuda_vertexes_count,*cpuVertexPtr,g_cuda_vertexes,
				g_cuda_normals_count,*cpuNormalsPtr,g_cuda_normals,
				g_cuda_faces_count,*cpuFacesPtr,g_cuda_faces);
			cudaDeviceSynchronize();
			//MyPause;
			return 0;
		}
	}

	
}//namesapce CUDA_SIMULATION
#endif//#if USE_CUDA_SIMULATION


