#include "VR_MACRO.h"
#include "VR_GPU_Geometry_OctreeLevelInfo.h"
#include <helper_math.h>
#include "VR_CUDA_GlobalDefine.cuh"
#include "cuda_runtime.h"//cudaMalloc

#if USE_CUDA
using namespace YC::Geometry::GPU;

//////////////////////////////////////////////////////////////   triangle overlap box  //////////////////////////////////////////////
#define X 0
#define Y 1
#define Z 2

#define CROSS(dest,v1,v2) \
	dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
	dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
	dest[2]=v1[0]*v2[1]-v1[1]*v2[0]; 

#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])

#define SUB(dest,v1,v2) \
	dest[0]=v1[0]-v2[0]; \
	dest[1]=v1[1]-v2[1]; \
	dest[2]=v1[2]-v2[2]; 

#define FINDMINMAX(x0,x1,x2,min,max) \
	min = max = x0;   \
	if(x1<min) min=x1;\
	if(x1>max) max=x1;\
	if(x2<min) min=x2;\
	if(x2>max) max=x2;

__device__ int planeBoxOverlap(float normal[3], float vert[3], float maxbox[3])	// -NJMP-
{
	//int q;
	float vmin[3],vmax[3],v;
	//for(q=X;q<=Z;q++)
	//{
	//	v=vert[q];					// -NJMP-
	//	if(normal[q]>0.0f)
	//	{
	//		vmin[q]=-maxbox[q] - v;	// -NJMP-
	//		vmax[q]= maxbox[q] - v;	// -NJMP-
	//	}
	//	else
	//	{
	//		vmin[q]= maxbox[q] - v;	// -NJMP-
	//		vmax[q]=-maxbox[q] - v;	// -NJMP-
	//	}
	//}

	{
		v=vert[X];					// -NJMP-
		if(normal[X]>0.0f)
		{
			vmin[X]=-maxbox[X] - v;	// -NJMP-
			vmax[X]= maxbox[X] - v;	// -NJMP-
		}
		else
		{
			vmin[X]= maxbox[X] - v;	// -NJMP-
			vmax[X]=-maxbox[X] - v;	// -NJMP-
		}
	}
	{
		v=vert[Y];					// -NJMP-
		if(normal[Y]>0.0f)
		{
			vmin[Y]=-maxbox[Y] - v;	// -NJMP-
			vmax[Y]= maxbox[Y] - v;	// -NJMP-
		}
		else
		{
			vmin[Y]= maxbox[Y] - v;	// -NJMP-
			vmax[Y]=-maxbox[Y] - v;	// -NJMP-
		}
	}
	{
		v=vert[Z];					// -NJMP-
		if(normal[Z]>0.0f)
		{
			vmin[Z]=-maxbox[Z] - v;	// -NJMP-
			vmax[Z]= maxbox[Z] - v;	// -NJMP-
		}
		else
		{
			vmin[Z]= maxbox[Z] - v;	// -NJMP-
			vmax[Z]=-maxbox[Z] - v;	// -NJMP-
		}
	}

	if(DOT(normal,vmin)>0.0f) return 0;	// -NJMP-
	if(DOT(normal,vmax)>=0.0f) return 1;	// -NJMP-

	return 0;
}


/*======================== X-tests ========================*/
#define AXISTEST_X01(a, b, fa, fb)			   \
	p0 = a*v0[Y] - b*v0[Z];			       	   \
	p2 = a*v2[Y] - b*v2[Z];			       	   \
	if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;} \
	rad = fa * boxhalfsize[Y] + fb * boxhalfsize[Z];   \
	if(min>rad || max<-rad) return 0;

#define AXISTEST_X2(a, b, fa, fb)			   \
	p0 = a*v0[Y] - b*v0[Z];			           \
	p1 = a*v1[Y] - b*v1[Z];			       	   \
	if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
	rad = fa * boxhalfsize[Y] + fb * boxhalfsize[Z];   \
	if(min>rad || max<-rad) return 0;

/*======================== Y-tests ========================*/
#define AXISTEST_Y02(a, b, fa, fb)			   \
	p0 = -a*v0[X] + b*v0[Z];		      	   \
	p2 = -a*v2[X] + b*v2[Z];	       	       	   \
	if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;} \
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Z];   \
	if(min>rad || max<-rad) return 0;

#define AXISTEST_Y1(a, b, fa, fb)			   \
	p0 = -a*v0[X] + b*v0[Z];		      	   \
	p1 = -a*v1[X] + b*v1[Z];	     	       	   \
	if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Z];   \
	if(min>rad || max<-rad) return 0;

/*======================== Z-tests ========================*/

#define AXISTEST_Z12(a, b, fa, fb)			   \
	p1 = a*v1[X] - b*v1[Y];			           \
	p2 = a*v2[X] - b*v2[Y];			       	   \
	if(p2<p1) {min=p2; max=p1;} else {min=p1; max=p2;} \
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Y];   \
	if(min>rad || max<-rad) return 0;

#define AXISTEST_Z0(a, b, fa, fb)			   \
	p0 = a*v0[X] - b*v0[Y];				   \
	p1 = a*v1[X] - b*v1[Y];			           \
	if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Y];   \
	if(min>rad || max<-rad) return 0;

__device__ int triBoxOverlap_2(float boxhalfsize[3],float v0[3],float v1[3],float v2[3],float e0[3],float e1[3],float e2[3])
{
	//float min,max,p0,p1,p2,rad,fex,fey,fez;		// -NJMP- "d" local variable removed
	float normal[3];

	CROSS(normal,e0,e1);
	// -NJMP- (line removed here)
	if(!planeBoxOverlap(normal,v0,boxhalfsize)) return 0;	// -NJMP-

	return 1;   /* box and triangle overlaps */
}
__device__ int triBoxOverlap_1(float boxhalfsize[3],float v0[3],float v1[3],float v2[3],float e0[3],float e1[3],float e2[3])
{
	/* Bullet 1: */
	/*  first test overlap in the {x,y,z}-directions */
	/*  find min, max of the triangle each direction, and test for overlap in */
	/*  that direction -- this is equivalent to testing a minimal AABB around */
	/*  the triangle against the AABB */
	float min,max,p0,p1,p2,rad,fex,fey,fez;		// -NJMP- "d" local variable removed
	//float normal[3];
	/* test in X-direction */
	FINDMINMAX(v0[X],v1[X],v2[X],min,max);
	if(min>boxhalfsize[X] || max<-boxhalfsize[X]) return 0;

	/* test in Y-direction */
	FINDMINMAX(v0[Y],v1[Y],v2[Y],min,max);
	if(min>boxhalfsize[Y] || max<-boxhalfsize[Y]) return 0;

	/* test in Z-direction */
	FINDMINMAX(v0[Z],v1[Z],v2[Z],min,max);
	if(min>boxhalfsize[Z] || max<-boxhalfsize[Z]) return 0;

	return 1;
}

__device__ int triBoxOverlap_0(float boxhalfsize[3],float v0[3],float v1[3],float v2[3],float e0[3],float e1[3],float e2[3])
{
	/*    use separating axis theorem to test overlap between triangle and box */
	/*    need to test for overlap in these directions: */
	/*    1) the {x,y,z}-directions (actually, since we use the AABB of the triangle */
	/*       we do not even need to test these) */
	/*    2) normal of the triangle */
	/*    3) crossproduct(edge from tri, {x,y,z}-directin) */
	/*       this gives 3x3=9 more tests */
//	float v0[3],v1[3],v2[3];
	//   float axis[3];
	float min,max,p0,p1,p2,rad,fex,fey,fez;		// -NJMP- "d" local variable removed
	//float normal[3];
//	float e0[3],e1[3],e2[3];

	

	/* Bullet 3:  */
	/*  test the 9 tests first (this was faster) */
	fex = fabsf(e0[X]);
	fey = fabsf(e0[Y]);
	fez = fabsf(e0[Z]);
	AXISTEST_X01(e0[Z], e0[Y], fez, fey);
	AXISTEST_Y02(e0[Z], e0[X], fez, fex);
	AXISTEST_Z12(e0[Y], e0[X], fey, fex);

	fex = fabsf(e1[X]);
	fey = fabsf(e1[Y]);
	fez = fabsf(e1[Z]);
	AXISTEST_X01(e1[Z], e1[Y], fez, fey);
	AXISTEST_Y02(e1[Z], e1[X], fez, fex);
	AXISTEST_Z0(e1[Y], e1[X], fey, fex);

	fex = fabsf(e2[X]);
	fey = fabsf(e2[Y]);
	fez = fabsf(e2[Z]);
	AXISTEST_X2(e2[Z], e2[Y], fez, fey);
	AXISTEST_Y1(e2[Z], e2[X], fez, fex);
	AXISTEST_Z12(e2[Y], e2[X], fey, fex);

	return 1;
}
__device__ int triBoxOverlap(float boxcenter[3],float boxhalfsize[3],float triverts[3][3])
{

	/*    use separating axis theorem to test overlap between triangle and box */
	/*    need to test for overlap in these directions: */
	/*    1) the {x,y,z}-directions (actually, since we use the AABB of the triangle */
	/*       we do not even need to test these) */
	/*    2) normal of the triangle */
	/*    3) crossproduct(edge from tri, {x,y,z}-directin) */
	/*       this gives 3x3=9 more tests */
	float v0[3],v1[3],v2[3];
	//   float axis[3];
	float min,max,p0,p1,p2,rad,fex,fey,fez;		// -NJMP- "d" local variable removed
	float normal[3],e0[3],e1[3],e2[3];

	/* This is the fastest branch on Sun */
	/* move everything so that the boxcenter is in (0,0,0) */
	SUB(v0,triverts[0],boxcenter);
	SUB(v1,triverts[1],boxcenter);
	SUB(v2,triverts[2],boxcenter);

	/* compute triangle edges */
	SUB(e0,v1,v0);      /* tri edge 0 */
	SUB(e1,v2,v1);      /* tri edge 1 */
	SUB(e2,v0,v2);      /* tri edge 2 */

	/* Bullet 3:  */
	/*  test the 9 tests first (this was faster) */
	fex = fabsf(e0[X]);
	fey = fabsf(e0[Y]);
	fez = fabsf(e0[Z]);
	AXISTEST_X01(e0[Z], e0[Y], fez, fey);
	AXISTEST_Y02(e0[Z], e0[X], fez, fex);
	AXISTEST_Z12(e0[Y], e0[X], fey, fex);

	fex = fabsf(e1[X]);
	fey = fabsf(e1[Y]);
	fez = fabsf(e1[Z]);
	AXISTEST_X01(e1[Z], e1[Y], fez, fey);
	AXISTEST_Y02(e1[Z], e1[X], fez, fex);
	AXISTEST_Z0(e1[Y], e1[X], fey, fex);

	fex = fabsf(e2[X]);
	fey = fabsf(e2[Y]);
	fez = fabsf(e2[Z]);
	AXISTEST_X2(e2[Z], e2[Y], fez, fey);
	AXISTEST_Y1(e2[Z], e2[X], fez, fex);
	AXISTEST_Z12(e2[Y], e2[X], fey, fex);

	/* Bullet 1: */
	/*  first test overlap in the {x,y,z}-directions */
	/*  find min, max of the triangle each direction, and test for overlap in */
	/*  that direction -- this is equivalent to testing a minimal AABB around */
	/*  the triangle against the AABB */

	/* test in X-direction */
	FINDMINMAX(v0[X],v1[X],v2[X],min,max);
	if(min>boxhalfsize[X] || max<-boxhalfsize[X]) return 0;

	/* test in Y-direction */
	FINDMINMAX(v0[Y],v1[Y],v2[Y],min,max);
	if(min>boxhalfsize[Y] || max<-boxhalfsize[Y]) return 0;

	/* test in Z-direction */
	FINDMINMAX(v0[Z],v1[Z],v2[Z],min,max);
	if(min>boxhalfsize[Z] || max<-boxhalfsize[Z]) return 0;

	/* Bullet 2: */
	/*  test if the box intersects the plane of the triangle */
	/*  compute plane equation of triangle: normal*x+d=0 */
	CROSS(normal,e0,e1);
	// -NJMP- (line removed here)
	if(!planeBoxOverlap(normal,v0,boxhalfsize)) return 0;	// -NJMP-

	return 1;   /* box and triangle overlaps */
}

__device__ float3 ClosestPtPointTriangle(float3 p, float3 a, float3 b, float3 c)
{
	// Check if P in vertex region outside A
	float3 ab = b - a;
	float3 ac = c - a;
	float3 ap = p - a;
	float d1 = dot(ab, ap);
	float d2 = dot(ac, ap);
	if (d1 <= 0.0f && d2 <= 0.0f) return a; // barycentric coordinates (1,0,0)
	// Check if P in vertex region outside B
	float3 bp = p - b;
	float d3 = dot(ab, bp);
	float d4 = dot(ac, bp);
	if (d3 >= 0.0f && d4 <= d3) return b; // barycentric coordinates (0,1,0)
	// Check if P in edge region of AB, if so return projection of P onto AB
	float vc = d1*d4 - d3*d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
		float v = d1 / (d1 - d3);
		return a + v * ab; // barycentric coordinates (1-v,v,0)
	}
	// Check if P in vertex region outside C
	float3 cp = p - c;
	float d5 = dot(ab, cp);
	float d6 = dot(ac, cp);
	if (d6 >= 0.0f && d5 <= d6) return c; // barycentric coordinates (0,0,1)
	// Check if P in edge region of AC, if so return projection of P onto AC
	float vb = d5*d2 - d1*d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
		float w = d2 / (d2 - d6);
		return a + w * ac; // barycentric coordinates (1-w,0,w)
	}
	// Check if P in edge region of BC, if so return projection of P onto BC
	float va = d3*d6 - d5*d4;
	if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
		float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return b + w * (c - b); // barycentric coordinates (0,1-w,w)
	}
	// P inside face region. Compute Q through its barycentric coordinates (u,v,w)
	float denom = 1.0f / (va + vb + vc);
	float v = vb * denom;
	float w = vc * denom;
	return a + ab * v + ac * w; // = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
}
///////////////////////////////////////////////////////////////////////////////////////   triangle overlap box  //////////////////////////////////////////////
int          g_OctreeInfoElement_Count = MyZero;
OctreeInfo * g_OctreeInfoElement_Cuda = MyNull;
int InitOctreeArray(int nSize, OctreeInfo** ptrOnCpu  )
{
	//OctreeInfo * g_OctreeInfoElement_Cuda
	g_OctreeInfoElement_Count = nSize;
	HANDLE_ERROR( cudaHostAlloc( (void**)ptrOnCpu, nSize * sizeof(OctreeInfo),cudaHostAllocMapped   )) ;
	HANDLE_ERROR(cudaHostGetDevicePointer((void **)&g_OctreeInfoElement_Cuda,(void *)(*ptrOnCpu),0));
	HANDLE_ERROR( cudaMemset( (void*)g_OctreeInfoElement_Cuda,	0, nSize * sizeof(OctreeInfo))) ;
	return 0;
}

int     g_VertexPosElement_Count = MyZero;
float3* g_VertexPosElement_Cuda = MyNull;
int		g_FaceIdxElement_Count = MyZero;
int3*	g_FaceIdxElement_Cuda = MyNull;
int		g_NormalDirectElement_Count = MyZero;
float3* g_NormalDirectElement_Cuda = MyNull;
int InitMesh_Vertex_Face_Normal(const int nVtxSize, float3 ** VtxPtr, const int nFaceSize, int3** FacePtr, const int nNormalSize, float3** NormalPtr)
{
	g_VertexPosElement_Count = nVtxSize;
	g_FaceIdxElement_Count = nFaceSize;
	g_NormalDirectElement_Count = nNormalSize;

	HANDLE_ERROR( cudaHostAlloc( (void**)VtxPtr, nVtxSize * sizeof(float3),cudaHostAllocMapped   )) ;
	HANDLE_ERROR(cudaHostGetDevicePointer((void **)&g_VertexPosElement_Cuda,(void *)(*VtxPtr),0));
	HANDLE_ERROR( cudaMemset( (void*)g_VertexPosElement_Cuda,	0, nVtxSize * sizeof(float3))) ;

	HANDLE_ERROR( cudaHostAlloc( (void**)FacePtr, nFaceSize * sizeof(int3),cudaHostAllocMapped   )) ;
	HANDLE_ERROR(cudaHostGetDevicePointer((void **)&g_FaceIdxElement_Cuda,(void *)(*FacePtr),0));
	HANDLE_ERROR( cudaMemset( (void*)g_FaceIdxElement_Cuda,	0, nFaceSize * sizeof(int3))) ;

	HANDLE_ERROR( cudaHostAlloc( (void**)NormalPtr, nNormalSize * sizeof(float3),cudaHostAllocMapped   )) ;
	HANDLE_ERROR(cudaHostGetDevicePointer((void **)&g_NormalDirectElement_Cuda,(void *)(*NormalPtr),0));
	HANDLE_ERROR( cudaMemset( (void*)g_NormalDirectElement_Cuda,	0, nNormalSize * sizeof(float3))) ;
	return 0;
}

__global__ void cuda_ParserGrid_Mesh(const int nSize, OctreeInfo* ptrOnCuda,
									 const int nVtxSize, float3 * VtxPtrOnCuda,
									 const int nFaceSize, int3* FacePtrOnCuda,
									 const int nNormalSize, float3* NormalPtrOnCuda,
									 const int nOctreeBegin, const int nOctreeEnd,
									 const int nFaceBegin, const int nFaceEnd)
{
	__shared__ int kernelFlag [MAX_KERNEL_PER_BLOCK];
	__shared__ float kernelDistance [MAX_KERNEL_PER_BLOCK];
	__shared__ int kernelDistanceId [MAX_KERNEL_PER_BLOCK];

	const int nHexId = blockIdx.x + nOctreeBegin;	
	//const int nFaceBase = threadIdx.x;
	const int kernelIdx = threadIdx.x;
	int nFaceIdx = nFaceBegin + threadIdx.x;
	const int nFaceStep = blockDim.x;
	//const int nKernelFlagIdx = blockIdx.x * blockDim.x + threadIdx.x;

	float boxcenter[3];
	float boxhalfsize[3];
	float triverts[3][3];
	 
	kernelFlag[kernelIdx] = 0;
	kernelDistance[kernelIdx] = FLT_MAX;

	OctreeInfo& curHex = ptrOnCuda[nHexId];

	boxcenter[0] = curHex.centerPos[0];
	boxcenter[1] = curHex.centerPos[1];
	boxcenter[2] = curHex.centerPos[2];

	boxhalfsize[0] = boxhalfsize[1] = boxhalfsize[2] = curHex.radius;

	float3 p,a,b,c;

	p = make_float3(boxcenter[0],boxcenter[1],boxcenter[2]);

	if ( (nFaceIdx < nFaceEnd) && 
		    (nHexId < nSize) && 
			(MyNo == (curHex.isContainMeshTriangle)) )
	{
		float3 * vtxPtr = &VtxPtrOnCuda[FacePtrOnCuda[nFaceIdx].x];
		triverts[0][0] = vtxPtr->x;triverts[0][1] = vtxPtr->y;triverts[0][2] = vtxPtr->z;
		a = make_float3(triverts[0][0],triverts[0][1],triverts[0][2]);

		vtxPtr = &VtxPtrOnCuda[FacePtrOnCuda[nFaceIdx].y];
		triverts[1][0] = vtxPtr->x;triverts[1][1] = vtxPtr->y;triverts[1][2] = vtxPtr->z;
		b = make_float3(triverts[1][0],triverts[1][1],triverts[1][2]);

		vtxPtr = &VtxPtrOnCuda[FacePtrOnCuda[nFaceIdx].z];
		triverts[2][0] = vtxPtr->x;triverts[2][1] = vtxPtr->y;triverts[2][2] = vtxPtr->z;
		c = make_float3(triverts[2][0],triverts[2][1],triverts[2][2]);

		if (1 == triBoxOverlap(boxcenter,boxhalfsize,triverts))
		{
			kernelFlag[kernelIdx] = 1;
		}

		float3 insect = (ClosestPtPointTriangle(p,a,b,c) - p);

		kernelDistance[kernelIdx] = dot(insect, insect);
		kernelDistanceId[kernelIdx] = nFaceIdx;
		/*if (dot(insect, insect) < curHex.distance)
		{
			curHex.distance = dot(insect, insect);
			curHex.nearestMeshTriangleId = nFaceIdx;
			curHex.center2nearestPtDirect = insect;
		}*/

#if 0
		float v0[3], v1[3], v2[3], e0[3], e1[3], e2[3];

		/* This is the fastest branch on Sun */
		/* move everything so that the boxcenter is in (0,0,0) */
		SUB(v0,triverts[0],boxcenter);
		SUB(v1,triverts[1],boxcenter);
		SUB(v2,triverts[2],boxcenter);

		/* compute triangle edges */
		SUB(e0,v1,v0);      /* tri edge 0 */
		SUB(e1,v2,v1);      /* tri edge 1 */
		SUB(e2,v0,v2);      /* tri edge 2 */

		if (1 == triBoxOverlap_1(boxhalfsize,v0, v1, v2, e0, e1, e2))
		{
			//kernelFlag[kernelIdx] = 1;
			if (1 == triBoxOverlap_0(boxhalfsize,v0, v1, v2, e0, e1, e2))
			{
				if (1 == triBoxOverlap_2(boxhalfsize,v0, v1, v2, e0, e1, e2))
				{
					kernelFlag[kernelIdx] = 1;
				}
				else
				{
					kernelFlag[kernelIdx] = 0;
				}
			}
			else
			{
				kernelFlag[kernelIdx] = 0;
			}
		}
		else
		{
			kernelFlag[kernelIdx] = 0;
		}
#endif
		__syncthreads();
		if (kernelIdx < WRAP_SIZE)
		{
			for (int lane = kernelIdx+WRAP_SIZE;lane < nFaceStep; lane+= WRAP_SIZE)
			{
				kernelFlag[kernelIdx] += kernelFlag[lane];	
				if (kernelDistance[kernelIdx] > kernelDistance[lane])
				{
					kernelDistance[kernelIdx] = kernelDistance[lane];
					kernelDistanceId[kernelIdx] = kernelDistanceId[lane];
				}
			}
			if ( kernelIdx < 16)
			{
				kernelFlag[kernelIdx] += kernelFlag[kernelIdx+16];
				if (kernelDistance[kernelIdx] > kernelDistance[kernelIdx+16])
				{
					kernelDistance[kernelIdx] = kernelDistance[kernelIdx+16];
					kernelDistanceId[kernelIdx] = kernelDistanceId[kernelIdx+16];
				}
			}
			if ( kernelIdx < 8)
			{
				kernelFlag[kernelIdx] += kernelFlag[kernelIdx+8];
				if (kernelDistance[kernelIdx] > kernelDistance[kernelIdx+8])
				{
					kernelDistance[kernelIdx] = kernelDistance[kernelIdx+8];
					kernelDistanceId[kernelIdx] = kernelDistanceId[kernelIdx+8];
				}
			}
			if ( kernelIdx < 4)
			{
				kernelFlag[kernelIdx] += kernelFlag[kernelIdx+4];
				if (kernelDistance[kernelIdx] > kernelDistance[kernelIdx+4])
				{
					kernelDistance[kernelIdx] = kernelDistance[kernelIdx+4];
					kernelDistanceId[kernelIdx] = kernelDistanceId[kernelIdx+4];
				}
			}
			if ( kernelIdx < 2)
			{
				kernelFlag[kernelIdx] += kernelFlag[kernelIdx+2];

				if (kernelDistance[kernelIdx] > kernelDistance[kernelIdx+2])
				{
					kernelDistance[kernelIdx] = kernelDistance[kernelIdx+2];
					kernelDistanceId[kernelIdx] = kernelDistanceId[kernelIdx+2];
				}
			}
			if ( kernelIdx < 1)
			{
				kernelFlag[kernelIdx] += kernelFlag[kernelIdx+1];//vals [ threadIdx.x ] += vals [ threadIdx.x + 1];
				curHex.isContainMeshTriangle = ((kernelFlag[0]>0) ? MyYES:MyNo);


				if (kernelDistance[kernelIdx] > kernelDistance[kernelIdx+1])
				{
					kernelDistance[kernelIdx] = kernelDistance[kernelIdx+1];
					kernelDistanceId[kernelIdx] = kernelDistanceId[kernelIdx+1];
				}
				if (curHex.distance > kernelDistance[0])
				{
					curHex.distance = kernelDistance[0];
					curHex.nearestMeshTriangleId = kernelDistanceId[0];
					/*if (curHex.nearestMeshTriangleId > nFaceSize)
					{
						CUPRINTF("curHex.nearestMeshTriangleId %d, kernelDistanceId[0] %d\n",curHex.nearestMeshTriangleId, kernelDistanceId[0]);
					}*/
				}
			}
			// first thread writes the result			
		}
		//nFaceIdx += nFaceStep;
	}
}

__global__ void parserOctreeInside(const int nSize, OctreeInfo* ptrOnCuda, const int nMaxLevel,
									const int nVtxSize, float3 * VtxPtrOnCuda,
									const int nFaceSize, int3* FacePtrOnCuda,
									const int nNormalSize, float3* NormalPtrOnCuda)
{
	const int hexId = blockIdx.x * blockDim.x + threadIdx.x;
	if (hexId < nSize)
	{
		OctreeInfo& curHexRef = ptrOnCuda[hexId];
		if (MyNo == curHexRef.isContainMeshTriangle)
		{
			float3 normals[3], triverts[3], center;
			center.x = curHexRef.centerPos[0];
			center.y = curHexRef.centerPos[1];
			center.z = curHexRef.centerPos[2];

			float3 * vtxPtr = &VtxPtrOnCuda[FacePtrOnCuda[curHexRef.nearestMeshTriangleId].x];
			triverts[0].x = vtxPtr->x;triverts[0].y = vtxPtr->y;triverts[0].z = vtxPtr->z;

			vtxPtr = &VtxPtrOnCuda[FacePtrOnCuda[curHexRef.nearestMeshTriangleId].y];
			triverts[1].x = vtxPtr->x;triverts[1].y = vtxPtr->y;triverts[1].z = vtxPtr->z;

			vtxPtr = &VtxPtrOnCuda[FacePtrOnCuda[curHexRef.nearestMeshTriangleId].z];
			triverts[2].x = vtxPtr->x;triverts[2].y = vtxPtr->y;triverts[2].z = vtxPtr->z;

			curHexRef.center2nearestPtDirect = (ClosestPtPointTriangle(center,triverts[0],triverts[1],triverts[2]) - center);

			vtxPtr = &NormalPtrOnCuda[FacePtrOnCuda[curHexRef.nearestMeshTriangleId].x];
			normals[0].x = vtxPtr->x;normals[0].y = vtxPtr->y;normals[0].z = vtxPtr->z;

			vtxPtr = &NormalPtrOnCuda[FacePtrOnCuda[curHexRef.nearestMeshTriangleId].y];
			normals[1].x = vtxPtr->x;normals[1].y = vtxPtr->y;normals[1].z = vtxPtr->z;

			vtxPtr = &NormalPtrOnCuda[FacePtrOnCuda[curHexRef.nearestMeshTriangleId].z];
			normals[2].x = vtxPtr->x;normals[2].y = vtxPtr->y;normals[2].z = vtxPtr->z;

			if (dot(normals[0] + normals[1] + normals[2],curHexRef.center2nearestPtDirect) > 0)
			{
				// cos < 90
				curHexRef.isInside = MyYES;
				curHexRef.isLeaf = MyYES;
			}
			else
			{
				curHexRef.isInside = MyNo;
				curHexRef.isLeaf = MyNo;
			}
		}
		else
		{
			if (curHexRef.nLevel == (nMaxLevel-1))
			{
				curHexRef.isLeaf = MyYES;
			}
			else
			{
				curHexRef.isLeaf = MyNo;
			}
			curHexRef.isInside = MyNo;
		}
	}
}

void ParserGrid_Mesh(const int nMaxLevel)
{
#define ParserGrid_Mesh_BlockSize (8192)
#define ParserGrid_Mesh_ThreadPerBlock (1024)
	LogInfo("OctreeCount %d, FaceCount %d\n",g_OctreeInfoElement_Count,g_FaceIdxElement_Count);
	for (int nOctreeBegin=0;nOctreeBegin < g_OctreeInfoElement_Count;nOctreeBegin += ParserGrid_Mesh_BlockSize)
	{
		const int nOctreeEnd = ((nOctreeBegin+ParserGrid_Mesh_BlockSize) < g_OctreeInfoElement_Count) ? (nOctreeBegin+ParserGrid_Mesh_BlockSize) : (g_OctreeInfoElement_Count);
		for (int nFaceBegin=0;nFaceBegin < g_FaceIdxElement_Count;nFaceBegin += ParserGrid_Mesh_ThreadPerBlock)
		{			
			const int nFaceEnd = ((nFaceBegin+ParserGrid_Mesh_ThreadPerBlock) < g_FaceIdxElement_Count) ? (nFaceBegin+ParserGrid_Mesh_ThreadPerBlock) : (g_FaceIdxElement_Count);
			LogInfo("octree[%d,%d] face[%d,%d]\n",nOctreeBegin,nOctreeEnd,nFaceBegin,nFaceEnd);
			//MyPause;
			cuda_ParserGrid_Mesh<<<ParserGrid_Mesh_BlockSize ,ParserGrid_Mesh_ThreadPerBlock>>>
				(g_OctreeInfoElement_Count, g_OctreeInfoElement_Cuda,
				g_VertexPosElement_Count, g_VertexPosElement_Cuda,
				g_FaceIdxElement_Count, g_FaceIdxElement_Cuda,
				g_NormalDirectElement_Count, g_NormalDirectElement_Cuda,
				nOctreeBegin, nOctreeEnd,
				nFaceBegin, nFaceEnd);
			cudaDeviceSynchronize();
			 
		}
	}

	;
	parserOctreeInside<<<GRIDCOUNT(g_OctreeInfoElement_Count,MAX_KERNEL_PER_BLOCK) ,MAX_KERNEL_PER_BLOCK>>>
		(g_OctreeInfoElement_Count, g_OctreeInfoElement_Cuda,nMaxLevel,
		g_VertexPosElement_Count, g_VertexPosElement_Cuda,
		g_FaceIdxElement_Count, g_FaceIdxElement_Cuda,
		g_NormalDirectElement_Count, g_NormalDirectElement_Cuda
		);
	cudaDeviceSynchronize();
	LogInfo("ParserGrid_Mesh Finish!\n");
}
#endif