#include "VR_GPU_Geometry_Octree.h"

#include <fstream>
#include <cmath>
#include <vector>
#include <map>
#include <queue>
#include "VR_Geometry.h"
#if USE_CUDA
#include "VR_GPU_Utils.h"
#include <boost/format.hpp>
#include <iostream>

extern void ParserGrid_Mesh(const int nMaxLevel);
extern int InitOctreeArray(int nSize, YC::Geometry::GPU::OctreeInfo** ptrOnCpu  );
extern int InitMesh_Vertex_Face_Normal(const int nVtxSize, float3 ** VtxPtr, const int nFaceSize, int3** FacePtr, const int nNormalSize, float3** NormalPtr);
namespace YC
{
	namespace Geometry
	{
		namespace GPU
		{
			///////////////////////////////////////////////////////////////////////////////////////////////////////
#if 1
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

			 int planeBoxOverlap(float normal[3], float vert[3], float maxbox[3])	// -NJMP-
			{
				int q;
				float vmin[3],vmax[3],v;
				for(q=X;q<=Z;q++)
				{
					v=vert[q];					// -NJMP-
					if(normal[q]>0.0f)
					{
						vmin[q]=-maxbox[q] - v;	// -NJMP-
						vmax[q]= maxbox[q] - v;	// -NJMP-
					}
					else
					{
						vmin[q]= maxbox[q] - v;	// -NJMP-
						vmax[q]=-maxbox[q] - v;	// -NJMP-
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


			 int triBoxOverlap(float boxcenter[3],float boxhalfsize[3],float triverts[3][3])
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

			 MyPoint ClosestPtPointTriangle(MyPoint p, MyPoint a, MyPoint b, MyPoint c)
			 {
				 // Check if P in vertex region outside A
				 MyPoint ab = b - a;
				 MyPoint ac = c - a;
				 MyPoint ap = p - a;
				 float d1 = ab.dot(ap);
				 float d2 = ac.dot(ap);
				 if (d1 <= 0.0f && d2 <= 0.0f) return a; // barycentric coordinates (1,0,0)
				 // Check if P in vertex region outside B
				 MyPoint bp = p - b;
				 float d3 = ab.dot(bp);
				 float d4 = ac.dot(bp);
				 if (d3 >= 0.0f && d4 <= d3) return b; // barycentric coordinates (0,1,0)
				 // Check if P in edge region of AB, if so return projection of P onto AB
				 float vc = d1*d4 - d3*d2;
				 if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
					 float v = d1 / (d1 - d3);
					 return a + v * ab; // barycentric coordinates (1-v,v,0)
				 }
				 // Check if P in vertex region outside C
				 MyPoint cp = p - c;
				 float d5 = ab.dot(cp);
				 float d6 = ac.dot(cp);
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
#endif
			//////////////////////////////////////////////////////////////////////////
			int VR_Octree::spliteCube(const MeshDataStruct& objMeshInfo, int nLevel)
			{
				Q_ASSERT(nLevel >3 && nLevel < 7);

				int nCount = 0;
				for (int i=0;i<nLevel;++i)
				{
					nCount += (int)std::pow((float)ChildrenCount,i);
				}
				const int nBufferSize = nCount;//(int)std::pow((float)ChildrenCount,nLevel-1)-1;
				m_OctreeInfoElement_Count = nBufferSize;
				std::vector<LevelInfo> LevelElement;
#define __MARK__
				int nBase=0;
				for (int i=1 __MARK__;i<=nLevel;++i)
				{
					LevelInfo tmpNode;
					tmpNode.nLevel = i-1;
					tmpNode.nBase = nBase;//((int)std::pow((float)ChildrenCount,(i-1))-1);
					tmpNode.nCount =((int)std::pow((float)ChildrenCount,(i-1)));
					nBase = tmpNode.nBase + tmpNode.nCount;
					tmpNode.radius = 1.0 / std::pow(2.0,i);
					tmpNode.radiusX2 = tmpNode.radius * 2.f;
					tmpNode.ElementCountPerDim = (int)std::pow(2.0,(i-1));
					LevelElement.push_back(tmpNode);					
					LogInfo("[level=%d] [base=%d] [count=%d] [radius=%f]\n",tmpNode.nLevel,tmpNode.nBase,tmpNode.nCount,tmpNode.radius);
				}
				//MyPause;
				//create GPU Memory
				//Utils::getCurrentGPUMemoryInfo();
				InitOctreeArray(nBufferSize,&m_OctreeInfoElement_Cpu);
				/*Utils::getCurrentGPUMemoryInfo();
				MyPause;*/
				std::queue< int > qq;

				//for (int i=0;i<nLevel;++i)
				{
					int i=0;
					const LevelInfo& curNode = LevelElement[i];
					for (int j=0;j<curNode.nCount;++j)
					{
						OctreeInfo& curOctreeNode = m_OctreeInfoElement_Cpu[curNode.nBase + j];
						curOctreeNode.isContainMeshTriangle = MyNo;
						curOctreeNode.isInside = MyNo;
						curOctreeNode.isLeaf = MyNo;
						curOctreeNode.nearestMeshTriangleId = Invalid_Id;
						curOctreeNode.distance = FLT_MAX;
						curOctreeNode.nId4CurrentLevel = j;
						curOctreeNode.nLevel = curNode.nLevel;
						curOctreeNode.radius = curNode.radius;
						
						curOctreeNode.nId = curNode.nBase + j;
						
						curOctreeNode.parentId = Invalid_Id;
						curOctreeNode.childId[0] = curOctreeNode.childId[1] = curOctreeNode.childId[2] = curOctreeNode.childId[3] = Invalid_Id;
						curOctreeNode.childId[4] = curOctreeNode.childId[5] = curOctreeNode.childId[6] = curOctreeNode.childId[7] = Invalid_Id;
						/*curOctreeNode.spaceIndexes[0] = j / (curNode.ElementCountPerDim*curNode.ElementCountPerDim);
						curOctreeNode.spaceIndexes[1] = (j % (curNode.ElementCountPerDim*curNode.ElementCountPerDim)) / curNode.ElementCountPerDim;
						curOctreeNode.spaceIndexes[2] = (j % (curNode.ElementCountPerDim*curNode.ElementCountPerDim)) % curNode.ElementCountPerDim;*/
						
						/*curOctreeNode.centerPos[0] = curOctreeNode.spaceIndexes[0] * curNode.radiusX2 + curNode.radius -0.5f;
						curOctreeNode.centerPos[1] = curOctreeNode.spaceIndexes[1] * curNode.radiusX2 + curNode.radius -0.5f;
						curOctreeNode.centerPos[2] = curOctreeNode.spaceIndexes[2] * curNode.radiusX2 + curNode.radius -0.5f;*/
						curOctreeNode.centerPos[0] = 0.5f -0.5f;
						curOctreeNode.centerPos[1] = 0.5f -0.5f;
						curOctreeNode.centerPos[2] = 0.5f -0.5f;
						
						qq.push(curOctreeNode.nId);
					}

					static MyVec3 directiion[8] = {MyVec3(-1.f,-1.f,-1.f),
													 MyVec3(-1.f,-1.f,1.f),
													 MyVec3(-1.f,1.f,-1.f),
													 MyVec3(-1.f,1.f,1.f),
													 MyVec3(1.f,-1.f,-1.f),
													 MyVec3(1.f,-1.f,1.f),
													 MyVec3(1.f,1.f,-1.f),
													 MyVec3(1.f,1.f,1.f)};

					std::map< int ,bool > mapId;
					while (!qq.empty())
					{
						int parentId = qq.front();
						OctreeInfo& parentOctreeNode = m_OctreeInfoElement_Cpu[parentId];
						qq.pop();

						if (parentOctreeNode.nLevel < (nLevel-1) )
						{
							const LevelInfo& curNode = LevelElement[parentOctreeNode.nLevel+1];
							for (int k=0;k<8;++k)
							{
								int j = parentOctreeNode.nId4CurrentLevel*8+k;
								//LogInfo("%d #\n",curNode.nBase + j);
								Q_ASSERT(mapId.find(curNode.nBase + j) == mapId.end());
								mapId[curNode.nBase + j] = true;
								OctreeInfo& curOctreeNode = m_OctreeInfoElement_Cpu[curNode.nBase + j];
								curOctreeNode.isContainMeshTriangle = MyNo;
								curOctreeNode.isInside = MyNo;
								curOctreeNode.isLeaf = MyNo;
								curOctreeNode.nearestMeshTriangleId = Invalid_Id;
								curOctreeNode.distance = FLT_MAX;
								curOctreeNode.nId4CurrentLevel = j;
								curOctreeNode.nLevel = parentOctreeNode.nLevel+1;
								curOctreeNode.radius = curNode.radius;

								curOctreeNode.centerPos[0] = parentOctreeNode.centerPos[0] + directiion[k][0] * curOctreeNode.radius;
								curOctreeNode.centerPos[1] = parentOctreeNode.centerPos[1] + directiion[k][1] * curOctreeNode.radius;
								curOctreeNode.centerPos[2] = parentOctreeNode.centerPos[2] + directiion[k][2] * curOctreeNode.radius;

								/*MyPoint centerParent(parentOctreeNode.centerPos[0],parentOctreeNode.centerPos[1],parentOctreeNode.centerPos[2]);
								MyPoint centerChild(curOctreeNode.centerPos[0],curOctreeNode.centerPos[1],curOctreeNode.centerPos[2]);
								LogInfo("%f -- %f\n",(centerParent-centerChild).norm(), parentOctreeNode.radius);
								Q_ASSERT((centerParent-centerChild).norm() < parentOctreeNode.radius);*/

								curOctreeNode.nId = curNode.nBase + j;
								curOctreeNode.nId = curNode.nBase + j;
								

								curOctreeNode.parentId = parentOctreeNode.nId;
								curOctreeNode.childId[0] = curOctreeNode.childId[1] = curOctreeNode.childId[2] = curOctreeNode.childId[3] = Invalid_Id;
								curOctreeNode.childId[4] = curOctreeNode.childId[5] = curOctreeNode.childId[6] = curOctreeNode.childId[7] = Invalid_Id;

								parentOctreeNode.childId[k] = curOctreeNode.nId;

								/*if (parentOctreeNode.nId == 102)
								{
									LogInfo("parent id %d --> child id %d\n",parentOctreeNode.nId,parentOctreeNode.childId[k]);
								}*/
								
								qq.push(curOctreeNode.nId);
							}
						}
					}
				}
				//MyPause;
				//////////////////////////////////////////////////////////////////////////
				

				//////////////////////////////////////////////////////////////////////////
				//Utils::getCurrentGPUMemoryInfo();
				
				//create mesh points faces normals gpu memory
				Q_ASSERT(objMeshInfo.normals.size() == objMeshInfo.points.size());
				InitMesh_Vertex_Face_Normal(objMeshInfo.points.size(), &m_VertexPosElement_Cpu, 
												 objMeshInfo.faces.size() / 3, &m_FaceIdxElement_Cpu, 
												 objMeshInfo.normals.size(), &m_NormalDirectElement_Cpu);
				/*Utils::getCurrentGPUMemoryInfo();
				MyPause;*/
				for (int i=0;i< objMeshInfo.points.size(); ++i)
				{
					m_VertexPosElement_Cpu[i].x = objMeshInfo.points[i].x;
					m_VertexPosElement_Cpu[i].y = objMeshInfo.points[i].y;
					m_VertexPosElement_Cpu[i].z = objMeshInfo.points[i].z;

					m_NormalDirectElement_Cpu[i].x = objMeshInfo.normals[i].x;
					m_NormalDirectElement_Cpu[i].y = objMeshInfo.normals[i].y;
					m_NormalDirectElement_Cpu[i].z = objMeshInfo.normals[i].z;
				}


				for (int idx=0,i=0;idx< objMeshInfo.faces.size();++i)
				{					
					m_FaceIdxElement_Cpu[i].x = objMeshInfo.faces[idx++];
					m_FaceIdxElement_Cpu[i].y = objMeshInfo.faces[idx++];
					m_FaceIdxElement_Cpu[i].z = objMeshInfo.faces[idx++];
				}
				LogInfo("call ParserGrid_Mesh\n");
				
				ParserGrid_Mesh(nLevel);
				//parserOnCpu(nBufferSize,objMeshInfo);

				//parserInside(nBufferSize,objMeshInfo);
				
				mergeInside(LevelElement);
				std::map< int, int> mapStatic;
				for (int i=0;i<nLevel;++i)
				{
					mapStatic[i] = 0;
					const LevelInfo& curNode = LevelElement[i];
					for (int j=0;j<curNode.nCount;++j)
					{
						OctreeInfo& curOctreeNode = m_OctreeInfoElement_Cpu[curNode.nBase + j];
						if (MyYES == curOctreeNode.isLeaf)
						{
							mapStatic[i]++;
						}
					}
					LogInfo("level %d : %d/%d \n",i,mapStatic[i],curNode.nCount);

					/*
					[bunny ]
					level 0 : 1/1
					level 1 : 8/8
					level 2 : 42/64
					level 3 : 195/512
					level 4 : 837/4096
					level 5 : 3463/32768
					*/
				}
				//MyPause;
				/*HANDLE_ERROR( cudaHostAlloc( (void**)&g_VertexOnCudaFlag4EFGOutBoundaryOnCpu, _nExternalMemory * g_nVertexOnCudaCount * sizeof(int),cudaHostAllocMapped   )) ;
				HANDLE_ERROR(cudaHostGetDevicePointer((void **)&g_VertexOnCudaFlag4EFGOutBoundaryOnCuda,(void *)g_VertexOnCudaFlag4EFGOutBoundaryOnCpu,0));
				HANDLE_ERROR( cudaMemset( (void*)g_VertexOnCudaFlag4EFGOutBoundaryOnCuda,	0, _nExternalMemory * g_nVertexOnCudaCount * sizeof(int))) ;	*/
				return 0;
			}

			void VR_Octree::parserOnCpu(const int nHexCount,const MeshDataStruct& objMeshInfo)
			{
				float boxcenter[3];
				float boxhalfsize[3];
				float triverts[3][3];

				const int nFaceCount = objMeshInfo.faces.size() / 3;
				for (int hexId = 0; hexId < nHexCount; ++hexId)
				{
					for (int faceIdx=0;faceIdx < nFaceCount; ++faceIdx)
					{

						OctreeInfo& curHex = m_OctreeInfoElement_Cpu[hexId];

						boxcenter[0] = curHex.centerPos[0];
						boxcenter[1] = curHex.centerPos[1];
						boxcenter[2] = curHex.centerPos[2];

						boxhalfsize[0] = boxhalfsize[1] = boxhalfsize[2] = curHex.radius;

						
						float3 * vtxPtr = &m_VertexPosElement_Cpu[m_FaceIdxElement_Cpu[faceIdx].x];
						triverts[0][0] = vtxPtr->x;triverts[0][1] = vtxPtr->y;triverts[0][2] = vtxPtr->z;

						vtxPtr = &m_VertexPosElement_Cpu[m_FaceIdxElement_Cpu[faceIdx].y];
						triverts[1][0] = vtxPtr->x;triverts[1][1] = vtxPtr->y;triverts[1][2] = vtxPtr->z;

						vtxPtr = &m_VertexPosElement_Cpu[m_FaceIdxElement_Cpu[faceIdx].z];
						triverts[2][0] = vtxPtr->x;triverts[2][1] = vtxPtr->y;triverts[2][2] = vtxPtr->z;

						if (1 == triBoxOverlap(boxcenter,boxhalfsize,triverts))
						{
							
							curHex.isContainMeshTriangle = MyYES;
						}
					}
				}
			}

			void makeCellVertex(MyPoint center, MyFloat radius, std::vector< MyPoint >& vertexes )
			{
				vertexes.resize(YC::Geometry::vertexs_per_cell);
				//FEM Ë³Ðò
				static MyDenseVector step[YC::Geometry::vertexs_per_cell] = 
				   {MyDenseVector(-1,-1,-1), MyDenseVector(1,-1,-1),
					MyDenseVector(-1,1,-1)	, MyDenseVector(1,1,-1),
					MyDenseVector(-1,-1,1)	, MyDenseVector(1,-1,1),
					MyDenseVector(-1,1,1)	, MyDenseVector(1,1,1)};

				for (int v=0;v < Geometry::vertexs_per_cell;++v)
				{
					vertexes[v] = (center + radius * step[v]);
					//printf("{%f,%f,%f}\n",vertexes[v]->getPos()[0],vertexes[v]->getPos()[1],vertexes[v]->getPos()[2]);
				}
				//MyPause;
			}

			MyVBOLineSet * VR_Octree::makeVBOLineSet()
			{
				static unsigned linePairs[12][2] = {{0,1},{4,5},{6,7},{2,3},
				{0,2},{1,3},{5,7},{4,6},
				{0,4},{1,5},{3,7},{2,6}};

				MyVBOLineSet * ret = new MyVBOLineSet;

				std::vector< float > lineSetPos;
				std::vector< int > lineSetIdx;
				std::vector< MyPoint > vertexes;

				int nLineCount = 0;

				//std::ofstream outfile("d:\\1.txt");
				for (int i=0;i<m_OctreeInfoElement_Count;++i)
				{
					OctreeInfo& curOctreeNode = m_OctreeInfoElement_Cpu[i];
					if (MyYES == curOctreeNode.isLeaf)
					{
						/*MyPoint tmp(curOctreeNode.centerPos[0],curOctreeNode.centerPos[1],curOctreeNode.centerPos[2]);
						if (tmp.dot(tmp) < 0.25f) continue;*/
						//outfile << curOctreeNode.nId << "," << curOctreeNode.nLevel << "," << curOctreeNode.spaceIndexes[0]<< "," << curOctreeNode.spaceIndexes[1]<< "," << curOctreeNode.spaceIndexes[2] << std::endl;
						nLineCount += 12;
						makeCellVertex(MyPoint(curOctreeNode.centerPos[0],curOctreeNode.centerPos[1],curOctreeNode.centerPos[2]),
							           curOctreeNode.radius,
									   vertexes);

						for (unsigned k=0;k<12;++k)
						{
							MyPoint & p0 = vertexes[linePairs[k][0]];
							MyPoint & p1 = vertexes[linePairs[k][1]];

							lineSetPos.push_back(p0.x());
							lineSetPos.push_back(p0.y());
							lineSetPos.push_back(p0.z());
							lineSetIdx.push_back(lineSetIdx.size());

							lineSetPos.push_back(p1.x());
							lineSetPos.push_back(p1.y());
							lineSetPos.push_back(p1.z());
							lineSetIdx.push_back(lineSetIdx.size());
						}
					}
				}
				//outfile.close();
				ret->initialize(nLineCount,&lineSetPos[0],&lineSetIdx[0]);
				return ret;
			}

			MyVBOLineSet * VR_Octree::makeVBOLineSet(const int nLevel)
			{
				static unsigned linePairs[12][2] = {{0,1},{4,5},{6,7},{2,3},
				{0,2},{1,3},{5,7},{4,6},
				{0,4},{1,5},{3,7},{2,6}};

				MyVBOLineSet * ret = new MyVBOLineSet;

				std::vector< float > lineSetPos;
				std::vector< int > lineSetIdx;
				std::vector< MyPoint > vertexes;

				int nLineCount = 0;

				//std::ofstream outfile("d:\\1.txt");
				for (int i=0;i<m_OctreeInfoElement_Count;++i)
				{
					OctreeInfo& curOctreeNode = m_OctreeInfoElement_Cpu[i];
					if (MyYES == curOctreeNode.isLeaf && nLevel == curOctreeNode.nLevel)
					{
						/*MyPoint tmp(curOctreeNode.centerPos[0],curOctreeNode.centerPos[1],curOctreeNode.centerPos[2]);
						if (tmp.dot(tmp) < 0.25f) continue;*/
						//outfile << curOctreeNode.nId << "," << curOctreeNode.nLevel << "," << curOctreeNode.spaceIndexes[0]<< "," << curOctreeNode.spaceIndexes[1]<< "," << curOctreeNode.spaceIndexes[2] << std::endl;
						nLineCount += 12;
						makeCellVertex(MyPoint(curOctreeNode.centerPos[0],curOctreeNode.centerPos[1],curOctreeNode.centerPos[2]),
							           curOctreeNode.radius,
									   vertexes);

						for (unsigned k=0;k<12;++k)
						{
							MyPoint & p0 = vertexes[linePairs[k][0]];
							MyPoint & p1 = vertexes[linePairs[k][1]];

							lineSetPos.push_back(p0.x());
							lineSetPos.push_back(p0.y());
							lineSetPos.push_back(p0.z());
							lineSetIdx.push_back(lineSetIdx.size());

							lineSetPos.push_back(p1.x());
							lineSetPos.push_back(p1.y());
							lineSetPos.push_back(p1.z());
							lineSetIdx.push_back(lineSetIdx.size());
						}
					}
				}
				//outfile.close();
				ret->initialize(nLineCount,&lineSetPos[0],&lineSetIdx[0]);
				return ret;
			}

			void VR_Octree::parserInside(const int nHexCount,const MeshDataStruct& objMeshInfo)
			{
				MyPoint center;
				MyPoint triverts[3];
				const int nFaceCount = objMeshInfo.faces.size() / 3;
				//std::ofstream outfile("d:\\111.txt");
				for (int hexId = 0; hexId < nHexCount; ++hexId)
				{
					OctreeInfo& curHexRef = m_OctreeInfoElement_Cpu[hexId];
					
					if (MyNo == curHexRef.isContainMeshTriangle)
					{
						center[0] = curHexRef.centerPos[0];
						center[1] = curHexRef.centerPos[1];
						center[2] = curHexRef.centerPos[2];

						LogInfo("curHexRef.nearestMeshTriangleId %d\n",curHexRef.nearestMeshTriangleId);
						Q_ASSERT(curHexRef.nearestMeshTriangleId < nFaceCount && curHexRef.nearestMeshTriangleId >= 0);

						float3 * vtxPtr = &m_VertexPosElement_Cpu[m_FaceIdxElement_Cpu[curHexRef.nearestMeshTriangleId].x];
						triverts[0][0] = vtxPtr->x;triverts[0][1] = vtxPtr->y;triverts[0][2] = vtxPtr->z;

						vtxPtr = &m_VertexPosElement_Cpu[m_FaceIdxElement_Cpu[curHexRef.nearestMeshTriangleId].y];
						triverts[1][0] = vtxPtr->x;triverts[1][1] = vtxPtr->y;triverts[1][2] = vtxPtr->z;

						vtxPtr = &m_VertexPosElement_Cpu[m_FaceIdxElement_Cpu[curHexRef.nearestMeshTriangleId].z];
						triverts[2][0] = vtxPtr->x;triverts[2][1] = vtxPtr->y;triverts[2][2] = vtxPtr->z;

						MyPoint direct = ClosestPtPointTriangle(center,triverts[0],triverts[1],triverts[2]) - center;
						MyPoint normals[3];
						/*Q_ASSERT(curHexRef.nearestMeshTriangleId < nFaceCount && curHexRef.nearestMeshTriangleId >= 0);
						outfile << curHexRef.nId << "," << curHexRef.nearestMeshTriangleId << "," << curHexRef.distance << std::endl;
						LogInfo("%d %f",curHexRef.nearestMeshTriangleId,curHexRef.distance);*/
						vtxPtr = &m_NormalDirectElement_Cpu[m_FaceIdxElement_Cpu[curHexRef.nearestMeshTriangleId].x];
						normals[0][0] = vtxPtr->x;normals[0][1] = vtxPtr->y;normals[0][2] = vtxPtr->z;

						vtxPtr = &m_NormalDirectElement_Cpu[m_FaceIdxElement_Cpu[curHexRef.nearestMeshTriangleId].y];
						normals[1][0] = vtxPtr->x;normals[1][1] = vtxPtr->y;normals[1][2] = vtxPtr->z;

						vtxPtr = &m_NormalDirectElement_Cpu[m_FaceIdxElement_Cpu[curHexRef.nearestMeshTriangleId].z];
						normals[2][0] = vtxPtr->x;normals[2][1] = vtxPtr->y;normals[2][2] = vtxPtr->z;

						
						if (direct.dot(normals[0] + normals[1] + normals[2]) > 0)
						{
							curHexRef.isInside = MyYES;
						}
						else
						{
							curHexRef.isInside = MyNo;
						}
					}
					else
					{
						curHexRef.isInside = MyNo;
					}
				}
				//outfile.close();
				
			}

			void VR_Octree::mergeInside(const std::vector<LevelInfo>& vecLevelInfo)
			{
				//return ;
				const int nMaxLevel = vecLevelInfo.size();
				for (int curLevel=nMaxLevel-2;curLevel >=0;--curLevel)
				{
					for (int i=0;i< vecLevelInfo[curLevel].nCount; ++i)
					{
						const int idx = vecLevelInfo[curLevel].nBase + i;
						OctreeInfo& ref = m_OctreeInfoElement_Cpu[idx];
						
						Q_ASSERT(ref.nLevel == curLevel);
						if (MyYES == ref.isInside)
						{
							//const int nChildIdx = ref.nId4CurrentLevel*8 + vecLevelInfo[curLevel+1].nBase;
							bool bFlag = true;
							for (int j=0;j<8;++j)
							{
								
								OctreeInfo& refChild = m_OctreeInfoElement_Cpu[ref.childId[j]];
								
								Q_ASSERT(refChild.parentId == ref.nId);
								if (/*MyYES == refChild.isLeaf &&*/ MyYES == refChild.isInside){}
								else
								{
									bFlag = false;
								}

								MyPoint centerParent(ref.centerPos[0],ref.centerPos[1],ref.centerPos[2]);
								MyPoint centerChild(refChild.centerPos[0],refChild.centerPos[1],refChild.centerPos[2]);
								
								Q_ASSERT((centerParent-centerChild).norm() < ref.radius);
							}

							if (bFlag)
							{
								//merge
								ref.isInside = MyYES;
								ref.isLeaf = MyYES;
								for (int j=0;j<8;++j)
								{
									OctreeInfo& refChild = m_OctreeInfoElement_Cpu[ref.childId[j]];
									refChild.isLeaf = MyNo;
								}
							} 
							else
							{
								ref.isInside = MyYES;
								ref.isLeaf = MyNo;
								
							}
						}
					}
				}
			}
		
			void VR_Octree::exportOctreeGrid(const char* lpszGridFile, const int nLevel)
			{
				//cout << boost::format("%2$3.1f - %1$.2f%%") % 10.0 % 12.5  << endl;
				// Êä³ö:12.5 - 10.00%
				boost::format fmter("{%1$.9ff,%2$.9ff,%3$.9ff,%4$.9ff},");
				
				std::ofstream outfile(lpszGridFile);
				outfile << "float obj_octree_grid[][4]={";
				for (int i=0;i<m_OctreeInfoElement_Count;++i)
				{
					OctreeInfo& ref = m_OctreeInfoElement_Cpu[i];
					if (MyYES == ref.isLeaf && nLevel == ref.nLevel)
					{
						outfile << fmter % ref.centerPos[0] % ref.centerPos[1] % ref.centerPos[2] % ref.radius;
					}
				}
				outfile << "};";
				outfile.close();
			}
		}//namespace GPU
	}//namespace Geometry
}
#endif