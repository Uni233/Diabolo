#include "vrBEM3D.h"
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "vrBase/vrLog.h"
#include "quad_rule.h"
#include "bemTriangleElem.h"
#include <fstream>
#include <sstream>
#include<iomanip>
#include "vrGlobalConf.h"
#include "RegionHandler.h"

#if USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/tbb_allocator.h>
#include "tbb_parallel_assemble_source_point.h"
using namespace tbb;


#endif
#include "MaterialModel.h"
#include "vrColor.h"
#include "vec3LessCompare.h"
#include "discontinuousColletionPoint.h"

#include "boost/date_time/posix_time/posix_time.hpp" // current time
#define Output_Matrix_Displacement (0)
#define LoadMatrixData (0)	

#include "vrTimer/TimingCPU.h"

#if USE_cuSolverDn
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#include <cuda_runtime.h>

#include "cublas_v2.h"
#include "cusolverDn.h"
#include "helper_cuda.h"

#include "helper_cusolver.h"

#include <sstream>

/*
 *  solve A*x = b by QR
 *
 */
int linearSolverQR(
    cusolverDnHandle_t handle,
    int n,
    const double *Acopy,
    int lda,
    const double *b,
    double *x)
{
    cublasHandle_t cublasHandle = NULL; // used in residual evaluation
    int bufferSize = 0;
    int bufferSize_geqrf = 0;
    int bufferSize_ormqr = 0;
    int *info = NULL;
    double *buffer = NULL;
    double *A = NULL;
    double *tau = NULL;
    int h_info = 0;
    double start, stop;
    double time_solve;
    const double one = 1.0;

    checkCudaErrors(cublasCreate(&cublasHandle));

    checkCudaErrors(cusolverDnDgeqrf_bufferSize(handle, n, n, (double*)Acopy, lda, &bufferSize_geqrf));
    checkCudaErrors(cusolverDnDormqr_bufferSize(
        handle,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_T,
        n,
        1,
        n,
        A,
        lda,
        NULL,
        x,
        n,
        &bufferSize_ormqr));

    printf("buffer_geqrf = %d, buffer_ormqr = %d \n", bufferSize_geqrf, bufferSize_ormqr);
    
    bufferSize = (bufferSize_geqrf > bufferSize_ormqr)? bufferSize_geqrf : bufferSize_ormqr ; 

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(double)*lda*n));
    checkCudaErrors(cudaMalloc ((void**)&tau, sizeof(double)*n));

// prepare a copy of A because getrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();
    start = second();

// compute QR factorization
    checkCudaErrors(cusolverDnDgeqrf(handle, n, n, A, lda, tau, buffer, bufferSize, info));

    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: LU factorization failed\n");
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(double)*n, cudaMemcpyDeviceToDevice));

    // compute Q^T*b
    checkCudaErrors(cusolverDnDormqr(
        handle,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_T,
        n,
        1,
        n,
        A,
        lda,
        tau,
        x,
        n,
        buffer,
        bufferSize,
        info));

    // x = R \ Q^T*b
    checkCudaErrors(cublasDtrsm(
         cublasHandle,
         CUBLAS_SIDE_LEFT,
         CUBLAS_FILL_MODE_UPPER,
         CUBLAS_OP_N,
         CUBLAS_DIAG_NON_UNIT,
         n,
         1,
         &one,
         A,
         lda,
         x,
         n));
    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();

    time_solve = stop - start;
    fprintf (stdout, "timing: QR = %10.6f sec\n", time_solve);

    if (cublasHandle) { checkCudaErrors(cublasDestroy(cublasHandle)); }
    if (info  ) { checkCudaErrors(cudaFree(info  )); }
    if (buffer) { checkCudaErrors(cudaFree(buffer)); }
    if (A     ) { checkCudaErrors(cudaFree(A)); }
    if (tau   ) { checkCudaErrors(cudaFree(tau)); }

    return 0;
}
#endif//USE_cuSolverDn

extern VR::vrBEM3D * g_BEM3D;
namespace VR
{



	MaterialModel* material=NULL;

#if USE_NEW_VERTEX
	struct TriangleSet4Dual
	{
		//           2
		//          / .
		//         /   .
		//        /     .
		//       0-------1


		//      ^
		//    1 | 2
		//      | |.
		//    Y | | .
		//      | |  .
		//    0 | 0---1
		//      +------->
		//        0 X 1

		//param space vertex order
		//vertex 0 :¡¾ 0.0, 0.0¡¿
		//vertex 1 :¡¾ 1.0, 0.0¡¿
		//vertex 2 :¡¾ 0.0, 1.0¡¿
		//notice right hand identity
		void setVertexBoundary(MyInt idx, bool flag)
		{
			vertex_isBoundary[idx] = flag;
		}

		bool getVertexBoundary(MyInt idx)const
		{
			return vertex_isBoundary[idx];
		}

		const MyVec3& getVertexPos(MyInt idx)const
		{
			return vertex[idx];
		}

		const MyVec3 getTriangleNormal()const
		{
			return triNormal;
		}

		void setVertexPts(const MyVec3& vtxPts0, const vrInt v0Id, 
			const MyVec3& vtxPts1, const vrInt v1Id, 
			const MyVec3& vtxPts2, const vrInt v2Id)
		{
			vertex[0] = vtxPts0;
			vertex[1] = vtxPts1;
			vertex[2] = vtxPts2;

			vertex_id[0] = v0Id;
			vertex_id[1] = v1Id;
			vertex_id[2] = v2Id;
		}

		const MyVec3I& getVertexIds()const
		{
			return vertex_id;
		}

		void computeTriNormal()
		{
			triNormal = compute_UnitNormal(vertex);
			/*triNormal = (vertex_end[1] - vertex_end[0]).cross(vertex_end[2] - vertex_end[0]);
			triNormal.normalize();*/
		}

		void setTriSetType(TriangleSetType type)
		{
			triSetType = type;
		}

		TriangleSetType getTriSetType()const
		{
			return triSetType;
		}

		vrInt getTriangleRegionId()const
		{
			Q_ASSERT(Invalid_Id != m_regionId);
			return m_regionId;
		}
		void setTriangleRegionId(vrInt id)
		{
			Q_ASSERT(Invalid_Id == m_regionId);
			m_regionId = id;
		}

		TriangleSet4Dual()
		{
			vertex_isBoundary[0] = false;
			vertex_isBoundary[1] = false;
			vertex_isBoundary[2] = false;
			m_regionId = Invalid_Id;
		}

		void computeTriElemContinuousType(const std::vector< VertexContinuousType >& vecVertexContinuousType)
		{
			bool isDisContinuousVertex[Geometry::vertexs_per_tri];
			for (int i=0;i<Geometry::vertexs_per_tri;++i)
			{
				isDisContinuousVertex[i] = (Vtx_DisContinuous == vecVertexContinuousType[vertex_id[i]] );
			}

			if ( isDisContinuousVertex[0] || 
				isDisContinuousVertex[1] || 
				isDisContinuousVertex[2] )
			{
				//MyError("triangle is discontinuous.");
				m_TriElemType = DisContinuous;

				if (!isDisContinuousVertex[0] && isDisContinuousVertex[1] && !isDisContinuousVertex[2])
				{
					m_ElemDisContinuousType = dis_1_1;
				}
				else if (!isDisContinuousVertex[0] && !isDisContinuousVertex[1] && isDisContinuousVertex[2])
				{
					m_ElemDisContinuousType = dis_1_2;
				}
				else if (isDisContinuousVertex[0] && !isDisContinuousVertex[1] && !isDisContinuousVertex[2])
				{
					m_ElemDisContinuousType = dis_1_3;
				}
				else if (!isDisContinuousVertex[0] && isDisContinuousVertex[1] && isDisContinuousVertex[2])
				{
					m_ElemDisContinuousType = dis_2_3;
				}
				else if (isDisContinuousVertex[0] && isDisContinuousVertex[1] && !isDisContinuousVertex[2])
				{
					m_ElemDisContinuousType = dis_2_2;
				}
				else if (isDisContinuousVertex[0] && !isDisContinuousVertex[1] && isDisContinuousVertex[2])
				{
					m_ElemDisContinuousType = dis_2_1;
				}
				else if (isDisContinuousVertex[0] && isDisContinuousVertex[1] && isDisContinuousVertex[2])
				{
					m_ElemDisContinuousType = dis_3_3;
				}
				else
				{
					MyError("Un-support discontinuous type!");
				}
			}
			else
			{
				//printf("triangle is Continuous\n");vrPause;
				m_TriElemType = Continuous;
				m_ElemDisContinuousType = dis_regular;
			}

			computeSourcePointInDisContinuousElem(isDisContinuousVertex);
		}

		void reference_to_physical_t3(const MyVec3& gloablCoords_0, const MyVec3& gloablCoords_1, const MyVec3& gloablCoords_2,
			const MyVec2ParamSpace& referenceCoords, MyVec3& phyCoords)
		{
			MyFloat shapeFunction[MyDim];
			shapeFunction[0] = 1.0 - referenceCoords[0] - referenceCoords[1];
			shapeFunction[1] = referenceCoords[0];
			shapeFunction[2] = referenceCoords[1];

			phyCoords = shapeFunction[0] * gloablCoords_0 + shapeFunction[1] * gloablCoords_1 + shapeFunction[2] * gloablCoords_2;

		}


		void computeSourcePointInDisContinuousElem(bool isDisContinuousVertex[])
		{

			static DisContinuousCollectionPoint tmp;
			static const VR::MyFloat s1 = tmp.s1;
			static const VR::MyFloat s2 = tmp.s2;
			MyVec2ParamSpace paramCoordsInDiscontinuous[MyDim];
			paramCoordsInDiscontinuous[0] = MyVec2ParamSpace(s2, s2);
			paramCoordsInDiscontinuous[1] = MyVec2ParamSpace(s1, s2);
			paramCoordsInDiscontinuous[2] = MyVec2ParamSpace(s2, s1);

			for (int v=0;v<Geometry::vertexs_per_tri;++v)
			{
				if (isDisContinuousVertex[v])
				{
					reference_to_physical_t3(vertex[0], vertex[1], vertex[2], paramCoordsInDiscontinuous[v], vertex_sourcePt[v]);
				}
				else
				{
					vertex_sourcePt[v] = vertex[v];
				}
			}

#if 0
			{
				MyVec3 tmpSourcePt;
				std::set< MyVec3, vec3LessCompare > srcPtSet; 
				for (int v=0;v<Geometry::vertexs_per_tri;++v)
				{
					reference_to_physical_t3(vertex[0], vertex[1], vertex[2], paramCoordsInDiscontinuous[v], tmpSourcePt);
					const MyVec3& refPos = tmpSourcePt;
					//printf("012 source point [%d][%f,%f,%f]\n",v, refPos[0], refPos[1], refPos[2]);
					srcPtSet.insert(refPos);
				}
				for (int v=0;v<Geometry::vertexs_per_tri;++v)
				{
					reference_to_physical_t3(vertex[2], vertex[0], vertex[1], paramCoordsInDiscontinuous[v], tmpSourcePt);
					const MyVec3& refPos = tmpSourcePt;
					//printf("201 source point [%d][%f,%f,%f]\n",v, refPos[0], refPos[1], refPos[2]);
					Q_ASSERT(srcPtSet.count(refPos)>0);
				}
				for (int v=0;v<Geometry::vertexs_per_tri;++v)
				{
					reference_to_physical_t3(vertex[1], vertex[2], vertex[0], paramCoordsInDiscontinuous[v], tmpSourcePt);
					const MyVec3& refPos = tmpSourcePt;
					//printf("120 source point [%d][%f,%f,%f]\n",v, refPos[0], refPos[1], refPos[2]);
					Q_ASSERT(srcPtSet.count(refPos)>0);
				}
				//vrPause;
			}
#endif
		}

		MyVec3 getSourcePoint(const vrInt idx)const
		{
			return vertex_sourcePt[idx];
		}

		vrInt get_vertex_id(const vrInt idx)const
		{
			return vertex_id[idx];
		}

	private:
		MyVec3 vertex[Geometry::vertexs_per_tri];
		MyVec3 vertex_sourcePt[Geometry::vertexs_per_tri];
		MyVec3I vertex_id;
		bool   vertex_isBoundary[Geometry::vertexs_per_tri];
		MyVec3 triNormal/*(v1-v0) X (v2-v0)*/;
		TriangleSetType triSetType;//Regular = 0, Positive = 1, Negative = 2

		vrInt m_regionId;
	public:
		TriElemType m_TriElemType;//Continuous = 1, DisContinuous = 2
		DisContinuousType m_ElemDisContinuousType;//dis_1_1=1, dis_1_2=2, dis_1_3=3, dis_2_3=4, dis_2_2=5, dis_2_1=6, dis_3_3=7, dis_regular=8
	};
#else
	//typedef enum{ Continuous, Discontinuous } TriElemType;
	struct TriangleSetWithType
	{
		//           2
		//          / .
		//         /   .
		//        /     .
		//       0-------1


		//      ^
		//    1 | 2
		//      | |.
		//    Y | | .
		//      | |  .
		//    0 | 0---1
		//      +------->
		//        0 X 1

		//param space vertex order
		//vertex 0 :¡¾ 0.0, 0.0¡¿
		//vertex 1 :¡¾ 1.0, 0.0¡¿
		//vertex 2 :¡¾ 0.0, 1.0¡¿
		//notice right hand identity

	public:

		void setVertexBoundary(MyInt idx, bool flag)
		{
			vertex_end_isBoundary[idx] = flag;
		}

		bool getVertexBoundary(MyInt idx)const
		{
			return vertex_end_isBoundary[idx];
		}

		const MyVec3& getEndVertex(MyInt idx)const
		{
			return vertex_end[idx];
		}

		const MyVec3& getCollectPoint(MyInt idx)const
		{
			return vertex_collection[idx];
		}
		const MyVec3 getTriangleNormal()const
		{
			return triNormal;
		}

		TriElemType getTriElemType()const
		{
			return type;
		}


		void setVertexPts(const MyVec3& vtxPts0, const MyVec3& vtxPts1, const MyVec3& vtxPts2)
		{
			vertex_end[0] = vtxPts0;
			vertex_end[1] = vtxPts1;
			vertex_end[2] = vtxPts2;
		}

		void setType(TriElemType _type)
		{
			type = _type;
		}

		void computeTriNormal()
		{
			triNormal = TriangleElemData::compute_UnitNormal(vertex_end);
			/*triNormal = (vertex_end[1] - vertex_end[0]).cross(vertex_end[2] - vertex_end[0]);
			triNormal.normalize();*/
		}


		void computeCollectionPts()
		{

			if (Continuous == type)
			{
				vertex_collection[0] = vertex_end[0];
				vertex_collection[1] = vertex_end[1];
				vertex_collection[2] = vertex_end[2];
			}
			else
			{
				MyVec2ParamSpace paramCoordsInDiscontinuous[MyDim] = { MyVec2ParamSpace(s2, s2), MyVec2ParamSpace(s1, s2), MyVec2ParamSpace(s2,s1 ) };

				reference_to_physical_t3(vertex_end[0], vertex_end[1], vertex_end[2], paramCoordsInDiscontinuous[0], vertex_collection[0]);
				reference_to_physical_t3(vertex_end[0], vertex_end[1], vertex_end[2], paramCoordsInDiscontinuous[1], vertex_collection[1]);
				reference_to_physical_t3(vertex_end[0], vertex_end[1], vertex_end[2], paramCoordsInDiscontinuous[2], vertex_collection[2]);
				/*infoLog << "param coords : " << paramCoordsInDiscontinuous[0].transpose() << " , " << paramCoordsInDiscontinuous[1].transpose() << " , " << paramCoordsInDiscontinuous[2].transpose() << std::endl;
				infoLog << "collect pt : " << vertex_collection[0].transpose() << " , " << vertex_collection[1].transpose() << " , " << vertex_collection[2].transpose() << std::endl;
				vrPause;*/
			}
		}

	private:
		MyVec3 vertex_end[Geometry::vertexs_per_tri];
		bool   vertex_end_isBoundary[Geometry::vertexs_per_tri];
		TriElemType type;
		MyVec3 triNormal;
		MyVec3 vertex_collection[Geometry::vertexs_per_tri];

	};
#endif

	MyFloat vrBEM3D::E = 0.0;
	MyFloat vrBEM3D::mu = 0.0;
	MyFloat vrBEM3D::shearMod = 0.0;

	vrBEM3D::vrBEM3D(const MyFloat _E , const MyFloat _mu ,
		const MyFloat _shearMod  )
	{
		E = (_E);
		mu = (_mu);
		shearMod = (_shearMod);

		/*{
			namespace pt = boost::posix_time;

			pt::ptime current_date_microseconds = pt::microsec_clock::local_time();

			long milliseconds = current_date_microseconds.time_of_day().total_seconds();

			pt::time_duration current_time_milliseconds = pt::milliseconds(milliseconds);

			pt::ptime current_date_milliseconds(current_date_microseconds.date(), 
				current_time_milliseconds);

			std::stringstream ss;
			ss << current_date_microseconds;
			m_str_currentDateTime = ss.str() ;

			std::cout << m_str_currentDateTime << std::endl;vrPause;
		}*/

		{
			// Get current system time
			boost::posix_time::ptime timeLocal = boost::posix_time::second_clock::local_time();

			std::stringstream ss;
			ss << timeLocal.date().year() << "_" << timeLocal.date().month() << "_"<< timeLocal.date().day() << "_" << timeLocal.time_of_day().hours()
				<< "_" << timeLocal.time_of_day().minutes() << "_" << timeLocal.time_of_day().seconds();
			m_str_currentDateTime = ss.str() ;

			std::cout << m_str_currentDateTime << std::endl;vrPause;
		}
		
	}
	vrBEM3D::~vrBEM3D(){}

	int vrBEM3D::initPhysicalSystem(vrLpsz lpszObjName, vrFloat resolution)
	{
#if 1

		m_strMeshFile = std::string(lpszObjName);

		printf("using [%s] material model [%f][%f][%f][%f][%f][%f]\n", GlobalConf::matMdl.c_str(), GlobalConf::youngsMod, GlobalConf::poissonsRatio, 
			GlobalConf::density, GlobalConf::strength, GlobalConf::toughness, GlobalConf::compress); 
		material = createMaterialModel(  GlobalConf::matMdl,
			GlobalConf::youngsMod, GlobalConf::poissonsRatio, GlobalConf::density,
			GlobalConf::strength,  GlobalConf::toughness,     GlobalConf::compress
			);
		//Q_ASSERT(material);

		crackMeshSize      =resolution;
		bemInitialized     =false;
		vdbInitialized     =false;
		fractureInitialized=false;
		lastVDBfile.clear();


		m_strNodeFile = m_strMeshFile;
		m_strNodeFile.append(".nodes");
		infoLog << "m_strNodeFile : " << m_strNodeFile << std::endl;

		m_strElemFile = m_strMeshFile;
		m_strElemFile.append(".elements");
		infoLog << "m_strElemFile : " << m_strElemFile << std::endl;

		m_strRegionFile = m_strMeshFile;
		m_strRegionFile.append(".regions");
		infoLog << "m_strRegionFile : " << m_strRegionFile << std::endl;

		int ret=readModel(m_reader_Elems, m_reader_Regions, m_reader_CrackTips, m_reader_CrackTipParents, m_reader_Nodes,
			TRI, m_reader_elemBodies, LINE, m_reader_bndryBodies);
		infoLog << "File : " << m_strElemFile << " has "<< ret << " elements." << std::endl;
		infoLog << "Node : " << m_reader_Nodes.size() << std::endl;//vrPause;

		//output_id_map(std::cout,"regions",getRegions());vrPause;
		if (GlobalConf::g_n_Obj_remesh > 3)
		{
			RegionHandler regionHandler;
			printf("%% initializing mesh regions from file %s ...\n",m_strRegionFile.c_str());


			if (regionHandler.readRegionDefinitions(m_strRegionFile) > 0)
			{
				regionHandler.assignRegions(getRegions(), getNodes(), getElems());
			}

		}
		//output_id_map(std::cout,"regions",getRegions());vrPause;

		m_ObjMesh_ptr = Geometry::MeshDataStructPtr(new Geometry::MeshDataStruct);

		std::vector< vrGLMVec3 >& points = m_ObjMesh_ptr->points;
		points.clear();
		vrGLMVec3 vertexNode;
		const nodeMap& curNodes = getNodes();
		const vrInt nVertexSize = curNodes.size();
		points.resize(nVertexSize);
		for (iterAllOf(ci,curNodes))
		{
			const vrInt v = (*ci).first;
			const std::vector<VR::MyFloat>& refVtx = (*ci).second;
			vertexNode[0] = refVtx[0];
			vertexNode[1] = refVtx[1];
			vertexNode[2] = refVtx[2];
			points[v-NODE_BASE_INDEX] = vertexNode;
		}

		std::vector<vrVec3I>& facesVec3I = m_ObjMesh_ptr->facesVec3I;
		facesVec3I.clear();
		const elemMap& curElems = getElems();
		const vrInt nTriSize = curElems.size();
		facesVec3I.resize(nTriSize);
		vrVec3I triNode;
		for (iterAllOf(ci,curElems))
		{
			const vrInt t = (*ci).first;
			const std::vector<unsigned int>& refTri = (*ci).second;
			triNode[0] = refTri[0]-NODE_BASE_INDEX;
			triNode[1] = refTri[1]-NODE_BASE_INDEX;
			triNode[2] = refTri[2]-NODE_BASE_INDEX;
			facesVec3I[t-NODE_BASE_INDEX] = triNode;
		}

		std::vector<vrInt>& facesVec3I_group = m_ObjMesh_ptr->facesVec3I_group;
		facesVec3I_group.clear();
		const id_map& regionMap = getRegions();
		const vrInt nRegionSize = regionMap.size();		
		facesVec3I_group.assign(nRegionSize,Invalid_Id);
		Q_ASSERT(nRegionSize == nTriSize);

		int curRegionId;
		for (iterAllOf(ci,regionMap))
		{
			const vrInt r = (*ci).first;
			curRegionId = (*ci).second;//regionMap.at(r);
			facesVec3I_group[r-NODE_BASE_INDEX] = curRegionId;
		}


		infoLog << "make triangle struct." << std::endl;
		std::vector< TriangleSet4Dual > vecTriSetWithType;

#endif
		elemMap nodeSharedElem;
		Q_ASSERT(facesVec3I_group.size() == facesVec3I.size());
		for (int f = 0; f < facesVec3I.size();f++)
		{
			const vrVec3I& curFace = facesVec3I[f];
			TriangleSet4Dual node;
			const vrGLMVec3& point0 = points[curFace[0]];nodeSharedElem[curFace[0]].push_back(f);
			const vrGLMVec3& point1 = points[curFace[1]];nodeSharedElem[curFace[1]].push_back(f);
			const vrGLMVec3& point2 = points[curFace[2]];nodeSharedElem[curFace[2]].push_back(f);
			node.setVertexPts(MyVec3(point0.x, point0.y, point0.z), curFace[0], 
				MyVec3(point1.x, point1.y, point1.z), curFace[1], 
				MyVec3(point2.x, point2.y, point2.z), curFace[2]);

			node.computeTriNormal();


			GlobalConf::boundarycondition_dc;
			const vrInt curTriRegionId = facesVec3I_group[f];
			Q_ASSERT(Invalid_Id != curTriRegionId);

			//printf("curTriRegionId %d boundarycondition_dc[%d]\n",curTriRegionId,GlobalConf::boundarycondition_dc.size());
			if (GlobalConf::boundarycondition_dc.find(curTriRegionId) != GlobalConf::boundarycondition_dc.end())
			{
				for (int v = 0; v < Geometry::vertexs_per_tri;++v)
				{
					node.setVertexBoundary(v, true);
				}
			}
			node.setTriangleRegionId(curTriRegionId);

#if USE_DIS
			if (1 == curTriRegionId || 4 == curTriRegionId || 5 == curTriRegionId || 6 == curTriRegionId || 6 < curTriRegionId)
			{
				node.setTriSetType(TriangleSetType::Regular);
			}
			else if (2 == curTriRegionId)
			{
				const MyVec3& refNormal = node.getTriangleNormal();
				Q_ASSERT(refNormal.y() < 0.0);
				node.setTriSetType(TriangleSetType::Positive);
			}
			else if (3 == curTriRegionId)
			{
				const MyVec3& refNormal = node.getTriangleNormal();
				Q_ASSERT(refNormal.y() > 0.0);
				node.setTriSetType(TriangleSetType::Negative);
			}
			else
			{
				MyError("Triangle Set Error.");
			}
#else
			node.setTriSetType(TriangleSetType::Regular);
			if (8 == curTriRegionId)
			{
				node.setTriSetType(TriangleSetType::Negative);
			}

			if (1 == curTriRegionId)
			{
				node.setTriSetType(TriangleSetType::Positive);
			}
			node.setTriSetType(TriangleSetType::Negative);
#endif


			vecTriSetWithType.push_back(node);
		}

		std::vector< VertexContinuousType > vecVertexContinuousType; vecVertexContinuousType.resize(nVertexSize);
		for (int vtxId=0;vtxId < nVertexSize;++vtxId)
		{
			VertexContinuousType& refType = vecVertexContinuousType[vtxId];
			const std::vector<unsigned int>&  refShareElement = nodeSharedElem.at(vtxId);
			refType = Vtx_Continuous;

#if !USE_DIS
			/*refType = Vtx_DisContinuous;
			continue;*/

			bool flag = true;
			for (int sf=0;flag && sf < refShareElement.size();++sf)
			{
				if (8 == vecTriSetWithType[refShareElement[sf]].getTriangleRegionId() ||
					8 == vecTriSetWithType[refShareElement[sf]].getTriangleRegionId())
				{
					refType = Vtx_DisContinuous;
					flag = false;
				}
			}


#else
			bool flag = true;
			for (int sf=0;flag && sf < refShareElement.size();++sf)
			{

				if (TriangleSetType::Positive == vecTriSetWithType[refShareElement[sf]].getTriSetType() ||
					TriangleSetType::Negative == vecTriSetWithType[refShareElement[sf]].getTriSetType() )
				{
					refType = Vtx_DisContinuous;
					flag = false;
				}
			}
#endif
		}
#if 1

		std::map< int,int > mapCount, mapDisContinuousTypeCount;
		mapCount[TriElemType::Continuous] = 0;
		mapCount[TriElemType::DisContinuous] = 0;

		mapDisContinuousTypeCount[DisContinuousType::dis_1_1 ] = 0;
		mapDisContinuousTypeCount[DisContinuousType::dis_1_2 ] = 0;
		mapDisContinuousTypeCount[DisContinuousType::dis_1_3 ] = 0;
		mapDisContinuousTypeCount[DisContinuousType::dis_2_1 ] = 0;
		mapDisContinuousTypeCount[DisContinuousType::dis_2_2 ] = 0;
		mapDisContinuousTypeCount[DisContinuousType::dis_2_3 ] = 0;
		mapDisContinuousTypeCount[DisContinuousType::dis_3_3 ] = 0;
		mapDisContinuousTypeCount[DisContinuousType::dis_regular ] = 0;
		for (int triId=0;triId < nTriSize;++triId)
		{
			//std::vector< VertexContinuousType > vecVertexContinuousType;
			TriangleSet4Dual& refTriElem =  vecTriSetWithType[triId];
			refTriElem.computeTriElemContinuousType(vecVertexContinuousType);

			mapCount.at(refTriElem.m_TriElemType)++;
			mapDisContinuousTypeCount.at(refTriElem.m_ElemDisContinuousType)++;
		}

		for (iterAllOf(itr, mapCount))
		{
			std::cout << "mapCount : " << (itr->first) << "  " << (itr->second) << std::endl;
		}

		for (iterAllOf(itr,mapDisContinuousTypeCount))
		{
			std::cout << "mapDisContinuousTypeCount : " << (itr->first) << "  " << (itr->second) << std::endl;
		}
		vrPause;
		infoLog << "make triangle element." << std::endl;
		m_vec_vertex_boundary.clear();
		for (int triId=0;triId < nTriSize;++triId)
		{
			const TriangleSet4Dual& node = vecTriSetWithType[triId];
			const TriangleSetType triType = node.getTriSetType(); 
			VertexPtr vtxPtr[Geometry::vertexs_per_tri];
			VertexPtr endVtxPtr[Geometry::vertexs_per_tri];

			for (int v=0;v<Geometry::vertexs_per_tri;++v)
			{
				vtxPtr[v] = Vertex::makeVertex4dualPlus(triType, node.getSourcePoint(v));


				vtxPtr[v]->setContinuousType(vecVertexContinuousType[node.get_vertex_id(v)]);

				endVtxPtr[v] = Vertex::makeVertex4dualPlusEndVtx(node.getVertexPos(v));
			}

			TriangleElemPtr newTriElemPtr = TriangleElem::makeTriangleElem4DualPlus(vtxPtr, endVtxPtr,node.getTriSetType());

			newTriElemPtr->setTriangleRegionId(node.getTriangleRegionId());

			for (int v=0;v<Geometry::vertexs_per_tri;++v)
			{
				if (node.getVertexBoundary(v))
				{
					m_vec_vertex_boundary.push_back( newTriElemPtr->getVertex(v) );
				}
			}

			newTriElemPtr->computeTriElemContinuousType();

			newTriElemPtr->setElemEndPoint(node.getVertexPos(0), node.getVertexPos(1), node.getVertexPos(2));
			newTriElemPtr->setElemVtxPoint(node.getSourcePoint(0), node.getSourcePoint(1), node.getSourcePoint(2));
			newTriElemPtr->setElemNormals(node.getTriangleNormal());
		}

		infoLog << "Vertex::size is " << Vertex::getVertexSize()  << " Triangle Size " << TriangleElem::getTriangleSize() << std::endl;
		vrPause;

		//TriangleElem::CountSurfaceType();
		//compute vertex normal for fundamental solution
		const vrInt nVertexSize_VtxNormal = Vertex::getVertexSize();
		for (vrInt v=0;v<nVertexSize_VtxNormal;++v)
		{
			Vertex::getVertex(v)->computeVertexNormal();
		}

		//compute mirror vertex
		Vertex::searchMirrorVertex_CrackTipVertex();
		Vertex::TestMirrorInfo();

#if !USE_Nagetive_InDebugBeam

		
		for (int triId=0;triId < nTriSize;++triId)
		{
			TriangleElemPtr curTriPtr = TriangleElem::getTriangle(triId);
			if (8 == curTriPtr->getTriangleRegionId())
			{
				for (int v=0;v<Geometry::vertexs_per_tri;++v)
				{
					curTriPtr->getVertex(v)->setVertexTypeInDual(VertexTypeInDual::Mirror_Negative);
					Q_ASSERT(curTriPtr->getVertex(v)->isDisContinuousVertex());
				}
			}

			if (1 == curTriPtr->getTriangleRegionId())
			{
				for (int v=0;v<Geometry::vertexs_per_tri;++v)
				{
					curTriPtr->getVertex(v)->setVertexTypeInDual(VertexTypeInDual::Mirror_Positive);
					Q_ASSERT(curTriPtr->getVertex(v)->isDisContinuousVertex());
				}
			}
		}
#endif//USE_Nagetive_InDebugBeam

		sortVertexConnectedVertexSurface();

		std::set< vrInt > boundary_counter;

		for (iterAllOf(ci,m_vec_vertex_boundary))
		{
			boundary_counter.insert( (*ci)->getId() );
		}

		printf("TriangleElement size %d, vertex size %d, boundary vertex size %d, m_vec_vertex_boundary.size = %d \n", TriangleElem::getTriangleSize(), Vertex::getVertexSize(), boundary_counter.size(), m_vec_vertex_boundary.size());
		vrPause;
		
		infoLog << "createForceBoundaryCondition2d();";
		createForceBoundaryCondition3d();
		infoLog << "distributeDof2d();";
		distributeDof3d();
		infoLog << "createGMatrixAndHMatrixBEM3d();";
#endif

		//TriangleElemData_DisContinuous::TestShapeFunction();vrExit;
		setDoFractrue(true);
		createGMatrixAndHMatrixBEM3d_SST();
		return 0;
	}

#if USE_Mantic_CMat

	vrInt vrBEM3D::sgn(vrFloat val)
	{//bool isEqual(MyFloat var1,MyFloat var2,MyFloat t = EPSINON)
		if (numbers::isEqual(val,0.0,1e-5))
		{
			return 0;
		}
		if (val > 0)
		{
			return 1;
		}
		else if (val <0)
		{
			return -1;
		}
		else
		{
			return 0;
		}
	}

	vrMat3 vrBEM3D::tensor_product(const MyVec3 vec_a, const MyVec3& vec_b)
	{
		vrMat3 retMat;


		for (int i = 0; i < 3;++i)
		{
			for (int j = 0; j < 3;++j)
			{
				retMat.coeffRef(i, j) = vec_a[i] * vec_b[j];
			}
		}
		return retMat;
	}

	void print_map_originId_dstId_set_trianleId(const std::map< std::pair<int, int>, std::set<int> >& map_originId_dstId_set_trianleId)
	{
		for (iterAllOf(ci, map_originId_dstId_set_trianleId))
		{
			auto myPair = (*ci).first;
			auto mySet = (*ci).second;


			std::cout << std::endl;
			for (iterAllOf(ci_in, mySet))
			{
				std::cout << "<< " << myPair.first << " , " << myPair.second << " >> " << *ci_in << std::endl;
			}
			//std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
		}

		//std::cout << "######################################################" << std::endl;
	}

	void vrBEM3D::compute_Guiggiani_CMatrix_for_Test()
	{
		const vrFloat fai_1 = (1.0 / 6.0) * numbers::MyPI;
		const vrFloat fai_2 = (2.0 / 6.0) * numbers::MyPI;
		const vrFloat theta = (62.632 / 180.0) * numbers::MyPI;
		/*const vrFloat fai_1 = (0.0 / 6.0) * numbers::MyPI;
		const vrFloat fai_2 = (3.0 / 6.0) * numbers::MyPI;
		const vrFloat theta = (90.0 / 180.0) * numbers::MyPI;*/

		const vrFloat C = 1.0 / (8*numbers::MyPI * 0.7);
		const vrFloat a = 0.4;
		vrMat3 CMat;
		CMat.coeffRef(0, 0) = C * (( ( ((fai_2 - fai_1) / 2.0) + ((std::sin(2*fai_2)-std::sin(2*fai_1))/4.0))* (2.0 - 3.0 * std::cos(theta) + std::pow(std::cos(theta),3))) + (a * (fai_2-fai_1) * (1 - std::cos(theta))));
		CMat.coeffRef(1, 1) = CMat.coeffRef(0, 0);
		CMat.coeffRef(2, 2) = C * (((fai_2 - fai_1)*(1 - std::pow(std::cos(theta), 3))) + (a * (fai_2 - fai_1) * (1 - std::cos(theta))));
		CMat.coeffRef(0, 1) = CMat.coeffRef(1, 0) = C * ((std::pow(std::sin(fai_2), 2) - std::pow(std::sin(fai_1), 2)) / 2.0) * (2.0 - 3.0 * std::cos(theta) + std::pow(std::cos(theta), 3));
		CMat.coeffRef(0, 2) = CMat.coeffRef(2, 0) = C * (std::sin(fai_2) - std::sin(fai_1)) * (std::pow(std::sin(theta),3));
		CMat.coeffRef(1, 2) = CMat.coeffRef(2, 1) = C * (vrNotice/*std::cos(fai_2) - std::cos(fai_1)*/std::cos(fai_1) - std::cos(fai_2)) * (std::pow(std::sin(theta), 3));

		std::cout << "compute_Guiggiani_CMatrix_for_Test : " << std::endl << CMat << std::endl;
	}

	bool CMatrixIsTrue(const vrMat3& CM)
	{
		for (int i=0;i<MyDim;++i)
		{
			for (int j=0;j<MyDim;++j)
			{
				const vrFloat Cij = CM.coeff(i,j);
				if (!(Cij < 1000.0 && Cij > -1000.0))
				{
					return false;
				}
			}
		}
		return true;
	}

#if 1
	void vrBEM3D::sortVertexConnectedVertexSurface()
	{
		printf("call sortVertexConnectedVertexSurface.\n");
#define DEBUG_sortVertexConnectedVertexSurface (0)
		/*
		1. r(v1) = v1 - x, the length of the vector r is one ? 
		2. the mantic matrix is suit for tangent surface is not along the coordinate see figure 5 in A General Algorithm for Multidimensional Cauchy Principal Value Integrals in the Boundary Element Method
		*/
		//std::ofstream outfile("D:/myDualBEM3D/new_out.txt");
		vrMat3 ContinuousCMatrix;
		ContinuousCMatrix.setZero();
		ContinuousCMatrix.coeffRef(0,0) = 0.5;
		ContinuousCMatrix.coeffRef(1,1) = 0.5;
		ContinuousCMatrix.coeffRef(2,2) = 0.5;
		const MyInt vertexSize = Vertex::getVertexSize();
		for (int v = 0; v < vertexSize; ++v)
		{

			VertexPtr curVtxPtr = Vertex::getVertex(v);

			if (curVtxPtr->isDisContinuousVertex())
			{
				curVtxPtr->setCMatrix(ContinuousCMatrix);
				continue;
			}
			const MyInt originVtxId = curVtxPtr->getId();
			const MyVec3 origin_x = curVtxPtr->getPos();

			std::map< std::pair<int, int>, std::set<int> > map_originId_dstId_set_trianleId;

			const std::vector< TriangleElemPtr >& vecShareElement = curVtxPtr->getShareElement();
			curVtxPtr->clearNearRegion();
			Q_ASSERT(vecShareElement.size()>1);

			//outfile << "vertex id = " << v << "  pos : " << origin_x.transpose() << std::endl;
#if DEBUG_sortVertexConnectedVertexSurface
			std::cout << "vertex id = " << v << "  pos : " << origin_x.transpose() << std::endl;
			std::cout << "vecShareElement.size is " << vecShareElement.size() << std::endl;
#endif
			for (iterAllOf(ci, vecShareElement))
			{
				const TriangleElemPtr triPtr = *ci;
				const MyInt triId = triPtr->getID();
				const MyInt originIndexInTri = triPtr->searchVtxIndexByVtxId(originVtxId);
				Q_ASSERT(Invalid_Id != originIndexInTri);

				{
					//line 1
					VertexPtr vtx_1_Ptr = triPtr->getEndVtxPtr((originIndexInTri + 1) % Geometry::vertexs_per_tri);
					const MyInt vtx_1_id = vtx_1_Ptr->getId();
					map_originId_dstId_set_trianleId[std::make_pair(originVtxId, vtx_1_id)].insert(triId);

					//outfile << (curVtxPtr->getPos()).transpose() << "  " << (vtx_1_Ptr->getPos()).transpose() << "  " << triId << std::endl;
				}
				{
					//line 2
					VertexPtr vtx_2_Ptr = triPtr->getEndVtxPtr((originIndexInTri + 2) % Geometry::vertexs_per_tri);
					const MyInt vtx_2_id = vtx_2_Ptr->getId();
					map_originId_dstId_set_trianleId[std::make_pair(originVtxId, vtx_2_id)].insert(triId);
					//outfile << (curVtxPtr->getPos()).transpose() << "  " << (vtx_2_Ptr->getPos()).transpose() << "  " << triId << std::endl;
				}
			}
#if DEBUG_sortVertexConnectedVertexSurface
			print_map_originId_dstId_set_trianleId(map_originId_dstId_set_trianleId);
#endif
			{

				std::map< vrInt, VertexPtr > mapSortedVertex;
				std::map< std::pair<vrInt, vrInt>, TriangleElemPtr > normal_i_i_plus_1;
				vrInt index_i = 0/*, index_i_plus_1 = 1*/;
				TriangleElemPtr triPtr = *(vecShareElement.begin()); curVtxPtr->addNearRegion(triPtr);
				MyInt currentTriangleId = triPtr->getID();
				MyInt originIndexInCurrentTriangle = triPtr->searchVtxIndexByVtxId(originVtxId);
				Q_ASSERT(Invalid_Id != originIndexInCurrentTriangle);

				MyInt vtx_1_index_InCurrentTriangle = (originIndexInCurrentTriangle + 1) % Geometry::vertexs_per_tri;
				VertexPtr vtx_1_Ptr = triPtr->getEndVtxPtr(vtx_1_index_InCurrentTriangle);
				MyInt vtx_1_id = vtx_1_Ptr->getId();

				mapSortedVertex[index_i] = vtx_1_Ptr;

				std::set<int>& refSet = map_originId_dstId_set_trianleId.at(std::make_pair(originVtxId, vtx_1_id));
				auto itr = refSet.find(currentTriangleId);
				Q_ASSERT(refSet.end() != itr);
				refSet.erase(itr);
				Q_ASSERT(1 == refSet.size());

				VertexPtr nextVtxPtr = triPtr->getEndVtxPtr(index2to1(originIndexInCurrentTriangle, vtx_1_index_InCurrentTriangle));
				MyInt nextVtxId = nextVtxPtr->getId();


				while (!map_originId_dstId_set_trianleId.empty())
				{
#if DEBUG_sortVertexConnectedVertexSurface
					print_map_originId_dstId_set_trianleId(map_originId_dstId_set_trianleId);
#endif
					{

						std::set<int>& refTriSet = map_originId_dstId_set_trianleId.at(std::make_pair(originVtxId, nextVtxId));
						auto eraseItr = refTriSet.find(currentTriangleId);
						Q_ASSERT(eraseItr != refTriSet.end());

						if (map_originId_dstId_set_trianleId.size() > 1 )
						{
							Q_ASSERT( 2 == refTriSet.size());
							refTriSet.erase(eraseItr);
							{
								//compute normal
								normal_i_i_plus_1[std::make_pair(vtx_1_id, nextVtxId)] = triPtr;
							}
							Q_ASSERT(1 == refTriSet.size());
							currentTriangleId = (*refTriSet.begin());

							auto eraseItr_1 = map_originId_dstId_set_trianleId.find(std::make_pair(originVtxId, nextVtxId));
							map_originId_dstId_set_trianleId.erase(eraseItr_1);

							mapSortedVertex[++index_i] = nextVtxPtr;
						}
						else
						{
							Q_ASSERT( 1 == refTriSet.size());
							currentTriangleId = (*refTriSet.begin());

							auto eraseItr_1 = map_originId_dstId_set_trianleId.find(std::make_pair(originVtxId, nextVtxId));
							map_originId_dstId_set_trianleId.erase(eraseItr_1);

							Q_ASSERT(mapSortedVertex.size() == (index_i + 1));
							mapSortedVertex[index_i + 1] = mapSortedVertex[0];
							mapSortedVertex[-1] = mapSortedVertex[index_i];

							{
								//compute normal
								normal_i_i_plus_1[std::make_pair(vtx_1_id, nextVtxId)] = triPtr;
							}
						}

					}
					triPtr = TriangleElem::getTriangle(currentTriangleId);curVtxPtr->addNearRegion(triPtr);
					vtx_1_id = nextVtxId;
					originIndexInCurrentTriangle = triPtr->searchVtxIndexByVtxId(originVtxId);
					vtx_1_index_InCurrentTriangle = triPtr->searchVtxIndexByEndVtxId(nextVtxId);
					Q_ASSERT(Invalid_Id != vtx_1_index_InCurrentTriangle && Invalid_Id != originIndexInCurrentTriangle && originIndexInCurrentTriangle != vtx_1_index_InCurrentTriangle);

					nextVtxPtr = triPtr->getEndVtxPtr(index2to1(originIndexInCurrentTriangle, vtx_1_index_InCurrentTriangle));
					nextVtxId = nextVtxPtr->getId();
				}

				std::map< vrInt, MyVec3 > map_ri;

				const vrInt n = mapSortedVertex.size() - 2;

				for (iterAllOf(ci, mapSortedVertex))
				{
					const vrInt vv  = (*ci).first;
					const MyVec3 vi = (*ci).second->getPos();
					//r(vi) = vi - x
					map_ri[vv] = (vi - origin_x);
					map_ri[vv].normalize(); vrNotice;
				}

				vrFloat value_0 = 0.0;
				vrMat3  value_1; value_1.setZero();
				for (int vv = 0; vv < n;++vv)
				{
					const vrInt i = vv;
					const vrInt vid_i = mapSortedVertex.at(i)->getId();
					const vrInt vid_iplus1 = mapSortedVertex.at(i + 1)->getId();
					const vrInt vid_isub1 = mapSortedVertex.at(i - 1)->getId();

					const MyVec3& n_isub1_i = normal_i_i_plus_1.at(std::make_pair(/*i - 1*/vid_isub1,/* i*/vid_i))->getElemNormals();
					const MyVec3& n_i_iplus1 = normal_i_i_plus_1.at(std::make_pair(/* i*/vid_i, /*i + 1*/vid_iplus1))->getElemNormals();
					const MyVec3& r_i		 = map_ri.at(i);
					const MyVec3& r_iplus1	 = map_ri.at(i+1);

					const vrInt curSGN = sgn((n_isub1_i.cross(n_i_iplus1)).dot(r_i));
					if (0 != curSGN)
					{
						value_0 += curSGN * std::acos(n_isub1_i.dot(n_i_iplus1));
					}


					value_1 += tensor_product((r_iplus1 - r_i).cross(n_i_iplus1), n_i_iplus1);
					//printf("value_0[%f] value_1[%f]\n",value_0,value_1);
				}

				vrMat3 Cmatrix = (1.0 / (4.0 * numbers::MyPI)) * (2 * numbers::MyPI + value_0) * vrMat3::Identity();
				Cmatrix -= (1.0 / (8.0*numbers::MyPI * (1 - mu))) * value_1;


				Q_ASSERT(curVtxPtr->getNearRegionSize() == curVtxPtr->getShareElement().size());

				//outfile << Cmatrix << std::endl;
#if DEBUG_sortVertexConnectedVertexSurface
				std::cout << Cmatrix << std::endl;
#endif


				if (!CMatrixIsTrue(Cmatrix))
				{
					printf("Cmatrix is error %d\n",curVtxPtr->getId());
					std::cout << "pos : " << (curVtxPtr->getPos().transpose()) << std::endl;
					vrPause;
					Cmatrix.setZero();
					Cmatrix.coeffRef(0,0) = 0.5;
					Cmatrix.coeffRef(1,1) = 0.5;
					Cmatrix.coeffRef(2,2) = 0.5;
				}
				curVtxPtr->setCMatrix(Cmatrix);

			}
		}
		//outfile.close();
	}
#else

	void vrBEM3D::sortVertexConnectedVertexSurface()
	{
#define DEBUG_sortVertexConnectedVertexSurface (0)
		/*
		1. r(v1) = v1 - x, the length of the vector r is one ? 
		2. the mantic matrix is suit for tangent surface is not along the coordinate see figure 5 in A General Algorithm for Multidimensional Cauchy Principal Value Integrals in the Boundary Element Method
		*/
		//std::ofstream outfile("D:/myDualBEM3D/out.txt");
		vrMat3 ContinuousCMatrix;
		ContinuousCMatrix.setZero();
		ContinuousCMatrix.coeffRef(0,0) = 0.5;
		ContinuousCMatrix.coeffRef(1,1) = 0.5;
		ContinuousCMatrix.coeffRef(2,2) = 0.5;
		const MyInt vertexSize = Vertex::getVertexSize();
		for (int v = 0; v < vertexSize; ++v)
		{

			VertexPtr curVtxPtr = Vertex::getVertex(v);

			if (curVtxPtr->isDisContinuousVertex())
			{
				curVtxPtr->setCMatrix(ContinuousCMatrix);
				continue;
			}
			const MyInt originVtxId = curVtxPtr->getId();
			const MyVec3 origin_x = curVtxPtr->getPos();

			std::map< std::pair<int, int>, std::set<int> > map_originId_dstId_set_trianleId;

			const std::vector< TriangleElemPtr >& vecShareElement = curVtxPtr->getShareElement();
			curVtxPtr->clearNearRegion();
			Q_ASSERT(vecShareElement.size()>1);

			//outfile << "vertex id = " << v << "  pos : " << origin_x.transpose() << std::endl;
#if DEBUG_sortVertexConnectedVertexSurface
			std::cout << "vertex id = " << v << "  pos : " << origin_x.transpose() << std::endl;
			std::cout << "vecShareElement.size is " << vecShareElement.size() << std::endl;
#endif
			for (iterAllOf(ci, vecShareElement))
			{
				const TriangleElemPtr triPtr = *ci;
				const MyInt triId = triPtr->getID();
				const MyInt originIndexInTri = triPtr->searchVtxIndexByVtxId(originVtxId);
				Q_ASSERT(Invalid_Id != originIndexInTri);

				{
					//line 1
					VertexPtr vtx_1_Ptr = triPtr->getVertex((originIndexInTri + 1) % Geometry::vertexs_per_tri);
					const MyInt vtx_1_id = vtx_1_Ptr->getId();
					map_originId_dstId_set_trianleId[std::make_pair(originVtxId, vtx_1_id)].insert(triId);

					//outfile << (curVtxPtr->getPos()).transpose() << "  " << (vtx_1_Ptr->getPos()).transpose() << "  " << triId << std::endl;
				}
				{
					//line 2
					VertexPtr vtx_2_Ptr = triPtr->getVertex((originIndexInTri + 2) % Geometry::vertexs_per_tri);
					const MyInt vtx_2_id = vtx_2_Ptr->getId();
					map_originId_dstId_set_trianleId[std::make_pair(originVtxId, vtx_2_id)].insert(triId);
					//outfile << (curVtxPtr->getPos()).transpose() << "  " << (vtx_2_Ptr->getPos()).transpose() << "  " << triId << std::endl;
				}
			}
#if DEBUG_sortVertexConnectedVertexSurface
			print_map_originId_dstId_set_trianleId(map_originId_dstId_set_trianleId);
#endif
			{

				std::map< vrInt, VertexPtr > mapSortedVertex;
				std::map< std::pair<vrInt, vrInt>, TriangleElemPtr > normal_i_i_plus_1;
				vrInt index_i = 0/*, index_i_plus_1 = 1*/;
				TriangleElemPtr triPtr = *(vecShareElement.begin()); curVtxPtr->addNearRegion(triPtr);
				MyInt currentTriangleId = triPtr->getID();
				MyInt originIndexInCurrentTriangle = triPtr->searchVtxIndexByVtxId(originVtxId);
				Q_ASSERT(Invalid_Id != originIndexInCurrentTriangle);

				MyInt vtx_1_index_InCurrentTriangle = (originIndexInCurrentTriangle + 1) % Geometry::vertexs_per_tri;
				VertexPtr vtx_1_Ptr = triPtr->getVertex(vtx_1_index_InCurrentTriangle);
				MyInt vtx_1_id = vtx_1_Ptr->getId();

				mapSortedVertex[index_i] = vtx_1_Ptr;

				std::set<int>& refSet = map_originId_dstId_set_trianleId.at(std::make_pair(originVtxId, vtx_1_id));
				auto itr = refSet.find(currentTriangleId);
				Q_ASSERT(refSet.end() != itr);
				refSet.erase(itr);
				Q_ASSERT(1 == refSet.size());

				VertexPtr nextVtxPtr = triPtr->getVertex(index2to1(originIndexInCurrentTriangle, vtx_1_index_InCurrentTriangle));
				MyInt nextVtxId = nextVtxPtr->getId();


				while (!map_originId_dstId_set_trianleId.empty())
				{
#if DEBUG_sortVertexConnectedVertexSurface
					print_map_originId_dstId_set_trianleId(map_originId_dstId_set_trianleId);
#endif
					{

						std::set<int>& refTriSet = map_originId_dstId_set_trianleId.at(std::make_pair(originVtxId, nextVtxId));
						auto eraseItr = refTriSet.find(currentTriangleId);
						Q_ASSERT(eraseItr != refTriSet.end());

						if (map_originId_dstId_set_trianleId.size() > 1 )
						{
							Q_ASSERT( 2 == refTriSet.size());
							refTriSet.erase(eraseItr);
							{
								//compute normal
								normal_i_i_plus_1[std::make_pair(vtx_1_id, nextVtxId)] = triPtr;
							}
							Q_ASSERT(1 == refTriSet.size());
							currentTriangleId = (*refTriSet.begin());

							auto eraseItr_1 = map_originId_dstId_set_trianleId.find(std::make_pair(originVtxId, nextVtxId));
							map_originId_dstId_set_trianleId.erase(eraseItr_1);

							mapSortedVertex[++index_i] = nextVtxPtr;
						}
						else
						{
							Q_ASSERT( 1 == refTriSet.size());
							currentTriangleId = (*refTriSet.begin());

							auto eraseItr_1 = map_originId_dstId_set_trianleId.find(std::make_pair(originVtxId, nextVtxId));
							map_originId_dstId_set_trianleId.erase(eraseItr_1);

							Q_ASSERT(mapSortedVertex.size() == (index_i + 1));
							mapSortedVertex[index_i + 1] = mapSortedVertex[0];
							mapSortedVertex[-1] = mapSortedVertex[index_i];

							{
								//compute normal
								normal_i_i_plus_1[std::make_pair(vtx_1_id, nextVtxId)] = triPtr;
							}
						}

					}
					triPtr = TriangleElem::getTriangle(currentTriangleId);curVtxPtr->addNearRegion(triPtr);
					vtx_1_id = nextVtxId;
					originIndexInCurrentTriangle = triPtr->searchVtxIndexByVtxId(originVtxId);
					vtx_1_index_InCurrentTriangle = triPtr->searchVtxIndexByVtxId(nextVtxId);
					Q_ASSERT(Invalid_Id != vtx_1_index_InCurrentTriangle && Invalid_Id != originIndexInCurrentTriangle && originIndexInCurrentTriangle != vtx_1_index_InCurrentTriangle);

					nextVtxPtr = triPtr->getVertex(index2to1(originIndexInCurrentTriangle, vtx_1_index_InCurrentTriangle));
					nextVtxId = nextVtxPtr->getId();
				}

				std::map< vrInt, MyVec3 > map_ri;

				const vrInt n = mapSortedVertex.size() - 2;

				for (iterAllOf(ci, mapSortedVertex))
				{
					const vrInt vv  = (*ci).first;
					const MyVec3 vi = (*ci).second->getPos();
					//r(vi) = vi - x
					map_ri[vv] = (vi - origin_x);
					map_ri[vv].normalize(); vrNotice;
				}

				vrFloat value_0 = 0.0;
				vrMat3  value_1; value_1.setZero();
				for (int vv = 0; vv < n;++vv)
				{
					const vrInt i = vv;
					const vrInt vid_i = mapSortedVertex.at(i)->getId();
					const vrInt vid_iplus1 = mapSortedVertex.at(i + 1)->getId();
					const vrInt vid_isub1 = mapSortedVertex.at(i - 1)->getId();

					const MyVec3& n_isub1_i = normal_i_i_plus_1.at(std::make_pair(/*i - 1*/vid_isub1,/* i*/vid_i))->getElemNormals();
					const MyVec3& n_i_iplus1 = normal_i_i_plus_1.at(std::make_pair(/* i*/vid_i, /*i + 1*/vid_iplus1))->getElemNormals();
					const MyVec3& r_i		 = map_ri.at(i);
					const MyVec3& r_iplus1	 = map_ri.at(i+1);

					const vrInt curSGN = sgn((n_isub1_i.cross(n_i_iplus1)).dot(r_i));
					if (0 != curSGN)
					{
						value_0 += curSGN * std::acos(n_isub1_i.dot(n_i_iplus1));
					}


					value_1 += tensor_product((r_iplus1 - r_i).cross(n_i_iplus1), n_i_iplus1);
					//printf("value_0[%f] value_1[%f]\n",value_0,value_1);
				}

				vrMat3 Cmatrix = (1.0 / (4.0 * numbers::MyPI)) * (2 * numbers::MyPI + value_0) * vrMat3::Identity();
				Cmatrix -= (1.0 / (8.0*numbers::MyPI * (1 - mu))) * value_1;


				Q_ASSERT(curVtxPtr->getNearRegionSize() == curVtxPtr->getShareElement().size());

				//outfile << Cmatrix << std::endl;
#if DEBUG_sortVertexConnectedVertexSurface
				std::cout << Cmatrix << std::endl;
#endif


				if (!CMatrixIsTrue(Cmatrix))
				{
					printf("Cmatrix is error %d\n",curVtxPtr->getId());
					std::cout << "pos : " << (curVtxPtr->getPos().transpose()) << std::endl;
					vrPause;
					Cmatrix.setZero();
					Cmatrix.coeffRef(0,0) = 0.5;
					Cmatrix.coeffRef(1,1) = 0.5;
					Cmatrix.coeffRef(2,2) = 0.5;
				}
				curVtxPtr->setCMatrix(Cmatrix);

			}
		}
	}
#endif
#endif

	bool vrBEM3D::isVertexInElement(const VertexPtr curVtxPtr, const TriangleElemPtr curElementPtr, vrInt& index)
	{
		//1. element has vertex
		index = curElementPtr->searchVtxIndexByVtxId(curVtxPtr->getId());

		//2. vertex shared element
		bool  vtxSharedElem = curVtxPtr->isSharedElementWithId(curElementPtr->getID());

		Q_ASSERT( vtxSharedElem == (index != Invalid_Id) );

		return vtxSharedElem;
	}

	void vrBEM3D::createGMatrixAndHMatrixBEM3d()
	{

	}

	void vrBEM3D::makeRigidH()
	{
		const MyInt nDofs = getDofs();
		MyMatrix& currentH = m_Hsubmatrix;
		//rigid H
		const MyInt nVtx = Vertex::getVertexSize();

		for (int row_v = 0; row_v < nVtx; ++row_v)
		{
			MyFloat sum = 0.0;
			//for each vertex
			const MyVec3I& rowDofs = Vertex::getVertex(row_v)->getDofs();
			for (int d = 0; d < MyDim; ++d)
			{
				//for each dimension
				const MyInt curRowId = rowDofs[d];
				for (int col_v = 0; col_v < nVtx; ++col_v)
				{
					if (col_v != row_v)
					{
						const MyInt curColId = Vertex::getVertex(col_v)->getDofs()[d];
						sum += currentH.coeff(curRowId, curColId);
					}
				}

				currentH.coeffRef(curRowId, curRowId) = -1.0* sum;
			}
		}
	}

	void vrBEM3D::createForceBoundaryCondition3d()
	{
#if 1
		std::map< vrInt, std::set< vrInt > > map_regionId_mapVertexId;
		const MyInt nTriSize = TriangleElem::getTriangleSize();
		for (MyInt c = 0; c < nTriSize; ++c)
		{
			TriangleElemPtr curTriPtr = TriangleElem::getTriangle(c);
			const vrInt regionId = curTriPtr->getTriangleRegionId();
			if (GlobalConf::boundarycondition_nm.find(regionId)!= GlobalConf::boundarycondition_nm.end())
			{
				map_regionId_mapVertexId[regionId].insert(curTriPtr->getVertex(0)->getId());
				map_regionId_mapVertexId[regionId].insert(curTriPtr->getVertex(1)->getId());
				map_regionId_mapVertexId[regionId].insert(curTriPtr->getVertex(2)->getId());
			}
		}

		for (iterAllOf(ci,map_regionId_mapVertexId))
		{
			std::vector< VertexPtr >& refVecForceVertex = map_regionId_vecVertex[(*ci).first];
			refVecForceVertex.clear();

			const std::set< vrInt >& refMapForceVtxId = (*ci).second;

			for (iterAllOf(mm,refMapForceVtxId))
			{
				refVecForceVertex.push_back(Vertex::getVertex(*mm));
			}
		}

		for (iterAllOf(ci,map_regionId_vecVertex))
		{
			printf("Region [%d] Force Boundary [%d]\n",(*ci).first, (*ci).second.size());
			/*for (iterAllOf(cci,(*ci).second))
			{
			printf("force source point id [%d]\n",(*cci)->getId());
			}*/
		}
		vrPause;
#endif
	}

	void vrBEM3D::distributeDof3d()
	{
#if USE_DebugGMatrix
		m_nGlobalDof = Geometry::first_dof_idx;

		for (iterAllOf(itr,map_regionId_vecVertex))
		{
			printf("distributeDof3d [%d]\n", (*itr).first );
			const std::vector< VertexPtr >& refVec = (*itr).second;

			for (iterAllOf(citr, refVec))
			{
				VertexPtr curVtxPtr = *citr;
				if (!(curVtxPtr->isValidDofs()))
				{
					if (0 == m_nGlobalDof)
					{
						m_vtx_id.insert(curVtxPtr->getId());
					}
					if (76 == m_nGlobalDof ||
						76 == (m_nGlobalDof+1) ||
						76 == (m_nGlobalDof+2) )
					{
						m_vtx_id.insert(curVtxPtr->getId());
					}
					curVtxPtr->setDof(m_nGlobalDof, m_nGlobalDof + 1, m_nGlobalDof + 2);
					m_nGlobalDof += MyDim;
				}
			}
		}
		const MyInt nTriSize = TriangleElem::getTriangleSize();
		for (MyInt c = 0; c < nTriSize; ++c)
		{
			TriangleElemPtr curTriPtr = TriangleElem::getTriangle(c);
			for (MyInt v = 0; v < Geometry::vertexs_per_tri; ++v)
			{
				VertexPtr curVtxPtr = curTriPtr->getVertex(v);
				if (!(curVtxPtr->isValidDofs()))
				{
					/*if (969 == m_nGlobalDof ||
						969 == (m_nGlobalDof+1) ||
						969 == (m_nGlobalDof+2) )
					{
						m_vtx_id.insert(curVtxPtr->getId());
					}*/
					curVtxPtr->setDof(m_nGlobalDof, m_nGlobalDof + 1, m_nGlobalDof + 2);
					m_nGlobalDof += MyDim;
				}
			}
		}
		m_displacement.resize(getDofs());m_displacement.setZero();
#else//USE_DebugGMatrix
		m_nGlobalDof = Geometry::first_dof_idx;
		const MyInt nTriSize = TriangleElem::getTriangleSize();
		for (MyInt c = 0; c < nTriSize; ++c)
		{
			TriangleElemPtr curTriPtr = TriangleElem::getTriangle(c);
			for (MyInt v = 0; v < Geometry::vertexs_per_tri; ++v)
			{
				VertexPtr curVtxPtr = curTriPtr->getVertex(v);
				if (!(curVtxPtr->isValidDofs()))
				{
					curVtxPtr->setDof(m_nGlobalDof, m_nGlobalDof + 1, m_nGlobalDof + 2);
					m_nGlobalDof += MyDim;
				}
			}
		}
		m_displacement.resize(getDofs());m_displacement.setZero();
#endif//USE_DebugGMatrix


		printf("global dof %d.\n", m_nGlobalDof);vrPause;
	}

	bool vrBEM3D::isForceCondition_up(const MyVec3& pos)
	{
		MyError("isForceCondition_up");
		return false;
		/*if (pos[GlobalConf::g_n_Obj_boundaryAxis] > (0.2) && pos[1] > (0.1))
		{
		return true;
		}
		else
		{
		return false;
		}*/
	}

	bool vrBEM3D::isForceCondition_down(const MyVec3& pos)
	{
		MyError("isForceCondition_down");
		return false;
		/*if (pos[GlobalConf::g_n_Obj_boundaryAxis] > (0.2) && pos[1] < (-0.1))
		{
		return true;
		}
		else
		{
		return false;
		}*/
	}

	bool vrBEM3D::isDCCondition(const MyVec3& pos)
	{
		MyError("isDCCondition");
		return false;
		/*if (pos[GlobalConf::g_n_Obj_boundaryAxis] < (-0.48))
		{
		return true;
		}
		else
		{
		return false;
		}*/
	}

	void Draw_Axes(void)
	{
		float ORG[3] = { 0, 0, 0 };

		float XP[3] = { 1, 0, 0 }, XN[3] = { -1, 0, 0 },
			YP[3] = { 0, 1, 0 }, YN[3] = { 0, -1, 0 },
			ZP[3] = { 0, 0, 1 }, ZN[3] = { 0, 0, -1 };

		glPushMatrix();
		glLineWidth(2.0);

		glBegin(GL_LINES);
		glColor3f(1, 0, 0); // X axis is red.
		glVertex3fv(ORG);
		glVertex3fv(XP);
		glColor3f(0, 1, 0); // Y axis is green.
		glVertex3fv(ORG);
		glVertex3fv(YP);
		glColor3f(0, 0, 1); // z axis is blue.
		glVertex3fv(ORG);
		glVertex3fv(ZP);
		glEnd();

		glPopMatrix();
	}

	

	void vrBEM3D::renderScene() const
	{
		static GLuint axes_list;
		static bool b_first_axis = true;
		if (b_first_axis)
		{
			/* Create a display list for drawing axes */
			axes_list = glGenLists(1);
			glNewList(axes_list, GL_COMPILE);
			const vrFloat axis_end = 2.0;
			const vrFloat axis_end_1 = 1.0;
			glColor4ub(0, 0, 255, 255);
			glBegin(GL_LINE_STRIP);
			glVertex3f(0.0f, 0.0f, 0.0f);
			glVertex3f(axis_end, 0.0f, 0.0f);
			glVertex3f(0.75f+axis_end_1, 0.25f, 0.0f);
			glVertex3f(0.75f+axis_end_1, -0.25f, 0.0f);
			glVertex3f(axis_end, 0.0f, 0.0f);
			glVertex3f(0.75f+axis_end_1, 0.0f, 0.25f);
			glVertex3f(0.75f+axis_end_1, 0.0f, -0.25f);
			glVertex3f(axis_end, 0.0f, 0.0f);
			glEnd();
			glBegin(GL_LINE_STRIP);
			glVertex3f(0.0f, 0.0f, 0.0f);
			glVertex3f(0.0f, axis_end, 0.0f);
			glVertex3f(0.0f, 0.75f+axis_end_1, 0.25f);
			glVertex3f(0.0f, 0.75f+axis_end_1, -0.25f);
			glVertex3f(0.0f, axis_end, 0.0f);
			glVertex3f(0.25f, 0.75f+axis_end_1, 0.0f);
			glVertex3f(-0.25f, 0.75f+axis_end_1, 0.0f);
			glVertex3f(0.0f, axis_end, 0.0f);
			glEnd();
			glBegin(GL_LINE_STRIP);
			glVertex3f(0.0f, 0.0f, 0.0f);
			glVertex3f(0.0f, 0.0f, axis_end);
			glVertex3f(0.25f, 0.0f, 0.75f+axis_end_1);
			glVertex3f(-0.25f, 0.0f, 0.75f+axis_end_1);
			glVertex3f(0.0f, 0.0f, axis_end);
			glVertex3f(0.0f, 0.25f, 0.75f+axis_end_1);
			glVertex3f(0.0f, -0.25f, 0.75f+axis_end_1);
			glVertex3f(0.0f, 0.0f, axis_end);
			glEnd();
			
			glColor4ub(255, 255, 0, 255);
			glRasterPos3f(axis_end*1.1, 0.0f, 0.0f);

			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, 'x');
			glRasterPos3f(0.0f, axis_end*1.1, 0.0f);
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, 'y');
			glRasterPos3f(0.0f, 0.0f, axis_end*1.1);
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, 'z');

			glEndList();
		}

		glPushMatrix();
		glCallList(axes_list);
		glPopMatrix();

#if DoublePrecision
		#define GLVertex3DF glVertex3d
#else//DoublePrecision
		#define GLVertex3DF glVertex3f
#endif//DoublePrecision
		//g_BEM3D->onlyUpdateExternalForceAndSolve();
		const std::vector< vrGLMVec3 >& points = m_ObjMesh_ptr->points;

		const vrFloat maxStress = Vertex::getMaxStress();
		const vrFloat minStress = Vertex::getMinStress();
		MyDenseVector curVtxColor;
		glBegin(GL_TRIANGLES);

		const std::vector< TriangleElemPtr >& sharedVec = TriangleElem::getTriangleVector();
		vrVec3 elemDisp_DisContinuous[3];
		vrVec3 elemDisp_EndVtx[3];

#if USE_Aliabadi_RegularSample

		std::set< MyInt > boundaryDofsSet;
		for (iterAllOf(ci, m_vec_vertex_boundary))
		{
			const VertexPtr  curVtxPtr = *ci;
			MyVec3I dofs = curVtxPtr->getDofs();
			for (int i = 0; i < dofs.size(); ++i)
			{
				boundaryDofsSet.insert(dofs[i]);
			}
		}
#endif//USE_Aliabadi_RegularSample

		for (iterAllOf(ci, sharedVec))
		{
			TriangleElemPtr ptr = *ci;
			//if (/*ptr->isDiscontinuous() &&*/ /*Mirror_Positive == ptr->getTriSetType()*/ 202 == ptr->getID())
			{
				for (int i = 0; i < MyDim;++i)
				{
					const MyVec3I dofs = ptr->getVertex(i)->getDofs();
					if (Iterator::has_elt(boundaryDofsSet, dofs[0]))
					{
						elemDisp_DisContinuous[i] = vrVec3( 0.0 , 0.0, 0.0);
					}
					else
					{
						elemDisp_DisContinuous[i] = vrVec3( m_displacement[dofs[0]], m_displacement[dofs[1]], m_displacement[dofs[2]]);
					}
				}

				for (int i = 0; i < MyDim;++i)
				{
					const MyVec3& vtx = ptr->getEndVtxPtr(i)->getPos();
					elemDisp_EndVtx[i] = ptr->get_m_data_SST_3D().interpolation_displacement(i, elemDisp_DisContinuous);
					
					glVertex3d(vtx[0] + elemDisp_EndVtx[i][0], vtx[1] + elemDisp_EndVtx[i][1], vtx[2] + elemDisp_EndVtx[i][2]);
					//GLVertex3DF(vtx[0] , vtx[1] , vtx[2] );
				}

			}
		}
		glEnd();

		
		//vrExit;
		glLineWidth(2.0);
		glColor3f(0,0,0);
		glBegin(GL_LINES);
		int linePair[MyDim][2] = { { 0, 1 }, { 1, 2 }, { 2, 0 } };
		const std::vector< TriangleElemPtr >& vec_triElem = TriangleElem::getTriangleVector();
		for (iterAllOf(ci, vec_triElem))
		{
			TriangleElemPtr ptr = *ci;
			for (int i = 0; i < MyDim; ++i)
			{
				for (int j = 0; j < 2;++j)
				{
					const MyVec3& vtx = ptr->getVertex(linePair[i][j])->getPos();
					const MyVec3I dofs = ptr->getVertex(linePair[i][j])->getDofs();
					GLVertex3DF(vtx[0] + m_displacement[dofs[0]], vtx[1] + m_displacement[dofs[1]], vtx[2] + m_displacement[dofs[2]]);
					//GLVertex3DF(vtx[0], vtx[1], vtx[2]);
				}
			}
		}
		glEnd();

		glLineWidth(2.0);
		glColor3f(1,0,0);
		glBegin(GL_LINES);
		const int nVtxSize = Vertex::getVertexSize();
		for (int v=0;v<nVtxSize;++v)
		{
			VertexPtr vtxPtr = Vertex::getVertex(v);
			const int dofs = vtxPtr->getDofs().y();
			vrFloat val = m_rhs[dofs] * 1.0;
			/*if (val < 1 && val > -1 )
			{
				val = 0.0;
			}*/
			const MyVec3& vtx = vtxPtr->getPos();
			const MyVec3& n_x = vtxPtr->getVertexNormal();
			const MyVec3 pos_0 = n_x * val * 10000 + vtx;
	//		GLVertex3DF(vtx[0], vtx[1], vtx[2]);
	//		GLVertex3DF(pos_0[0], pos_0[1], pos_0[2]);
		}
		glEnd();


		glColor3f(1,0.5,0);
		glPointSize(10.0);
		glBegin(GL_POINTS);

		for (iterAllOf(ci, m_vec_vertex_boundary))
		{
			VertexPtr vtxPtr = (*ci);

			const MyVec3& vtx = vtxPtr->getPos();
			const MyVec3I& dofs = vtxPtr->getDofs();
			GLVertex3DF(vtx[0] + m_displacement[dofs[0]], vtx[1] + m_displacement[dofs[1]], vtx[2] + m_displacement[dofs[2]]);
		}
		glEnd();

		glColor3f(1,0,1);
		glPointSize(10.0);
		glBegin(GL_POINTS);

		for (iterAllOf(ni,map_regionId_vecVertex))
		{
			for (iterAllOf(ci, (*ni).second))
			{
				VertexPtr vtxPtr = (*ci);

				//if (319 == vtxPtr->getId())
				{
					const MyVec3& vtx = vtxPtr->getPos();
					const MyVec3I& dofs = vtxPtr->getDofs();
					//printf("vtx 319 (%f,%f,%f)\n",vtx[0],vtx[1],vtx[2]);
					GLVertex3DF(vtx[0] + m_displacement[dofs[0]], vtx[1] + m_displacement[dofs[1]], vtx[2] + m_displacement[dofs[2]]);
				}

			}
		}

		glEnd();

#if USE_DebugGMatrix
		glColor3f(0.56,0,0.5);
		glPointSize(15.0);
		glBegin(GL_POINTS);

		//const int nVtxSize = Vertex::getVertexSize();
		//for (int v=0;v<nVtxSize;++v)
		//{
		//	VertexPtr curVtxPtr = Vertex::getVertex(v);
		//	const MyVec3I& refDofs = curVtxPtr->getDofs();
		//	if ( (m_nouse_set_dof.count(refDofs[0])>0)||
		//		 (m_nouse_set_dof.count(refDofs[1])>0)||
		//		 (m_nouse_set_dof.count(refDofs[2])>0) )
		//	{
		//		const MyVec3& vtx = curVtxPtr->getPos();
		//		const MyVec3I& dofs = curVtxPtr->getDofs();
		//		//printf("vtx 319 (%f,%f,%f)\n",vtx[0],vtx[1],vtx[2]);
		//		glVertex3d(vtx[0] + m_displacement[dofs[0]], vtx[1] + m_displacement[dofs[1]], vtx[2] + m_displacement[dofs[2]]);
		//	}
		//}
		//glColor3f(0.56,0.5,0.5);
		for (iterAllOf(itr,m_vtx_id))
		{
			VertexPtr curVtxPtr = Vertex::getVertex(*itr);
			const MyVec3& vtx = curVtxPtr->getPos();
			const MyVec3I& dofs = curVtxPtr->getDofs();
			//printf("vtx 319 (%f,%f,%f)\n",vtx[0],vtx[1],vtx[2]);
			GLVertex3DF(vtx[0] + m_displacement[dofs[0]], vtx[1] + m_displacement[dofs[1]], vtx[2] + m_displacement[dofs[2]]);
		}
		glEnd();
#endif//USE_DebugGMatrix

	}

	MyVector vrBEM3D::GaussElimination(const MyMatrix& K, MyVector& b)
	{
		const MyInt n = K.rows();

		MyMatrix AK(n, n + 1); AK.setZero();
		AK.block(0, 0, n, n) = K;
		AK.block(0, n, n, 1) = b;
		for (MyInt i = 0; i < n; i++) {
			// Search for maximum in this column
			MyFloat maxEl = abs(AK.coeff(i, i)/*[i][i]*/);
			MyInt maxRow = i;
			for (MyInt k = i + 1; k<n; k++) {
				if (abs(AK.coeff(k, i)/*[k][i]*/) > maxEl) {
					maxEl = abs(AK.coeff(k, i)/*[k][i]*/);
					maxRow = k;
				}
			}

			// Swap maximum row with current row (column by column)
			for (MyInt k = i; k < n + 1; k++) {
				MyFloat tmp = AK.coeff(maxRow, k);//A[maxRow][k];
				/*A[maxRow][k]*/AK.coeffRef(maxRow, k) = AK.coeff(i, k);//A[i][k];
				/*A[i][k]*/AK.coeffRef(i, k) = tmp;
			}

			// Make all rows below this one 0 in current column
			for (MyInt k = i + 1; k < n; k++) {
				MyFloat c = -AK.coeff(k, i) / AK.coeff(i, i);/*A[k][i]/A[i][i]*/;
				for (MyInt j = i; j < n + 1; j++) {
					if (i == j) {
						/*A[k][j]*/ AK.coeffRef(k, j) = 0;
					}
					else {
						/*A[k][j]*/AK.coeffRef(k, j) += c * AK.coeff(i, j) /*A[i][j]*/;
					}
				}
			}
		}

		// Solve equation Ax=b for an upper triangular matrix A
		MyVector x(n);
		for (MyInt i = n - 1; i >= 0; i--) {
			x[i] = AK.coeff(i, n) / AK.coeff(i, i);//A[i][n]/A[i][i];
			for (MyInt k = i - 1; k >= 0; k--) {
				/*A[k][n]*/AK.coeffRef(k, n) -= /*A[k][i]*/AK.coeff(k, i) * x[i];
			}
		}
		return x;
	}

#if USE_TBB

#if USE_DUAL
	void vrBEM3D::Parallel_AssembleSystem_DualEquation()
	{
		const MyInt nPts = Vertex::getVertexSize();
		const vrInt nSize = nPts*MyDim*MyDim;
		parallel_for(blocked_range<size_t>(0, nSize), TBB::AssembleSystem_DualEquation(this, nSize), auto_partitioner());

		//MyVector row = m_Gsubmatrix.row(1);
		
		/*for (int v=0;v<nPts;++v)
		{
			VertexPtr vtxPtr = Vertex::getVertex(v);
			const MyVec3I dofs = vtxPtr->getDofs();
			if (1 == dofs[1])
			{
				for (int idx_i=0;idx_i<MyDim;++idx_i)
				{
					for (int idx_j=0;idx_j<MyDim;++idx_j)
					{
						AssembleSystem_DisContinuous_DualEquation_Aliabadi_Nouse(v,idx_i,idx_j);
						++g_atomix_count;
						std::cout << "Processing " << g_atomix_count << " of " << nSize << std::endl;
					}
				}
			}

		}

		printf("dofs %d  colidx %d \n",getDofs(), m_set_colIdx.size());
		for (int i=0;i<getDofs();++i)
		{
			if (m_set_colIdx.count(i) == 0)
			{
				printf("miss col %d \n",i);
			}
		}
		vrExit;*/
	}





	vrFloat vrBEM3D::compute_K_ij_I_k_SST(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D, const vrInt idx_i, const vrInt idx_j, const  vrInt idx_I)
	{

		MyError("vrBEM3D::compute_K_ij_I_k_SST.");
		vrFloat retVal = 0.0;
#if 0
		MyNotice;//using ETA space
		//
		vrFloat doubleLayer_Term_k = 0.0, singleLayer_Term_k = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();
		const MyVec3& unitNormal_srcPt = refDataSST3D.unitNormal_fieldPt;MyNotice
			const vrInt n_gpts = TriangleElem::GaussPointSize_xi_In_Theta * TriangleElem::GaussPointSize_xi_In_Rho;
		Q_ASSERT(n_gpts == refDataSST3D.m_gaussQuadrature_xi_eta_polar.rows());

		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);

		for (int idx_k=0; idx_k < MyDim; ++idx_k)
		{
			const vrFloat delta_jk = TriangleElemData::delta_ij(idx_j,idx_k);
			const vrFloat delta_ik = TriangleElemData::delta_ij(idx_i,idx_k);

			for (int index_theta=0,idx_g=0;index_theta<TriangleElem::GaussPointSize_xi_In_Theta;++index_theta)
			{
				const vrFloat cur_theta_singlelayer = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer.row(index_theta)[0];
				const vrFloat curWeight_singleLayer = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer.row(index_theta)[1];

#if 1 //SST
				const vrFloat A = refDataSST3D.A_theta(cur_theta_singlelayer);
				const vrFloat N_I_0 = refDataSST3D.N_I_0_eta(idx_I);
				//const vrFloat N_I_1 = refDataSST3D.N_I_1_eta(idx_I,cur_theta_singlelayer);
				const MyVec3& n_x = refDataSST3D.unitNormal_fieldPt;
				const vrFloat jacob_eta = refDataSST3D.Jacobi_eta;
				MyVec3 sst_dr;
				vrFloat sst_drdn = 0.0;
				for (int m=0;m<MyDim;++m)
				{
					sst_dr[m] = refDataSST3D.r_i(m,cur_theta_singlelayer); 
					sst_drdn += (sst_dr[m]*n_x[m]);
				}

				const vrFloat M0 = (
					(1.0-2.0*mu)*(delta_ij * sst_dr[idx_k] + delta_jk * sst_dr[idx_i] MyNotice - delta_ik * sst_dr[idx_j]) + 
					3.0 * sst_dr[idx_i] * sst_dr[idx_j] * sst_dr[idx_k]
				) * unitNormal_srcPt[idx_k];

				const vrFloat M1 = ((1.0) / (8.0*numbers::MyPI*(1-mu))) * M0 * jacob_eta;

				const vrFloat F_1_ij_I_k = (M1*N_I_0)/(A*A);
#endif
				for (int index_rho=0;index_rho<TriangleElem::GaussPointSize_xi_In_Rho;++index_rho,++idx_g)
				{
					MyNotice;/*the order of theta and rho,(theta,rho)*/
					auto curRows = refDataSST3D.m_gaussQuadrature_xi_eta_polar.row(idx_g);
					MyVec2ParamSpace theta_rho;
					theta_rho[0] = curRows[0];theta_rho[1]=curRows[1];
					const vrFloat cur_rho = theta_rho[1];
					const vrFloat curWeight_doubleLayer = curRows[2];
					Q_ASSERT(numbers::isEqual(theta_rho[0], refDataSST3D.m_gaussQuadrature_xi_eta_polar.row(idx_g).x()));
					Q_ASSERT(numbers::isEqual(theta_rho[1], refDataSST3D.m_gaussQuadrature_xi_eta_polar.row(idx_g).y()));
					Q_ASSERT(numbers::isEqual(cur_theta_singlelayer,theta_rho[0]));

					const MyVec2ParamSpace cur_eta = refDataSST3D.pc2eta(refDataSST3D.m_SrcPt_in_eta MYNOTICE,theta_rho);
					const MyVec2ParamSpace cur_xi = refDataSST3D.eta2xi(cur_eta);
					const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

					MyFloat jacob_eta_nouse;/*jacob_eta = jacob_xi * mat_T_Inv*/
					MyVec3 unitNormals_fieldPt;
					MyFloat r;
					MyVec3 dr;
					MyFloat drdn;
					getKernelParameters_3D_SST(srcPos,fieldPoint,refDataSST3D,jacob_eta_nouse,unitNormals_fieldPt,r,dr,drdn);

					const MyVec3& unitNormals_srcPt = unitNormals_fieldPt;MyNotice;
					Q_ASSERT(numbers::isEqual(jacob_eta, jacob_eta_nouse));
					Q_ASSERT(numbers::isEqual(r,cur_rho*A));

					const vrFloat Kij_k = get_Kij_SST_3D_k(idx_i, idx_j, idx_k, r, dr, drdn, unitNormals_fieldPt, unitNormals_srcPt);
					const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);

					const vrFloat SingularTerm_Kij_I_k = Kij_k * N_I * jacob_eta * theta_rho[TriangleElem::idx_rho];
					const vrFloat SingularTerm_F_1_ij_I_k = (1.0/cur_rho)*(F_1_ij_I_k);
					doubleLayer_Term_k += (SingularTerm_Kij_I_k - SingularTerm_F_1_ij_I_k) * curWeight_doubleLayer;

				}//for (int index_rho=0;index_rho<TriangleElem::GaussPointSize_xi_In_Rho;++index_rho,++idx_g)

				const vrFloat beta = 1.0 / A;
				const vrFloat cur_Rho_hat = refDataSST3D.rho_hat(cur_theta_singlelayer);
				singleLayer_Term_k += F_1_ij_I_k * log( abs(cur_Rho_hat/beta) ) * curWeight_singleLayer;

			}//for (int index_theta=0,idx_g=0;index_theta<TriangleElem::GaussPointSize_xi_In_Theta;++index_theta)
			retVal += (doubleLayer_Term_k + singleLayer_Term_k);
		}//for (int idx_k=0; idx_k < MyDim; ++idx_k) 

#endif
		return retVal;
	}

	vrFloat vrBEM3D::compute_K_ij_I(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D, const vrInt idx_i, const vrInt idx_j, const  vrInt idx_I)
	{
		/*MyError("vrBEM3D::compute_K_ij_I.");
		return 0.0;*/
#if 0


		//MyError("vrBEM3D::compute_K_ij_I.");
		vrFloat retVal = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();

		const MyVec3& unitNormals_srcPt = curSourcePtr->getVertexNormal();MyNotice;
#if SPEEDUP_5_31
		int tmp_GaussPointSize_xi_In_Theta = 0;
		int tmp_GaussPointSize_xi_In_Rho = 0;
		if (dis_regular == refDataSST3D.m_DisContinuousType)
		{
			tmp_GaussPointSize_xi_In_Theta = TriangleElem::GaussPointSize_xi_In_Theta;
			tmp_GaussPointSize_xi_In_Rho = TriangleElem::GaussPointSize_xi_In_Rho;
		}
		else
		{
			tmp_GaussPointSize_xi_In_Theta = TriangleElem::GaussPointSize_xi_In_Theta_DisContinuous;
			tmp_GaussPointSize_xi_In_Rho = TriangleElem::GaussPointSize_xi_In_Rho_DisContinuous;
		}
		const vrInt nGaussPointSize_xi_In_Theta = tmp_GaussPointSize_xi_In_Theta;
		const vrInt nGaussPointSize_xi_In_Rho = tmp_GaussPointSize_xi_In_Rho;
#endif
		const vrInt n_gpts = nGaussPointSize_xi_In_Theta * nGaussPointSize_xi_In_Rho;
		Q_ASSERT(n_gpts == refDataSST3D.m_gaussQuadrature_xi_polar.rows());

		for (int idx_k=0; idx_k < MyDim; ++idx_k)
		{
			for (int idx_g=0;idx_g < n_gpts;++idx_g)
			{
				MyNotice;/*the order of theta and rho,(theta,rho)*/
				auto curRows = refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g);
				MyVec2ParamSpace theta_rho;
				theta_rho[TriangleElem::idx_theta_doubleLayer] = curRows[TriangleElem::idx_theta_doubleLayer];
				theta_rho[TriangleElem::idx_rho_doubleLayer]=curRows[TriangleElem::idx_rho_doubleLayer];
				const vrFloat curWeight = curRows[TriangleElem::idx_weight_doubleLayer];
				Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_theta_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g).x()));
				Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_rho_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g).y()));

				const MyVec2ParamSpace currentSrcPtInParam /*in xi space*/ = refDataSST3D.m_SrcPt_in_xi;
				Q_ASSERT( ((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) /*compute_T_ij_I*/);
				if (!((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) )
				{
					printf("compute_T_ij_I : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
				}
				const MyVec2ParamSpace cur_xi = refDataSST3D.pc2xi( currentSrcPtInParam MYNOTICE,theta_rho);
				const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

				MyFloat jacob_xi;
				MyVec3 unitNormals_fieldPt;
				MyFloat r;
				MyVec3 dr;
				MyFloat drdn;
				getKernelParameters_3D(srcPos,fieldPoint,refDataSST3D,jacob_xi,unitNormals_fieldPt,r,dr,drdn);


				const vrFloat Kij = get_Kij_SST_3D_k(idx_i, idx_j, idx_k, r, dr, drdn, unitNormals_fieldPt, unitNormals_srcPt);
				const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);
				retVal += Kij * N_I * jacob_xi * theta_rho[TriangleElem::idx_rho_doubleLayer] * curWeight;
			}
		}

		return retVal;
#endif
	}

	

	

	vrFloat vrBEM3D::compute_H_ij_I(const VertexPtr curSourcePtr, const TriangleElemData& curTriElemData, const vrInt idx_i, const vrInt idx_j, const  vrInt idx_I)
	{
		return compute_S_ij_I(curSourcePtr, curTriElemData, idx_i, idx_j, idx_I);
	}

	vrFloat vrBEM3D::compute_S_ij_I(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D, const vrInt idx_i, const vrInt idx_j, const  vrInt idx_I)
	{
		/*MyError("vrBEM3D::compute_S_ij_I.");
		return 0.0;*/
#if 0
		//MyError("vrBEM3D::compute_S_ij_I.");
		vrFloat retVal = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();

		const MyVec3& unitNormals_srcPt = curSourcePtr->getVertexNormal();MyNotice;
#if SPEEDUP_5_31
		int tmp_GaussPointSize_xi_In_Theta = 0;
		int tmp_GaussPointSize_xi_In_Rho = 0;
		if (dis_regular == refDataSST3D.m_DisContinuousType)
		{
			tmp_GaussPointSize_xi_In_Theta = TriangleElem::GaussPointSize_xi_In_Theta;
			tmp_GaussPointSize_xi_In_Rho = TriangleElem::GaussPointSize_xi_In_Rho;
		}
		else
		{
			tmp_GaussPointSize_xi_In_Theta = TriangleElem::GaussPointSize_xi_In_Theta_DisContinuous;
			tmp_GaussPointSize_xi_In_Rho = TriangleElem::GaussPointSize_xi_In_Rho_DisContinuous;
		}
		const vrInt nGaussPointSize_xi_In_Theta = tmp_GaussPointSize_xi_In_Theta;
		const vrInt nGaussPointSize_xi_In_Rho = tmp_GaussPointSize_xi_In_Rho;
#endif
		const vrInt n_gpts = nGaussPointSize_xi_In_Theta * nGaussPointSize_xi_In_Rho;
		Q_ASSERT(n_gpts == refDataSST3D.m_gaussQuadrature_xi_polar.rows());

		for (int idx_k=0; idx_k < MyDim; ++idx_k)
		{
			for (int idx_g=0;idx_g < n_gpts;++idx_g)
			{
				MyNotice;/*the order of theta and rho,(theta,rho)*/
				auto curRows = refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g);
				MyVec2ParamSpace theta_rho;
				theta_rho[TriangleElem::idx_theta_doubleLayer] = curRows[TriangleElem::idx_theta_doubleLayer];
				theta_rho[TriangleElem::idx_rho_doubleLayer]=curRows[TriangleElem::idx_rho_doubleLayer];
				const vrFloat curWeight = curRows[TriangleElem::idx_weight_doubleLayer];
				Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_theta_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g).x()));
				Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_rho_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g).y()));

				const MyVec2ParamSpace currentSrcPtInParam /*in xi space*/ = refDataSST3D.m_SrcPt_in_xi;
				Q_ASSERT( ((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) /*compute_T_ij_I*/);
				if (!((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) )
				{
					printf("compute_S_ij_I : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
				}
				const MyVec2ParamSpace cur_xi = refDataSST3D.pc2xi(currentSrcPtInParam MYNOTICE, theta_rho);
				const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

				MyFloat jacob_xi;
				MyVec3 unitNormals_fieldPt;
				MyFloat r;
				MyVec3 dr;
				MyFloat drdn;
				getKernelParameters_3D(srcPos,fieldPoint,refDataSST3D,jacob_xi,unitNormals_fieldPt,r,dr,drdn);



				const vrFloat Sij_k = get_Sij_SST_3D_k(idx_i, idx_j, idx_k, r, dr, drdn, unitNormals_fieldPt, unitNormals_srcPt);
				const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);
				retVal += Sij_k * N_I * jacob_xi * theta_rho[TriangleElem::idx_rho_doubleLayer] * curWeight;

			}//for (int idx_g=0;idx_g < n_gpts;++idx_g)
		}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
		return retVal;
#endif
	}

	vrFloat vrBEM3D::compute_H_ij_I_k_SST(const VertexPtr curSourcePtr, const TriangleElemData& curTriElemData, const vrInt idx_i, const vrInt idx_j, const  vrInt idx_I)
	{
		return compute_S_ij_I_k_SST( curSourcePtr, curTriElemData, idx_i, idx_j, idx_I);
	}

	vrFloat vrBEM3D::compute_S_ij_I_k_SST(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D, const vrInt idx_i, const vrInt idx_j, const  vrInt idx_I)
	{
		MyError("vrBEM3D::compute_S_ij_I_k_SST.");
		vrFloat retVal = 0.0;
#if 0

		MyNotice;//using ETA space
		//
		vrFloat doubleLayer_Term_k = 0.0, singleLayer_Term_k = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();
		const MyVec3& unitNormal_srcPt = refDataSST3D.unitNormal_fieldPt;MyNotice
			const vrInt n_gpts = TriangleElem::GaussPointSize_xi_In_Theta * TriangleElem::GaussPointSize_xi_In_Rho;
		Q_ASSERT(n_gpts == refDataSST3D.m_gaussQuadrature_xi_eta_polar.rows());

		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);

		for (int idx_k=0; idx_k < MyDim; ++idx_k)
		{
			const vrFloat delta_jk = TriangleElemData::delta_ij(idx_j,idx_k);
			const vrFloat delta_ik = TriangleElemData::delta_ij(idx_i,idx_k);

			for (int index_theta=0,idx_g=0;index_theta<TriangleElem::GaussPointSize_xi_In_Theta;++index_theta)
			{
				const vrFloat cur_theta_singlelayer = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer.row(index_theta)[0];
				const vrFloat curWeight_singleLayer = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer.row(index_theta)[1];

#if 1 // SST
				const vrFloat A = refDataSST3D.A_theta(cur_theta_singlelayer);
				const vrFloat N_I_0 = refDataSST3D.N_I_0_eta(idx_I);
				const vrFloat N_I_1 = refDataSST3D.N_I_1_eta(idx_I,cur_theta_singlelayer);
				const MyVec3& n_x = refDataSST3D.unitNormal_fieldPt;
				const vrFloat jacob_eta = refDataSST3D.Jacobi_eta;
				MyVec3 sst_dr;
				vrFloat sst_drdn = 0.0;
				for (int m=0;m<MyDim;++m)
				{
					sst_dr[m] = refDataSST3D.r_i(m,cur_theta_singlelayer); 
					sst_drdn += (sst_dr[m]*n_x[m]);
				}

				const vrFloat M0 = (
					3.0 * sst_drdn * ( (1.0-2.0*mu)*delta_ik*sst_dr[idx_j] + mu*(delta_ij*sst_dr[idx_k]+delta_jk*sst_dr[idx_i]) MyNotice - 5.0*sst_dr[idx_i]*sst_dr[idx_j]*sst_dr[idx_k] ) +
					3.0 * mu * (n_x[idx_i]*sst_dr[idx_j]*sst_dr[idx_k]+n_x[idx_k]*sst_dr[idx_i]*sst_dr[idx_j]) MyNotice - 
					(1.0-4.0*mu) * delta_ik * n_x[idx_j] + 
					(1.0-2.0*mu) * (3.0 * n_x[idx_j] * sst_dr[idx_i] * sst_dr[idx_k] + delta_ij * n_x[idx_k] + delta_jk * n_x[idx_i])
					) * unitNormal_srcPt[idx_k];

				const vrFloat M1 = ( (shearMod)/(4.0*numbers::MyPI*(1.0-mu)) ) * M0 * jacob_eta;

				const vrFloat F_2_ij_I_k = (M1*N_I_0)/(A*A*A);
				const vrFloat F_1_ij_I_k = (M1*N_I_1)/(A*A);
#endif
				for (int index_rho=0;index_rho<TriangleElem::GaussPointSize_xi_In_Rho;++index_rho,++idx_g)
				{
					MyNotice;/*the order of theta and rho,(theta,rho)*/
					auto curRows = refDataSST3D.m_gaussQuadrature_xi_eta_polar.row(idx_g);
					MyVec2ParamSpace theta_rho;
					theta_rho[0] = curRows[0];theta_rho[1]=curRows[1];
					const vrFloat cur_rho = theta_rho[1];
					const vrFloat curWeight_doubleLayer = curRows[2];
					Q_ASSERT(numbers::isEqual(theta_rho[0], refDataSST3D.m_gaussQuadrature_xi_eta_polar.row(idx_g).x()));
					Q_ASSERT(numbers::isEqual(theta_rho[1], refDataSST3D.m_gaussQuadrature_xi_eta_polar.row(idx_g).y()));
					Q_ASSERT(numbers::isEqual(cur_theta_singlelayer,theta_rho[0]));

					const MyVec2ParamSpace cur_eta = refDataSST3D.pc2eta(refDataSST3D.m_SrcPt_in_eta MYNOTICE,theta_rho);
					const MyVec2ParamSpace cur_xi = refDataSST3D.eta2xi(cur_eta);
					const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

					MyFloat jacob_eta_nouse;/*jacob_eta = jacob_xi * mat_T_Inv*/
					MyVec3 unitNormals_fieldPt;
					MyFloat r;
					MyVec3 dr;
					MyFloat drdn;
					getKernelParameters_3D_SST(srcPos,fieldPoint,refDataSST3D,jacob_eta_nouse,unitNormals_fieldPt,r,dr,drdn);

					const MyVec3& unitNormals_srcPt = unitNormals_fieldPt;MyNotice;
					Q_ASSERT(numbers::isEqual(jacob_eta, jacob_eta_nouse));
					Q_ASSERT(numbers::isEqual(r,cur_rho*A));

					const vrFloat Sij_k = get_Sij_SST_3D_k(idx_i, idx_j, idx_k, r, dr, drdn, unitNormals_fieldPt, unitNormals_srcPt);
					const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);

					const vrFloat SingularTerm_Sij_I_k = Sij_k * N_I * jacob_eta * theta_rho[TriangleElem::idx_rho];

					const vrFloat SingularTerm_F_1_ij_I_k = (1.0/(cur_rho))*(F_1_ij_I_k);
					const vrFloat SingularTerm_F_2_ij_I_k = (1.0/(cur_rho*cur_rho)) * (F_2_ij_I_k);

					doubleLayer_Term_k += (SingularTerm_Sij_I_k - (SingularTerm_F_1_ij_I_k + SingularTerm_F_2_ij_I_k)) * curWeight_doubleLayer;

				}//for (int index_rho=0;index_rho<TriangleElem::GaussPointSize_xi_In_Rho;++index_rho,++idx_g)

				const vrFloat beta = 1.0 / A;
				const vrFloat cur_Rho_hat = refDataSST3D.rho_hat(cur_theta_singlelayer);
				singleLayer_Term_k += ( (F_1_ij_I_k * log( abs(cur_Rho_hat/beta) )) MyNotice - (F_2_ij_I_k * (1.0/(cur_Rho_hat))) )* curWeight_singleLayer;


			}//for (int index_theta=0,idx_g=0;index_theta<TriangleElem::GaussPointSize_xi_In_Theta;++index_theta)
			retVal += (doubleLayer_Term_k + singleLayer_Term_k);
		}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
		return retVal;
#endif
	}

#endif//USE_DUAL

	


#endif//USE_TBB

	void vrBEM3D::resizeSystemMatrix(const vrInt nDofs)
	{
		infoLog << "nDofs " << nDofs << std::endl;
		m_Hsubmatrix.resize(nDofs, nDofs); m_Hsubmatrix.setZero();
		m_Gsubmatrix.resize(nDofs, nDofs); m_Gsubmatrix.setZero();
		m_A_matrix.resize(nDofs, nDofs); m_A_matrix.setZero();
		m_rhs.resize(nDofs); m_rhs.setZero();
	}

	void vrBEM3D::HG2A(const vrInt nDofs)
	{
		vrFloat trace_H = 0.0;
		vrFloat trace_G = 0.0;

#if Output_Matrix_Displacement

		
		{
			const int nRows = m_Hsubmatrix.rows();
			const int nCols = m_Hsubmatrix.cols();
			std::stringstream ss;
			ss << "d:/m_Hsubmatrix_360_mi_[" << m_str_currentDateTime << "]_" << nRows <<".txt";
			std::ofstream outfileH(ss.str().c_str());
			for (int r=0;r<nRows;++r)
			{
				for (int c=0;c<nCols;++c)
				{
					outfileH << r << " " << c << " " << std::setprecision(10)<< m_Hsubmatrix.coeff(r,c) << std::endl;
				}
			}
			outfileH.close();
		}

		{
			const int nRows = m_Gsubmatrix.rows();
			const int nCols = m_Gsubmatrix.cols();
			std::stringstream ss;
			ss << "d:/m_m_Gsubmatrix_360_mi_[" << m_str_currentDateTime << "]_"  << nRows <<".txt";
			std::ofstream outfileG(ss.str().c_str());
			for (int r=0;r<nRows;++r)
			{
				for (int c=0;c<nCols;++c)
				{
					outfileG << r << " " << c << " " << std::setprecision(10) << m_Gsubmatrix.coeff(r,c) << std::endl;
				}
			}
			outfileG.close();
		}

#endif//Output_Matrix_Displacement
		for (int row = 0; row < nDofs; ++row)
		{
			trace_H += m_Hsubmatrix.coeff(row, row);			
			trace_G += m_Gsubmatrix.coeff(row, row);
		}
		const vrFloat SF = abs(trace_H / trace_G);
		printf("SF(%f) trace_H(%f) trace_G(%f)\n",SF,trace_H,trace_G);
		Q_ASSERT(!isOutofRange(SF));

		std::set< MyInt > boundaryDofsSet;

#if USE_Mantic_CMat
		for (iterAllOf(ci, m_vec_vertex_boundary))
		{
			const VertexPtr  curVtxPtr = *ci;
			MyVec3I dofs = curVtxPtr->getDofs();
			for (int i = 0; i < dofs.size(); ++i)
			{
				boundaryDofsSet.insert(dofs[i]);
			}
		}

#endif
		//move displacement unknow
#if USE_Aliabadi_RegularSample
		//boundaryDofsSet.clear();
#endif

		for (int d = 0; d < nDofs; ++d)
		{
			if (!Iterator::has_elt(boundaryDofsSet, d))
			{
				MyNotice;//notice is col, not row
				m_A_matrix.col(d) = m_Hsubmatrix.col(d);
			}
		}

		for (iterAllOf(ci, boundaryDofsSet))
		{
			const MyInt traceId = *ci;
			m_A_matrix.col(traceId) = m_Gsubmatrix.col(traceId) * SF * -1.0;
		}
	}

	void trimString( vrString & str ) 
	{
		vrLpsz whiteSpace = " \t\n\r";
		vrSizt_t location;
		location = str.find_first_not_of(whiteSpace);
		str.erase(0,location);
		location = str.find_last_not_of(whiteSpace);
		str.erase(location + 1);
	}

	void vrBEM3D::createGMatrixAndHMatrixBEM3d_SST()
	{

		if (!isDoFractrue())
		{
			return ;
		}

		if (m_fractureStep >= m_maxFractureStep)
		{
			MyError("m_fractureStep >= m_maxFractureStep");

		}
		else
		{
			m_fractureStep++;
		}
		setDoFractrue(false);
		//

		const MyInt nDofs = getDofs();
		resizeSystemMatrix(nDofs);


		const MyInt ne = TriangleElem::getTriangleSize();
		for (int element = 0; element < ne; ++element)
		{
			infoLog << "element " << element << " of " << ne;
			TriangleElem::getTriangle(element)->compute_Shape_Deris_Jacobi_SST_3D();//jacobi error, is right! 3-17
		}
		infoLog << "generating matrix...." << std::endl;

#if 0&&LoadMatrixData
		std::ifstream infile("D:/gpu_result.txt");
		//std::ifstream infile("D:/myDualBEM3D/Release_SST_Debug/DebugData/displacement_360_peng.txt");
		
		vrString line;
		vrFloat curData;
		getline( infile, line );
		int nIdx = 0;
		while (!infile.eof())
		{
			trimString(line);
			istringstream lineStream( line );
			lineStream >> curData;
			//printf("curData = [%f]\n",curData);
			m_displacement[nIdx] = curData;
			nIdx++;
			getline( infile, line );
		}

		Q_ASSERT(nIdx == getDofs());
		return ;
#endif//LoadMatrixData
		//return ;

#if USE_TBB

#if LoadMatrixData
		{
			GlobalConf::g_str_Obj_DebugDisplacement;
			GlobalConf::g_str_Obj_DebugGsubmatrix;
			GlobalConf::g_str_Obj_DebugHsubmatrix;
			std::ifstream infile_Hsubmatrix(GlobalConf::g_str_Obj_DebugHsubmatrix);
			vrInt nRow,nCol;
			vrFloat val;
			vrInt nCount = 0;
			while (!infile_Hsubmatrix.eof())
			{
				++nCount;
				infile_Hsubmatrix >> nRow >> nCol >> val;
				m_Hsubmatrix.coeffRef(nRow, nCol) = val;
			}
			
			printf("%d == (%d)\n",nCount, nDofs*nDofs);
			Q_ASSERT(nCount == (nDofs*nDofs));


			std::ifstream infile_Gsubmatrix(GlobalConf::g_str_Obj_DebugGsubmatrix);
			//vrInt nRow,nCol;
			//vrFloat val;
			nCount = 0;
			while (!infile_Gsubmatrix.eof())
			{
				++nCount;
				infile_Gsubmatrix >> nRow >> nCol >> val;
				m_Gsubmatrix.coeffRef(nRow, nCol) = val;
			}
			printf("%d == (%d)\n",nCount, nDofs*nDofs);
			Q_ASSERT(nCount == (nDofs*nDofs));
		}
		

#else//LoadMatrixData
		Parallel_AssembleSystem_DualEquation();
#endif//LoadMatrixData
		
		//Parallel_AssembleSystem();
#endif

		HG2A(nDofs);
		//make right hand side
		applyBoundaryCondition(nDofs);

		applyForceCondition(nDofs);

		solve();

#if Output_Matrix_Displacement
		std::stringstream ss;
		ss << "d:/displacement_360_mi_[" << m_str_currentDateTime << "]_" << m_displacement.rows() <<".txt";

		std::ofstream outfile(ss.str().c_str());
		outfile << m_displacement << std::endl;
		outfile.close();
#endif//Output_Matrix_Displacement
		//generateSurfaceStresses();

	}

	void vrBEM3D::onlyUpdateExternalForceAndSolve()
	{
		if (!isDoFractrue())
		{
			return ;
		}
		setDoFractrue(false);

		static vrFloat s_Scale = 1.0;
		s_Scale *= 1.2;
		printf("applyForceCondition \n");
		applyForceCondition(getDofs(),s_Scale);
		printf("solve \n");
		solve();
		printf("generateSurfaceStresses \n");
		generateSurfaceStresses();
		printf("onlyUpdateExternalForceAndSolve end \n");
	}

	void vrBEM3D::applyBoundaryCondition(const vrInt nDofs)
	{
		MyVector dirichletVals(nDofs);
		dirichletVals.setZero();

#if USE_Mantic_CMat
		for (iterAllOf(ci, m_vec_vertex_boundary))
		{
			const MyVec3I& dofs = (*ci)->getDofs();
			for (int v = 0; v < dofs.size(); ++v)
			{
				dirichletVals[dofs[v]] = 0.0;
			}
		}
#endif
		/*MyVector tmpVec = m_Hsubmatrix * dirichletVals;
		
		{
			std::stringstream ss;
			ss << "d:/m_Hsubmatrix_dirichletVals_[" << m_str_currentDateTime << "]_" << m_displacement.rows() <<".txt";

			std::ofstream outfile(ss.str().c_str());
			outfile << tmpVec << std::endl;
			outfile.close();
		}*/

		m_rhs = m_rhs - m_Hsubmatrix * dirichletVals;
		//printf("m_Hsubmatrix * dirichletVals norm(%f) sum(%f) m_rhs.sum(%f)\n",tmpVec.norm(), tmpVec.colwise().sum(),m_rhs.colwise().sum());
	}

	struct sortNode
	{
		vrFloat val;
		vrInt dofs;
	};

	class sortNode_LessCompare
	{
	public:
		bool operator()(const sortNode& lhs,const sortNode& rhs)const
		{
			return lhs.val < rhs.val;
		}
	};

	void vrBEM3D::applyForceCondition(const vrInt nDofs, const vrFloat scale)
	{
		MyVector externalForceVals(nDofs);
		externalForceVals.setZero();


		GlobalConf::boundarycondition_nm;
		map_regionId_vecVertex;
		for (iterAllOf(ci,map_regionId_vecVertex))
		{
			const vrInt regionId = (*ci).first;
			const MyVec3& unitForce = GlobalConf::boundarycondition_nm.at(regionId);

			for (iterAllOf(ni, (*ci).second))
			{
				MyVec3I dofs = (*ni)->getDofs();
				externalForceVals[dofs[0]] = unitForce[0] * scale;
				externalForceVals[dofs[1]] = unitForce[1] * scale;
				externalForceVals[dofs[2]] = unitForce[2] * scale;
			}
		}

		Q_ASSERT(numbers::isZero(m_rhs.norm()));
		m_rhs = m_rhs + m_Gsubmatrix * externalForceVals;

#if 1
		/*const int ne = TriangleElem::getTriangleSize();
		for (int t = 0; t<ne;++t)
		{
			TriangleElemPtr triPtr = TriangleElem::getTriangle(t);
			const int regionId = triPtr->getTriangleRegionId();
			if (6 == regionId)
			{
				for (int v=0;v<Geometry::vertexs_per_tri;++v)
				{
					const MyVec3I& refDofs = triPtr->getVertex(v)->getDofs();
					m_rhs[refDofs[0]] = 0.0 * m_rhs[refDofs[0]];
					m_rhs[refDofs[1]] = 0.0 * m_rhs[refDofs[1]];
					m_rhs[refDofs[2]] = 0.0 * m_rhs[refDofs[2]];
				}
			}
		}*/

		MyVector tmpVec = m_Gsubmatrix * externalForceVals;
		MyMatrix tmpMat; tmpMat.resize(tmpVec.rows(),5);
		tmpMat.col(0) = m_rhs;
		tmpMat.col(1) = tmpVec;
		tmpMat.col(2) = externalForceVals;

		{
			const MyVector& refRow = m_Gsubmatrix.row(1);
			tmpMat.col(3) = refRow.transpose();

			const MyVector& refRow76 = m_Gsubmatrix.row(76);
			tmpMat.col(4) = refRow76.transpose();
			const vrFloat val = refRow.dot(externalForceVals);
			printf("row 969 val %f \n",val);

			std::stringstream ss;
			ss << "d:/m_Gsubmatrix_externalForceVals_[" << m_str_currentDateTime << "]_" << m_displacement.rows() <<".txt";

			std::ofstream outfile(ss.str().c_str());
			outfile << tmpMat << std::endl;
			outfile.close();

			/*const vrFloat EPSINON = 1.0e-16;
			for (int d=0;d<refRow.size();++d)
			{
				const vrFloat var = refRow[d];
				if(var < EPSINON  && var > -EPSINON)
				{
					m_nouse_set_dof.insert(d);
				}
			}*/
		}
		/*{
			std::vector< sortNode > vecSortNode;
			sortNode tmpNode;
			for (int v=0;v<m_rhs.size();++v)
			{
				tmpNode.val = m_rhs[v];
				tmpNode.dofs = v;
				vecSortNode.push_back(tmpNode);
			}
			std::sort(vecSortNode.begin(), vecSortNode.end(),sortNode_LessCompare());
			tmpMat.resize(vecSortNode.size(),2);

			for (int v=0;v<vecSortNode.size();++v)
			{
				tmpMat.coeffRef(v,0) = vecSortNode[v].val;
				tmpMat.coeffRef(v,1) = vecSortNode[v].dofs;
			}

			std::stringstream ss;
			ss << "d:/m_Gsubmatrix_SortNode_[" << m_str_currentDateTime << "]_" << m_displacement.rows() <<".txt";

			std::ofstream outfile(ss.str().c_str());
			outfile << tmpMat << std::endl;
			outfile.close();
		}*/

#endif

		

		/*for (int v=0;v<m_rhs.size();++v)
		{
			if (m_rhs[v]>1 || m_rhs[v] < -1)
			{
				m_rhs[v] = -1.0 * m_rhs[v];
				printf("#################################################################\n");
			}
		}*/
	}

	void vrBEM3D::solve()
	{
#if 0
		{
			const int nRows = m_A_matrix.rows();
			const int nCols = m_A_matrix.cols();
			std::stringstream ss;
			ss << "d:/m_A_matrix_[" << m_str_currentDateTime << "]_" << nRows <<".mtx";
			std::ofstream outfileA(ss.str().c_str());
			
			outfileA << nRows << " " << nCols << " " << nRows*nCols << std::endl;
			for (int r=0;r<nRows;++r)
			{
				for (int c=0;c<nCols;++c)
				{
					outfileA << r << " " << c << " " << m_A_matrix.coeff(r,c) << std::endl;
				}
			}
			outfileA.close();
		}

		{
			std::stringstream ss;
			ss << "d:/m_rhs_vector__[" << m_str_currentDateTime << "]_" << m_rhs.rows() <<".txt";

			std::ofstream outfile_rhs(ss.str().c_str());
			outfile_rhs << m_rhs << std::endl;
			outfile_rhs.close();
		}
#endif//Output_Matrix_Displacement

		TimingCPU timerCPU;
		timerCPU.StartCounter();
#if USE_cuSolverDn

		printf("step 1: make dense matrix\n");
		cusolverDnHandle_t handle = NULL;
		cublasHandle_t cublasHandle = NULL; // used in residual evaluation
		cudaStream_t stream = NULL;

		checkCudaErrors(cusolverDnCreate(&handle));
		checkCudaErrors(cublasCreate(&cublasHandle));
		checkCudaErrors(cudaStreamCreate(&stream));

		checkCudaErrors(cusolverDnSetStream(handle, stream));
		checkCudaErrors(cublasSetStream(cublasHandle, stream));

		const vrInt rowsA = getDofs(); // number of rows of A
		const vrInt colsA = getDofs(); // number of columns of A
		const vrInt lda   = rowsA; // leading dimension in dense matrix

		double *cpu_A = m_A_matrix.data(); // dense matrix from CSR(A)
		double *cpu_displacement = m_displacement.data(); // a copy of d_x
		double *cpu_rhs = m_rhs.data(); // b = ones(m,1)

		static MyVector vec_cpu_residual(rowsA);
		double *cpu_residual = vec_cpu_residual.data(); // r = b - A*x, a copy of d_r

		double *gpu_A = NULL; // a copy of h_A
		double *gpu_displacement = NULL; // x = A \ b
		double *gpu_rhs = NULL; // a copy of h_b
		double *gpu_residual = NULL; // r = b - A*x

		checkCudaErrors(cudaMalloc((void **)&gpu_A, sizeof(double)*lda*colsA));
		checkCudaErrors(cudaMalloc((void **)&gpu_displacement, sizeof(double)*colsA));
		checkCudaErrors(cudaMalloc((void **)&gpu_rhs, sizeof(double)*rowsA));
		checkCudaErrors(cudaMalloc((void **)&gpu_residual, sizeof(double)*rowsA));

		printf("step 2: prepare data on device\n");
		checkCudaErrors(cudaMemcpy(gpu_A, cpu_A, sizeof(double)*lda*colsA, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(gpu_rhs, cpu_rhs, sizeof(double)*rowsA, cudaMemcpyHostToDevice));

		printf("step 3: solve A*x = b \n");
		linearSolverQR(handle, rowsA, gpu_A, lda, gpu_rhs, gpu_displacement);

		printf("step 4: evaluate residual\n");
		checkCudaErrors(cudaMemcpy(gpu_residual, gpu_rhs, sizeof(double)*rowsA, cudaMemcpyDeviceToDevice));

		// r = b - A*x
		const double minus_one = -1.0;
		const double one = 1.0;
		checkCudaErrors(cublasDgemm_v2(
			cublasHandle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			rowsA,
			1,
			colsA,
			&minus_one,
			gpu_A,
			lda,
			gpu_displacement,
			rowsA,
			&one,
			gpu_residual,
			rowsA));

		checkCudaErrors(cudaMemcpy(cpu_displacement, gpu_displacement, sizeof(double)*colsA, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(cpu_residual, gpu_residual, sizeof(double)*rowsA, cudaMemcpyDeviceToHost));

		double x_inf = 0.0;
		double r_inf = 0.0;
		double A_inf = 0.0;
		x_inf = vec_norminf(colsA, cpu_displacement);
		r_inf = vec_norminf(rowsA, cpu_residual);
		A_inf = mat_norminf(rowsA, colsA, cpu_A, lda);

		printf("|b - A*x| = %E \n", r_inf);
		printf("|A| = %E \n", A_inf);
		printf("|x| = %E \n", x_inf);
		printf("|b - A*x|/(|A|*|x|) = %E \n", r_inf/(A_inf * x_inf));

		if (handle) { checkCudaErrors(cusolverDnDestroy(handle)); }
		if (cublasHandle) { checkCudaErrors(cublasDestroy(cublasHandle)); }
		if (stream) { checkCudaErrors(cudaStreamDestroy(stream)); }

		if (gpu_A) { checkCudaErrors(cudaFree(gpu_A)); }
		if (gpu_displacement) { checkCudaErrors(cudaFree(gpu_displacement)); }
		if (gpu_rhs) { checkCudaErrors(cudaFree(gpu_rhs)); }
		if (gpu_residual) { checkCudaErrors(cudaFree(gpu_residual)); }
#else //USE_cuSolverDn
		m_displacement = GaussElimination(m_A_matrix, m_rhs);
		//m_displacement = m_A_matrix.householderQr().solve(m_rhs);
		m_displacement = m_A_matrix.fullPivLu().solve(m_rhs);
		
		vrFloat relative_error = (m_A_matrix*m_displacement - m_rhs).norm() / m_rhs.norm(); // norm() is L2 norm
		cout << "The relative error is:\n" << relative_error << endl;
#endif//USE_cuSolverDn
		printf("CPU time [ms]: %f\n",timerCPU.GetCounter());
		

#if 0
		{
			std::stringstream ss;
			ss << "d:/m_displacement_vector_[" << m_str_currentDateTime << "]_" << m_displacement.rows() <<".txt";

			std::ofstream outfile(ss.str().c_str());
			outfile << m_displacement << std::endl;
			outfile.close();
		}
#endif
	}

	bool vrBEM3D::isOutofRange(const vrFloat val)const
	{
		if (val < FLT_MAX && val > (-1.0*FLT_MAX))
		{
			return false;
		}
		else
		{
			return true;
		}
	}

#if USE_Sigmoidal

	vrFloat vrBEM3D::compute_K_ij_I_SST_DisContinuous_Sigmoidal(const vrInt nSubTriIdx, const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,vrInt idx_i,vrInt idx_j,vrInt idx_I)
	{
#if 0

		vrFloat retVal = 0.0;
		vrFloat doubleLayer_Term_k = 0.0, singleLayer_Term_k = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();
		const MyVec3& unitNormal_srcPt = curSourcePtr->getVertexNormal();MyNotice;

		const vrInt n_gpts = TriangleElem::GaussPointSize_eta_In_Theta_SubTri * TriangleElem::GaussPointSize_eta_In_Rho_SubTri;

#if USE_Sigmoidal
		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal[nSubTriIdx];
		const MyMatrix& cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal[nSubTriIdx];
#endif//USE_Sigmoidal

		if (n_gpts != cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows())
		{
			printf("compute_K_ij_I_SST_DisContinuous_Sigmoidal : n_gpts[%d] == cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows()[%d] \n",n_gpts, cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows());
		}
		Q_ASSERT(n_gpts == cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows());
		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);

		for (int idx_k=0; idx_k < MyDim; ++idx_k)
		{
			const vrFloat delta_jk = TriangleElemData::delta_ij(idx_j,idx_k);
			const vrFloat delta_ik = TriangleElemData::delta_ij(idx_i,idx_k);

			for (int index_theta=0,idx_g=0;index_theta<TriangleElem::GaussPointSize_eta_In_Theta_SubTri;++index_theta)
			{
				const vrFloat cur_theta_singlelayer_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_theta_singleLayer];

#if 1 //SST
				const vrFloat A = refDataSST3D.A_theta_SubTri(cur_theta_singlelayer_Sigmoidal);
				const vrFloat N_I_0 = refDataSST3D.N_I_0_eta_SubTri(idx_I);
				//const vrFloat N_I_1 = refDataSST3D.N_I_1_eta(idx_I,cur_theta_singlelayer);
				const MyVec3& n_x = refDataSST3D.unitNormal_fieldPt;
				const vrFloat jacob_eta = refDataSST3D.Jacobi_eta_SubTri;
				MyVec3 sst_dr;
				vrFloat sst_drdn = 0.0;
				for (int m=0;m<MyDim;++m)
				{
					sst_dr[m] = refDataSST3D.r_i_SubTri(m,cur_theta_singlelayer_Sigmoidal); 
					sst_drdn += (sst_dr[m]*n_x[m]);
				}

				const vrFloat M0 = (
					(1.0-2.0*mu)*(delta_ij * sst_dr[idx_k] + delta_jk * sst_dr[idx_i] MyNotice - delta_ik * sst_dr[idx_j]) + 
					3.0 * sst_dr[idx_i] * sst_dr[idx_j] * sst_dr[idx_k]
				) * unitNormal_srcPt[idx_k];

				const vrFloat M1 = ((1.0) / (8.0*numbers::MyPI*(1-mu))) * M0 * jacob_eta;

				const vrFloat F_1_ij_I_k = (M1*N_I_0)/(A*A);
#endif//SST
				const vrFloat curWeight_singleLayer_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_weight_singleLayer];
				const vrFloat cur_rho_bar_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_rho_bar_singleLayer];
				const vrFloat cur_jacobi_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_Jacobi_singleLayer];
				for (int index_rho=0;index_rho<TriangleElem::GaussPointSize_eta_In_Rho_SubTri;++index_rho,++idx_g)
				{
					auto curRows_Sigmoidal = cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g);
					MyVec2ParamSpace theta_rho_Sigmoidal;
					theta_rho_Sigmoidal[TriangleElem::idx_theta_doubleLayer] = curRows_Sigmoidal[TriangleElem::idx_theta_doubleLayer];
					theta_rho_Sigmoidal[TriangleElem::idx_rho_doubleLayer]=curRows_Sigmoidal[TriangleElem::idx_rho_doubleLayer];
					const vrFloat cur_rho = theta_rho_Sigmoidal[TriangleElem::idx_rho_doubleLayer];
					const vrFloat curWeight_doubleLayer = curRows_Sigmoidal[TriangleElem::idx_weight_doubleLayer];

					Q_ASSERT(numbers::isEqual(theta_rho_Sigmoidal[TriangleElem::idx_theta_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g).x()));
					Q_ASSERT(numbers::isEqual(theta_rho_Sigmoidal[TriangleElem::idx_rho_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g).y()));
					Q_ASSERT(numbers::isEqual(cur_theta_singlelayer_Sigmoidal,theta_rho_Sigmoidal[TriangleElem::idx_theta_doubleLayer]));

					const MyVec2ParamSpace currentSrcPtInParam /*in eta sub triangle space*/ = refDataSST3D.m_SrcPt_in_eta_SubTri;
					Q_ASSERT( ((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))) );
					if (!((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))))
					{
						printf("compute_K_ij_I_SST_DisContinuous_Sigmoidal : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
					}

					const MyVec2ParamSpace cur_eta = refDataSST3D.pc2eta(currentSrcPtInParam MYNOTICE, theta_rho_Sigmoidal);
					const MyVec2ParamSpace cur_xi = refDataSST3D.eta2xi_SubTri(cur_eta);
					const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

					MyFloat jacob_eta_nouse;/*jacob_eta = jacob_xi * mat_T_Inv*/
					MyVec3 normals_fieldpoint;
					MyFloat r;
					MyVec3 dr;
					MyFloat drdn;
					getKernelParameters_3D_SST_SubTri(srcPos,fieldPoint,refDataSST3D,jacob_eta_nouse,normals_fieldpoint,r,dr,drdn);

					Q_ASSERT(numbers::isEqual(jacob_eta, jacob_eta_nouse));
					Q_ASSERT(numbers::isEqual(r,cur_rho*A));


					const vrFloat Kij_k = get_Kij_SST_3D_k(idx_i, idx_j, idx_k, r, dr, drdn, normals_fieldpoint, unitNormal_srcPt);
					const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);

					//////////////////////////////////////////////////////////////////////////

					const vrFloat SingularTerm_Kij_I_k = Kij_k * N_I * jacob_eta * theta_rho_Sigmoidal[TriangleElem::idx_rho_doubleLayer];
					const vrFloat SingularTerm_F_1_ij_I_k = (1.0/cur_rho)*(F_1_ij_I_k);
					doubleLayer_Term_k += (SingularTerm_Kij_I_k - SingularTerm_F_1_ij_I_k) * curWeight_doubleLayer* cur_jacobi_Sigmoidal MYNOTICE;

				}//for (int index_rho=0;index_rho<TriangleElem::GaussPointSize_eta_In_Rho_SubTri;++index_rho,++idx_g)

				const vrFloat beta = 1.0 / A;

				singleLayer_Term_k += F_1_ij_I_k * log( abs(cur_rho_bar_Sigmoidal/beta) ) * curWeight_singleLayer_Sigmoidal * cur_jacobi_Sigmoidal MyNotice;
			}//for (int index_theta=0,idx_g=0;index_theta<TriangleElem::GaussPointSize_eta_In_Theta_SubTri;++index_theta)

			retVal += (doubleLayer_Term_k + singleLayer_Term_k);

		}//for (int idx_k=0; idx_k < MyDim; ++idx_k)

		return retVal;
#endif
	}

	vrFloat vrBEM3D::compute_S_ij_I_SST_DisContinuous_Sigmoidal(const vrInt nSubTriIdx, const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,vrInt idx_i,vrInt idx_j,vrInt idx_I)
	{
#if 0

		vrFloat retVal = 0.0;
		vrFloat doubleLayer_Term_k = 0.0, singleLayer_Term_k = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();
		const MyVec3& unitNormal_srcPt = curSourcePtr->getVertexNormal();MyNotice

			const vrInt n_gpts = TriangleElem::GaussPointSize_eta_In_Theta_SubTri * TriangleElem::GaussPointSize_eta_In_Rho_SubTri;

#if USE_Sigmoidal
		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal[nSubTriIdx];
		const MyMatrix& cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal[nSubTriIdx];
#endif//USE_Sigmoidal

		if (n_gpts != cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows())
		{
			printf("compute_S_ij_I_SST_DisContinuous_Sigmoidal : n_gpts[%d] == cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows()[%d] \n",n_gpts, cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows());
		}
		Q_ASSERT(n_gpts == cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows());
		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
		for (int idx_k=0; idx_k < MyDim; ++idx_k)
		{
			const vrFloat delta_jk = TriangleElemData::delta_ij(idx_j,idx_k);
			const vrFloat delta_ik = TriangleElemData::delta_ij(idx_i,idx_k);

			for (int index_theta=0,idx_g=0;index_theta<TriangleElem::GaussPointSize_eta_In_Theta_SubTri;++index_theta)
			{
				const vrFloat cur_theta_singlelayer_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_theta_singleLayer];

#if 1 // SST
				const vrFloat A = refDataSST3D.A_theta_SubTri(cur_theta_singlelayer_Sigmoidal);
				const vrFloat N_I_0 = refDataSST3D.N_I_0_eta_SubTri(idx_I);
				const vrFloat N_I_1 = refDataSST3D.N_I_1_eta_SubTri(idx_I,cur_theta_singlelayer_Sigmoidal);
				const MyVec3& n_x = refDataSST3D.unitNormal_fieldPt;
				const vrFloat jacob_eta = refDataSST3D.Jacobi_eta_SubTri;
				MyVec3 sst_dr;
				vrFloat sst_drdn = 0.0;
				for (int m=0;m<MyDim;++m)
				{
					sst_dr[m] = refDataSST3D.r_i_SubTri(m,cur_theta_singlelayer_Sigmoidal); 
					sst_drdn += (sst_dr[m]*n_x[m]);
				}
				const vrFloat M0 = (
					3.0 * sst_drdn * ( (1.0-2.0*mu)*delta_ik*sst_dr[idx_j] + mu*(delta_ij*sst_dr[idx_k]+delta_jk*sst_dr[idx_i]) MyNotice - 5.0*sst_dr[idx_i]*sst_dr[idx_j]*sst_dr[idx_k] ) +
					3.0 * mu * (n_x[idx_i]*sst_dr[idx_j]*sst_dr[idx_k]+n_x[idx_k]*sst_dr[idx_i]*sst_dr[idx_j]) MyNotice - 
					(1.0-4.0*mu) * delta_ik * n_x[idx_j] + 
					(1.0-2.0*mu) * (3.0 * n_x[idx_j] * sst_dr[idx_i] * sst_dr[idx_k] + delta_ij * n_x[idx_k] + delta_jk * n_x[idx_i])
					) * unitNormal_srcPt[idx_k];
				const vrFloat M1 = ( (shearMod)/(4.0*numbers::MyPI*(1.0-mu)) ) * M0 * jacob_eta;

				const vrFloat F_2_ij_I_k = (M1*N_I_0)/(A*A*A);
				const vrFloat F_1_ij_I_k = (M1*N_I_1)/(A*A);
#endif//SST
				const vrFloat curWeight_singleLayer_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_weight_singleLayer];
				const vrFloat cur_rho_bar_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_rho_bar_singleLayer];
				const vrFloat cur_jacobi_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_Jacobi_singleLayer];
				for (int index_rho=0;index_rho<TriangleElem::GaussPointSize_eta_In_Rho_SubTri;++index_rho,++idx_g)
				{
					MyNotice;/*the order of theta and rho,(theta,rho)*/
					auto curRows_Sigmoidal = cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g);
					MyVec2ParamSpace theta_rho_Sigmoidal;
					theta_rho_Sigmoidal[TriangleElem::idx_theta_doubleLayer] = curRows_Sigmoidal[TriangleElem::idx_theta_doubleLayer];
					theta_rho_Sigmoidal[TriangleElem::idx_rho_doubleLayer]=curRows_Sigmoidal[TriangleElem::idx_rho_doubleLayer];
					const vrFloat cur_rho = theta_rho_Sigmoidal[TriangleElem::idx_rho_doubleLayer];
					const vrFloat curWeight_doubleLayer = curRows_Sigmoidal[TriangleElem::idx_weight_doubleLayer];

					Q_ASSERT(numbers::isEqual(theta_rho_Sigmoidal[TriangleElem::idx_theta_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g).x()));
					Q_ASSERT(numbers::isEqual(theta_rho_Sigmoidal[TriangleElem::idx_rho_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g).y()));
					Q_ASSERT(numbers::isEqual(cur_theta_singlelayer_Sigmoidal,theta_rho_Sigmoidal[TriangleElem::idx_theta_doubleLayer]));

					const MyVec2ParamSpace currentSrcPtInParam /*in eta sub triangle space*/ = refDataSST3D.m_SrcPt_in_eta_SubTri;
					Q_ASSERT( ((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))) );
					if (!((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))))
					{
						printf("compute_S_ij_I_SST_DisContinuous_Sigmoidal : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
					}
					const MyVec2ParamSpace cur_eta = refDataSST3D.pc2eta(currentSrcPtInParam /*MyVec2ParamSpace(0.0,0.0)*/MYNOTICE, theta_rho_Sigmoidal);
					const MyVec2ParamSpace cur_xi = refDataSST3D.eta2xi_SubTri(cur_eta);
					const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

					MyFloat jacob_eta_nouse;/*jacob_eta = jacob_xi * mat_T_Inv*/
					MyVec3 normals_fieldpoint;
					MyFloat r;
					MyVec3 dr;
					MyFloat drdn;
					getKernelParameters_3D_SST_SubTri(srcPos,fieldPoint,refDataSST3D,jacob_eta_nouse,normals_fieldpoint,r,dr,drdn);

					Q_ASSERT(numbers::isEqual(jacob_eta, jacob_eta_nouse));

					//printf("[%d,%d] r(%f)  rho*A(%f) \n",index_theta,index_rho,r,cur_rho*A);
					Q_ASSERT(numbers::isEqual(r,cur_rho*A));

					const vrFloat Sij_k = get_Sij_SST_3D_k(idx_i, idx_j, idx_k, r, dr, drdn, normals_fieldpoint, unitNormal_srcPt);
					const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);

					const vrFloat SingularTerm_Sij_I_k = Sij_k * N_I * jacob_eta * theta_rho_Sigmoidal[TriangleElem::idx_rho_doubleLayer] ;

					const vrFloat SingularTerm_F_1_ij_I_k = (1.0/(cur_rho))*(F_1_ij_I_k);
					const vrFloat SingularTerm_F_2_ij_I_k = (1.0/(cur_rho*cur_rho)) * (F_2_ij_I_k);

					doubleLayer_Term_k += (SingularTerm_Sij_I_k - (SingularTerm_F_1_ij_I_k + SingularTerm_F_2_ij_I_k)) * curWeight_doubleLayer* cur_jacobi_Sigmoidal MYNOTICE;

				}//for (int index_rho=0;index_rho<TriangleElem::GaussPointSize_eta_In_Rho_SubTri;++index_rho,++idx_g)

				const vrFloat beta = 1.0 / A;
#if DEBUG_5_28
				const vrFloat cur_Rho_hat = refDataSST3D.rho_hat_SubTri(nSubTriIdx, cur_theta_singlelayer);//refDataSST3D.rho_hat(cur_theta_singlelayer);
#else

#if USE_Sigmoidal
				cur_rho_bar_Sigmoidal;
				singleLayer_Term_k += ( (F_1_ij_I_k * log( abs(cur_rho_bar_Sigmoidal/beta) )) MyNotice - (F_2_ij_I_k * (1.0/(cur_rho_bar_Sigmoidal))) )* curWeight_singleLayer_Sigmoidal * cur_jacobi_Sigmoidal MyNotice;
#else
				const vrInt triId = refDataSST3D.search_Theta_in_eta_belong_SubTri_Index( cur_theta_singlelayer);
				const vrFloat cur_Rho_hat = refDataSST3D.rho_hat_SubTri(triId, cur_theta_singlelayer);//refDataSST3D.rho_hat(cur_theta_singlelayer);
				singleLayer_Term_k += ( (F_1_ij_I_k * log( abs(cur_Rho_hat/beta) )) MyNotice - (F_2_ij_I_k * (1.0/(cur_Rho_hat))) )* curWeight_singleLayer;
#endif//USE_Sigmoidal

#endif//DEBUG_5_28
			}//for (int index_theta=0,idx_g=0;index_theta<TriangleElem::GaussPointSize_eta_In_Theta_SubTri;++index_theta)

			retVal += (doubleLayer_Term_k + singleLayer_Term_k);

		}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
		return retVal;
#endif
	}
#else//USE_Sigmoidal
	vrFloat vrBEM3D::compute_S_ij_I_SST_DisContinuous(const vrInt nSubTriIdx, const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,vrInt idx_i,vrInt idx_j,vrInt idx_I)
	{
		vrFloat retVal = 0.0;
		vrFloat doubleLayer_Term_k = 0.0, singleLayer_Term_k = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();
		const MyVec3& unitNormal_srcPt = curSourcePtr->getVertexNormal();MyNotice

			const vrInt n_gpts = TriangleElem::GaussPointSize_eta_In_Theta_SubTri * TriangleElem::GaussPointSize_eta_In_Rho_SubTri;
#if USE_SUBTRI_INTE

#if USE_Sigmoidal

		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal[nSubTriIdx];
		const MyMatrix& cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal[nSubTriIdx];
#else

		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri[nSubTriIdx];
		const MyMatrix& cur_gaussQuadrature_eta_theta_singleLayer_SubTri = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer_SubTri[nSubTriIdx];
#endif//USE_Sigmoidal

#else

		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri;
		const MyMatrix& cur_gaussQuadrature_eta_theta_singleLayer_SubTri = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer_SubTri;
#endif
		if (n_gpts != cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows())
		{
			printf("compute_S_ij_I_SST_DisContinuous : n_gpts[%d] == cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows()[%d] \n",n_gpts, cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows());
		}
		Q_ASSERT(n_gpts == cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows());
		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
		for (int idx_k=0; idx_k < MyDim; ++idx_k)
		{
			const vrFloat delta_jk = TriangleElemData::delta_ij(idx_j,idx_k);
			const vrFloat delta_ik = TriangleElemData::delta_ij(idx_i,idx_k);

			for (int index_theta=0,idx_g=0;index_theta<TriangleElem::GaussPointSize_eta_In_Theta_SubTri;++index_theta)
			{
				const vrFloat cur_theta_singlelayer_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_theta_singleLayer];
				const vrFloat curWeight_singleLayer_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_weight_singleLayer];
				const vrFloat cur_rho_bar_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_rho_bar_singleLayer];
				const vrFloat cur_jacobi_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_Jacobi_singleLayer];
#if 1 // SST
				const vrFloat A = refDataSST3D.A_theta_SubTri(cur_theta_singlelayer_Sigmoidal);
				const vrFloat N_I_0 = refDataSST3D.N_I_0_eta_SubTri(idx_I);
				const vrFloat N_I_1 = refDataSST3D.N_I_1_eta_SubTri(idx_I,cur_theta_singlelayer_Sigmoidal);
				const MyVec3& n_x = refDataSST3D.unitNormal_fieldPt;
				const vrFloat jacob_eta = refDataSST3D.Jacobi_eta_SubTri;
				MyVec3 sst_dr;
				vrFloat sst_drdn = 0.0;
				for (int m=0;m<MyDim;++m)
				{
					sst_dr[m] = refDataSST3D.r_i_SubTri(m,cur_theta_singlelayer_Sigmoidal); 
					sst_drdn += (sst_dr[m]*n_x[m]);
				}
				const vrFloat M0 = (
					3.0 * sst_drdn * ( (1.0-2.0*mu)*delta_ik*sst_dr[idx_j] + mu*(delta_ij*sst_dr[idx_k]+delta_jk*sst_dr[idx_i]) MyNotice - 5.0*sst_dr[idx_i]*sst_dr[idx_j]*sst_dr[idx_k] ) +
					3.0 * mu * (n_x[idx_i]*sst_dr[idx_j]*sst_dr[idx_k]+n_x[idx_k]*sst_dr[idx_i]*sst_dr[idx_j]) MyNotice - 
					(1.0-4.0*mu) * delta_ik * n_x[idx_j] + 
					(1.0-2.0*mu) * (3.0 * n_x[idx_j] * sst_dr[idx_i] * sst_dr[idx_k] + delta_ij * n_x[idx_k] + delta_jk * n_x[idx_i])
					) * unitNormal_srcPt[idx_k];
				const vrFloat M1 = ( (shearMod)/(4.0*numbers::MyPI*(1.0-mu)) ) * M0 * jacob_eta;

				const vrFloat F_2_ij_I_k = (M1*N_I_0)/(A*A*A);
				const vrFloat F_1_ij_I_k = (M1*N_I_1)/(A*A);
#endif//SST
				for (int index_rho=0;index_rho<TriangleElem::GaussPointSize_eta_In_Rho_SubTri;++index_rho,++idx_g)
				{
					MyNotice;/*the order of theta and rho,(theta,rho)*/
					auto curRows_Sigmoidal = cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g);
					MyVec2ParamSpace theta_rho_Sigmoidal;
					theta_rho_Sigmoidal[TriangleElem::idx_theta_doubleLayer] = curRows_Sigmoidal[TriangleElem::idx_theta_doubleLayer];
					theta_rho_Sigmoidal[TriangleElem::idx_rho_doubleLayer]=curRows_Sigmoidal[TriangleElem::idx_rho_doubleLayer];
					const vrFloat cur_rho = theta_rho_Sigmoidal[TriangleElem::idx_rho_doubleLayer];
					const vrFloat curWeight_doubleLayer = curRows_Sigmoidal[TriangleElem::idx_weight_doubleLayer];

					Q_ASSERT(numbers::isEqual(theta_rho_Sigmoidal[TriangleElem::idx_theta_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g).x()));
					Q_ASSERT(numbers::isEqual(theta_rho_Sigmoidal[TriangleElem::idx_rho_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g).y()));
					Q_ASSERT(numbers::isEqual(cur_theta_singlelayer_Sigmoidal,theta_rho_Sigmoidal[TriangleElem::idx_theta_doubleLayer]));

					const MyVec2ParamSpace currentSrcPtInParam /*in eta sub triangle space*/ = refDataSST3D.m_SrcPt_in_eta_SubTri;
					Q_ASSERT( ((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))) );
					if (!((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))))
					{
						printf("compute_S_ij_I_SST_DisContinuous : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
					}
					const MyVec2ParamSpace cur_eta = refDataSST3D.pc2eta(currentSrcPtInParam /*MyVec2ParamSpace(0.0,0.0)*/MYNOTICE, theta_rho_Sigmoidal);
					const MyVec2ParamSpace cur_xi = refDataSST3D.eta2xi_SubTri(cur_eta);
					const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

					MyFloat jacob_eta_nouse;/*jacob_eta = jacob_xi * mat_T_Inv*/
					MyVec3 normals_fieldpoint;
					MyFloat r;
					MyVec3 dr;
					MyFloat drdn;
					getKernelParameters_3D_SST_SubTri(srcPos,fieldPoint,refDataSST3D,jacob_eta_nouse,normals_fieldpoint,r,dr,drdn);

					Q_ASSERT(numbers::isEqual(jacob_eta, jacob_eta_nouse));

					//printf("[%d,%d] r(%f)  rho*A(%f) \n",index_theta,index_rho,r,cur_rho*A);
					Q_ASSERT(numbers::isEqual(r,cur_rho*A));

					const vrFloat Sij_k = get_Sij_SST_3D_k(idx_i, idx_j, idx_k, r, dr, drdn, normals_fieldpoint, unitNormal_srcPt);
					const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);

					const vrFloat SingularTerm_Sij_I_k = Sij_k * N_I * jacob_eta * theta_rho_Sigmoidal[TriangleElem::idx_rho_doubleLayer] * cur_jacobi_Sigmoidal MYNOTICE;

					const vrFloat SingularTerm_F_1_ij_I_k = (1.0/(cur_rho))*(F_1_ij_I_k);
					const vrFloat SingularTerm_F_2_ij_I_k = (1.0/(cur_rho*cur_rho)) * (F_2_ij_I_k);

					doubleLayer_Term_k += (SingularTerm_Sij_I_k - (SingularTerm_F_1_ij_I_k + SingularTerm_F_2_ij_I_k)) * curWeight_doubleLayer;

				}//for (int index_rho=0;index_rho<TriangleElem::GaussPointSize_eta_In_Rho_SubTri;++index_rho,++idx_g)

				const vrFloat beta = 1.0 / A;
#if DEBUG_5_28
				const vrFloat cur_Rho_hat = refDataSST3D.rho_hat_SubTri(nSubTriIdx, cur_theta_singlelayer);//refDataSST3D.rho_hat(cur_theta_singlelayer);
#else

#if USE_Sigmoidal
				cur_rho_bar_Sigmoidal;
				singleLayer_Term_k += ( (F_1_ij_I_k * log( abs(cur_rho_bar_Sigmoidal/beta) )) MyNotice - (F_2_ij_I_k * (1.0/(cur_rho_bar_Sigmoidal))) )* curWeight_singleLayer_Sigmoidal * cur_jacobi_Sigmoidal MyNotice;
#else
				const vrInt triId = refDataSST3D.search_Theta_in_eta_belong_SubTri_Index( cur_theta_singlelayer);
				const vrFloat cur_Rho_hat = refDataSST3D.rho_hat_SubTri(triId, cur_theta_singlelayer);//refDataSST3D.rho_hat(cur_theta_singlelayer);
				singleLayer_Term_k += ( (F_1_ij_I_k * log( abs(cur_Rho_hat/beta) )) MyNotice - (F_2_ij_I_k * (1.0/(cur_Rho_hat))) )* curWeight_singleLayer;
#endif//USE_Sigmoidal

#endif//DEBUG_5_28
			}//for (int index_theta=0,idx_g=0;index_theta<TriangleElem::GaussPointSize_eta_In_Theta_SubTri;++index_theta)

			retVal += (doubleLayer_Term_k + singleLayer_Term_k);

		}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
		return retVal;
	}
#endif//USE_Sigmoidal

	

#if USE_Sigmoidal
	vrFloat vrBEM3D::compute_T_ij_I_SST_DisContinuous_Sigmoidal(const vrInt nSubTriIdx, const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,vrInt idx_i,vrInt idx_j,vrInt idx_I)
	{
#if 0

		vrFloat doubleLayer_Term = 0.0, singleLayer_Term = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();

		const vrInt n_gpts = TriangleElem::GaussPointSize_eta_In_Theta_SubTri * TriangleElem::GaussPointSize_eta_In_Rho_SubTri;
		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal[nSubTriIdx];
		const MyMatrix& cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal[nSubTriIdx];

		if (n_gpts != cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows())
		{
			printf("compute_T_ij_I_SST_DisContinuous_Sigmoidal : n_gpts[%d] == cur_gaussQuadrature_xi_eta_polar_SubTri.rows()[%d] \n",n_gpts, cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows());
		}
		Q_ASSERT(n_gpts == cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows());

		for (int index_theta=0,idx_g=0;index_theta<TriangleElem::GaussPointSize_eta_In_Theta_SubTri;++index_theta)
		{
			const vrFloat cur_theta_singlelayer_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_theta_singleLayer];


#if 1 // SST
			const vrFloat A = refDataSST3D.A_theta_SubTri(cur_theta_singlelayer_Sigmoidal);
			const vrFloat N_I_0 = refDataSST3D.N_I_0_eta_SubTri(idx_I);

			const MyVec3& n_x = refDataSST3D.unitNormal_fieldPt;
			const vrFloat jacob_eta = refDataSST3D.Jacobi_eta_SubTri;
			MyVec3 sst_dr;
			vrFloat sst_drdn = 0.0;
			for (int m=0;m<MyDim;++m)
			{
				sst_dr[m] = refDataSST3D.r_i_SubTri(m,cur_theta_singlelayer_Sigmoidal); 
				sst_drdn += (sst_dr[m]*n_x[m]);
			}
			const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
			const vrFloat M0 = (sst_drdn)*( (1.0-2.0*mu)*delta_ij + 3.0* sst_dr[idx_i] * sst_dr[idx_j]) - (1.0-2.0*mu)*(sst_dr[idx_i]*n_x[idx_j]-sst_dr[idx_j]*n_x[idx_i]);
			const vrFloat M1 = ((-1.0) / (8.0*numbers::MyPI*(1-mu))) * M0 * jacob_eta ;
			const vrFloat F_1_ij_I = (M1*N_I_0)/(A*A);
#endif // SST

			const vrFloat curWeight_singleLayer_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_weight_singleLayer];
			const vrFloat cur_rho_bar_singleLayer_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_rho_bar_singleLayer];
			const vrFloat cur_jacobi_singleLayer_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_Jacobi_singleLayer];
			for (int index_rho=0;index_rho<TriangleElem::GaussPointSize_eta_In_Rho_SubTri;++index_rho,++idx_g)
			{
				MyNotice;/*the order of theta and rho,(theta,rho)*/
				auto curRows = cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g);
				MyVec2ParamSpace theta_rho;
				theta_rho[TriangleElem::idx_theta_doubleLayer] = curRows[TriangleElem::idx_theta_doubleLayer];
				theta_rho[TriangleElem::idx_rho_doubleLayer]=curRows[TriangleElem::idx_rho_doubleLayer];
				const vrFloat cur_rho = theta_rho[TriangleElem::idx_rho_doubleLayer];
				const vrFloat curWeight_doubleLayer = curRows[TriangleElem::idx_weight_doubleLayer];

				Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_theta_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g).x()));
				Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_rho_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g).y()));
				Q_ASSERT(numbers::isEqual(cur_theta_singlelayer_Sigmoidal,theta_rho[TriangleElem::idx_theta_doubleLayer]));

				const MyVec2ParamSpace currentSrcPtInParam /*in eta sub triangle space*/ = refDataSST3D.m_SrcPt_in_eta_SubTri;
				Q_ASSERT( ((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))) );
				if (!((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))))
				{
					printf("compute_T_ij_I_SST_DisContinuous : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
				}
				const MyVec2ParamSpace cur_eta = refDataSST3D.pc2eta(currentSrcPtInParam /*MyVec2ParamSpace(0.0,0.0)*/MYNOTICE, theta_rho);
				const MyVec2ParamSpace cur_xi = refDataSST3D.eta2xi_SubTri(cur_eta);
				const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

				MyFloat jacob_eta_nouse;/*jacob_eta = jacob_xi * mat_T_Inv*/
				MyVec3 normals_fieldpoint;
				MyFloat r;
				MyVec3 dr;
				MyFloat drdn;
				getKernelParameters_3D_SST_SubTri(srcPos,fieldPoint,refDataSST3D,jacob_eta_nouse,normals_fieldpoint,r,dr,drdn);

				Q_ASSERT(numbers::isEqual(jacob_eta, jacob_eta_nouse));

				//printf("[%d,%d] r(%f)  rho*A(%f) \n",index_theta,index_rho,r,cur_rho*A);
				Q_ASSERT(numbers::isEqual(r,cur_rho*A));

				const vrFloat Tij = get_Tij_SST_3D(idx_i, idx_j, r, dr,drdn,normals_fieldpoint);
				const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);

				const vrFloat SingularTerm_Tij_I = Tij * N_I * jacob_eta * theta_rho[TriangleElem::idx_rho_doubleLayer] ; 


				const vrFloat SingularTerm_F_1_ij_I = (1.0/cur_rho)*(F_1_ij_I);

				doubleLayer_Term += (SingularTerm_Tij_I - SingularTerm_F_1_ij_I) * curWeight_doubleLayer * cur_jacobi_singleLayer_Sigmoidal MYNOTICE;
			}

			const vrFloat beta = 1.0 / A;

#if USE_Sigmoidal
			singleLayer_Term += F_1_ij_I * log( abs(cur_rho_bar_singleLayer_Sigmoidal/beta) ) * curWeight_singleLayer_Sigmoidal * cur_jacobi_singleLayer_Sigmoidal MYNOTICE ;
#else//USE_Sigmoidal
			const vrInt triId = refDataSST3D.search_Theta_in_eta_belong_SubTri_Index( cur_theta_singlelayer_Sigmoidal);
			const vrFloat cur_Rho_hat = refDataSST3D.rho_hat_SubTri(triId, cur_theta_singlelayer_Sigmoidal);//refDataSST3D.rho_hat(cur_theta_singlelayer);
			singleLayer_Term += F_1_ij_I * log( abs(cur_Rho_hat/beta) ) * curWeight_singleLayer_Sigmoidal;
#endif//USE_Sigmoidal




		}
		return (doubleLayer_Term + singleLayer_Term);
#endif
	}
#else//USE_Sigmoidal

	vrFloat vrBEM3D::compute_T_ij_I_SST_DisContinuous(const vrInt nSubTriIdx, const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,vrInt idx_i,vrInt idx_j,vrInt idx_I)
	{//MyError("vrBEM3D::compute_T_ij_I_SST_DisContinuous.");
		vrFloat doubleLayer_Term = 0.0, singleLayer_Term = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();

		const vrInt n_gpts = TriangleElem::GaussPointSize_eta_In_Theta_SubTri * TriangleElem::GaussPointSize_eta_In_Rho_SubTri;
#if USE_SUBTRI_INTE

#if USE_Sigmoidal
		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal[nSubTriIdx];
		const MyMatrix& cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal[nSubTriIdx];
#else//USE_Sigmoidal
		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri[nSubTriIdx];
		const MyMatrix& cur_gaussQuadrature_eta_theta_singleLayer_SubTri = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer_SubTri[nSubTriIdx];
#endif//USE_Sigmoidal

#else//USE_SUBTRI_INTE

		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri;
		const MyMatrix& cur_gaussQuadrature_eta_theta_singleLayer_SubTri = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer_SubTri;
#endif//USE_SUBTRI_INTE
		if (n_gpts != cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows())
		{
			printf("compute_T_ij_I_SST_DisContinuous : n_gpts[%d] == cur_gaussQuadrature_xi_eta_polar_SubTri.rows()[%d] \n",n_gpts, cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows());
		}
		Q_ASSERT(n_gpts == cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows());

		for (int index_theta=0,idx_g=0;index_theta<TriangleElem::GaussPointSize_eta_In_Theta_SubTri;++index_theta)
		{
			const vrFloat cur_theta_singlelayer_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_theta_singleLayer];
			const vrFloat curWeight_singleLayer_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_weight_singleLayer];
			const vrFloat cur_rho_bar_singleLayer_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_rho_bar_singleLayer];
			const vrFloat cur_jacobi_singleLayer_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_Jacobi_singleLayer];

#if 1 // SST
			const vrFloat A = refDataSST3D.A_theta_SubTri(cur_theta_singlelayer_Sigmoidal);
			const vrFloat N_I_0 = refDataSST3D.N_I_0_eta_SubTri(idx_I);

			const MyVec3& n_x = refDataSST3D.unitNormal_fieldPt;
			const vrFloat jacob_eta = refDataSST3D.Jacobi_eta_SubTri;
			MyVec3 sst_dr;
			vrFloat sst_drdn = 0.0;
			for (int m=0;m<MyDim;++m)
			{
				sst_dr[m] = refDataSST3D.r_i_SubTri(m,cur_theta_singlelayer_Sigmoidal); 
				sst_drdn += (sst_dr[m]*n_x[m]);
			}
			const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
			const vrFloat M0 = (sst_drdn)*( (1.0-2.0*mu)*delta_ij + 3.0* sst_dr[idx_i] * sst_dr[idx_j]) - (1.0-2.0*mu)*(sst_dr[idx_i]*n_x[idx_j]-sst_dr[idx_j]*n_x[idx_i]);
			const vrFloat M1 = ((-1.0) / (8.0*numbers::MyPI*(1-mu))) * M0 * jacob_eta * cur_jacobi_singleLayer_Sigmoidal MyNotice;
			const vrFloat F_1_ij_I = (M1*N_I_0)/(A*A);
#endif // SST

			for (int index_rho=0;index_rho<TriangleElem::GaussPointSize_eta_In_Rho_SubTri;++index_rho,++idx_g)
			{
				MyNotice;/*the order of theta and rho,(theta,rho)*/
				auto curRows = cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g);
				MyVec2ParamSpace theta_rho;
				theta_rho[TriangleElem::idx_theta_doubleLayer] = curRows[TriangleElem::idx_theta_doubleLayer];
				theta_rho[TriangleElem::idx_rho_doubleLayer]=curRows[TriangleElem::idx_rho_doubleLayer];
				const vrFloat cur_rho = theta_rho[TriangleElem::idx_rho_doubleLayer];
				const vrFloat curWeight_doubleLayer = curRows[TriangleElem::idx_weight_doubleLayer];

				Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_theta_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g).x()));
				Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_rho_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g).y()));
				Q_ASSERT(numbers::isEqual(cur_theta_singlelayer_Sigmoidal,theta_rho[TriangleElem::idx_theta_doubleLayer]));

				const MyVec2ParamSpace currentSrcPtInParam /*in eta sub triangle space*/ = refDataSST3D.m_SrcPt_in_eta_SubTri;
				Q_ASSERT( ((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))) );
				if (!((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))))
				{
					printf("compute_T_ij_I_SST_DisContinuous : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
				}
				const MyVec2ParamSpace cur_eta = refDataSST3D.pc2eta(currentSrcPtInParam /*MyVec2ParamSpace(0.0,0.0)*/MYNOTICE, theta_rho);
				const MyVec2ParamSpace cur_xi = refDataSST3D.eta2xi_SubTri(cur_eta);
				const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

				MyFloat jacob_eta_nouse;/*jacob_eta = jacob_xi * mat_T_Inv*/
				MyVec3 normals_fieldpoint;
				MyFloat r;
				MyVec3 dr;
				MyFloat drdn;
				getKernelParameters_3D_SST_SubTri(srcPos,fieldPoint,refDataSST3D,jacob_eta_nouse,normals_fieldpoint,r,dr,drdn);

				Q_ASSERT(numbers::isEqual(jacob_eta, jacob_eta_nouse));

				//printf("[%d,%d] r(%f)  rho*A(%f) \n",index_theta,index_rho,r,cur_rho*A);
				Q_ASSERT(numbers::isEqual(r,cur_rho*A));

				const vrFloat Tij = get_Tij_SST_3D(idx_i, idx_j, r, dr,drdn,normals_fieldpoint);
				const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);

				const vrFloat SingularTerm_Tij_I = Tij * N_I * jacob_eta * theta_rho[TriangleElem::idx_rho_doubleLayer] * cur_jacobi_singleLayer_Sigmoidal MYNOTICE; 


				const vrFloat SingularTerm_F_1_ij_I = (1.0/cur_rho)*(F_1_ij_I);

				doubleLayer_Term += (SingularTerm_Tij_I - SingularTerm_F_1_ij_I) * curWeight_doubleLayer;
			}

			const vrFloat beta = 1.0 / A;
#if DEBUG_5_28
			const vrFloat cur_Rho_hat = refDataSST3D.rho_hat_SubTri(nSubTriIdx, cur_theta_singlelayer);
#else//DEBUG_5_28

#if USE_Sigmoidal
			singleLayer_Term += F_1_ij_I * log( abs(cur_rho_bar_singleLayer_Sigmoidal/beta) ) * curWeight_singleLayer_Sigmoidal  ;
#else//USE_Sigmoidal
			const vrInt triId = refDataSST3D.search_Theta_in_eta_belong_SubTri_Index( cur_theta_singlelayer);
			const vrFloat cur_Rho_hat = refDataSST3D.rho_hat_SubTri(triId, cur_theta_singlelayer);//refDataSST3D.rho_hat(cur_theta_singlelayer);
			singleLayer_Term += F_1_ij_I * log( abs(cur_Rho_hat/beta) ) * curWeight_singleLayer;
#endif//USE_Sigmoidal

#endif//DEBUG_5_28


		}
		return (doubleLayer_Term + singleLayer_Term);
	}
#endif//USE_Sigmoidal



	

	

	vrFloat vrBEM3D::compute_U_ij_I_DisContinuous(const vrInt nSubTriIdx, const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,vrInt idx_i,vrInt idx_j,vrInt idx_I)
	{
#if 0

		vrFloat retVal = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();
		const bool isDisContinuousVtx = curSourcePtr->isDisContinuousVertex();
		Q_ASSERT(isDisContinuousVtx);

		const vrInt n_gpts = TriangleElem::GaussPointSize_eta_In_Theta_SubTri * TriangleElem::GaussPointSize_eta_In_Rho_SubTri;
#if USE_SUBTRI_INTE

#if USE_Sigmoidal
		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal[nSubTriIdx];
		const MyMatrix& cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal[nSubTriIdx];
#else//USE_Sigmoidal
		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri[nSubTriIdx];
		const MyMatrix& cur_gaussQuadrature_eta_theta_singleLayer_SubTri = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer_SubTri[nSubTriIdx];
#endif//USE_Sigmoidal

#else//USE_SUBTRI_INTE

		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri;
		const MyMatrix& cur_gaussQuadrature_eta_theta_singleLayer_SubTri = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer_SubTri;
#endif//USE_SUBTRI_INTE

		const vrFloat jacob_eta = refDataSST3D.Jacobi_eta_SubTri;
		if (n_gpts != cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows())
		{
			printf("compute_U_ij_I_SST_DisContinuous : n_gpts[%d] == cur_gaussQuadrature_xi_eta_polar_SubTri.rows()[%d] \n",n_gpts, cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows());
		}
		Q_ASSERT(n_gpts == cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows());

		for (int idx_g=0;idx_g < n_gpts;++idx_g)
		{
			MyNotice/*the order of theta and rho,(theta,rho)*/
				auto curRows = cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g);
			MyVec2ParamSpace theta_rho;
			theta_rho[TriangleElem::idx_theta_doubleLayer] = curRows[TriangleElem::idx_theta_doubleLayer];
			theta_rho[TriangleElem::idx_rho_doubleLayer]=curRows[TriangleElem::idx_rho_doubleLayer];
			const vrFloat curWeight = curRows[TriangleElem::idx_weight_doubleLayer];


			Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_theta_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g).x()));
			Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_rho_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g).y()));

			const MyVec2ParamSpace currentSrcPtInParam /*in eta sub triangle space*/ = refDataSST3D.m_SrcPt_in_eta_SubTri;
			Q_ASSERT( ((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))) );
			if (!((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))))
			{
				printf("compute_U_ij_I_DisContinuous : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
			}

			const MyVec2ParamSpace cur_eta = refDataSST3D.pc2eta(currentSrcPtInParam /*MyVec2ParamSpace(0.0,0.0)*/MYNOTICE, theta_rho);
			const MyVec2ParamSpace cur_xi = refDataSST3D.eta2xi_SubTri(cur_eta);
			const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

			MyFloat jacob_eta_nouse;/*jacob_eta = jacob_xi * mat_T_Inv*/
			MyVec3 normals_fieldpoint;
			MyFloat r;
			MyVec3 dr;
			MyFloat drdn;
			getKernelParameters_3D_SST_SubTri(srcPos,fieldPoint,refDataSST3D,jacob_eta_nouse,normals_fieldpoint,r,dr,drdn);

			Q_ASSERT(numbers::isEqual(jacob_eta, jacob_eta_nouse));


			//Q_ASSERT(numbers::isEqual(r,theta_rho[TriangleElemData::idx_rho]));
			const vrFloat Uij = get_Uij_SST_3D(idx_i, idx_j, r, dr);

			const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);

			retVal += Uij * N_I * jacob_eta * theta_rho[TriangleElem::idx_rho_doubleLayer] * curWeight;
		}


		return retVal;
#endif
	}

	

	

	void vrBEM3D::getKernelParameters_3D(const MyVec3& srcPos, const MyVec3& fieldPoint,const TriangleElemData& curTriElemData,
		MyFloat& jacob_xi,MyVec3& normals_fieldpoint,MyFloat& r,MyVec3& dr,MyFloat& drdn)
	{
		jacob_xi = curTriElemData.Jacobi_xi;
		normals_fieldpoint = curTriElemData.unitNormal_fieldPt;
		/*const MyVec3& normalsTmp = curTriElemData.unitNormal_srcPt;
		Q_ASSERT(numbers::isEqual(normalsTmp[0],normals_fieldpoint[0])&&
		numbers::isEqual(normalsTmp[1],normals_fieldpoint[1])&&
		numbers::isEqual(normalsTmp[2],normals_fieldpoint[2]));*/
		MyVec3 relDist = fieldPoint - srcPos;
		r=relDist.norm();

		dr[0] = relDist[0] / r;
		dr[1] = relDist[1] / r;
		dr[2] = relDist[2] / r;

		drdn =  dr[0]*normals_fieldpoint[0] + dr[1]*normals_fieldpoint[1] +dr[2]*normals_fieldpoint[2];//dr.transpose()*normals
	}

#if DEBUG_3_3

	void vrBEM3D::getKernelParameters_3D_SST(const MyVec3& srcPos, const MyVec3& fieldPoint,const TriangleElemData& curTriElemData,
		MyFloat& jacob_eta,MyVec3& normals_fieldpoint,MyFloat& r,MyVec3& dr,MyFloat& drdn)
	{
		getKernelParameters_3D(srcPos, fieldPoint, curTriElemData, jacob_eta, normals_fieldpoint, r, dr, drdn);
		jacob_eta = curTriElemData.Jacobi_eta;
	}
#endif

	void vrBEM3D::getKernelParameters_3D_SST_SubTri(const MyVec3& srcPos, const MyVec3& fieldPoint,const TriangleElemData& curTriElemData,
		MyFloat& jacob_eta,MyVec3& normals_fieldpoint,MyFloat& r,MyVec3& dr,MyFloat& drdn)
	{
		getKernelParameters_3D(srcPos, fieldPoint, curTriElemData, jacob_eta, normals_fieldpoint, r, dr, drdn);
		jacob_eta = curTriElemData.Jacobi_eta_SubTri;
	}

#if USE_Fracture
	int vrBEM3D::readElems(elemMap& elems, ELEM_TYPE elemType, std::set<unsigned int> bodies,
		int typeColumn, idMap* bodyIDs, elemMap* parentIDs, bool strictlyListedBodies)
	{
		using namespace std;
		unsigned int elemId,bodyId,buffer,column;
		bool flag;
		istringstream strstream;
		string line, token;
		std::vector<unsigned int> nodes; // store node IDs
		std::vector<unsigned int> parents; // store parents when reading boundary elements
		elems.clear();
		if(bodyIDs!=NULL) bodyIDs->clear();

		ifstream in(m_strElemFile.c_str());
		if(!in.is_open()) return -1;
		while(in.good()){
			getline(in, line);
			strstream.clear();
			strstream.str(line);
			nodes.clear();
			parents.clear();
			//            printf("parsing line \"%s\"\n", line.c_str());
			//            printf("strstream holds \"%s\"\n", strstream.str().c_str());
			// first column is element ID
			getline(strstream, token, ' ');
			sscanf(token.c_str(), "%u", &elemId);
			//            printf(" token 1 is \"%s\"\n",token.c_str());
			// second column is body id
			getline(strstream, token, ' ');
			sscanf(token.c_str(), "%u", &bodyId);
			//            printf(" token 2 is \"%s\"\n",token.c_str());
			//            printf("body id is %u", bodyId);
			flag= (!strictlyListedBodies && bodies.empty()) || (bodies.count(bodyId)>0); // only read further if body ID matches
			//            printf(" -- %s found\n", flag?"":"not");
			column=3;
			while(getline(strstream, token, ' ') && flag){
				sscanf(token.c_str(), "%u", &buffer);
				//                printf(" token %u is \"%s\"\n",column, token.c_str());
				if((parentIDs!=NULL) && (column < typeColumn)){ // columns 3 and 4 in a boundary file are the parent elements of the boundary element (face)
					parents.push_back(buffer);
				}else if(column == typeColumn){
					// check the element type
					flag = (buffer==elemType); // stop reading if the elment type is wrong
				}else if(column > typeColumn){
					// store node ID
					nodes.push_back(buffer);
				}
				++column;
			}
			if(flag && !nodes.empty()){
				elems[elemId]=nodes;
				if(bodyIDs!=NULL) (*bodyIDs)[elemId]=bodyId;
				if(parentIDs!=NULL) (*parentIDs)[elemId]=parents;
				//                printf("added element %u\n", elemId);
			}
		}
		in.close();
		return elems.size();
	}

	int vrBEM3D::readNodes(nodeMap& nodes)
	{
		using namespace std;
		string line;
		double coords[3]; // store node coords
		int id, buffer, tokens;
		char test;
		nodes.clear();

		//		printf("reading nodes from \"%s\"\n",nodeFile.c_str());
		ifstream in(m_strNodeFile.c_str());
		if(!in.is_open()) return -1;
		//		printf("file opened successfully\n");
		while(in.good()){
			getline(in, line); test=0;
			// format for nodes is "node-id unused x y z"
			tokens=sscanf(line.c_str(),"%d %d %lf %lf %lf%c",
				&id, &buffer, coords, coords+1, coords+2, &test);
			//printf("read %d tokens: %d %d %lf %lf %lf, test is %d\n", tokens, id, buffer, coords[0], coords[1], coords[2], test);vrPause;
			if(tokens==5 && test==0){ //line format is correct
				nodes[id]=vector<double>(coords, coords+3);
			}
		}
		in.close();
		//		printf("read %d nodes from %s\n", nodes.size(),nodeFile.c_str());
		return nodes.size();
	}

	int vrBEM3D::readModel(elemMap& elems/* sim.getElems() */, idMap& bodyIDs/* sim.getRegions() */,
		elemMap& bndrys/* sim.getCrackTips() */, elemMap& bndryParents/* sim.getCrackTipParents() */, 
		nodeMap& nodes/* sim.getNodes() */,
		const ELEM_TYPE elemType/* FractureSim::TRI */,  const std::set<unsigned int> elemBodies/* region_ids */,
		const ELEM_TYPE bndryType/* FractureSim::LINE */, const std::set<unsigned int> bndryBodies/* crack_tip_ids */)
	{

		nodeMap nodes_in;
		//elem_map elems_in;
		int ret;
		// read elements and nodes
		ret=readElems(elems, elemType, elemBodies,ELEMENTS_FILE,&bodyIDs);
		//output_elem_map(std::cout, "elems", elems);vrPause;
		if(ret<0) return ret;
		ret=readNodes(nodes_in);
		if(ret<0) return ret;
		// now switch to .boundary file

		/*string tmp=elemFile;
		elemFile=meshFile;
		elemFile.append(".boundary");
		ret=readElems(bndrys,bndryType,bndryBodies,BOUNDARY_FILE,NULL,&bndryParents,true);
		elemFile=tmp;*/

		//if(ret<0) return ret; // it's ok to not have a boundary file, if there's no pre-defined cracks

		// reading is complete, but for HyENA-BEM compatibility the nodes need to be numbered
		// in the same order as they appear in the element map.

		// run through the element map and decide new node numbers
		idMap fwd;// bkwd; // fwd[old_id]=new_id, bkwd[new_id]=old_id
		unsigned int new_id=NODE_BASE_INDEX;
		for(elemMap::iterator i = elems.begin(); i!=elems.end(); ++i){
			//run through all nodes of the element
			for(elemMap::mapped_type::iterator j = i->second.begin(); j!=i->second.end(); ++j){
				if(fwd.count(*j)==0){ // assing new number at first occurence of node
					fwd[*j]=new_id; //bkwd[new_id]=*j;
					new_id++;
				}
				(*j)=fwd[*j]; //update element
			}
		}
		nodes.clear();
		// copy from nodes_in to nodes while applying new numbering
		for(nodeMap::iterator i = nodes_in.begin(); i!= nodes_in.end(); ++i){
			nodes[fwd[i->first]] = i->second;
		}
		// apply new numbering to bndry elements
		for(elemMap::iterator i = bndrys.begin(); i!=bndrys.end(); ++i){
			for(elemMap::mapped_type::iterator j = i->second.begin(); j!=i->second.end(); ++j){
				(*j)=fwd[*j]; //update element
			}
		}
		//output_elem_map(std::cout, "elems", elems);vrPause;
		printf("readModel success!\n");
		return elems.size();
	}
#if USE_SST_DEBUG
	int vrBEM3D::initVDB(double voxelSize, bool noSI, double nbHWidth)
	{
		printf("vrBEM3D::initVDB 1\n");
		levelSet.reset( new FractureSim::VDBWrapper(voxelSize,nbHWidth,noSI) );
		printf("vrBEM3D::initVDB 2\n");
		int ret = levelSet->init(m_reader_Nodes,m_reader_Elems,m_reader_Regions,m_reader_Cracks/*cracks.size = 0 yangchen*/);
		if(ret==0){
			// initialize crackTipStates based on inside/outside object
			/*printf("[DEBUG] crackTips.SIZE = %d\n",crackTips.size()); system("pause");
			for(elem_map::const_iterator it = crackTips.begin();
			it!= crackTips.end(); ++it
			){
			unsigned int nd_a, nd_b;
			nd_a = it->second[0]; nd_b = it->second[1];
			vdb::Vec3d a(nodes[nd_a][0],nodes[nd_a][1],nodes[nd_a][2]);
			vdb::Vec3d b(nodes[nd_b][0],nodes[nd_b][1],nodes[nd_b][2]);
			if((levelSet->isInside(a) ||
			levelSet->isInside(b))&&
			levelSet->isInside((a+b)/2.0)
			)
			crackTipStates[it->first]=ACTIVE;
			else
			crackTipStates[it->first]=INACTIVE;
			}*/
			vdbInitialized=true;
		}
		else
		{
			printf("initVDB fail!\n");
		}
		printf("vrBEM3D::initVDB 3\n");
		return ret;
	}
	int vrBEM3D::remesh(int nTris, double adaptive, double offsetVoxels)
	{
		if(vdbInitialized && nTris>3)
			return levelSet->mesh(getNodes(),getElems(),nTris,adaptive,offsetVoxels);
		return -1;
	}
#endif//USE_SST_DEBUG

	void vrBEM3D::outputMesh2Obj(const std::string& filename, const elem_map& elems, const node_map& nodes)
	{
		std::ofstream outfile(filename+".remesh.obj");
		const nodeMap& curNodes = nodes;
		const vrInt nVertexSize = curNodes.size();
		for (int v=NODE_BASE_INDEX;v<(nVertexSize+NODE_BASE_INDEX);++v)
		{
			const std::vector<VR::MyFloat>& refVtx = curNodes.at(v);
			Q_ASSERT(3 == refVtx.size());
			outfile << "v  " << refVtx[0] << " "  << refVtx[1] << " "  << refVtx[2] << std::endl;
		}
		const elemMap& curElems = elems;
		const vrInt nTriSize = curElems.size();
		for (int t=1;t<=nTriSize;++t)
		{
			const std::vector<unsigned int>& refTri = curElems.at(t);
			outfile << "f  " << refTri[0] << " "  << refTri[1] << " "  << refTri[2] << std::endl;
		}
		outfile.close();
	}

	void vrBEM3D::outputMeshInfo(const std::string& filename, const elem_map& elems, const id_map& bodyIDs, const elem_map& bndrys, const elem_map& bndryParents, const node_map& nodes)
	{
		ofstream outfile(filename);
		output_elem_map(outfile, "elems", elems);
		output_id_map(outfile, "bodyIDs", bodyIDs);
		output_elem_map(outfile, "bndrys", bndrys);
		output_elem_map(outfile, "bndryParents", bndryParents);
		output_node_map(outfile, "nodes", nodes);
	}

	void vrBEM3D::output_elem_map(ostream& out, std::string name, const elem_map& elems)
	{
		//typedef std::map<unsigned int, std::vector<unsigned int>> elem_map
		out << "name : " << name << std::endl;
		for (iterAllOf(ci, elems))
		{
			const elem_map::key_type& firstVal = (*ci).first;
			const elem_map::referent_type& secondVal = (*ci).second;
			out << firstVal << " : ";
			std::copy(secondVal.begin(), secondVal.end(), std::ostream_iterator<elem_map::referent_type::value_type>(out, " "));
			out << std::endl;
		}
	}

	void vrBEM3D::output_id_map(ostream& out, std::string name, const id_map& ids)
	{
		out << "name : " << name << std::endl;
		//typedef std::map<unsigned int, unsigned int> id_map
		for (iterAllOf(ci, ids))
		{
			const id_map::key_type& firstVal = (*ci).first;
			const id_map::referent_type& secondVal = (*ci).second;

			out << firstVal << " : " << secondVal << std::endl;
		}
	}

	void vrBEM3D::output_node_map(ostream& out, std::string name, const node_map& nodes)
	{
		out << "name : " << name << std::endl;
		//typedef std::map<unsigned int, std::vector<double>> node_map
		for (iterAllOf(ci, nodes))
		{
			const node_map::key_type& firstVal = (*ci).first;
			const node_map::referent_type& secondVal = (*ci).second;
			out << firstVal << " : ";
			std::copy(secondVal.begin(), secondVal.end(), std::ostream_iterator<node_map::referent_type::value_type>(out, " "));
			out << std::endl;
		}
	}

	void vrBEM3D::output_id_set(ofstream& out, std::string name, const id_set& ids)
	{
		out << "name : " << name << std::endl;
		//typedef std::map<unsigned int, unsigned int> id_map
		for (iterAllOf(ci, ids))
		{
			const id_map::key_type& firstVal = (*ci);

			out << firstVal << std::endl;
		}
	}


	const char* getName(CRACK_STATE states)
	{
		switch(states)
		{
		case CRACK_STATE::ACTIVE:
			return "CRACK_STATE::ACTIVE";
			break;
		case CRACK_STATE::ACTIVE_A:
			return "CRACK_STATE::ACTIVE_A";
			break;
		case CRACK_STATE::ACTIVE_B:
			return "CRACK_STATE::ACTIVE_B";
			break;
		case CRACK_STATE::INACTIVE:
			return "CRACK_STATE::INACTIVE";
			break;
		case CRACK_STATE::UPDATE:
			return "CRACK_STATE::UPDATE";
			break;
		case CRACK_STATE::UPDATE_A:
			return "CRACK_STATE::UPDATE_A";
			break;
		case CRACK_STATE::UPDATE_B:
			return "CRACK_STATE::UPDATE_B";
			break;
		default:
			return "unknow state!";
		}
	}

	void vrBEM3D::output_state_map(ofstream& out, std::string name, const state_map& states)
	{
		out << "name : " << name << std::endl;
		//typedef std::map<unsigned int, std::vector<double>> node_map
		for (iterAllOf(ci, states))
		{
			const state_map::key_type& firstVal = (*ci).first;
			const state_map::referent_type& secondVal = (*ci).second;
			out << firstVal << " : " << getName(secondVal) << std::endl;
		}
	}

	/* Compute stresses on triangles in the mesh based on nodal displacements
	* and add to VTK data
	* output-param retValues       (terminology of "node_map" is misused here)
	* is a map<int, vector<double> > that maps element-IDs to 9 double values:
	* the first value is the max. principal stress value,
	* the next 3 values are the plane-normal across which the principal stress is given;
	* the following 4 values are magnitude & normal vector for the min. principal stress
	* the last value is a flag: 0 for regular surface elements, >0 for fracture elements
	* for fractures, we specify whether max. and min. principal stress reach their largest
	* magnitude for the positive or negative side of the fracture (sign of applied COD):
	* 1: both positive; 2: max. negative, min. positive; 3: max. pos., min. neg.; 4: both neg.
	*/
	int vrBEM3D::computeSurfaceStresses(
		node_map& maxPrincipalStress, const vector_type& u,
		const vector_type& crackBaseDisplacements,
		double E, double nu)
	{
		// compute principal and cartesian stresses based on the linear FEM formulation found in
		// Bargteil et al. 2007 "A Finite Element Method for Animating Large Viscoplastic Flow"
		// by extending every triangle to a tet with no out-of-plane deformation
		// ...
		elemMap& elems = getElems();
		nodeMap& nodes = getNodes();
		const unsigned int n = elems.size();
		unsigned int i=0;

		maxPrincipalStress.clear();

		//matrices are: B inverse of edge vector matrix in material space, X edge vector matrix in world space, F deformation gradient
		// U*S*V' SVD of F, P stress (Piola-Kirchhoff)
		Eigen::Matrix3d B,F,U,S,Vt,P,X, I=Eigen::Matrix3d::Identity();
		const double mu=E/(2*(1+nu));
		const double lambda=E*nu/((1+nu)*(1-2*nu)); // convert (E, nu) to Lame parameters (mu = shear modulus, lambda) ; as usual expect trouble for nu=0.5
		for(elem_map::iterator it=elems.begin(); it!=elems.end(); ++it, ++i)
		{ // loop over elements

			Eigen::Vector3d // a,b,c are node coordinates in material space; ua,ub,uc are nodal displacements, so world coordinates are a+ua etc.
				a (nodes[it->second[0]][0], nodes[it->second[0]][1], nodes[it->second[0]][2]),
				b (nodes[it->second[1]][0], nodes[it->second[1]][1], nodes[it->second[1]][2]),
				c (nodes[it->second[2]][0], nodes[it->second[2]][1], nodes[it->second[2]][2]),
				ua(u[3*(it->second[0]-NODE_BASE_INDEX)], u[3*(it->second[0]-NODE_BASE_INDEX)+1], u[3*(it->second[0]-NODE_BASE_INDEX)+2]),
				ub(u[3*(it->second[1]-NODE_BASE_INDEX)], u[3*(it->second[1]-NODE_BASE_INDEX)+1], u[3*(it->second[1]-NODE_BASE_INDEX)+2]),
				uc(u[3*(it->second[2]-NODE_BASE_INDEX)], u[3*(it->second[2]-NODE_BASE_INDEX)+1], u[3*(it->second[2]-NODE_BASE_INDEX)+2]);

			if( computeElementPrincipalStresses(U,S,Vt,P, a,b,c, ua,ub,uc, mu,lambda) !=0) return -1;MYNOTICE;



			maxPrincipalStress[it->first].assign(9,0.0);
			// store max. principal stress and its plane-normal vector for each element (also min. for compressive fracture?)
			// in the SVD, V is the pre-rotation and U is the post-rotation
			// so the direction of max. principal stress is the first column of V
			// which becomes the first row of V.transpose()
			maxPrincipalStress[it->first][0]=P(0,0);  // first entry is the stress value
			maxPrincipalStress[it->first][1]=Vt(0,0); // next 3 entries
			maxPrincipalStress[it->first][2]=Vt(0,1); // are the plane-
			maxPrincipalStress[it->first][3]=Vt(0,2); // normal vector
			// same for min. principal stress used for compressive fractures
			maxPrincipalStress[it->first][4]=P(2,2);
			maxPrincipalStress[it->first][5]=Vt(2,0); // next 3 entries
			maxPrincipalStress[it->first][6]=Vt(2,1); // are the plane-
			maxPrincipalStress[it->first][7]=Vt(2,2); // normal vector
			//s_xx[i]=P(0,0); s_yy[i]=P(1,1); s_zz[i]=P(2,2); // for testing

			P = U*P*Vt; // P is now a matrix of cartesian stresses


		}
		return 0;
	}

	// computes SVD of the deformation gradient F = U*S*Vt
	// and principal stresses P = 2 mu (S-I) + lambda tr(S-I);
	int vrBEM3D::computeElementPrincipalStresses(
		Eigen::Matrix3d& U, Eigen::Matrix3d& S, Eigen::Matrix3d& Vt, Eigen::Matrix3d& P,
		const Eigen::Vector3d&  a, const Eigen::Vector3d&  b, const Eigen::Vector3d&  c,
		const Eigen::Vector3d& ua, const Eigen::Vector3d& ub, const Eigen::Vector3d& uc,
		double mu, double lambda
		)
	{
		Eigen::Matrix3d B,F,X, I=Eigen::Matrix3d::Identity();
		bool flag=true;
		X.col(0) = a - c; // using world space matrix as temporary storage before inversion
		X.col(1) = b - c;
		X.col(2) = X.col(0).cross(X.col(1)).normalized();
		X.computeInverseWithCheck(B,flag);
		if(!flag) return -1; // matrix not invertible ~> possibly degenerate element
		// now build the actual world space matrix
		X.col(0) = a+ua -c-uc;
		X.col(1) = b+ub -c-uc;
		X.col(2) = X.col(0).cross(X.col(1)).normalized();
		F=X*B;
		// compute singular value decomposition of F --> U*S*V'
		Eigen::JacobiSVD<Eigen::Matrix3d> svd(F,Eigen::ComputeFullU | Eigen::ComputeFullV);
		S = svd.singularValues().asDiagonal(); // these will be returned in order (highest to lowest)
		U = svd.matrixU();
		Vt= svd.matrixV().transpose();
		P = 2*mu*(S-I) + lambda*(S-I).trace()*I; // P is now a diagonal matrix of principal stresses
		//printf("\n%% diag. deform: %.3le, %.3le, %.3le", S(0,0), S(1,1), S(2,2));
		//printf("\n%% stresses for element %d:",it->first);
		//printf("\n%% -- principal: %.3le, %.3le, %.3le", P(0,0), P(1,1), P(2,2));
		return 0;
	}

	void vrBEM3D::generateSurfaceStresses()
	{
		nodeMap psn; // maps node-/element-IDs to 4 double values each, first value is the max. principal stress, last 3 values are it's direction (unit vector)
		MyVector  nouseVec;

		int ret = computeSurfaceStresses(
			psn,m_displacement, nouseVec,
			GlobalConf::youngsMod,GlobalConf::poissonsRatio
			);


		Vertex::clearStress();

		vrFloat maxStress = FLT_MIN, minStress = FLT_MAX;
		for (iterAllOf(ci,psn))
		{
			const nodeMap::key_type& triId_VDB = (*ci).first;
			const nodeMap::referent_type& stressVec = (*ci).second;
			//printf("nodeId_VDB [%d]\n",triId_VDB);

			TriangleElemPtr  curTriPtr = TriangleElem::getTriangle(triId_VDB - NODE_BASE_INDEX);

			curTriPtr->getVertex(0)->setStress(stressVec[0]);
			curTriPtr->getVertex(1)->setStress(stressVec[0]);
			curTriPtr->getVertex(2)->setStress(stressVec[0]);
		}

		vrFloat curStress;
		for (int v=0;v<Vertex::getVertexSize();++v)
		{
			curStress = Vertex::getVertex(v)->getStressInfo();

			if (curStress > maxStress)
			{
				maxStress = curStress;
			}

			if (curStress <  minStress)
			{
				minStress = curStress;
			}
		}

		Vertex::setStressRange(maxStress,minStress);
	}

#if USE_VDB

#if USE_SST_DEBUG
	vrInt vrBEM3D::seedCracksAndPropagate(int maxSeed)
	{
		nodeMap psn; // maps node-/element-IDs to 4 double values each, first value is the max. principal stress, last 3 values are it's direction (unit vector)
		MyVector  nouseVec;
		printf("\n");
		printf("1[%d]\n",psn.size());
		int ret = postPro->computeSurfaceStresses(
			psn,m_displacement, nouseVec,
			GlobalConf::youngsMod,GlobalConf::poissonsRatio
			);
		printf("2[%d]\n",Vertex::getVertexSize());

		Vertex::clearStress();

		vrFloat maxStress = FLT_MIN, minStress = FLT_MAX;
		for (iterAllOf(ci,psn))
		{
			const nodeMap::key_type& triId_VDB = (*ci).first;
			const nodeMap::referent_type& stressVec = (*ci).second;
			printf("nodeId_VDB [%d]\n",triId_VDB);

			TriangleElemPtr  curTriPtr = TriangleElem::getTriangle(triId_VDB - NODE_BASE_INDEX);

			curTriPtr->getVertex(0)->setStress(stressVec[0]);
			curTriPtr->getVertex(1)->setStress(stressVec[0]);
			curTriPtr->getVertex(2)->setStress(stressVec[0]);
		}

		vrFloat curStress;
		for (int v=0;v<Vertex::getVertexSize();++v)
		{
			curStress = Vertex::getVertex(v)->getStressInfo();

			if (curStress > maxStress)
			{
				maxStress = curStress;
			}

			if (curStress <  minStress)
			{
				minStress = curStress;
			}
		}
		printf("3\n");
		Vertex::setStressRange(maxStress,minStress);
		return ret;
	}
#endif//USE_SST_DEBUG

#endif//USE_VDB

#if USE_Aliabadi


	
	vrFloat vrBEM3D::get_Sij_SST_3D_k_Aliabadi_Peng(vrInt idx_i,vrInt idx_j,vrInt idx_k,vrFloat r,const MyVector& dr, const vrFloat drdn, const MyVec3& n_x, const MyVec3& n_s)
	{
		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
		const vrFloat delta_jk = TriangleElemData::delta_ij(idx_j,idx_k);
		const vrFloat delta_ik = TriangleElemData::delta_ij(idx_i,idx_k);

		const vrFloat retVal = (shearMod / (4.0*numbers::MyPI*(1.0-mu)*r*r*r)) * 
			(
			3.0*drdn*( (1.0-2.0*mu)*delta_ik*dr[idx_j] + mu * (delta_ij*dr[idx_k] + delta_jk*dr[idx_i]) MyNoticeMsg("Minus.") - 5.0 * dr[idx_i] * dr[idx_j] * dr[idx_k]) + 
			3.0 * mu * (n_x[idx_i] * dr[idx_j] * dr[idx_k] + n_x[idx_k] * dr[idx_i] * dr[idx_j]) MyNoticeMsg("Minus.") - 
			(1.0 - 4.0*mu) * delta_ik * n_x[idx_j] + 
			(1.0 - 2.0*mu) * ( (3.0 * n_x[idx_j] * dr[idx_i] * dr[idx_k]) + delta_ij * n_x[idx_k] + delta_jk * n_x[idx_i])
			) * n_s[idx_k];
		return retVal;
	}

	vrFloat vrBEM3D::get_Kij_SST_3D_k_Aliabadi_Peng(vrInt idx_i,vrInt idx_j,vrInt idx_k,vrFloat r,const MyVector& dr, const vrFloat drdn, const MyVec3& n_s)
	{
		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
		const vrFloat delta_jk = TriangleElemData::delta_ij(idx_j,idx_k);
		const vrFloat delta_ik = TriangleElemData::delta_ij(idx_i,idx_k);
		//const vrFloat One = -1.0;
		const vrFloat One = 1.0;
		const vrFloat retVal = ((One) / (8.0*numbers::MyPI*(1-mu)*r*r)) * 
			(
			(1.0-2.0*mu)*(delta_ij*dr[idx_k]+delta_jk*dr[idx_i] MyNoticeMsg("minus.")-delta_ik*dr[idx_j]) + 
			3.0*dr[idx_i]*dr[idx_j]*dr[idx_k]
		) * n_s[idx_k];

		return retVal;
	}
#if USE_Peng_Kernel
	vrFloat vrBEM3D::get_Sij_SST_3D_k_Aliabadi_Peng(vrInt idx_i,vrInt idx_j,vrInt idx_k,vrFloat r,const MyVector& dr, const vrFloat drdn, const MyVec3& n_x, const MyVec3& n_s)
	{
		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
		const vrFloat delta_jk = TriangleElemData::delta_ij(idx_j,idx_k);
		const vrFloat delta_ik = TriangleElemData::delta_ij(idx_i,idx_k);

		const vrFloat retVal = (shearMod / (4.0*numbers::MyPI*(1.0-mu)*r*r*r)) * 
			(
			3.0*drdn*( (1.0-2.0*mu)*delta_ik*dr[idx_j] + mu * (delta_ij*dr[idx_k] + delta_jk*dr[idx_i]) MyNoticeMsg("Minus.") - 5.0 * dr[idx_i] * dr[idx_j] * dr[idx_k]) + 
			3.0 * mu * (n_x[idx_i] * dr[idx_j] * dr[idx_k] + n_x[idx_k] * dr[idx_i] * dr[idx_j]) MyNoticeMsg("Minus.") - 
			(1.0 - 4.0*mu) * delta_ik * n_x[idx_j] + 
			(1.0 - 2.0*mu) * ( (3.0 * n_x[idx_j] * dr[idx_i] * dr[idx_k]) + delta_ij * n_x[idx_k] + delta_jk * n_x[idx_i])
			) * n_s[idx_k];
		return retVal;
	}

	vrFloat vrBEM3D::get_Kij_SST_3D_k_Aliabadi_Peng(vrInt idx_i,vrInt idx_j,vrInt idx_k,vrFloat r,const MyVector& dr, const vrFloat drdn, const MyVec3& n_s)
	{
		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
		const vrFloat delta_jk = TriangleElemData::delta_ij(idx_j,idx_k);
		const vrFloat delta_ik = TriangleElemData::delta_ij(idx_i,idx_k);
		//const vrFloat One = -1.0;
		const vrFloat One = 1.0;
		const vrFloat retVal = ((One) / (8.0*numbers::MyPI*(1-mu)*r*r)) * 
			(
			(1.0-2.0*mu)*(delta_ij*dr[idx_k]+delta_jk*dr[idx_i] MyNoticeMsg("minus.")-delta_ik*dr[idx_j]) + 
			3.0*dr[idx_i]*dr[idx_j]*dr[idx_k]
			) * n_s[idx_k];

		return retVal;
	}
#else
	vrFloat vrBEM3D::get_Kij_SST_3D_k_Aliabadi(vrInt idx_i,vrInt idx_j,vrInt idx_k,vrFloat r,const MyVector& dr, const vrFloat drdn)
	{
		vrFloat retVal = 0.0;
		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
		const vrFloat delta_jk = TriangleElemData::delta_ij(idx_j,idx_k);
		const vrFloat delta_ik = TriangleElemData::delta_ij(idx_i,idx_k);
		//const vrFloat One = -1.0;
		const vrFloat One = 1.0;
		retVal = ((One) / (8.0*numbers::MyPI*(1-mu)*r*r)) * 
			(
			(1.0-2.0*mu)*(delta_ij*dr[idx_k]+delta_ik*dr[idx_j]-delta_jk*dr[idx_i]) + 
			3.0*dr[idx_i]*dr[idx_j]*dr[idx_k]
		);
		//printf("get_Kij_SST_3D_k_Aliabadi %f\n",retVal);
		return retVal;
	}

	vrFloat vrBEM3D::get_Sij_SST_3D_k_Aliabadi(vrInt idx_i,vrInt idx_j,vrInt idx_k,vrFloat r,const MyVector& dr, const vrFloat drdn, const MyVec3& unitNormal_fieldPoint)
	{
		vrFloat retVal = 0.0;
		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);

		const MyVec3& n_x = unitNormal_fieldPoint;
		
		const vrFloat delta_jk = TriangleElemData::delta_ij(idx_j,idx_k);
		const vrFloat delta_ik = TriangleElemData::delta_ij(idx_i,idx_k);
		retVal = (shearMod / (4.0*numbers::MyPI*(1.0-mu)*r*r*r)) * 
			(
			3.0*drdn*( (1.0-2.0*mu)*delta_ij*dr[idx_k] + mu * (delta_ik*dr[idx_j] + delta_jk*dr[idx_i]) MyNotice - 5.0 * dr[idx_i] * dr[idx_j] * dr[idx_k]) + 
			3.0 * mu * (n_x[idx_i] * dr[idx_j] * dr[idx_k] + n_x[idx_j] * dr[idx_i] * dr[idx_k]) MyNotice + 
			(1.0 - 2.0*mu) * ( (3.0 * n_x[idx_k] * dr[idx_i] * dr[idx_j]) + delta_ik * n_x[idx_j] + delta_jk * n_x[idx_i]) -
			(1.0 - 4.0*mu) * delta_ij * n_x[idx_k]				
		);
		//printf("get_Sij_SST_3D_k_Aliabadi %f\n",retVal);
		return retVal;
	}
#endif

	vrFloat vrBEM3D::get_Tij_SST_3D_Aliabadi(vrInt idx_i,vrInt idx_j,vrFloat r,const MyVector& dr, const vrFloat drdn, const MyVec3& normal_fieldPoint)
	{
		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
		return ((-1.0) / (8.0*numbers::MyPI*(1-mu)*r*r)) * 
			(
			(drdn*( (1.0-2.0*mu)*delta_ij+3.0*dr[idx_i]*dr[idx_j] )) -
			(1.0-2.0*mu)*(dr[idx_i]*normal_fieldPoint[idx_j]-dr[idx_j]*normal_fieldPoint[idx_i])
			);
	}

	vrFloat vrBEM3D::get_Uij_SST_3D_Aliabadi(vrInt idx_i,vrInt idx_j,vrFloat r,const MyVector& dr)
	{
		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
		return ((1.0) / (16.0 * numbers::MyPI * shearMod*(1-mu)*r)) * 	( (3.0-4.0*mu)*delta_ij + dr[idx_i] * dr[idx_j] );
	}

	vrFloat vrBEM3D::compute_U_ij_I_Aliabadi(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,vrInt idx_i,vrInt idx_j,vrInt idx_I)
	{
	vrFloat retVal = 0.0;
	const MyVec3 srcPos = curSourcePtr->getPos();
	const bool isDisContinuousVtx = curSourcePtr->isDisContinuousVertex();

#if SPEEDUP_5_31
	int tmp_GaussPointSize_xi_In_Theta = 0;
	int tmp_GaussPointSize_xi_In_Rho = 0;
	if (dis_regular == refDataSST3D.m_DisContinuousType)
	{
		tmp_GaussPointSize_xi_In_Theta = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Theta;
		tmp_GaussPointSize_xi_In_Rho = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Rho;
	}
	else
	{
		tmp_GaussPointSize_xi_In_Theta = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Theta_DisContinuous;
		tmp_GaussPointSize_xi_In_Rho = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Rho_DisContinuous;
	}
	const vrInt nGaussPointSize_xi_In_Theta = tmp_GaussPointSize_xi_In_Theta;
	const vrInt nGaussPointSize_xi_In_Rho = tmp_GaussPointSize_xi_In_Rho;
#endif
	const vrInt n_gpts = nGaussPointSize_xi_In_Theta * nGaussPointSize_xi_In_Rho;
	Q_ASSERT(n_gpts == refDataSST3D.m_gaussQuadrature_xi_polar.rows());
	for (int idx_g=0;idx_g < n_gpts;++idx_g)
	{
		MyNotice;/*the order of theta and rho,(theta,rho)*/
		auto curRows = refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g);
		MyVec2ParamSpace theta_rho;
		theta_rho[TriangleElem::idx_theta_doubleLayer] = curRows[TriangleElem::idx_theta_doubleLayer];
		theta_rho[TriangleElem::idx_rho_doubleLayer]=curRows[TriangleElem::idx_rho_doubleLayer];
		const vrFloat curWeight = curRows[TriangleElem::idx_weight_doubleLayer];
		/*std::cout << theta_rho << std::endl;
		std::cout << refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g) << std::endl;*/
		Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_theta_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g).x()));
		Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_rho_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g).y()));

		const MyVec2ParamSpace currentSrcPtInParam /*in xi space*/ = refDataSST3D.m_SrcPt_in_xi;
		Q_ASSERT( ((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) /*compute_U_ij_I*/ );
		if (!((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) )
		{
			printf("compute_U_ij_I : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
		}
		const MyVec2ParamSpace cur_xi = refDataSST3D.pc2xi(currentSrcPtInParam /*MyVec2ParamSpace(0.0,0.0)*/ MYNOTICE, theta_rho);
		const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

		MyFloat jacob_xi; MyVec3 normals_fieldpoint; MyFloat r; MyVec3 dr; MyFloat drdn;
		getKernelParameters_3D(srcPos,fieldPoint,refDataSST3D,jacob_xi,normals_fieldpoint,r,dr,drdn);
		const vrFloat Uij = get_Uij_SST_3D_Aliabadi(idx_i, idx_j, r, dr);
		const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);
		retVal += Uij * N_I * jacob_xi * theta_rho[TriangleElem::idx_rho_doubleLayer] * curWeight;
	}


	return retVal;
}


	vrFloat vrBEM3D::compute_U_ij_I_SST_Aliabadi(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,vrInt idx_i,vrInt idx_j,vrInt idx_I)
	{
		vrFloat retVal = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();
		const bool isDisContinuousVtx = curSourcePtr->isDisContinuousVertex();

#if SPEEDUP_5_31
		int tmp_GaussPointSize_eta_In_Theta = 0;
		int tmp_GaussPointSize_eta_In_Rho = 0;
		if (dis_regular == refDataSST3D.m_DisContinuousType)
		{
			tmp_GaussPointSize_eta_In_Theta = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta;
			tmp_GaussPointSize_eta_In_Rho = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Rho;
		}
		else
		{
			tmp_GaussPointSize_eta_In_Theta = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_DisContinuous;
			tmp_GaussPointSize_eta_In_Rho = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Rho_DisContinuous;
		}
		const vrInt nGaussPointSize_xi_In_Theta = tmp_GaussPointSize_eta_In_Theta;
		const vrInt nGaussPointSize_xi_In_Rho = tmp_GaussPointSize_eta_In_Rho;
#endif
		const vrInt n_gpts = nGaussPointSize_xi_In_Theta * nGaussPointSize_xi_In_Rho;
		Q_ASSERT(n_gpts == refDataSST3D.m_gaussQuadrature_xi_eta_polar.rows());
		for (int idx_g=0;idx_g < n_gpts;++idx_g)
		{                          
			auto curRows = refDataSST3D.m_gaussQuadrature_xi_eta_polar.row(idx_g);
			MyVec2ParamSpace theta_rho;
			theta_rho[TriangleElem::idx_theta_doubleLayer] = curRows[TriangleElem::idx_theta_doubleLayer];
			theta_rho[TriangleElem::idx_rho_doubleLayer]=curRows[TriangleElem::idx_rho_doubleLayer];
			const vrFloat cur_rho = theta_rho[TriangleElem::idx_rho_doubleLayer];
			const vrFloat A = refDataSST3D.A_theta(theta_rho[TriangleElem::idx_theta_doubleLayer]);
			const vrFloat curWeight = curRows[TriangleElem::idx_weight_doubleLayer];
			/*std::cout << theta_rho << std::endl;
			std::cout << refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g) << std::endl;*/
			Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_theta_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_eta_polar.row(idx_g).x()));
			Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_rho_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_eta_polar.row(idx_g).y()));

			const MyVec2ParamSpace currentSrcPtInParam /*in xi space*/ = refDataSST3D.m_SrcPt_in_eta;
			Q_ASSERT( ((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) /*compute_U_ij_I*/ );
			if (!((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) )
			{
				printf("compute_U_ij_I_SST_Aliabadi : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
			}
			const MyVec2ParamSpace cur_eta = refDataSST3D.pc2eta(currentSrcPtInParam /*MyVec2ParamSpace(0.0,0.0)*/ MYNOTICE,theta_rho);
			const MyVec2ParamSpace cur_xi = refDataSST3D.eta2xi(cur_eta);
			const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);


			MyFloat jacob_eta_nouse; MyVec3 normals_fieldpoint; MyFloat r; MyVec3 dr; MyFloat drdn;
			getKernelParameters_3D_SST(srcPos,fieldPoint,refDataSST3D,jacob_eta_nouse,normals_fieldpoint,r,dr,drdn);
			Q_ASSERT(numbers::isEqual(refDataSST3D.Jacobi_eta, jacob_eta_nouse));
			Q_ASSERT(numbers::isEqual(r,cur_rho*A));
			const vrFloat Uij = get_Uij_SST_3D_Aliabadi(idx_i, idx_j, r, dr);
			const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);
			retVal += Uij * N_I * jacob_eta_nouse * theta_rho[TriangleElem::idx_rho_doubleLayer] * curWeight;
		}


		return retVal;
	}

	vrFloat vrBEM3D::compute_T_ij_I_SST_Aliabadi(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,vrInt idx_i,vrInt idx_j,vrInt idx_I)
	{
		vrFloat doubleLayer_Term = 0.0, singleLayer_Term = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();
#if SPEEDUP_5_31
		int tmp_GaussPointSize_eta_In_Theta = 0;
		int tmp_GaussPointSize_eta_In_Rho = 0;
		if (dis_regular == refDataSST3D.m_DisContinuousType)
		{
			tmp_GaussPointSize_eta_In_Theta = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta;
			tmp_GaussPointSize_eta_In_Rho = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Rho;
		}
		else
		{
			tmp_GaussPointSize_eta_In_Theta = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_DisContinuous;
			tmp_GaussPointSize_eta_In_Rho = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Rho_DisContinuous;
		}
		const vrInt nGaussPointSize_eta_In_Theta = tmp_GaussPointSize_eta_In_Theta;
		const vrInt nGaussPointSize_eta_In_Rho = tmp_GaussPointSize_eta_In_Rho;
#endif

		const vrInt n_gpts = nGaussPointSize_eta_In_Theta * nGaussPointSize_eta_In_Rho;
		Q_ASSERT(n_gpts == refDataSST3D.m_gaussQuadrature_xi_eta_polar.rows());

		for (int index_theta=0,idx_g=0;index_theta<nGaussPointSize_eta_In_Theta;++index_theta)
		{
			const vrFloat cur_theta_singlelayer = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer.row(index_theta)[TriangleElem::idx_theta_singleLayer];
			const vrFloat curWeight_singleLayer = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer.row(index_theta)[TriangleElem::idx_weight_singleLayer];
#if 1 // SST
			const vrFloat A = refDataSST3D.A_theta(cur_theta_singlelayer);
			const vrFloat N_I_0 = refDataSST3D.N_I_0_eta(idx_I);
			//const vrFloat N_I_1 = refDataSST3D.N_I_1_eta(idx_I,cur_theta_singlelayer);
			const MyVec3& n_x = refDataSST3D.unitNormal_fieldPt;
			const vrFloat jacob_eta = refDataSST3D.Jacobi_eta;
			MyVec3 sst_dr;
			vrFloat sst_drdn = 0.0;
			for (int m=0;m<MyDim;++m)
			{
				sst_dr[m] = refDataSST3D.r_i(m,cur_theta_singlelayer); 
				sst_drdn += (sst_dr[m]*n_x[m]);
			}
#if 0
			const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
			const vrFloat M0 = (sst_drdn)*( (1.0-2.0*mu)*delta_ij + 3.0* sst_dr[idx_i] * sst_dr[idx_j]) - (1.0-2.0*mu)*(sst_dr[idx_i]*n_x[idx_j]-sst_dr[idx_j]*n_x[idx_i]);
			const vrFloat M1 = ((-1.0) / (8.0*numbers::MyPI*(1-mu))) * M0 * jacob_eta;
#else
			const vrFloat nouse_r = 1.0;
			const vrFloat M0 = get_Tij_SST_3D_Aliabadi(idx_i, idx_j, nouse_r, sst_dr, sst_drdn, n_x);
			const vrFloat M1 = M0 * jacob_eta ;
#endif

			const vrFloat F_1_ij_I = (M1*N_I_0)/(A*A);
#endif
			for (int index_rho=0;index_rho<nGaussPointSize_eta_In_Rho;++index_rho,++idx_g)
			{
				MyNotice;/*the order of theta and rho,(theta,rho)*/
				auto curRows = refDataSST3D.m_gaussQuadrature_xi_eta_polar.row(idx_g);
				MyVec2ParamSpace theta_rho;
				theta_rho[TriangleElem::idx_theta_doubleLayer] = curRows[TriangleElem::idx_theta_doubleLayer];
				theta_rho[TriangleElem::idx_rho_doubleLayer]=curRows[TriangleElem::idx_rho_doubleLayer];
				const vrFloat cur_rho = theta_rho[TriangleElem::idx_rho_doubleLayer];
				const vrFloat curWeight_doubleLayer = curRows[TriangleElem::idx_weight_doubleLayer];
				Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_theta_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_eta_polar.row(idx_g).x()));
				Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_rho_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_eta_polar.row(idx_g).y()));
				Q_ASSERT(numbers::isEqual(cur_theta_singlelayer,theta_rho[TriangleElem::idx_theta_doubleLayer]));

				const MyVec2ParamSpace currentSrcPtInParam/* in eta space*/ = refDataSST3D.m_SrcPt_in_eta;
				Q_ASSERT( ((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) /*compute_T_ij_I_SST*/);
				if (!((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))))
				{
					printf("compute_T_ij_I_SST : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
				}
				const MyVec2ParamSpace cur_eta = refDataSST3D.pc2eta(currentSrcPtInParam /*MyVec2ParamSpace(0.0,0.0)*/ MYNOTICE,theta_rho);
				const MyVec2ParamSpace cur_xi = refDataSST3D.eta2xi(cur_eta);
				const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

				MyFloat jacob_eta_nouse;/*jacob_eta = jacob_xi * mat_T_Inv*/
				MyVec3 normals_fieldpoint;
				MyFloat r;
				MyVec3 dr;
				MyFloat drdn;
				getKernelParameters_3D_SST(srcPos,fieldPoint,refDataSST3D,jacob_eta_nouse,normals_fieldpoint,r,dr,drdn);

				Q_ASSERT(numbers::isEqual(jacob_eta, jacob_eta_nouse));

				//printf("[%d,%d] r(%f)  rho*A(%f) \n",index_theta,index_rho,r,cur_rho*A);
				Q_ASSERT(numbers::isEqual(r,cur_rho*A));

				const vrFloat Tij = get_Tij_SST_3D_Aliabadi(idx_i, idx_j, r, dr,drdn,normals_fieldpoint);
				const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);

				const vrFloat SingularTerm_Tij_I = Tij * N_I * jacob_eta * theta_rho[TriangleElem::idx_rho_doubleLayer];


				const vrFloat SingularTerm_F_1_ij_I = (1.0/cur_rho)*(F_1_ij_I);

				doubleLayer_Term += (SingularTerm_Tij_I - SingularTerm_F_1_ij_I) * curWeight_doubleLayer;
			}

			const vrFloat beta = 1.0 / A;
			const vrFloat cur_Rho_hat = refDataSST3D.rho_hat(cur_theta_singlelayer);
			singleLayer_Term += F_1_ij_I * log( abs(cur_Rho_hat/beta) ) * curWeight_singleLayer;
		}

		return (doubleLayer_Term + singleLayer_Term);
	}

	vrFloat vrBEM3D::compute_T_ij_I_Aliabadi(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,vrInt idx_i,vrInt idx_j,vrInt idx_I)
	{
		vrFloat retVal = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();

#if SPEEDUP_5_31
		int tmp_GaussPointSize_xi_In_Theta = 0;
		int tmp_GaussPointSize_xi_In_Rho = 0;
		if (dis_regular == refDataSST3D.m_DisContinuousType)
		{
			tmp_GaussPointSize_xi_In_Theta = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Theta;
			tmp_GaussPointSize_xi_In_Rho = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Rho;
		}
		else
		{
			tmp_GaussPointSize_xi_In_Theta = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Theta_DisContinuous;
			tmp_GaussPointSize_xi_In_Rho = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Rho_DisContinuous;
		}
		const vrInt nGaussPointSize_xi_In_Theta = tmp_GaussPointSize_xi_In_Theta;
		const vrInt nGaussPointSize_xi_In_Rho = tmp_GaussPointSize_xi_In_Rho;
#endif
		const vrInt n_gpts = nGaussPointSize_xi_In_Theta * nGaussPointSize_xi_In_Rho;
		Q_ASSERT(n_gpts == refDataSST3D.m_gaussQuadrature_xi_polar.rows());

		for (int idx_g=0;idx_g < n_gpts;++idx_g)
		{
			MyNotice;/*the order of theta and rho,(theta,rho)*/
			auto curRows = refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g);
			MyVec2ParamSpace theta_rho;
			theta_rho[TriangleElem::idx_theta_doubleLayer] = curRows[TriangleElem::idx_theta_doubleLayer];
			theta_rho[TriangleElem::idx_rho_doubleLayer]=curRows[TriangleElem::idx_rho_doubleLayer];
			const vrFloat curWeight = curRows[TriangleElem::idx_weight_doubleLayer];
			Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_theta_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g).x()));
			Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_rho_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g).y()));

			const MyVec2ParamSpace currentSrcPtInParam /*in xi space*/ = refDataSST3D.m_SrcPt_in_xi;
			Q_ASSERT( ((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) /*compute_T_ij_I*/);
			if (!((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) )
			{
				printf("compute_T_ij_I : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
			}
			const MyVec2ParamSpace cur_xi = refDataSST3D.pc2xi(currentSrcPtInParam /*MyVec2ParamSpace(0.0,0.0)*/ MYNOTICE, theta_rho);
			const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

			MyFloat jacob_xi;
			MyVec3 normals_fieldpoint;
			MyFloat r;
			MyVec3 dr;
			MyFloat drdn;
			getKernelParameters_3D(srcPos,fieldPoint,refDataSST3D,jacob_xi,normals_fieldpoint,r,dr,drdn);
			//Q_ASSERT(numbers::isEqual(r,theta_rho[TriangleElemData::idx_rho]));


			const vrFloat Tij = get_Tij_SST_3D_Aliabadi(idx_i, idx_j, r, dr,drdn,normals_fieldpoint);
			const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);
			retVal += Tij * N_I * jacob_xi * theta_rho[TriangleElem::idx_rho_doubleLayer] * curWeight;
		}
		return retVal;
	}
#if 0

	void vrBEM3D::AssembleSystem_DisContinuous_DualEquation_Aliabadi(const vrInt v, const vrInt idx_i, const vrInt idx_j)
	{
#define DualEquation_Aliabadi_RegularSection (1)
#define DualEquation_Aliabadi_PositiveSection (1)
#define DualEquation_Aliabadi_NegativeSection (1)
		const MyInt ne = TriangleElem::getTriangleSize();		
		VertexPtr curSourcePtr = Vertex::getVertex(v);	
		const vrMat3& CMatrix = curSourcePtr->getCMatrix();
		const MyVec3I& srcDofs = curSourcePtr->getDofs();
		const bool isDisContinuousVtx = curSourcePtr->isDisContinuousVertex();
		const vrFloat Cij = CMatrix.coeff(idx_i,idx_j);


		const VertexTypeInDual srcPtType = curSourcePtr->getVertexTypeInDual();//Regular=0,Mirror_Positive=1, Mirror_Negative=2,CrackTip=3
		//const VertexTypeInDual srcPtType = Mirror_Negative;//Regular=0,Mirror_Positive=1, Mirror_Negative=2,CrackTip=3

		if (VertexTypeInDual::Regular == srcPtType)
		{
			//MyError("VertexTypeInDual::Regular == srcPtType || VertexTypeInDual::Mirror_Positive == srcPtType");
#if DualEquation_Aliabadi_RegularSection

			vrFloat Tij_I,Uij_I;
			for (int idx_e=0;idx_e<ne;++idx_e)
			{
				TriangleElemPtr curTriElemPtr =  TriangleElem::getTriangle(idx_e);
				const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();//dis_1_1=1, dis_1_2=2, dis_1_3=3, dis_2_3=4, dis_2_2=5, dis_2_1=6, dis_3_3=7, dis_regular=8

				vrInt srcPtIdx;
				bool ptInElem = isVertexInElement(curSourcePtr, curTriElemPtr, srcPtIdx);
				MyVec3I srcPt_SST_LookUp;			
				TriangleElemData data4SST_with_DisContinuous;
				if (ptInElem)
				{
					const DisContinuousType tmpTriElemDisContinuousType = TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,srcPtIdx);
					MyVec3 vtx_globalCoord[Geometry::vertexs_per_tri];

					for (int v=0;v<Geometry::vertexs_per_tri;++v)
					{
						srcPt_SST_LookUp[v] = srcPtIdx;
						vtx_globalCoord[v] = curTriElemPtr->getVertex(srcPtIdx)->getPos();
						srcPtIdx = (srcPtIdx+1) % Geometry::vertexs_per_tri;
					}
					data4SST_with_DisContinuous.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord);
				}	

				for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				{
					if (ptInElem)
					{
						//source point in triangle element.
						if (isDisContinuousVtx)
						{
							Q_ASSERT( (DisContinuousType::dis_regular != curTriElemDisContinuousType) );
#if USE_Aliabadi_RegularSample
							Tij_I = 0.0;
							Tij_I += compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(0, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
							Tij_I += compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(1, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);
							Tij_I += compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(2, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
#else//USE_Aliabadi_RegularSample
							Tij_I = 0.0;
							Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(0, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
							Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(1, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);
							Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(2, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
#endif//USE_Aliabadi_RegularSample

							/////////////////////////////////////////////////////////////////////////////////////////////////////
							Uij_I = compute_U_ij_I_Aliabadi( curSourcePtr, curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
							Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
							m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;

						}
						else
						{
							Tij_I = compute_T_ij_I_SST_Aliabadi(curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;

							/////////////////////////////////////////////////////////////////////////////////////////////////////
							Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);	
							Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
							m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;

						}
					}
					else
					{
						//source point do not locate in triangle element.
						Tij_I = compute_T_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);

						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;
						//printf("Tij_I_Aliabadi[%f]  Tij_I[%f]\n",Tij_I_tmp,Tij_I);vrPause;
						/////////////////////////////////////////////////////////////////////////////////////////////////////
						Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);	
						Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
						m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
					}
				}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)

			}//for (int idx_e=0;idx_e<ne;++idx_e)

			m_Hsubmatrix.coeffRef(srcDofs[idx_i],srcDofs[idx_j]) += Cij;

#endif//DualEquation_Aliabadi_RegularSection
		}
		else if (VertexTypeInDual::Mirror_Positive == srcPtType)
		{
#if DualEquation_Aliabadi_PositiveSection
			Q_ASSERT(curSourcePtr->isMirrorVertex());
#if USE_MI_NegativeSingular
			VertexPtr curSourcePtr_Positive_Ptr = curSourcePtr;
			VertexPtr curSource_Mirror_Negative_Ptr = curSourcePtr_Positive_Ptr->getMirrorVertex();
			const MyVec3I& src_Mirror_Negative_Dofs = curSource_Mirror_Negative_Ptr->getDofs();
			vrInt n_pt_Mirror_Negative_InElem = 0;
			vrInt n_ptInElem = 0;
#endif//USE_MI_NegativeSingular
			vrFloat Tij_I,Uij_I;
			for (int idx_e=0;idx_e<ne;++idx_e)
			{
				TriangleElemPtr curTriElemPtr =  TriangleElem::getTriangle(idx_e);

#if USE_MI_NegativeSingular
				const TriangleSetType curTriSetType = curTriElemPtr->getTriSetType();//Regular = 0, Positive = 1, Negative

				if (TriangleSetType::Regular == curTriSetType)
				{

					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						//source point do not locate in triangle element.
						Tij_I = compute_T_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);

						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;
						//printf("Tij_I_Aliabadi[%f]  Tij_I[%f]\n",Tij_I_tmp,Tij_I);vrPause;
						/////////////////////////////////////////////////////////////////////////////////////////////////////
						Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);	
						Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
						m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else if (TriangleSetType::Positive == curTriSetType)
				{
					const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();//dis_1_1=1, dis_1_2=2, dis_1_3=3, dis_2_3=4, dis_2_2=5, dis_2_1=6, dis_3_3=7, dis_regular=8
					Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);

					vrInt srcPtIdx;
					bool ptInElem = isVertexInElement(curSourcePtr, curTriElemPtr, srcPtIdx);
					MyVec3I srcPt_SST_LookUp;			
					TriangleElemData data4SST_with_DisContinuous;
					if (ptInElem)
					{
						n_ptInElem++; 
						const DisContinuousType tmpTriElemDisContinuousType = TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,srcPtIdx);
						MyVec3 vtx_globalCoord[Geometry::vertexs_per_tri];

						for (int v=0;v<Geometry::vertexs_per_tri;++v)
						{
							srcPt_SST_LookUp[v] = srcPtIdx;
							vtx_globalCoord[v] = curTriElemPtr->getVertex(srcPtIdx)->getPos();
							srcPtIdx = (srcPtIdx+1) % Geometry::vertexs_per_tri;
						}
						data4SST_with_DisContinuous.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord);
					}

					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						if (ptInElem)
						{
							Q_ASSERT(isDisContinuousVtx);
							Q_ASSERT( (DisContinuousType::dis_regular != curTriElemDisContinuousType) );
#if USE_Aliabadi_RegularSample
							Tij_I = 0.0;
							Tij_I += compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(0, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
							Tij_I += compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(1, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);
							Tij_I += compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(2, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
#else//USE_Aliabadi_RegularSample
							Tij_I = 0.0;
							Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(0, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
							Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(1, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);
							Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(2, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
#endif//USE_Aliabadi_RegularSample

							/////////////////////////////////////////////////////////////////////////////////////////////////////
							Uij_I = compute_U_ij_I_Aliabadi( curSourcePtr, curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
							Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
							m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
						}
						else
						{
							//source point do not locate in triangle element.
							Tij_I = compute_T_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;
							//printf("Tij_I_Aliabadi[%f]  Tij_I[%f]\n",Tij_I_tmp,Tij_I);vrPause;
							/////////////////////////////////////////////////////////////////////////////////////////////////////
							Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);	
							Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
							m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
						}
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else if (TriangleSetType::Negative == curTriSetType)
				{
					const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();//dis_1_1=1, dis_1_2=2, dis_1_3=3, dis_2_3=4, dis_2_2=5, dis_2_1=6, dis_3_3=7, dis_regular=8
					vrInt src_Mirror_Negative_PtIdx;
					bool pt_Mirror_Negative_InElem = isVertexInElement(curSource_Mirror_Negative_Ptr, curTriElemPtr, src_Mirror_Negative_PtIdx);
					const bool isDisContinuousVtx_Mirror_Negative = curSource_Mirror_Negative_Ptr->isDisContinuousVertex();
					Q_ASSERT(isDisContinuousVtx_Mirror_Negative);

					MyVec3 srcPt_Mirror_Negative_SST_LookUp;
					TriangleElemData data4SST_Mirror_Negative_DisContinuous;

					if (pt_Mirror_Negative_InElem)
					{
						n_pt_Mirror_Negative_InElem++;

						const DisContinuousType tmpTriElemDisContinuousType = 
							TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,src_Mirror_Negative_PtIdx);
						MyVec3 vtx_globalCoord[Geometry::vertexs_per_tri];

						for (int v=0;v<Geometry::vertexs_per_tri;++v)
						{
							srcPt_Mirror_Negative_SST_LookUp[v] = src_Mirror_Negative_PtIdx;
							vtx_globalCoord[v] = curTriElemPtr->getVertex(src_Mirror_Negative_PtIdx)->getPos();
							src_Mirror_Negative_PtIdx = (src_Mirror_Negative_PtIdx + 1) % Geometry::vertexs_per_tri;
						}
						data4SST_Mirror_Negative_DisContinuous.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord);
					}

					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						if (pt_Mirror_Negative_InElem)
						{
							Q_ASSERT(isDisContinuousVtx_Mirror_Negative);
							Q_ASSERT( (DisContinuousType::dis_regular != curTriElemDisContinuousType) );
#if USE_Aliabadi_RegularSample
							Tij_I = 0.0;
							Tij_I += compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(0, curSource_Mirror_Negative_Ptr,data4SST_Mirror_Negative_DisContinuous,idx_i,idx_j,idx_I);	
							Tij_I += compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(1, curSource_Mirror_Negative_Ptr,data4SST_Mirror_Negative_DisContinuous,idx_i,idx_j,idx_I);
							Tij_I += compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(2, curSource_Mirror_Negative_Ptr,data4SST_Mirror_Negative_DisContinuous,idx_i,idx_j,idx_I);	
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_Mirror_Negative_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I ;
#else//USE_Aliabadi_RegularSample
							Tij_I = 0.0;
							Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(0, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
							Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(1, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);
							Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(2, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
#endif//USE_Aliabadi_RegularSample

							/////////////////////////////////////////////////////////////////////////////////////////////////////
							Uij_I = compute_U_ij_I_Aliabadi( curSourcePtr, curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
							Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
							m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
						}
						else
						{
							//source point do not locate in triangle element.
							Tij_I = compute_T_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;
							//printf("Tij_I_Aliabadi[%f]  Tij_I[%f]\n",Tij_I_tmp,Tij_I);vrPause;
							/////////////////////////////////////////////////////////////////////////////////////////////////////
							Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);	
							Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
							m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
						}
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else
				{
					MyError("Unsupport Triangle element type.");
				}


#else//USE_MI_NegativeSingular

				const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();//dis_1_1=1, dis_1_2=2, dis_1_3=3, dis_2_3=4, dis_2_2=5, dis_2_1=6, dis_3_3=7, dis_regular=8

				vrInt srcPtIdx;
				bool ptInElem = isVertexInElement(curSourcePtr, curTriElemPtr, srcPtIdx);
				MyVec3I srcPt_SST_LookUp;			
				TriangleElemData data4SST_with_DisContinuous;
				if (ptInElem)
				{
					const DisContinuousType tmpTriElemDisContinuousType = TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,srcPtIdx);
					MyVec3 vtx_globalCoord[Geometry::vertexs_per_tri];

					for (int v=0;v<Geometry::vertexs_per_tri;++v)
					{
						srcPt_SST_LookUp[v] = srcPtIdx;
						vtx_globalCoord[v] = curTriElemPtr->getVertex(srcPtIdx)->getPos();
						srcPtIdx = (srcPtIdx+1) % Geometry::vertexs_per_tri;
					}
					data4SST_with_DisContinuous.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord);
				}	

				for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				{
					if (ptInElem)
					{
						//source point in triangle element.
						if (isDisContinuousVtx)
						{
							Q_ASSERT( (DisContinuousType::dis_regular != curTriElemDisContinuousType) );
#if USE_Aliabadi_RegularSample
							Tij_I = 0.0;
							Tij_I += compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(0, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
							Tij_I += compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(1, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);
							Tij_I += compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(2, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
#else//USE_Aliabadi_RegularSample
							Tij_I = 0.0;
							Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(0, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
							Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(1, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);
							Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(2, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
#endif//USE_Aliabadi_RegularSample

							/////////////////////////////////////////////////////////////////////////////////////////////////////
							Uij_I = compute_U_ij_I_Aliabadi( curSourcePtr, curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
							Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
							m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;

						}
						else
						{
							Tij_I = compute_T_ij_I_SST_Aliabadi(curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;

							/////////////////////////////////////////////////////////////////////////////////////////////////////
							Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);	
							Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
							m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;

						}
					}
					else
					{
						//source point do not locate in triangle element.
						Tij_I = compute_T_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);

						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;
						//printf("Tij_I_Aliabadi[%f]  Tij_I[%f]\n",Tij_I_tmp,Tij_I);vrPause;
						/////////////////////////////////////////////////////////////////////////////////////////////////////
						Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);	
						Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
						m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
					}
				}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)


#endif//USE_MI_NegativeSingular
			}//for (int idx_e=0;idx_e<ne;++idx_e)
			Q_ASSERT(1 == n_ptInElem && 1 == n_pt_Mirror_Negative_InElem);
			Q_ASSERT(curSourcePtr->isMirrorVertex());
			//VertexPtr curSource_Mirror_Negative_Ptr = curSourcePtr->getMirrorVertex();
			const MyVec3I& mirrorDofs_Negative = curSource_Mirror_Negative_Ptr->getDofs();

			const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
			m_Hsubmatrix.coeffRef(srcDofs[idx_i],srcDofs[idx_j]) += 0.5 * delta_ij;
			m_Hsubmatrix.coeffRef(srcDofs[idx_i],mirrorDofs_Negative[idx_j]) += 0.5 * delta_ij;

#endif//DualEquation_Aliabadi_PositiveSection
		}
		else if (VertexTypeInDual::Mirror_Negative  == srcPtType)
		{

			static std::set< vrInt > ptIdSet;
			ptIdSet.insert(v);
			printf("VertexTypeInDual::Mirror_Negative == srcPtType  [%d] \n", ptIdSet.size());
#if DualEquation_Aliabadi_NegativeSection
			Q_ASSERT(curSourcePtr->isMirrorVertex());

#if USE_MI_NegativeSingular
			vrInt n_ptInElem_negative = 0;
			vrInt n_ptInElem_positive = 0;
#endif//USE_MI_NegativeSingular

			VertexPtr curSourcePtr_negative = curSourcePtr; MYNOTICE;//current source point locate on the negative element
			const MyVec3I& srcDofs_negative = curSourcePtr_negative->getDofs();
			Q_ASSERT(isDisContinuousVtx);
			Q_ASSERT(1 == (curSourcePtr_negative->getShareElement().size()));

			const bool isDisContinuousVtx_source_negative = curSourcePtr_negative->isDisContinuousVertex();

			const vrFloat n_j_s = (curSourcePtr_negative->getVertexNormal())[idx_j];
			vrFloat Kij_I,Sij_I;
			for (int idx_e=0;idx_e<ne;++idx_e)
			{
				TriangleElemPtr curTriElemPtr =  TriangleElem::getTriangle(idx_e);
				const TriangleSetType curTriSetType = curTriElemPtr->getTriSetType();//Regular = 0, Positive = 1, Negative
#if USE_MI_NegativeSingular
				if (TriangleSetType::Regular == curTriSetType)
				{
					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						for (int idx_k=0; idx_k < MyDim; ++idx_k)
						{
							Sij_I = compute_S_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I,idx_k);
							Sij_I = n_j_s * Sij_I;
							m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Sij_I;

							Kij_I = compute_K_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I,idx_k);
							Kij_I = n_j_s * Kij_I;
							m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Kij_I;
							//printf("Sij_I[%f]  Kij_I[%f]\n",Sij_I,Kij_I); vrPause;
						}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else if (TriangleSetType::Positive == curTriSetType)
				{
					const MyVec3I& srcDofs_negative = curSourcePtr_negative->getDofs();
					VertexPtr curSource_Mirror_Ptr_Positive = curSourcePtr_negative->getMirrorVertex();
					const MyVec3I& src_Mirror_Dofs_Positive = curSource_Mirror_Ptr_Positive->getDofs();
					const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();
					vrInt src_Mirror_PtIdx_Positive;
					bool pt_Mirror_InElem_Positive = isVertexInElement(curSource_Mirror_Ptr_Positive, curTriElemPtr, src_Mirror_PtIdx_Positive);

					MyVec3 srcPt_Mirror_SST_LookUp_Positive;
					TriangleElemData data4SST_Mirror_Positive;

					if (pt_Mirror_InElem_Positive)
					{
						n_ptInElem_positive++;
						Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
						const DisContinuousType tmpTriElemDisContinuousType = TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,src_Mirror_PtIdx_Positive);
						MyVec3 vtx_globalCoord_Positive[Geometry::vertexs_per_tri];

						for (int v=0;v<Geometry::vertexs_per_tri;++v)
						{
							srcPt_Mirror_SST_LookUp_Positive[v] = src_Mirror_PtIdx_Positive;
							vtx_globalCoord_Positive[v] = curTriElemPtr->getVertex(src_Mirror_PtIdx_Positive)->getPos();
							src_Mirror_PtIdx_Positive = (src_Mirror_PtIdx_Positive + 1) %Geometry::vertexs_per_tri;
						}
						data4SST_Mirror_Positive.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord_Positive);
					}

					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						if (pt_Mirror_InElem_Positive)
						{
							Q_ASSERT( curSource_Mirror_Ptr_Positive->isDisContinuousVertex());
							for (int idx_k=0; idx_k < MyDim; ++idx_k)
							{
								Sij_I = 0.0;
								Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(0, curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive,idx_i,idx_j,idx_I,idx_k);	
								Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(1, curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive,idx_i,idx_j,idx_I,idx_k);
								Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(2, curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive,idx_i,idx_j,idx_I,idx_k);	

								Sij_I = n_j_s * Sij_I;
								m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_Mirror_SST_LookUp_Positive[idx_I])->getDof(idx_k)) += Sij_I;

								Kij_I = 0.0;
								Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(0, curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive,idx_i,idx_j,idx_I,idx_k);
								Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(1, curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive,idx_i,idx_j,idx_I,idx_k);
								Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(2, curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive,idx_i,idx_j,idx_I,idx_k);
								Kij_I = n_j_s * Kij_I;
								m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_Mirror_SST_LookUp_Positive[idx_I])->getDof(idx_k)) += Kij_I;

							}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
						}
						else
						{
							for (int idx_k=0; idx_k < MyDim; ++idx_k)
							{
								Sij_I = compute_S_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I,idx_k);
								Sij_I = n_j_s * Sij_I;
								m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Sij_I;

								Kij_I = compute_K_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I,idx_k);
								Kij_I = n_j_s * Kij_I;
								m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Kij_I;
								//printf("Sij_I[%f]  Kij_I[%f]\n",Sij_I,Kij_I); vrPause;
							}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
						}
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)

				}
				else if (TriangleSetType::Negative == curTriSetType)
				{					
					const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();
					Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
					vrInt srcPtIdx_negative;
					bool ptInElem_negative = isVertexInElement(curSourcePtr_negative, curTriElemPtr, srcPtIdx_negative);
					MyVec3 srcPt_SST_LookUp_negative;
					TriangleElemData data4SST_negative;

					if (ptInElem_negative)
					{
						n_ptInElem_negative++;
						Q_ASSERT(TriangleSetType::Negative == curTriSetType);
						Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
						const DisContinuousType tmpTriElemDisContinuousType = TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,srcPtIdx_negative);
						Q_ASSERT(dis_3_3 == tmpTriElemDisContinuousType);
						MyVec3 vtx_globalCoord_negative[Geometry::vertexs_per_tri];

						for (int v=0;v<Geometry::vertexs_per_tri;++v)
						{
							srcPt_SST_LookUp_negative[v] = srcPtIdx_negative;
							vtx_globalCoord_negative[v] = curTriElemPtr->getVertex(srcPtIdx_negative)->getPos();
							srcPtIdx_negative = (srcPtIdx_negative+1) %Geometry::vertexs_per_tri;
						}
						data4SST_negative.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord_negative);
					}

					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						if (ptInElem_negative)
						{
							Q_ASSERT(isDisContinuousVtx_source_negative);
							for (int idx_k=0; idx_k < MyDim; ++idx_k)
							{
#if USE_Aliabadi_RegularSample
								Sij_I = 0.0;
								Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(0, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);	
								Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(1, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
								Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(2, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);	

								Sij_I = n_j_s * Sij_I;
								m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_k)) += Sij_I;

								Kij_I = 0.0;
								Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(0, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
								Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(1, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
								Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(2, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
								Kij_I = n_j_s * Kij_I;
								m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_k)) += Kij_I;

#endif//USE_Aliabadi_RegularSample
								//printf("SST : Sij_I[%f]  Kij_I[%f]\n",Sij_I,Kij_I); vrPause;
							}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
						}
						else
						{
							for (int idx_k=0; idx_k < MyDim; ++idx_k)
							{
								Sij_I = compute_S_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I,idx_k);
								Sij_I = n_j_s * Sij_I;
								m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Sij_I;

								Kij_I = compute_K_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I,idx_k);
								Kij_I = n_j_s * Kij_I;
								m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Kij_I;
								//printf("Sij_I[%f]  Kij_I[%f]\n",Sij_I,Kij_I); vrPause;
							}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
						}
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else
				{
					MyError("Unsupport Triangle element type.");
				}
#else//USE_MI_NegativeSingular

				const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();

				//Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
				vrInt srcPtIdx_negative;
				bool ptInElem_negative = isVertexInElement(curSourcePtr_negative, curTriElemPtr, srcPtIdx_negative);
				//Q_ASSERT(ptInElem_negative);

				MyVec3 srcPt_SST_LookUp_negative;
				TriangleElemData data4SST_negative;

				if (ptInElem_negative)
				{
					Q_ASSERT(TriangleSetType::Negative == curTriSetType);
					Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
					const DisContinuousType tmpTriElemDisContinuousType = TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,srcPtIdx_negative);
					Q_ASSERT(dis_3_3 == tmpTriElemDisContinuousType);
					MyVec3 vtx_globalCoord_negative[Geometry::vertexs_per_tri];

					for (int v=0;v<Geometry::vertexs_per_tri;++v)
					{
						srcPt_SST_LookUp_negative[v] = srcPtIdx_negative;
						vtx_globalCoord_negative[v] = curTriElemPtr->getVertex(srcPtIdx_negative)->getPos();
						srcPtIdx_negative = (srcPtIdx_negative+1) %Geometry::vertexs_per_tri;
					}
					data4SST_negative.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord_negative);
				}

				for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				{
					if (ptInElem_negative)
					{
						Q_ASSERT(TriangleSetType::Negative == curTriSetType);
						Q_ASSERT(isDisContinuousVtx_source_negative);
#if USE_Peng_Kernel
						int idx_k = 999;
						{
							Sij_I = 0.0;
							Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(0, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);	
							Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(1, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
							Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(2, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);	


							m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_j)) += Sij_I;

							Kij_I = 0.0;
							Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(0, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
							Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(1, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
							Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(2, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);

							m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_j)) += Kij_I;

							//printf("SST : Sij_I[%f]  Kij_I[%f]\n",Sij_I,Kij_I); vrPause;
						}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
#else
						for (int idx_k=0; idx_k < MyDim; ++idx_k)
						{
#if USE_Aliabadi_RegularSample
							Sij_I = 0.0;
							Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(0, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);	
							Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(1, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
							Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(2, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);	

							Sij_I = n_j_s * Sij_I;
							m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_k)) += Sij_I;

							Kij_I = 0.0;
							Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(0, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
							Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(1, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
							Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(2, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
							Kij_I = n_j_s * Kij_I;
							m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_k)) += Kij_I;

#else//USE_Aliabadi_RegularSample
							Sij_I = 0.0;
							Sij_I += compute_S_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(0, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);	
							Sij_I += compute_S_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(1, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
							Sij_I += compute_S_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(2, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);	

							Sij_I = n_j_s * Sij_I;
							m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_k)) += Sij_I;

							Kij_I = 0.0;
							Kij_I += compute_K_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(0, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
							Kij_I += compute_K_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(1, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
							Kij_I += compute_K_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(2, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
							Kij_I = n_j_s * Kij_I;
							m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_k)) += Kij_I;

#endif//USE_Aliabadi_RegularSample
							//printf("SST : Sij_I[%f]  Kij_I[%f]\n",Sij_I,Kij_I); vrPause;
						}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
#endif
					}
					else
					{
#if USE_Peng_Kernel
						int idx_k=999;
						{
							Sij_I = compute_S_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I,idx_k);

							m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Sij_I;

							Kij_I = compute_K_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I,idx_k);

							m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Kij_I;
							//printf("Sij_I[%f]  Kij_I[%f]\n",Sij_I,Kij_I); vrPause;
						}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
#else
						for (int idx_k=0; idx_k < MyDim; ++idx_k)
						{
							Sij_I = compute_S_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I,idx_k);
							Sij_I = n_j_s * Sij_I;
							m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Sij_I;

							Kij_I = compute_K_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I,idx_k);
							Kij_I = n_j_s * Kij_I;
							m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Kij_I;
							//printf("Sij_I[%f]  Kij_I[%f]\n",Sij_I,Kij_I); vrPause;
						}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
#endif
					}
				}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
#endif//USE_MI_NegativeSingular
			}//for (int idx_e=0;idx_e<ne;++idx_e)
			Q_ASSERT(1 == n_ptInElem_positive && 1 == n_ptInElem_negative);
			//const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
			const vrFloat delta_ij = 1.0;
			//MyNoticeMsg("G Matrix.") m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],srcDofs_negative[idx_j]) +=  0.5 MyNoticeMsg("No delta ij.");
			MyNoticeMsg("G Matrix.") m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],srcDofs_negative[idx_j]) +=  0.5 * delta_ij MyNoticeMsg("No delta ij.");

			Q_ASSERT(curSourcePtr->isMirrorVertex());
			VertexPtr curSource_Mirror_Positive_Ptr = curSourcePtr->getMirrorVertex();
			const MyVec3I& mirrorDofs_Positive = curSource_Mirror_Positive_Ptr->getDofs();
			MyNoticeMsg("G Matrix.") m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],mirrorDofs_Positive[idx_j]) +=  -0.5 * delta_ij MyNoticeMsg("No delta ij.");
#endif//DualEquation_Aliabadi_NegativeSection
		}
		else
		{
			MyError("Unsupport source point type in AssembleSystem_DisContinuous_DualEquation_Aliabadi.");
		}
	}

#else//if 0

#define DualEquation_Aliabadi_RegularSection (1)
#define DualEquation_Aliabadi_PositiveSection (1)
#define DualEquation_Aliabadi_NegativeSection (1)

#if USE_UniformSampling
	std::set<int> vrBEM3D::m_set_colIdx;

	void vrBEM3D::DualEquation_Regular_Peng(const vrInt idx_i, const vrInt idx_j, const int ne, const vrFloat Cij, VertexPtr curSourcePtr)
	{
		const MyVec3I& srcDofs = curSourcePtr->getDofs();
		const bool isDisContinuousVtx = curSourcePtr->isDisContinuousVertex();		
		vrFloat Tij_I,Uij_I;
		const VertexTypeInDual srcPtType = curSourcePtr->getVertexTypeInDual();//Regular=0,Mirror_Positive=1, Mirror_Negative=2,CrackTip=3
		Q_ASSERT(VertexTypeInDual::Regular == srcPtType);
		for (int idx_e=0;idx_e<ne;++idx_e)
		{
			TriangleElemPtr curTriElemPtr =  TriangleElem::getTriangle(idx_e);
			const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();//dis_1_1=1, dis_1_2=2, dis_1_3=3, dis_2_3=4, dis_2_2=5, dis_2_1=6, dis_3_3=7, dis_regular=8

			vrInt srcPtIdx;
			bool ptInElem = isVertexInElement(curSourcePtr, curTriElemPtr, srcPtIdx);
			MyVec3I srcPt_SST_LookUp;			
			TriangleElemData * ptr_data4SST_with_DisContinuous = NULL;
			if (ptInElem)
			{
				for (int v=0;v<Geometry::vertexs_per_tri;++v)
				{
					srcPt_SST_LookUp[v] = (srcPtIdx+v) % Geometry::vertexs_per_tri;
				}
				ptr_data4SST_with_DisContinuous = &(curTriElemPtr->get_m_data_SST_3D(srcPtIdx));					
			}	

			for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
			{
				if (ptInElem)
				{
					//source point in triangle element.
					if (isDisContinuousVtx)
					{
						Q_ASSERT( (DisContinuousType::dis_regular != curTriElemDisContinuousType) );
						Tij_I = compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(curSourcePtr,(*ptr_data4SST_with_DisContinuous),idx_i,idx_j,idx_I);	
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
						/////////////////////////////////////////////////////////////////////////////////////////////////////							
						Uij_I = compute_U_ij_I_SST_DisContinuous_Regular_Aliabadi(curSourcePtr,(*ptr_data4SST_with_DisContinuous),idx_i,idx_j,idx_I);	
						m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Uij_I;

					}
					else
					{
						Tij_I = compute_T_ij_I_SST_Aliabadi(curSourcePtr, (*ptr_data4SST_with_DisContinuous),idx_i,idx_j,idx_I);	
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
						/////////////////////////////////////////////////////////////////////////////////////////////////////
						Uij_I = compute_U_ij_I_SST_Aliabadi(curSourcePtr, (*ptr_data4SST_with_DisContinuous),idx_i,idx_j,idx_I);	
						Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
						m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Uij_I;
						//m_set_colIdx.insert(curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j));
					}
				}
				else
				{
					//source point do not locate in triangle element.
					Tij_I = compute_T_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I);
					m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;
					/////////////////////////////////////////////////////////////////////////////////////////////////////
					Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I);	
					Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
					m_Gsubmatrix.coeffRef(srcDofs[idx_i], curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
					//m_set_colIdx.insert(curTriElemPtr->getVertex(idx_I)->getDof(idx_j));
				}
			}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)

		}//for (int idx_e=0;idx_e<ne;++idx_e)
		m_Hsubmatrix.coeffRef(srcDofs[idx_i], srcDofs[idx_j]) += Cij;
	}

	void vrBEM3D::DualEquation_Positive_Peng(const vrInt idx_i, const vrInt idx_j, const int ne, const vrFloat Cij, VertexPtr curSourcePtr)
	{
		const MyVec3I& srcDofs = curSourcePtr->getDofs();
		const bool isDisContinuousVtx = curSourcePtr->isDisContinuousVertex();		
		vrFloat Tij_I,Uij_I;
		const VertexTypeInDual srcPtType = curSourcePtr->getVertexTypeInDual();//Regular=0,Mirror_Positive=1, Mirror_Negative=2,CrackTip=3
		Q_ASSERT(VertexTypeInDual::Mirror_Positive == srcPtType);

		Q_ASSERT(curSourcePtr->isMirrorVertex());
		VertexPtr curSourcePtr_Positive_Ptr = curSourcePtr;
		VertexPtr curSource_Mirror_Negative_Ptr = curSourcePtr_Positive_Ptr->getMirrorVertex();
		const MyVec3I& src_Mirror_Negative_Dofs = curSource_Mirror_Negative_Ptr->getDofs();
		vrInt n_pt_Mirror_Negative_InElem = 0;
		vrInt n_ptInElem = 0;
		for (int idx_e=0;idx_e<ne;++idx_e)
		{
			TriangleElemPtr curTriElemPtr =  TriangleElem::getTriangle(idx_e);
#if USE_MI_NegativeSingular
			const TriangleSetType curTriSetType = curTriElemPtr->getTriSetType();//Regular = 0, Positive = 1, Negative

			if (TriangleSetType::Regular == curTriSetType)
			{
				for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				{
					//source point do not locate in triangle element.
					Tij_I = compute_T_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I);
					m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;
					/////////////////////////////////////////////////////////////////////////////////////////////////////
					Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I);	
					Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
					m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
				}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
			}
			else if (TriangleSetType::Positive == curTriSetType)
			{
				const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();//dis_1_1=1, dis_1_2=2, dis_1_3=3, dis_2_3=4, dis_2_2=5, dis_2_1=6, dis_3_3=7, dis_regular=8
				Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);

				vrInt srcPtIdx;
				bool ptInElem = isVertexInElement(curSourcePtr, curTriElemPtr, srcPtIdx);
				MyVec3I srcPt_SST_LookUp;			
				TriangleElemData * ptr_data4SST_with_DisContinuous = NULL;
				if (ptInElem)
				{
					n_ptInElem++; 
					for (int v=0;v<Geometry::vertexs_per_tri;++v)
					{
						srcPt_SST_LookUp[v] = (srcPtIdx+v) % Geometry::vertexs_per_tri;
					}
					ptr_data4SST_with_DisContinuous = &(curTriElemPtr->get_m_data_SST_3D(srcPtIdx));
				}

				for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				{
					if (ptInElem)
					{
						Q_ASSERT(isDisContinuousVtx);
						Q_ASSERT( (DisContinuousType::dis_regular != curTriElemDisContinuousType) );
						Tij_I = compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(curSourcePtr, (*ptr_data4SST_with_DisContinuous),idx_i,idx_j,idx_I);	
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
						/////////////////////////////////////////////////////////////////////////////////////////////////////
						//				Uij_I = compute_U_ij_I_SST_DisContinuous_Regular_Aliabadi(curSourcePtr,(*ptr_data4SST_with_DisContinuous),idx_i,idx_j,idx_I);	
						//				m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Uij_I;
					}
					else
					{
						//source point do not locate in triangle element.
						Tij_I = compute_T_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I);
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;
						/////////////////////////////////////////////////////////////////////////////////////////////////////
						//				Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I);	
						//				Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
						//				m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
					}
				}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
			}
			else if (TriangleSetType::Negative == curTriSetType)
			{
				const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();//dis_1_1=1, dis_1_2=2, dis_1_3=3, dis_2_3=4, dis_2_2=5, dis_2_1=6, dis_3_3=7, dis_regular=8
				vrInt src_Mirror_Negative_PtIdx;
				bool pt_Mirror_Negative_InElem = isVertexInElement(curSource_Mirror_Negative_Ptr, curTriElemPtr, src_Mirror_Negative_PtIdx);
				const bool isDisContinuousVtx_Mirror_Negative = curSource_Mirror_Negative_Ptr->isDisContinuousVertex();
				Q_ASSERT(isDisContinuousVtx_Mirror_Negative);

				MyVec3 srcPt_Mirror_Negative_SST_LookUp;

				TriangleElemData * ptr_data4SST_Mirror_Negative_DisContinuous = NULL;
				if (pt_Mirror_Negative_InElem)
				{
					n_pt_Mirror_Negative_InElem++;
					for (int v=0;v<Geometry::vertexs_per_tri;++v)
					{
						srcPt_Mirror_Negative_SST_LookUp[v] = (src_Mirror_Negative_PtIdx + v) % Geometry::vertexs_per_tri;
					}
					ptr_data4SST_Mirror_Negative_DisContinuous = &(curTriElemPtr->get_m_data_SST_3D(src_Mirror_Negative_PtIdx));
				}

				for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				{
					if (pt_Mirror_Negative_InElem)
					{
						Q_ASSERT(isDisContinuousVtx_Mirror_Negative);
						Q_ASSERT( (DisContinuousType::dis_regular != curTriElemDisContinuousType) );
						Tij_I = compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(curSource_Mirror_Negative_Ptr, (*ptr_data4SST_Mirror_Negative_DisContinuous),idx_i,idx_j,idx_I);	
						m_Hsubmatrix.coeffRef(srcDofs[idx_i] MyNoticeMsg("Use Mirror negative point."),curTriElemPtr->getVertex(srcPt_Mirror_Negative_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
						/////////////////////////////////////////////////////////////////////////////////////////////////////
						//				Uij_I = compute_U_ij_I_SST_DisContinuous_Regular_Aliabadi(curSource_Mirror_Negative_Ptr,(*ptr_data4SST_Mirror_Negative_DisContinuous),idx_i,idx_j,idx_I);	
						//				m_Gsubmatrix.coeffRef(srcDofs[idx_i] MyNoticeMsg("Use Mirror negative point."),curTriElemPtr->getVertex(srcPt_Mirror_Negative_SST_LookUp[idx_I])->getDof(idx_j)) += Uij_I;
					} 
					else
					{
						//source point do not locate in triangle element.
						Tij_I = compute_T_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I);
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;
						/////////////////////////////////////////////////////////////////////////////////////////////////////
						//				Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I);	
						//				Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
						//				m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
					}
				}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
			}
			else
			{
				MyError("Unsupport Triangle element type.");
			}
#endif//USE_MI_NegativeSingular
		}//for (int idx_e=0;idx_e<ne;++idx_e)
		Q_ASSERT(1 == n_ptInElem && 1 == n_pt_Mirror_Negative_InElem);
		Q_ASSERT(curSourcePtr->isMirrorVertex());
		//VertexPtr curSource_Mirror_Negative_Ptr = curSourcePtr->getMirrorVertex();
		const MyVec3I& mirrorDofs_Negative = curSource_Mirror_Negative_Ptr->getDofs();

		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
		m_Hsubmatrix.coeffRef(srcDofs[idx_i],srcDofs[idx_j]) += 0.5 * delta_ij;
		m_Hsubmatrix.coeffRef(srcDofs[idx_i],mirrorDofs_Negative[idx_j]) += 0.5 * delta_ij;
	}

	vrFloat vrBEM3D::compute_K_ij_I_Peng(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D, const vrInt idx_i, const vrInt idx_j, const  vrInt idx_I)
	{
		vrFloat retVal = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();
		const MyVec3& n_s = curSourcePtr->getVertexNormal();MyNotice;
#if SPEEDUP_5_31
		int tmp_GaussPointSize_xi_In_Theta = 0;
		int tmp_GaussPointSize_xi_In_Rho = 0;
		if (dis_regular == refDataSST3D.m_DisContinuousType)
		{
			tmp_GaussPointSize_xi_In_Theta = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Theta;
			tmp_GaussPointSize_xi_In_Rho = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Rho;
		}
		else
		{
			tmp_GaussPointSize_xi_In_Theta = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Theta_DisContinuous;
			tmp_GaussPointSize_xi_In_Rho = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Rho_DisContinuous;
		}
		const vrInt nGaussPointSize_xi_In_Theta = tmp_GaussPointSize_xi_In_Theta;
		const vrInt nGaussPointSize_xi_In_Rho = tmp_GaussPointSize_xi_In_Rho;
#endif
		const vrInt n_gpts = nGaussPointSize_xi_In_Theta * nGaussPointSize_xi_In_Rho;
		Q_ASSERT(n_gpts == refDataSST3D.m_gaussQuadrature_xi_polar.rows());

		for (int idx_g=0;idx_g < n_gpts;++idx_g)
		{
			MyNotice;/*the order of theta and rho,(theta,rho)*/
			auto curRows = refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g);
			MyVec2ParamSpace theta_rho;
			theta_rho[TriangleElem::idx_theta_doubleLayer] = curRows[TriangleElem::idx_theta_doubleLayer];
			theta_rho[TriangleElem::idx_rho_doubleLayer]=curRows[TriangleElem::idx_rho_doubleLayer];
			const vrFloat curWeight = curRows[TriangleElem::idx_weight_doubleLayer];
			Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_theta_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g).x()));
			Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_rho_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g).y()));

			const MyVec2ParamSpace currentSrcPtInParam /*in xi space*/ = refDataSST3D.m_SrcPt_in_xi;
			Q_ASSERT( ((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) /*compute_T_ij_I*/);
			if (!((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) )
			{
				printf("compute_T_ij_I : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
			}
			const MyVec2ParamSpace cur_xi = refDataSST3D.pc2xi( currentSrcPtInParam MYNOTICE,theta_rho);
			const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

			MyFloat jacob_xi;
			MyVec3 unitNormals_fieldPt;
			MyFloat r;
			MyVec3 dr;
			MyFloat drdn;
			getKernelParameters_3D(srcPos,fieldPoint,refDataSST3D,jacob_xi,unitNormals_fieldPt,r,dr,drdn);

			vrFloat Kij = 0.0;
			for (int idx_k=0; idx_k < MyDim; ++idx_k)
			{
				Kij += get_Kij_SST_3D_k_Aliabadi_Peng(idx_i, idx_j, idx_k, r, dr, drdn,n_s);
			}

			const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);
			retVal += Kij * N_I * jacob_xi * theta_rho[TriangleElem::idx_rho_doubleLayer] * curWeight;
		}

		return retVal;
	}

	vrFloat vrBEM3D::compute_S_ij_I_Peng(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D, const vrInt idx_i, const vrInt idx_j, const  vrInt idx_I)
	{
		vrFloat retVal = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();
		const MyVec3& n_s = curSourcePtr->getVertexNormal();MyNotice;

		int tmp_GaussPointSize_xi_In_Theta = 0;
		int tmp_GaussPointSize_xi_In_Rho = 0;
		if (dis_regular == refDataSST3D.m_DisContinuousType)
		{
			tmp_GaussPointSize_xi_In_Theta = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Theta;
			tmp_GaussPointSize_xi_In_Rho = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Rho;
		}
		else
		{
			tmp_GaussPointSize_xi_In_Theta = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Theta_DisContinuous;
			tmp_GaussPointSize_xi_In_Rho = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Rho_DisContinuous;
		}
		const vrInt nGaussPointSize_xi_In_Theta = tmp_GaussPointSize_xi_In_Theta;
		const vrInt nGaussPointSize_xi_In_Rho = tmp_GaussPointSize_xi_In_Rho;
		const vrInt n_gpts = nGaussPointSize_xi_In_Theta * nGaussPointSize_xi_In_Rho;
		Q_ASSERT(n_gpts == refDataSST3D.m_gaussQuadrature_xi_polar.rows());

		for (int idx_g=0;idx_g < n_gpts;++idx_g)
		{
			auto curRows = refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g);
			MyVec2ParamSpace theta_rho;
			theta_rho[TriangleElem::idx_theta_doubleLayer] = curRows[TriangleElem::idx_theta_doubleLayer];
			theta_rho[TriangleElem::idx_rho_doubleLayer]=curRows[TriangleElem::idx_rho_doubleLayer];
			const vrFloat curWeight = curRows[TriangleElem::idx_weight_doubleLayer];
			Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_theta_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g).x()));
			Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_rho_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g).y()));

			const MyVec2ParamSpace currentSrcPtInParam /*in xi space*/ = refDataSST3D.m_SrcPt_in_xi;
			Q_ASSERT( ((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) /*compute_T_ij_I*/);
			if (!((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) )
			{
				printf("compute_S_ij_I : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
			}
			const MyVec2ParamSpace cur_xi = refDataSST3D.pc2xi(currentSrcPtInParam MYNOTICE, theta_rho);
			const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

			MyFloat jacob_xi;
			MyVec3 unitNormals_fieldPt;
			MyFloat r;
			MyVec3 dr;
			MyFloat drdn;
			getKernelParameters_3D(srcPos,fieldPoint,refDataSST3D,jacob_xi,unitNormals_fieldPt,r,dr,drdn);
			vrFloat Sij_k = 0.0;
			for (int idx_k=0; idx_k < MyDim; ++idx_k)
			{
				Sij_k += get_Sij_SST_3D_k_Aliabadi_Peng(idx_i, idx_j, idx_k, r, dr, drdn, unitNormals_fieldPt, n_s);
			}
			const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);
			retVal += Sij_k * N_I * jacob_xi * theta_rho[TriangleElem::idx_rho_doubleLayer] * curWeight;

		}//for (int idx_g=0;idx_g < n_gpts;++idx_g)
		return retVal;
	}

	vrFloat vrBEM3D::compute_S_ij_I_SST_DisContinuous_Peng(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,const vrInt idx_i,const vrInt idx_j,const vrInt idx_I)
	{
		vrFloat retVal = 0.0;
		vrFloat doubleLayer_Term_k = 0.0, singleLayer_Term_k = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();
		const MyVec3& n_s = curSourcePtr->getVertexNormal();MyNotice;

		const vrInt nGaussPointSize_eta_In_Theta_SubTri_Regular = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_SubTri;
		const vrInt nGaussPointSize_eta_In_Rho_SubTri_Regular = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Rho_SubTri;
		const vrInt n_gpts = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_SubTri * GlobalConf::g_n_Sample_GaussPointSize_eta_In_Rho_SubTri;
#if USE_360_Sample
		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri_Regular = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal;
		const MyMatrix& cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Regular = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal;
#else//USE_360_Sample
		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri_Regular = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal[nSubTriIdx];
		const MyMatrix& cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Regular = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal[nSubTriIdx];
#endif//USE_360_Sample
		if (n_gpts != cur_gaussQuadrature_xi_eta_polar_SubTri_Regular.rows())
		{
			printf("compute_S_ij_I_SST_DisContinuous_Sigmoidal : n_gpts[%d] == cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows()[%d] \n",n_gpts, cur_gaussQuadrature_xi_eta_polar_SubTri_Regular.rows());
		}
		Q_ASSERT(n_gpts == cur_gaussQuadrature_xi_eta_polar_SubTri_Regular.rows());
		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
		for (int index_theta=0,idx_g=0;index_theta<nGaussPointSize_eta_In_Theta_SubTri_Regular;++index_theta)
		{
			const vrFloat cur_theta_singlelayer_Regular = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Regular.row(index_theta)[TriangleElem::idx_theta_singleLayer];
			const vrFloat A = refDataSST3D.A_theta_SubTri(cur_theta_singlelayer_Regular);
			const vrFloat N_I_0 = refDataSST3D.N_I_0_eta_SubTri(idx_I);
			const vrFloat N_I_1 = refDataSST3D.N_I_1_eta_SubTri(idx_I,cur_theta_singlelayer_Regular);
			const MyVec3& n_x = refDataSST3D.unitNormal_fieldPt;
			const vrFloat jacob_eta = refDataSST3D.Jacobi_eta_SubTri;
			MyVec3 sst_dr;
			vrFloat sst_drdn = 0.0;
			for (int m=0;m<MyDim;++m)
			{
				sst_dr[m] = refDataSST3D.r_i_SubTri(m,cur_theta_singlelayer_Regular); 
				sst_drdn += (sst_dr[m]*n_x[m]);
			}
			const vrFloat nouse_r = 1.0;
			vrFloat M0 = 0.0;
			for (int idx_k=0; idx_k < MyDim; ++idx_k)
			{
				M0 += get_Sij_SST_3D_k_Aliabadi_Peng(idx_i, idx_j, idx_k, nouse_r, sst_dr, sst_drdn, n_x, n_s);
			}
			//const vrFloat M0 = get_Sij_SST_3D_k_Aliabadi(idx_i, idx_j, idx_k, nouse_r, sst_dr, sst_drdn, n_x);
			const vrFloat M1 = M0 * jacob_eta ;
			const vrFloat F_2_ij_I_k = (M1*N_I_0)/(A*A*A);
			const vrFloat F_1_ij_I_k = (M1*N_I_1)/(A*A);

			const vrFloat curWeight_singleLayer_Regular = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Regular.row(index_theta)[TriangleElem::idx_weight_singleLayer];
			const vrFloat cur_rho_bar_Regular = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Regular.row(index_theta)[TriangleElem::idx_rho_bar_singleLayer];

			for (int index_rho=0; index_rho<nGaussPointSize_eta_In_Rho_SubTri_Regular; ++index_rho,++idx_g)
			{
				auto curRows_Regular = cur_gaussQuadrature_xi_eta_polar_SubTri_Regular.row(idx_g);
				MyVec2ParamSpace theta_rho_Regular;
				theta_rho_Regular[TriangleElem::idx_theta_doubleLayer] = curRows_Regular[TriangleElem::idx_theta_doubleLayer];
				theta_rho_Regular[TriangleElem::idx_rho_doubleLayer]=curRows_Regular[TriangleElem::idx_rho_doubleLayer];
				const vrFloat cur_rho = theta_rho_Regular[TriangleElem::idx_rho_doubleLayer];
				const vrFloat curWeight_doubleLayer = curRows_Regular[TriangleElem::idx_weight_doubleLayer];

				Q_ASSERT(numbers::isEqual(theta_rho_Regular[TriangleElem::idx_theta_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Regular.row(idx_g).x()));
				Q_ASSERT(numbers::isEqual(theta_rho_Regular[TriangleElem::idx_rho_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Regular.row(idx_g).y()));
				Q_ASSERT(numbers::isEqual(cur_theta_singlelayer_Regular,theta_rho_Regular[TriangleElem::idx_theta_doubleLayer]));

				const MyVec2ParamSpace currentSrcPtInParam /*in eta sub triangle space*/ = refDataSST3D.m_SrcPt_in_eta_SubTri;
				Q_ASSERT( ((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))) );
				if (!((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))))
				{
					printf("compute_S_ij_I_SST_DisContinuous_Sigmoidal : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
				}
				const MyVec2ParamSpace cur_eta = refDataSST3D.pc2eta(currentSrcPtInParam /*MyVec2ParamSpace(0.0,0.0)*/MYNOTICE, theta_rho_Regular);
				const MyVec2ParamSpace cur_xi = refDataSST3D.eta2xi_SubTri(cur_eta);
				const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

				MyFloat jacob_eta_nouse;/*jacob_eta = jacob_xi * mat_T_Inv*/
				MyVec3 normals_fieldpoint;
				MyFloat r;
				MyVec3 dr;
				MyFloat drdn;
				getKernelParameters_3D_SST_SubTri(srcPos,fieldPoint,refDataSST3D,jacob_eta_nouse,normals_fieldpoint,r,dr,drdn);
				Q_ASSERT(numbers::isEqual(r,cur_rho*A));

				vrFloat Sij_k = 0.0;
				for (int idx_k=0; idx_k < MyDim; ++idx_k)
				{
					Sij_k += get_Sij_SST_3D_k_Aliabadi_Peng(idx_i, idx_j, idx_k, r, dr, drdn, normals_fieldpoint, n_s);
				}
				
				const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);

				const vrFloat SingularTerm_Sij_I_k = Sij_k * N_I * jacob_eta * theta_rho_Regular[TriangleElem::idx_rho_doubleLayer] ;

				const vrFloat SingularTerm_F_1_ij_I_k = (1.0/(cur_rho))*(F_1_ij_I_k);
				const vrFloat SingularTerm_F_2_ij_I_k = (1.0/(cur_rho*cur_rho)) * (F_2_ij_I_k);

				doubleLayer_Term_k += (SingularTerm_Sij_I_k - (SingularTerm_F_1_ij_I_k + SingularTerm_F_2_ij_I_k)) * curWeight_doubleLayer;

			}//for (int index_rho=0; index_rho<nGaussPointSize_eta_In_Rho_SubTri_Regular; ++index_rho,++idx_g)

			const vrFloat beta = 1.0 / A;
			singleLayer_Term_k += ( (F_1_ij_I_k * log( abs(cur_rho_bar_Regular/beta) )) MyNotice - (F_2_ij_I_k * (1.0/(cur_rho_bar_Regular))) )* curWeight_singleLayer_Regular;

		}//for (int index_theta=0,idx_g=0;index_theta<nGaussPointSize_eta_In_Theta_SubTri_Regular;++index_theta)
		retVal += (doubleLayer_Term_k + singleLayer_Term_k);
		return retVal;
	}

	void vrBEM3D::DualEquation_Negative_Peng(const vrInt idx_i, const vrInt idx_j, const int ne, const vrFloat Cij, VertexPtr curSourcePtr)
	{
		const MyVec3I& srcDofs = curSourcePtr->getDofs();
		const bool isDisContinuousVtx = curSourcePtr->isDisContinuousVertex();		
		vrFloat Tij_I,Uij_I;
		const VertexTypeInDual srcPtType = curSourcePtr->getVertexTypeInDual();//Regular=0,Mirror_Positive=1, Mirror_Negative=2,CrackTip=3
		Q_ASSERT(VertexTypeInDual::Mirror_Negative == srcPtType);

			Q_ASSERT(curSourcePtr->isMirrorVertex());
			vrInt n_ptInElem_negative = 0;
			vrInt n_ptInElem_positive = 0;
			VertexPtr curSourcePtr_negative = curSourcePtr; MYNOTICE;//current source point locate on the negative element
			const MyVec3I& srcDofs_negative = curSourcePtr_negative->getDofs();
			Q_ASSERT(isDisContinuousVtx);
			Q_ASSERT(1 == (curSourcePtr_negative->getShareElement().size()));
			const bool isDisContinuousVtx_source_negative = curSourcePtr_negative->isDisContinuousVertex();			
			vrFloat Kij_I,Sij_I;
			for (int idx_e=0;idx_e<ne;++idx_e)
			{
				TriangleElemPtr curTriElemPtr =  TriangleElem::getTriangle(idx_e);
				const TriangleSetType curTriSetType = curTriElemPtr->getTriSetType();//Regular = 0, Positive = 1, Negative
#if USE_MI_NegativeSingular
				if (TriangleSetType::Regular == curTriSetType)
				{					
					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						Sij_I = compute_S_ij_I_Peng(curSourcePtr_negative,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I);
						m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Sij_I;
						/////////////////////////////////////////////////////////////////////////////////////////////////////////////
						Kij_I = compute_K_ij_I_Peng(curSourcePtr_negative,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I);
						m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Kij_I;
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else if (TriangleSetType::Positive == curTriSetType)
				{
					const MyVec3I& srcDofs_negative = curSourcePtr_negative->getDofs();
					VertexPtr curSource_Mirror_Ptr_Positive = curSourcePtr_negative->getMirrorVertex();					
					const MyVec3I& src_Mirror_Dofs_Positive = curSource_Mirror_Ptr_Positive->getDofs();
					const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();
					vrInt src_Mirror_PtIdx_Positive;
					bool pt_Mirror_InElem_Positive = isVertexInElement(curSource_Mirror_Ptr_Positive, curTriElemPtr, src_Mirror_PtIdx_Positive);
					MyVec3 srcPt_Mirror_SST_LookUp_Positive;
					TriangleElemData * ptr_data4SST_Mirror_Positive = NULL;
					if (pt_Mirror_InElem_Positive)
					{
						n_ptInElem_positive++;
						Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
						for (int v=0;v<Geometry::vertexs_per_tri;++v)
						{
							srcPt_Mirror_SST_LookUp_Positive[v] = (src_Mirror_PtIdx_Positive + v) %Geometry::vertexs_per_tri;
						}
						ptr_data4SST_Mirror_Positive = &(curTriElemPtr->get_m_data_SST_3D(src_Mirror_PtIdx_Positive));
					}

					//const vrFloat n_j_s_negative = (curSourcePtr_negative->getVertexNormal())[idx_j];
					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						if (pt_Mirror_InElem_Positive)
						{							
							Q_ASSERT( curSource_Mirror_Ptr_Positive->isDisContinuousVertex());
							Sij_I = compute_S_ij_I_SST_DisContinuous_Peng(curSource_Mirror_Ptr_Positive, (*ptr_data4SST_Mirror_Positive),idx_i,idx_j,idx_I);
							m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_Mirror_SST_LookUp_Positive[idx_I])->getDof(idx_j)) += Sij_I;
							
						}
						else
						{
							Sij_I = compute_S_ij_I_Peng(curSourcePtr_negative,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I);
							m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Sij_I;							
						}
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else if (TriangleSetType::Negative == curTriSetType)
				{						
					const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();
					Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
					vrInt srcPtIdx_negative;
					bool ptInElem_negative = isVertexInElement(curSourcePtr_negative, curTriElemPtr, srcPtIdx_negative);
					MyVec3 srcPt_SST_LookUp_negative;
					TriangleElemData * ptr_data4SST_negative = NULL;

					const vrFloat n_j_s_negative = (curSourcePtr_negative->getVertexNormal())[idx_j];
					if (ptInElem_negative)
					{
						n_ptInElem_negative++;
						Q_ASSERT(TriangleSetType::Negative == curTriSetType);
						Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
						for (int v=0;v<Geometry::vertexs_per_tri;++v)
						{
							srcPt_SST_LookUp_negative[v] = (srcPtIdx_negative+v) %Geometry::vertexs_per_tri;
						}
						ptr_data4SST_negative = &(curTriElemPtr->get_m_data_SST_3D(srcPtIdx_negative));
					}

					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						if (ptInElem_negative)
						{
							Q_ASSERT(isDisContinuousVtx_source_negative);
							Sij_I = compute_S_ij_I_SST_DisContinuous_Peng(curSourcePtr_negative, (*ptr_data4SST_negative),idx_i,idx_j,idx_I);
							m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_j)) += Sij_I;
							
						}
						else
						{
							Sij_I = compute_S_ij_I_Peng(curSourcePtr_negative,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I);
							m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Sij_I;
							
						}
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else
				{
					MyError("Unsupport Triangle element type.");
				}
#endif//USE_MI_NegativeSingular
			}//for (int idx_e=0;idx_e<ne;++idx_e)
			Q_ASSERT(1 == n_ptInElem_positive && 1 == n_ptInElem_negative);
			/*const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);			
			MyNoticeMsg("G Matrix.") m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],srcDofs_negative[idx_j]) +=  0.5 * delta_ij MyNoticeMsg("No delta ij.");
			Q_ASSERT(curSourcePtr->isMirrorVertex());
			VertexPtr curSource_Mirror_Positive_Ptr = curSourcePtr->getMirrorVertex();
			const MyVec3I& mirrorDofs_Positive = curSource_Mirror_Positive_Ptr->getDofs();
			MyNoticeMsg("G Matrix.") m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],mirrorDofs_Positive[idx_j]) +=  -0.5 * delta_ij MyNoticeMsg("No delta ij.");*/

	}

	void vrBEM3D::AssembleSystem_DisContinuous_DualEquation_Aliabadi_Peng(const vrInt v, const vrInt idx_i, const vrInt idx_j)
	{
		const MyInt ne = TriangleElem::getTriangleSize();		
		VertexPtr curSourcePtr = Vertex::getVertex(v);	
		const vrMat3& CMatrix = curSourcePtr->getCMatrix();
		const vrFloat Cij = CMatrix.coeff(idx_i,idx_j);
		const VertexTypeInDual srcPtType = curSourcePtr->getVertexTypeInDual();//Regular=0,Mirror_Positive=1, Mirror_Negative=2,CrackTip=3

		if (VertexTypeInDual::Regular == srcPtType)
		{
			DualEquation_Regular_Peng(idx_i, idx_j, ne, Cij, curSourcePtr);
		}
		else if (VertexTypeInDual::Mirror_Positive == srcPtType)
		{
			DualEquation_Positive_Peng(idx_i, idx_j, ne, Cij, curSourcePtr);
		}
		else if (VertexTypeInDual::Mirror_Negative  == srcPtType)
		{
			DualEquation_Negative_Peng(idx_i, idx_j, ne, Cij, curSourcePtr);
		}
		else
		{
			MyError("Unsupport source point type in AssembleSystem_DisContinuous_DualEquation_Aliabadi.");
		}
	}

	void vrBEM3D::AssembleSystem_DisContinuous_DualEquation_Aliabadi_Nouse(const vrInt v, const vrInt idx_i, const vrInt idx_j)
	{
		const MyInt ne = TriangleElem::getTriangleSize();		
		VertexPtr curSourcePtr = Vertex::getVertex(v);	
		const vrMat3& CMatrix = curSourcePtr->getCMatrix();
		const MyVec3I& srcDofs = curSourcePtr->getDofs();
		const bool isDisContinuousVtx = curSourcePtr->isDisContinuousVertex();
		const vrFloat Cij = CMatrix.coeff(idx_i,idx_j);
		vrFloat Tij_I,Uij_I;
		const VertexTypeInDual srcPtType = curSourcePtr->getVertexTypeInDual();//Regular=0,Mirror_Positive=1, Mirror_Negative=2,CrackTip=3
		
		//Q_ASSERT(VertexTypeInDual::Regular == srcPtType);
		//Q_ASSERT(!isDisContinuousVtx);
		if (VertexTypeInDual::Regular == srcPtType)
		{
#if DualEquation_Aliabadi_RegularSection
			for (int idx_e=0;idx_e<ne;++idx_e)
			{
				TriangleElemPtr curTriElemPtr =  TriangleElem::getTriangle(idx_e);
				const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();//dis_1_1=1, dis_1_2=2, dis_1_3=3, dis_2_3=4, dis_2_2=5, dis_2_1=6, dis_3_3=7, dis_regular=8

				vrInt srcPtIdx;
				bool ptInElem = isVertexInElement(curSourcePtr, curTriElemPtr, srcPtIdx);
				MyVec3I srcPt_SST_LookUp;			
				TriangleElemData * ptr_data4SST_with_DisContinuous = NULL;
				if (ptInElem)
				{
					for (int v=0;v<Geometry::vertexs_per_tri;++v)
					{
						srcPt_SST_LookUp[v] = (srcPtIdx+v) % Geometry::vertexs_per_tri;
					}
					ptr_data4SST_with_DisContinuous = &(curTriElemPtr->get_m_data_SST_3D(srcPtIdx));					
				}	
				
				for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				{
					if (ptInElem)
					{
						//source point in triangle element.
						if (isDisContinuousVtx)
						{
							Q_ASSERT( (DisContinuousType::dis_regular != curTriElemDisContinuousType) );
							Tij_I = compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(curSourcePtr,(*ptr_data4SST_with_DisContinuous),idx_i,idx_j,idx_I);	
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
							/////////////////////////////////////////////////////////////////////////////////////////////////////							
							Uij_I = compute_U_ij_I_SST_DisContinuous_Regular_Aliabadi(curSourcePtr,(*ptr_data4SST_with_DisContinuous),idx_i,idx_j,idx_I);	
							m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Uij_I;

						}
						else
						{
							Tij_I = compute_T_ij_I_SST_Aliabadi(curSourcePtr, (*ptr_data4SST_with_DisContinuous),idx_i,idx_j,idx_I);	
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
							/////////////////////////////////////////////////////////////////////////////////////////////////////
							Uij_I = compute_U_ij_I_SST_Aliabadi(curSourcePtr, (*ptr_data4SST_with_DisContinuous),idx_i,idx_j,idx_I);	
							Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
							m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Uij_I;
							//m_set_colIdx.insert(curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j));
						}
					}
					else
					{
						//source point do not locate in triangle element.
						Tij_I = compute_T_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I);
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;
						/////////////////////////////////////////////////////////////////////////////////////////////////////
						Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I);	
						Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
						m_Gsubmatrix.coeffRef(srcDofs[idx_i], curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
						//m_set_colIdx.insert(curTriElemPtr->getVertex(idx_I)->getDof(idx_j));
					}
				}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)

			}//for (int idx_e=0;idx_e<ne;++idx_e)
			m_Hsubmatrix.coeffRef(srcDofs[idx_i], srcDofs[idx_j]) += Cij;
			//m_set_colIdx.insert(srcDofs[idx_j]);
#endif//DualEquation_Aliabadi_RegularSection
		}
		else if (VertexTypeInDual::Mirror_Positive == srcPtType)
		{
#if DualEquation_Aliabadi_PositiveSection
			Q_ASSERT(curSourcePtr->isMirrorVertex());
			VertexPtr curSourcePtr_Positive_Ptr = curSourcePtr;
			VertexPtr curSource_Mirror_Negative_Ptr = curSourcePtr_Positive_Ptr->getMirrorVertex();
			const MyVec3I& src_Mirror_Negative_Dofs = curSource_Mirror_Negative_Ptr->getDofs();
			vrInt n_pt_Mirror_Negative_InElem = 0;
			vrInt n_ptInElem = 0;
			for (int idx_e=0;idx_e<ne;++idx_e)
			{
				TriangleElemPtr curTriElemPtr =  TriangleElem::getTriangle(idx_e);
#if USE_MI_NegativeSingular
				const TriangleSetType curTriSetType = curTriElemPtr->getTriSetType();//Regular = 0, Positive = 1, Negative

				if (TriangleSetType::Regular == curTriSetType)
				{
					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						//source point do not locate in triangle element.
						Tij_I = compute_T_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I);
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;
						/////////////////////////////////////////////////////////////////////////////////////////////////////
						Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I);	
						Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
						m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else if (TriangleSetType::Positive == curTriSetType)
				{
					const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();//dis_1_1=1, dis_1_2=2, dis_1_3=3, dis_2_3=4, dis_2_2=5, dis_2_1=6, dis_3_3=7, dis_regular=8
					Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);

					vrInt srcPtIdx;
					bool ptInElem = isVertexInElement(curSourcePtr, curTriElemPtr, srcPtIdx);
					MyVec3I srcPt_SST_LookUp;			
					TriangleElemData * ptr_data4SST_with_DisContinuous = NULL;
					if (ptInElem)
					{
						n_ptInElem++; 
						for (int v=0;v<Geometry::vertexs_per_tri;++v)
						{
							srcPt_SST_LookUp[v] = (srcPtIdx+v) % Geometry::vertexs_per_tri;
						}
						ptr_data4SST_with_DisContinuous = &(curTriElemPtr->get_m_data_SST_3D(srcPtIdx));
					}

					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						if (ptInElem)
						{
							Q_ASSERT(isDisContinuousVtx);
							Q_ASSERT( (DisContinuousType::dis_regular != curTriElemDisContinuousType) );
							Tij_I = compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(curSourcePtr, (*ptr_data4SST_with_DisContinuous),idx_i,idx_j,idx_I);	
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
							/////////////////////////////////////////////////////////////////////////////////////////////////////
							Uij_I = compute_U_ij_I_SST_DisContinuous_Regular_Aliabadi(curSourcePtr,(*ptr_data4SST_with_DisContinuous),idx_i,idx_j,idx_I);	
							m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Uij_I;
						}
						else
						{
							//source point do not locate in triangle element.
							Tij_I = compute_T_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I);
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;
							/////////////////////////////////////////////////////////////////////////////////////////////////////
							Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I);	
			//				Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
							m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
						}
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else if (TriangleSetType::Negative == curTriSetType)
				{
					const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();//dis_1_1=1, dis_1_2=2, dis_1_3=3, dis_2_3=4, dis_2_2=5, dis_2_1=6, dis_3_3=7, dis_regular=8
					vrInt src_Mirror_Negative_PtIdx;
					bool pt_Mirror_Negative_InElem = isVertexInElement(curSource_Mirror_Negative_Ptr, curTriElemPtr, src_Mirror_Negative_PtIdx);
					const bool isDisContinuousVtx_Mirror_Negative = curSource_Mirror_Negative_Ptr->isDisContinuousVertex();
					Q_ASSERT(isDisContinuousVtx_Mirror_Negative);

					MyVec3 srcPt_Mirror_Negative_SST_LookUp;
					
					TriangleElemData * ptr_data4SST_Mirror_Negative_DisContinuous = NULL;
					if (pt_Mirror_Negative_InElem)
					{
						n_pt_Mirror_Negative_InElem++;
						for (int v=0;v<Geometry::vertexs_per_tri;++v)
						{
							srcPt_Mirror_Negative_SST_LookUp[v] = (src_Mirror_Negative_PtIdx + v) % Geometry::vertexs_per_tri;
						}
						ptr_data4SST_Mirror_Negative_DisContinuous = &(curTriElemPtr->get_m_data_SST_3D(src_Mirror_Negative_PtIdx));
					}

					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						if (pt_Mirror_Negative_InElem)
						{
							Q_ASSERT(isDisContinuousVtx_Mirror_Negative);
							Q_ASSERT( (DisContinuousType::dis_regular != curTriElemDisContinuousType) );
							Tij_I = compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(curSource_Mirror_Negative_Ptr, (*ptr_data4SST_Mirror_Negative_DisContinuous),idx_i,idx_j,idx_I);	
							m_Hsubmatrix.coeffRef(srcDofs[idx_i] MyNoticeMsg("Use Mirror negative point."),curTriElemPtr->getVertex(srcPt_Mirror_Negative_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
							/////////////////////////////////////////////////////////////////////////////////////////////////////
							Uij_I = compute_U_ij_I_SST_DisContinuous_Regular_Aliabadi(curSource_Mirror_Negative_Ptr,(*ptr_data4SST_Mirror_Negative_DisContinuous),idx_i,idx_j,idx_I);	
							m_Gsubmatrix.coeffRef(srcDofs[idx_i] MyNoticeMsg("Use Mirror negative point."),curTriElemPtr->getVertex(srcPt_Mirror_Negative_SST_LookUp[idx_I])->getDof(idx_j)) += Uij_I;
						} 
						else
						{
							//source point do not locate in triangle element.
							Tij_I = compute_T_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I);
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;
							/////////////////////////////////////////////////////////////////////////////////////////////////////
							Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I);	
			//				Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
							m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
						}
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else
				{
					MyError("Unsupport Triangle element type.");
				}
#endif//USE_MI_NegativeSingular
			}//for (int idx_e=0;idx_e<ne;++idx_e)
			Q_ASSERT(1 == n_ptInElem && 1 == n_pt_Mirror_Negative_InElem);
			Q_ASSERT(curSourcePtr->isMirrorVertex());
			//VertexPtr curSource_Mirror_Negative_Ptr = curSourcePtr->getMirrorVertex();
			const MyVec3I& mirrorDofs_Negative = curSource_Mirror_Negative_Ptr->getDofs();

			const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
			m_Hsubmatrix.coeffRef(srcDofs[idx_i],srcDofs[idx_j]) += 0.5 * delta_ij;
			m_Hsubmatrix.coeffRef(srcDofs[idx_i],mirrorDofs_Negative[idx_j]) += 0.5 * delta_ij;

#endif//DualEquation_Aliabadi_PositiveSection
		}
		else if (VertexTypeInDual::Mirror_Negative  == srcPtType)
		{
			const int idx_i_1 = idx_i;
			const int idx_j_1 = idx_j;
#if DualEquation_Aliabadi_NegativeSection
			Q_ASSERT(curSourcePtr->isMirrorVertex());
			vrInt n_ptInElem_negative = 0;
			vrInt n_ptInElem_positive = 0;
			VertexPtr curSourcePtr_negative = curSourcePtr; MYNOTICE;//current source point locate on the negative element
			const MyVec3I& srcDofs_negative = curSourcePtr_negative->getDofs();
			Q_ASSERT(isDisContinuousVtx);
			Q_ASSERT(1 == (curSourcePtr_negative->getShareElement().size()));
			const bool isDisContinuousVtx_source_negative = curSourcePtr_negative->isDisContinuousVertex();			
			vrFloat Kij_I,Sij_I;
			for (int idx_e=0;idx_e<ne;++idx_e)
			{
				TriangleElemPtr curTriElemPtr =  TriangleElem::getTriangle(idx_e);
				const TriangleSetType curTriSetType = curTriElemPtr->getTriSetType();//Regular = 0, Positive = 1, Negative
#if USE_MI_NegativeSingular
				if (TriangleSetType::Regular == curTriSetType)
				{
					const vrFloat n_j_s_negative = (curSourcePtr_negative->getVertexNormal())[idx_j];
					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						for (int idx_k=0; idx_k < MyDim; ++idx_k)
						{
							Sij_I = compute_S_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->get_m_data_SST_3D(),idx_i_1,idx_j_1,idx_I,idx_k);
							Sij_I = n_j_s_negative * Sij_I;
							m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Sij_I;
							/////////////////////////////////////////////////////////////////////////////////////////////////////////////
							Kij_I = compute_K_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->get_m_data_SST_3D(),idx_i_1,idx_j_1,idx_I,idx_k);
							Kij_I = n_j_s_negative * Kij_I;
							m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Kij_I;
						}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else if (TriangleSetType::Positive == curTriSetType)
				{
					const MyVec3I& srcDofs_negative = curSourcePtr_negative->getDofs();
					VertexPtr curSource_Mirror_Ptr_Positive = curSourcePtr_negative->getMirrorVertex();					
					const MyVec3I& src_Mirror_Dofs_Positive = curSource_Mirror_Ptr_Positive->getDofs();
					const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();
					vrInt src_Mirror_PtIdx_Positive;
					bool pt_Mirror_InElem_Positive = isVertexInElement(curSource_Mirror_Ptr_Positive, curTriElemPtr, src_Mirror_PtIdx_Positive);
					MyVec3 srcPt_Mirror_SST_LookUp_Positive;
					TriangleElemData * ptr_data4SST_Mirror_Positive = NULL;
					if (pt_Mirror_InElem_Positive)
					{
						n_ptInElem_positive++;
						Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
						for (int v=0;v<Geometry::vertexs_per_tri;++v)
						{
							srcPt_Mirror_SST_LookUp_Positive[v] = (src_Mirror_PtIdx_Positive + v) %Geometry::vertexs_per_tri;
						}
						ptr_data4SST_Mirror_Positive = &(curTriElemPtr->get_m_data_SST_3D(src_Mirror_PtIdx_Positive));
					}

					const vrFloat n_j_s_negative = (curSourcePtr_negative->getVertexNormal())[idx_j];
					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						if (pt_Mirror_InElem_Positive)
						{	
							//const vrFloat n_j_s_positive = (curSource_Mirror_Ptr_Positive->getVertexNormal())[idx_j];
							Q_ASSERT( curSource_Mirror_Ptr_Positive->isDisContinuousVertex());
							for (int idx_k=0; idx_k < MyDim; ++idx_k)
							{
								Sij_I = compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(curSource_Mirror_Ptr_Positive, (*ptr_data4SST_Mirror_Positive),idx_i_1,idx_j_1,idx_I,idx_k);	
								Sij_I = n_j_s_negative * Sij_I;
								//Sij_I = n_j_s_positive * Sij_I;//71

								m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_Mirror_SST_LookUp_Positive[idx_I])->getDof(idx_k)) += Sij_I;
								///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
								Kij_I = compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(curSource_Mirror_Ptr_Positive,(*ptr_data4SST_Mirror_Positive),idx_i,idx_j,idx_I,idx_k);
								Kij_I = n_j_s_negative * Kij_I;//Kij_I = n_j_s_positive * Kij_I;
								m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_Mirror_SST_LookUp_Positive[idx_I])->getDof(idx_k)) += Kij_I;
							}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
						}
						else
						{
							for (int idx_k=0; idx_k < MyDim; ++idx_k)
							{
								Sij_I = compute_S_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->get_m_data_SST_3D(),idx_i_1,idx_j_1,idx_I,idx_k);
								Sij_I = n_j_s_negative * Sij_I;
								m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Sij_I;
								/////////////////////////////////////////////////////////////////////////////////////////////////////////////
								Kij_I = compute_K_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I,idx_k);
								Kij_I = n_j_s_negative * Kij_I;
								m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Kij_I;
							}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
						}
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else if (TriangleSetType::Negative == curTriSetType)
				{						
					const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();
					Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
					vrInt srcPtIdx_negative;
					bool ptInElem_negative = isVertexInElement(curSourcePtr_negative, curTriElemPtr, srcPtIdx_negative);
					MyVec3 srcPt_SST_LookUp_negative;
					TriangleElemData * ptr_data4SST_negative = NULL;

					const vrFloat n_j_s_negative = (curSourcePtr_negative->getVertexNormal())[idx_j];
					if (ptInElem_negative)
					{
						n_ptInElem_negative++;
						Q_ASSERT(TriangleSetType::Negative == curTriSetType);
						Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
						for (int v=0;v<Geometry::vertexs_per_tri;++v)
						{
							srcPt_SST_LookUp_negative[v] = (srcPtIdx_negative+v) %Geometry::vertexs_per_tri;
						}
						ptr_data4SST_negative = &(curTriElemPtr->get_m_data_SST_3D(srcPtIdx_negative));
					}

					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						if (ptInElem_negative)
						{
							Q_ASSERT(isDisContinuousVtx_source_negative);
							for (int idx_k=0; idx_k < MyDim; ++idx_k)
							{
								Sij_I = compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(curSourcePtr_negative, (*ptr_data4SST_negative),idx_i_1,idx_j_1,idx_I,idx_k);	
								Sij_I = n_j_s_negative * Sij_I;
								m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_k)) += Sij_I;
								//////////////////////////////////////////////////////////////////////////////////////
								Kij_I = compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(curSourcePtr_negative, (*ptr_data4SST_negative),idx_i,idx_j,idx_I,idx_k);
								Kij_I = n_j_s_negative * Kij_I;
								m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_k)) += Kij_I;
							}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
						}
						else
						{
							for (int idx_k=0; idx_k < MyDim; ++idx_k)
							{
								Sij_I = compute_S_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->get_m_data_SST_3D(),idx_i_1,idx_j_1,idx_I,idx_k);
								Sij_I = n_j_s_negative * Sij_I;
								m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Sij_I;
								//////////////////////////////////////////////////////////////////////////////////////////////////////////////
								Kij_I = compute_K_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->get_m_data_SST_3D(),idx_i,idx_j,idx_I,idx_k);
								Kij_I = n_j_s_negative * Kij_I;
								m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Kij_I;
							}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
						}
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else
				{
					MyError("Unsupport Triangle element type.");
				}
#endif//USE_MI_NegativeSingular
			}//for (int idx_e=0;idx_e<ne;++idx_e)
			Q_ASSERT(1 == n_ptInElem_positive && 1 == n_ptInElem_negative);
			const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);			
			MyNoticeMsg("G Matrix.") m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],srcDofs_negative[idx_j]) +=  0.5 /** delta_ij*/ MyNoticeMsg("No delta ij.");
			Q_ASSERT(curSourcePtr->isMirrorVertex());
			VertexPtr curSource_Mirror_Positive_Ptr = curSourcePtr->getMirrorVertex();
			const MyVec3I& mirrorDofs_Positive = curSource_Mirror_Positive_Ptr->getDofs();
			MyNoticeMsg("G Matrix.") m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],mirrorDofs_Positive[idx_j]) +=  -0.5 /** delta_ij*/ MyNoticeMsg("No delta ij.");
#endif//DualEquation_Aliabadi_NegativeSection
		}
		else
		{
			MyError("Unsupport source point type in AssembleSystem_DisContinuous_DualEquation_Aliabadi.");
		}
	}
#else//USE_UniformSampling

void vrBEM3D::AssembleSystem_DisContinuous_DualEquation_Aliabadi(const vrInt v, const vrInt idx_i, const vrInt idx_j)
{
#define DualEquation_Aliabadi_RegularSection (1)
#define DualEquation_Aliabadi_PositiveSection (1)
#define DualEquation_Aliabadi_NegativeSection (1)
	const MyInt ne = TriangleElem::getTriangleSize();		
	VertexPtr curSourcePtr = Vertex::getVertex(v);	
	const vrMat3& CMatrix = curSourcePtr->getCMatrix();
	const MyVec3I& srcDofs = curSourcePtr->getDofs();
	const bool isDisContinuousVtx = curSourcePtr->isDisContinuousVertex();
	const vrFloat Cij = CMatrix.coeff(idx_i,idx_j);


	const VertexTypeInDual srcPtType = curSourcePtr->getVertexTypeInDual();//Regular=0,Mirror_Positive=1, Mirror_Negative=2,CrackTip=3
	//const VertexTypeInDual srcPtType = Mirror_Negative;//Regular=0,Mirror_Positive=1, Mirror_Negative=2,CrackTip=3

	if (VertexTypeInDual::Regular == srcPtType)
	{
		//MyError("VertexTypeInDual::Regular == srcPtType || VertexTypeInDual::Mirror_Positive == srcPtType");
#if DualEquation_Aliabadi_RegularSection

		vrFloat Tij_I,Uij_I;
		for (int idx_e=0;idx_e<ne;++idx_e)
		{
			TriangleElemPtr curTriElemPtr =  TriangleElem::getTriangle(idx_e);
			const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();//dis_1_1=1, dis_1_2=2, dis_1_3=3, dis_2_3=4, dis_2_2=5, dis_2_1=6, dis_3_3=7, dis_regular=8

			vrInt srcPtIdx;
			bool ptInElem = isVertexInElement(curSourcePtr, curTriElemPtr, srcPtIdx);
			MyVec3I srcPt_SST_LookUp;			
			TriangleElemData data4SST_with_DisContinuous;
			if (ptInElem)
			{
				const DisContinuousType tmpTriElemDisContinuousType = TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,srcPtIdx);
				MyVec3 vtx_globalCoord[Geometry::vertexs_per_tri];

				for (int v=0;v<Geometry::vertexs_per_tri;++v)
				{
					srcPt_SST_LookUp[v] = srcPtIdx;
					vtx_globalCoord[v] = curTriElemPtr->getVertex(srcPtIdx)->getPos();
					srcPtIdx = (srcPtIdx+1) % Geometry::vertexs_per_tri;
				}
				data4SST_with_DisContinuous.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord);
			}	

			for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
			{
				if (ptInElem)
				{
					//source point in triangle element.
					if (isDisContinuousVtx)
					{
						Q_ASSERT( (DisContinuousType::dis_regular != curTriElemDisContinuousType) );

						Tij_I = 0.0;
#if USE_360_Sample
						Tij_I += compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
#else//USE_360_Sample
						for (vrInt subTriIdx=0; subTriIdx < Geometry::vertexs_per_tri; ++subTriIdx)
						{
							Tij_I += compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(subTriIdx, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
						}
#endif//USE_360_Sample
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
						/////////////////////////////////////////////////////////////////////////////////////////////////////
						Uij_I = 0.0;
#if USE_360_Sample
						Uij_I += compute_U_ij_I_SST_DisContinuous_Regular_Aliabadi(curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
#else//USE_360_Sample
						for (vrInt subTriIdx=0; subTriIdx < Geometry::vertexs_per_tri; ++subTriIdx)
						{
							Uij_I += compute_U_ij_I_SST_DisContinuous_Regular_Aliabadi(subTriIdx, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
						}
#endif//USE_360_Sample
						m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Uij_I;

					}
					else
					{
						Tij_I = compute_T_ij_I_SST_Aliabadi(curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;

						/////////////////////////////////////////////////////////////////////////////////////////////////////
						Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);	
						Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
						m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;

					}
				}
				else
				{
					//source point do not locate in triangle element.
					Tij_I = compute_T_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);

					m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;
					//printf("Tij_I_Aliabadi[%f]  Tij_I[%f]\n",Tij_I_tmp,Tij_I);vrPause;
					/////////////////////////////////////////////////////////////////////////////////////////////////////
					Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);	
					Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
					m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
				}
			}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)

		}//for (int idx_e=0;idx_e<ne;++idx_e)

		m_Hsubmatrix.coeffRef(srcDofs[idx_i],srcDofs[idx_j]) += Cij;

#endif//DualEquation_Aliabadi_RegularSection
	}
	else if (VertexTypeInDual::Mirror_Positive == srcPtType)
	{
#if DualEquation_Aliabadi_PositiveSection
		Q_ASSERT(curSourcePtr->isMirrorVertex());
#if USE_MI_NegativeSingular
		VertexPtr curSourcePtr_Positive_Ptr = curSourcePtr;
		VertexPtr curSource_Mirror_Negative_Ptr = curSourcePtr_Positive_Ptr->getMirrorVertex();
		const MyVec3I& src_Mirror_Negative_Dofs = curSource_Mirror_Negative_Ptr->getDofs();
		vrInt n_pt_Mirror_Negative_InElem = 0;
		vrInt n_ptInElem = 0;
#endif//USE_MI_NegativeSingular
		vrFloat Tij_I,Uij_I;
		for (int idx_e=0;idx_e<ne;++idx_e)
		{
			TriangleElemPtr curTriElemPtr =  TriangleElem::getTriangle(idx_e);

#if USE_MI_NegativeSingular
			const TriangleSetType curTriSetType = curTriElemPtr->getTriSetType();//Regular = 0, Positive = 1, Negative

			if (TriangleSetType::Regular == curTriSetType)
			{

				for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				{
					//source point do not locate in triangle element.
					Tij_I = compute_T_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);

					m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;
					//printf("Tij_I_Aliabadi[%f]  Tij_I[%f]\n",Tij_I_tmp,Tij_I);vrPause;
					/////////////////////////////////////////////////////////////////////////////////////////////////////
					Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);	
					Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
					m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
				}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
			}
			else if (TriangleSetType::Positive == curTriSetType)
			{
				const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();//dis_1_1=1, dis_1_2=2, dis_1_3=3, dis_2_3=4, dis_2_2=5, dis_2_1=6, dis_3_3=7, dis_regular=8
				Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);

				vrInt srcPtIdx;
				bool ptInElem = isVertexInElement(curSourcePtr, curTriElemPtr, srcPtIdx);
				MyVec3I srcPt_SST_LookUp;			
				TriangleElemData data4SST_with_DisContinuous;
				if (ptInElem)
				{
					n_ptInElem++; 
					const DisContinuousType tmpTriElemDisContinuousType = TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,srcPtIdx);
					MyVec3 vtx_globalCoord[Geometry::vertexs_per_tri];

					for (int v=0;v<Geometry::vertexs_per_tri;++v)
					{
						srcPt_SST_LookUp[v] = srcPtIdx;
						vtx_globalCoord[v] = curTriElemPtr->getVertex(srcPtIdx)->getPos();
						srcPtIdx = (srcPtIdx+1) % Geometry::vertexs_per_tri;
					}
					data4SST_with_DisContinuous.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord);
				}

				for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				{
					if (ptInElem)
					{
						Q_ASSERT(isDisContinuousVtx);
						Q_ASSERT( (DisContinuousType::dis_regular != curTriElemDisContinuousType) );
						Tij_I = 0.0;
#if USE_360_Sample
						Tij_I += compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
#else//USE_360_Sample
						for (vrInt subTriIdx=0; subTriIdx < Geometry::vertexs_per_tri; ++subTriIdx)
						{
							Tij_I += compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(subTriIdx, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
						}
#endif//USE_360_Sample
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;


						/////////////////////////////////////////////////////////////////////////////////////////////////////
						Uij_I = 0.0;
#if USE_360_Sample
						Uij_I += compute_U_ij_I_SST_DisContinuous_Regular_Aliabadi(curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
#else//USE_360_Sample
						for (vrInt subTriIdx=0; subTriIdx < Geometry::vertexs_per_tri; ++subTriIdx)
						{
							Uij_I += compute_U_ij_I_SST_DisContinuous_Regular_Aliabadi(subTriIdx, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
						}
#endif//USE_360_Sample
						m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Uij_I;

					}
					else
					{
						//source point do not locate in triangle element.
						Tij_I = compute_T_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;
						//printf("Tij_I_Aliabadi[%f]  Tij_I[%f]\n",Tij_I_tmp,Tij_I);vrPause;
						/////////////////////////////////////////////////////////////////////////////////////////////////////
						Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);	
						Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
						m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
					}
				}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
			}
			else if (TriangleSetType::Negative == curTriSetType)
			{
				const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();//dis_1_1=1, dis_1_2=2, dis_1_3=3, dis_2_3=4, dis_2_2=5, dis_2_1=6, dis_3_3=7, dis_regular=8
				vrInt src_Mirror_Negative_PtIdx;
				bool pt_Mirror_Negative_InElem = isVertexInElement(curSource_Mirror_Negative_Ptr, curTriElemPtr, src_Mirror_Negative_PtIdx);
				const bool isDisContinuousVtx_Mirror_Negative = curSource_Mirror_Negative_Ptr->isDisContinuousVertex();
				Q_ASSERT(isDisContinuousVtx_Mirror_Negative);

				MyVec3 srcPt_Mirror_Negative_SST_LookUp;
				TriangleElemData data4SST_Mirror_Negative_DisContinuous;

				if (pt_Mirror_Negative_InElem)
				{
					n_pt_Mirror_Negative_InElem++;

					const DisContinuousType tmpTriElemDisContinuousType = 
						TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,src_Mirror_Negative_PtIdx);
					MyVec3 vtx_globalCoord[Geometry::vertexs_per_tri];

					for (int v=0;v<Geometry::vertexs_per_tri;++v)
					{
						srcPt_Mirror_Negative_SST_LookUp[v] = src_Mirror_Negative_PtIdx;
						vtx_globalCoord[v] = curTriElemPtr->getVertex(src_Mirror_Negative_PtIdx)->getPos();
						src_Mirror_Negative_PtIdx = (src_Mirror_Negative_PtIdx + 1) % Geometry::vertexs_per_tri;
					}
					data4SST_Mirror_Negative_DisContinuous.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord);
				}

				for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				{
					if (pt_Mirror_Negative_InElem)
					{
						Q_ASSERT(isDisContinuousVtx_Mirror_Negative);
						Q_ASSERT( (DisContinuousType::dis_regular != curTriElemDisContinuousType) );
						Tij_I = 0.0;
#if USE_360_Sample
						Tij_I += compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(curSource_Mirror_Negative_Ptr,data4SST_Mirror_Negative_DisContinuous,idx_i,idx_j,idx_I);	
#else//USE_360_Sample
						for (vrInt subTriIdx=0; subTriIdx < Geometry::vertexs_per_tri; ++subTriIdx)
						{
							Tij_I += compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(subTriIdx, curSource_Mirror_Negative_Ptr,data4SST_Mirror_Negative_DisContinuous,idx_i,idx_j,idx_I);	
						}
#endif//USE_360_Sample
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_Mirror_Negative_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;


						/////////////////////////////////////////////////////////////////////////////////////////////////////

						Uij_I = 0.0;
#if USE_360_Sample
						Uij_I += compute_U_ij_I_SST_DisContinuous_Regular_Aliabadi(curSource_Mirror_Negative_Ptr,data4SST_Mirror_Negative_DisContinuous,idx_i,idx_j,idx_I);	
#else//USE_360_Sample
						for (vrInt subTriIdx=0; subTriIdx < Geometry::vertexs_per_tri; ++subTriIdx)
						{
							Uij_I += compute_U_ij_I_SST_DisContinuous_Regular_Aliabadi(subTriIdx, curSource_Mirror_Negative_Ptr,data4SST_Mirror_Negative_DisContinuous,idx_i,idx_j,idx_I);	
						}
#endif//USE_360_Sample
						m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_Mirror_Negative_SST_LookUp[idx_I])->getDof(idx_j)) += Uij_I;

					}
					else
					{
						//source point do not locate in triangle element.
						Tij_I = compute_T_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;
						//printf("Tij_I_Aliabadi[%f]  Tij_I[%f]\n",Tij_I_tmp,Tij_I);vrPause;
						/////////////////////////////////////////////////////////////////////////////////////////////////////
						Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);	
						Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
						m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
					}
				}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
			}
			else
			{
				MyError("Unsupport Triangle element type.");
			}


#else//USE_MI_NegativeSingular

			const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();//dis_1_1=1, dis_1_2=2, dis_1_3=3, dis_2_3=4, dis_2_2=5, dis_2_1=6, dis_3_3=7, dis_regular=8

			vrInt srcPtIdx;
			bool ptInElem = isVertexInElement(curSourcePtr, curTriElemPtr, srcPtIdx);
			MyVec3I srcPt_SST_LookUp;			
			TriangleElemData data4SST_with_DisContinuous;
			if (ptInElem)
			{
				const DisContinuousType tmpTriElemDisContinuousType = TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,srcPtIdx);
				MyVec3 vtx_globalCoord[Geometry::vertexs_per_tri];

				for (int v=0;v<Geometry::vertexs_per_tri;++v)
				{
					srcPt_SST_LookUp[v] = srcPtIdx;
					vtx_globalCoord[v] = curTriElemPtr->getVertex(srcPtIdx)->getPos();
					srcPtIdx = (srcPtIdx+1) % Geometry::vertexs_per_tri;
				}
				data4SST_with_DisContinuous.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord);
			}	

			for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
			{
				if (ptInElem)
				{
					//source point in triangle element.
					if (isDisContinuousVtx)
					{
						Q_ASSERT( (DisContinuousType::dis_regular != curTriElemDisContinuousType) );
#if USE_Aliabadi_RegularSample
						Tij_I = 0.0;
						Tij_I += compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(0, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
						Tij_I += compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(1, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);
						Tij_I += compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(2, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
#else//USE_Aliabadi_RegularSample
						Tij_I = 0.0;
						Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(0, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
						Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(1, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);
						Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(2, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
#endif//USE_Aliabadi_RegularSample

						/////////////////////////////////////////////////////////////////////////////////////////////////////
						Uij_I = compute_U_ij_I_Aliabadi( curSourcePtr, curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
						Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
						m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;

					}
					else
					{
						Tij_I = compute_T_ij_I_SST_Aliabadi(curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;

						/////////////////////////////////////////////////////////////////////////////////////////////////////
						Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);	
						Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
						m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;

					}
				}
				else
				{
					//source point do not locate in triangle element.
					Tij_I = compute_T_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);

					m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;
					//printf("Tij_I_Aliabadi[%f]  Tij_I[%f]\n",Tij_I_tmp,Tij_I);vrPause;
					/////////////////////////////////////////////////////////////////////////////////////////////////////
					Uij_I = compute_U_ij_I_Aliabadi(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);	
					Q_ASSERT(numbers::isEqual(0.0,Uij_I)); 
					m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
				}
			}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)


#endif//USE_MI_NegativeSingular
		}//for (int idx_e=0;idx_e<ne;++idx_e)
		Q_ASSERT(1 == n_ptInElem && 1 == n_pt_Mirror_Negative_InElem);
		Q_ASSERT(curSourcePtr->isMirrorVertex());
		//VertexPtr curSource_Mirror_Negative_Ptr = curSourcePtr->getMirrorVertex();
		const MyVec3I& mirrorDofs_Negative = curSource_Mirror_Negative_Ptr->getDofs();

		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
		m_Hsubmatrix.coeffRef(srcDofs[idx_i],srcDofs[idx_j]) += 0.5 * delta_ij;
		m_Hsubmatrix.coeffRef(srcDofs[idx_i],mirrorDofs_Negative[idx_j]) += 0.5 * delta_ij;

#endif//DualEquation_Aliabadi_PositiveSection
	}
	else if (VertexTypeInDual::Mirror_Negative  == srcPtType)
	{

		static std::set< vrInt > ptIdSet;
		ptIdSet.insert(v);
		printf("VertexTypeInDual::Mirror_Negative == srcPtType  [%d] \n", ptIdSet.size());
#if DualEquation_Aliabadi_NegativeSection
		Q_ASSERT(curSourcePtr->isMirrorVertex());

#if USE_MI_NegativeSingular
		vrInt n_ptInElem_negative = 0;
		vrInt n_ptInElem_positive = 0;
#endif//USE_MI_NegativeSingular

		VertexPtr curSourcePtr_negative = curSourcePtr; MYNOTICE;//current source point locate on the negative element
		const MyVec3I& srcDofs_negative = curSourcePtr_negative->getDofs();
		Q_ASSERT(isDisContinuousVtx);
		Q_ASSERT(1 == (curSourcePtr_negative->getShareElement().size()));

		const bool isDisContinuousVtx_source_negative = curSourcePtr_negative->isDisContinuousVertex();

		const vrFloat n_j_s = (curSourcePtr_negative->getVertexNormal())[idx_j];
		vrFloat Kij_I,Sij_I;
		for (int idx_e=0;idx_e<ne;++idx_e)
		{
			TriangleElemPtr curTriElemPtr =  TriangleElem::getTriangle(idx_e);
			const TriangleSetType curTriSetType = curTriElemPtr->getTriSetType();//Regular = 0, Positive = 1, Negative
#if USE_MI_NegativeSingular
			if (TriangleSetType::Regular == curTriSetType)
			{
				for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				{
					for (int idx_k=0; idx_k < MyDim; ++idx_k)
					{
						Sij_I = compute_S_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I,idx_k);
						Sij_I = n_j_s * Sij_I;
						m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Sij_I;

						Kij_I = compute_K_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I,idx_k);
						Kij_I = n_j_s * Kij_I;
						m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Kij_I;
						//printf("Sij_I[%f]  Kij_I[%f]\n",Sij_I,Kij_I); vrPause;
					}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
				}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
			}
			else if (TriangleSetType::Positive == curTriSetType)
			{
				const MyVec3I& srcDofs_negative = curSourcePtr_negative->getDofs();
				VertexPtr curSource_Mirror_Ptr_Positive = curSourcePtr_negative->getMirrorVertex();
				const MyVec3I& src_Mirror_Dofs_Positive = curSource_Mirror_Ptr_Positive->getDofs();
				const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();
				vrInt src_Mirror_PtIdx_Positive;
				bool pt_Mirror_InElem_Positive = isVertexInElement(curSource_Mirror_Ptr_Positive, curTriElemPtr, src_Mirror_PtIdx_Positive);

				MyVec3 srcPt_Mirror_SST_LookUp_Positive;
				TriangleElemData data4SST_Mirror_Positive;

				if (pt_Mirror_InElem_Positive)
				{
					n_ptInElem_positive++;
					Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
					const DisContinuousType tmpTriElemDisContinuousType = TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,src_Mirror_PtIdx_Positive);
					MyVec3 vtx_globalCoord_Positive[Geometry::vertexs_per_tri];

					for (int v=0;v<Geometry::vertexs_per_tri;++v)
					{
						srcPt_Mirror_SST_LookUp_Positive[v] = src_Mirror_PtIdx_Positive;
						vtx_globalCoord_Positive[v] = curTriElemPtr->getVertex(src_Mirror_PtIdx_Positive)->getPos();
						src_Mirror_PtIdx_Positive = (src_Mirror_PtIdx_Positive + 1) %Geometry::vertexs_per_tri;
					}
					data4SST_Mirror_Positive.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord_Positive);
				}

				for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				{
					if (pt_Mirror_InElem_Positive)
					{
						Q_ASSERT( curSource_Mirror_Ptr_Positive->isDisContinuousVertex());
						for (int idx_k=0; idx_k < MyDim; ++idx_k)
						{
							Sij_I = 0.0;
#if USE_360_Sample
							Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive,idx_i,idx_j,idx_I,idx_k);	
#else//USE_360_Sample
							for (vrInt subTriIdx=0; subTriIdx < Geometry::vertexs_per_tri; ++subTriIdx)
							{
								Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(subTriIdx, curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive,idx_i,idx_j,idx_I,idx_k);	
							}//for (vrInt subTriIdx=0; subTriIdx < Geometry::vertexs_per_tri; ++subTriIdx)
#endif//USE_360_Sample
							Sij_I = n_j_s * Sij_I;
							m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_Mirror_SST_LookUp_Positive[idx_I])->getDof(idx_k)) += Sij_I;

							Kij_I = 0.0;
#if USE_360_Sample
							Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive,idx_i,idx_j,idx_I,idx_k);
#else//USE_360_Sample
							for (vrInt subTriIdx=0; subTriIdx < Geometry::vertexs_per_tri; ++subTriIdx)
							{
								Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(subTriIdx, curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive,idx_i,idx_j,idx_I,idx_k);
							}//for (vrInt subTriIdx=0; subTriIdx < Geometry::vertexs_per_tri; ++subTriIdx)
#endif//USE_360_Sample
							Kij_I = n_j_s * Kij_I;
							m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_Mirror_SST_LookUp_Positive[idx_I])->getDof(idx_k)) += Kij_I;

						}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
					}
					else
					{
						for (int idx_k=0; idx_k < MyDim; ++idx_k)
						{
							Sij_I = compute_S_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I,idx_k);
							Sij_I = n_j_s * Sij_I;
							m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Sij_I;

							Kij_I = compute_K_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I,idx_k);
							Kij_I = n_j_s * Kij_I;
							m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Kij_I;
							//printf("Sij_I[%f]  Kij_I[%f]\n",Sij_I,Kij_I); vrPause;
						}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
					}
				}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)

			}
			else if (TriangleSetType::Negative == curTriSetType)
			{					
				const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();
				Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
				vrInt srcPtIdx_negative;
				bool ptInElem_negative = isVertexInElement(curSourcePtr_negative, curTriElemPtr, srcPtIdx_negative);
				MyVec3 srcPt_SST_LookUp_negative;
				TriangleElemData data4SST_negative;

				if (ptInElem_negative)
				{
					n_ptInElem_negative++;
					Q_ASSERT(TriangleSetType::Negative == curTriSetType);
					Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
					const DisContinuousType tmpTriElemDisContinuousType = TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,srcPtIdx_negative);
					Q_ASSERT(dis_3_3 == tmpTriElemDisContinuousType);
					MyVec3 vtx_globalCoord_negative[Geometry::vertexs_per_tri];

					for (int v=0;v<Geometry::vertexs_per_tri;++v)
					{
						srcPt_SST_LookUp_negative[v] = srcPtIdx_negative;
						vtx_globalCoord_negative[v] = curTriElemPtr->getVertex(srcPtIdx_negative)->getPos();
						srcPtIdx_negative = (srcPtIdx_negative+1) %Geometry::vertexs_per_tri;
					}
					data4SST_negative.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord_negative);
				}

				for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				{
					if (ptInElem_negative)
					{
						Q_ASSERT(isDisContinuousVtx_source_negative);
						for (int idx_k=0; idx_k < MyDim; ++idx_k)
						{
							Sij_I = 0.0;
#if USE_360_Sample
							Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);	
#else//USE_360_Sample
							for (vrInt subTriIdx=0; subTriIdx < Geometry::vertexs_per_tri; ++subTriIdx)
							{
								Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(subTriIdx, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);	
							}//for (vrInt subTriIdx=0; subTriIdx < Geometry::vertexs_per_tri; ++subTriIdx)
#endif//USE_360_Sample
							Sij_I = n_j_s * Sij_I;
							m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_k)) += Sij_I;

							Kij_I = 0.0;
#if USE_360_Sample
							Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
#else//USE_360_Sample
							for (vrInt subTriIdx=0; subTriIdx < Geometry::vertexs_per_tri; ++subTriIdx)
							{
								Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(subTriIdx, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
							}//for (vrInt subTriIdx=0; subTriIdx < Geometry::vertexs_per_tri; ++subTriIdx)
#endif//USE_360_Sample
							Kij_I = n_j_s * Kij_I;
							m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_k)) += Kij_I;
							//printf("SST : Sij_I[%f]  Kij_I[%f]\n",Sij_I,Kij_I); vrPause;
						}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
					}
					else
					{
						for (int idx_k=0; idx_k < MyDim; ++idx_k)
						{
							Sij_I = compute_S_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I,idx_k);
							Sij_I = n_j_s * Sij_I;
							m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Sij_I;

							Kij_I = compute_K_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I,idx_k);
							Kij_I = n_j_s * Kij_I;
							m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Kij_I;
							//printf("Sij_I[%f]  Kij_I[%f]\n",Sij_I,Kij_I); vrPause;
						}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
					}
				}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
			}
			else
			{
				MyError("Unsupport Triangle element type.");
			}
#else//USE_MI_NegativeSingular

			const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();

			//Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
			vrInt srcPtIdx_negative;
			bool ptInElem_negative = isVertexInElement(curSourcePtr_negative, curTriElemPtr, srcPtIdx_negative);
			//Q_ASSERT(ptInElem_negative);

			MyVec3 srcPt_SST_LookUp_negative;
			TriangleElemData data4SST_negative;

			if (ptInElem_negative)
			{
				Q_ASSERT(TriangleSetType::Negative == curTriSetType);
				Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
				const DisContinuousType tmpTriElemDisContinuousType = TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,srcPtIdx_negative);
				Q_ASSERT(dis_3_3 == tmpTriElemDisContinuousType);
				MyVec3 vtx_globalCoord_negative[Geometry::vertexs_per_tri];

				for (int v=0;v<Geometry::vertexs_per_tri;++v)
				{
					srcPt_SST_LookUp_negative[v] = srcPtIdx_negative;
					vtx_globalCoord_negative[v] = curTriElemPtr->getVertex(srcPtIdx_negative)->getPos();
					srcPtIdx_negative = (srcPtIdx_negative+1) %Geometry::vertexs_per_tri;
				}
				data4SST_negative.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord_negative);
			}

			for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
			{
				if (ptInElem_negative)
				{
					Q_ASSERT(TriangleSetType::Negative == curTriSetType);
					Q_ASSERT(isDisContinuousVtx_source_negative);
#if USE_Peng_Kernel
					int idx_k = 999;
					{
						Sij_I = 0.0;
						Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(0, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);	
						Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(1, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
						Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(2, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);	


						m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_j)) += Sij_I;

						Kij_I = 0.0;
						Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(0, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
						Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(1, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
						Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(2, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);

						m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_j)) += Kij_I;

						//printf("SST : Sij_I[%f]  Kij_I[%f]\n",Sij_I,Kij_I); vrPause;
					}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
#else
					for (int idx_k=0; idx_k < MyDim; ++idx_k)
					{
#if USE_Aliabadi_RegularSample
						Sij_I = 0.0;
						Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(0, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);	
						Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(1, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
						Sij_I += compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(2, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);	

						Sij_I = n_j_s * Sij_I;
						m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_k)) += Sij_I;

						Kij_I = 0.0;
						Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(0, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
						Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(1, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
						Kij_I += compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(2, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
						Kij_I = n_j_s * Kij_I;
						m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_k)) += Kij_I;

#else//USE_Aliabadi_RegularSample
						Sij_I = 0.0;
						Sij_I += compute_S_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(0, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);	
						Sij_I += compute_S_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(1, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
						Sij_I += compute_S_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(2, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);	

						Sij_I = n_j_s * Sij_I;
						m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_k)) += Sij_I;

						Kij_I = 0.0;
						Kij_I += compute_K_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(0, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
						Kij_I += compute_K_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(1, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
						Kij_I += compute_K_ij_I_SST_DisContinuous_Sigmoidal_Aliabadi(2, curSourcePtr_negative,data4SST_negative,idx_i,idx_j,idx_I,idx_k);
						Kij_I = n_j_s * Kij_I;
						m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_k)) += Kij_I;

#endif//USE_Aliabadi_RegularSample
						//printf("SST : Sij_I[%f]  Kij_I[%f]\n",Sij_I,Kij_I); vrPause;
					}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
#endif
				}
				else
				{
#if USE_Peng_Kernel
					int idx_k=999;
					{
						Sij_I = compute_S_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I,idx_k);

						m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Sij_I;

						Kij_I = compute_K_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I,idx_k);

						m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Kij_I;
						//printf("Sij_I[%f]  Kij_I[%f]\n",Sij_I,Kij_I); vrPause;
					}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
#else
					for (int idx_k=0; idx_k < MyDim; ++idx_k)
					{
						Sij_I = compute_S_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I,idx_k);
						Sij_I = n_j_s * Sij_I;
						m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Sij_I;

						Kij_I = compute_K_ij_I_Aliabadi(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I,idx_k);
						Kij_I = n_j_s * Kij_I;
						m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_k)) += Kij_I;
						//printf("Sij_I[%f]  Kij_I[%f]\n",Sij_I,Kij_I); vrPause;
					}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
#endif
				}
			}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
#endif//USE_MI_NegativeSingular
		}//for (int idx_e=0;idx_e<ne;++idx_e)
		Q_ASSERT(1 == n_ptInElem_positive && 1 == n_ptInElem_negative);
		//const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
		const vrFloat delta_ij = 1.0;
		//MyNoticeMsg("G Matrix.") m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],srcDofs_negative[idx_j]) +=  0.5 MyNoticeMsg("No delta ij.");
		MyNoticeMsg("G Matrix.") m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],srcDofs_negative[idx_j]) +=  0.5 * delta_ij MyNoticeMsg("No delta ij.");

		Q_ASSERT(curSourcePtr->isMirrorVertex());
		VertexPtr curSource_Mirror_Positive_Ptr = curSourcePtr->getMirrorVertex();
		const MyVec3I& mirrorDofs_Positive = curSource_Mirror_Positive_Ptr->getDofs();
		MyNoticeMsg("G Matrix.") m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],mirrorDofs_Positive[idx_j]) +=  -0.5 * delta_ij MyNoticeMsg("No delta ij.");
#endif//DualEquation_Aliabadi_NegativeSection
	}
	else
	{
		MyError("Unsupport source point type in AssembleSystem_DisContinuous_DualEquation_Aliabadi.");
	}
}
#endif//USE_UniformSampling

	

#endif//if 0
	

#if USE_Aliabadi_RegularSample

	vrFloat vrBEM3D::compute_U_ij_I_SST_DisContinuous_Regular_Aliabadi(
#if !USE_360_Sample
		const vrInt nSubTriIdx,
#endif//USE_360_Sample
		const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,vrInt idx_i,vrInt idx_j,vrInt idx_I)
	{
		vrFloat retVal = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();
		const bool isDisContinuousVtx = curSourcePtr->isDisContinuousVertex();

		const vrInt nGaussPointSize_xi_In_Theta = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_SubTri;
		const vrInt nGaussPointSize_xi_In_Rho = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Rho_SubTri;

		const vrInt n_gpts = nGaussPointSize_xi_In_Theta * nGaussPointSize_xi_In_Rho;
#if USE_360_Sample
		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal;
#else//USE_360_Sample
		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal[nSubTriIdx];
#endif//USE_360_Sample
		
		//const MyMatrix& cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal[nSubTriIdx];

		if (n_gpts != cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows())
		{
			printf("compute_U_ij_I_SST_DisContinuous_Sigmoidal : n_gpts[%d] == cur_gaussQuadrature_xi_eta_polar_SubTri.rows()[%d] \n",n_gpts, cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows());
		}
		Q_ASSERT(n_gpts == cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows());

		

		for (int index_theta=0,idx_g=0;index_theta<nGaussPointSize_xi_In_Theta;++index_theta)
		{
			
			for (int index_rho=0;index_rho<nGaussPointSize_xi_In_Rho;++index_rho,++idx_g)
			{
				MyNotice;/*the order of theta and rho,(theta,rho)*/
				auto curRows = cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g);
				MyVec2ParamSpace theta_rho;
				theta_rho[TriangleElem::idx_theta_doubleLayer] = curRows[TriangleElem::idx_theta_doubleLayer];
				theta_rho[TriangleElem::idx_rho_doubleLayer]=curRows[TriangleElem::idx_rho_doubleLayer];
				const vrFloat cur_rho = theta_rho[TriangleElem::idx_rho_doubleLayer];
				const vrFloat curWeight_doubleLayer = curRows[TriangleElem::idx_weight_doubleLayer];

				Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_theta_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g).x()));
				Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_rho_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g).y()));
				

				const MyVec2ParamSpace currentSrcPtInParam /*in eta sub triangle space*/ = refDataSST3D.m_SrcPt_in_eta_SubTri;
				Q_ASSERT( ((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))) );
				if (!((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))))
				{
					printf("compute_T_ij_I_SST_DisContinuous : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
				}
				const MyVec2ParamSpace cur_eta = refDataSST3D.pc2eta(currentSrcPtInParam /*MyVec2ParamSpace(0.0,0.0)*/MYNOTICE, theta_rho);
				const MyVec2ParamSpace cur_xi = refDataSST3D.eta2xi_SubTri(cur_eta);
				const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

				MyFloat jacob_eta_nouse;/*jacob_eta = jacob_xi * mat_T_Inv*/
				MyVec3 normals_fieldpoint;
				MyFloat r;
				MyVec3 dr;
				MyFloat drdn;
				getKernelParameters_3D_SST_SubTri(srcPos,fieldPoint,refDataSST3D,jacob_eta_nouse,normals_fieldpoint,r,dr,drdn);

				

				//printf("[%d,%d] r(%f)  rho*A(%f*%f=%f) \n",index_theta,index_rho,r,cur_rho,A,cur_rho*A);vrPause;
#if USE_Jacobi_Weight
				const vrInt subTriId = refDataSST3D.search_Theta_in_eta_belong_SubTri_Index(theta_rho[TriangleElem::idx_theta_doubleLayer]);
				const vrFloat jacob_eta = refDataSST3D.Jacobi_eta_SubTri[subTriId];
#else//USE_Jacobi_Weight
				const vrFloat jacob_eta = refDataSST3D.Jacobi_eta_SubTri;
				//Q_ASSERT(numbers::isEqual(jacob_eta, jacob_eta_nouse));
#endif//USE_Jacobi_Weight

				
				const vrFloat Uij = get_Uij_SST_3D_Aliabadi(idx_i, idx_j, r, dr);
				const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);
				
				retVal += Uij * N_I * jacob_eta * theta_rho[TriangleElem::idx_rho_doubleLayer] * curWeight_doubleLayer;
			}
		}
		return retVal;
	}

	vrFloat vrBEM3D::compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(
#if !USE_360_Sample
		const vrInt nSubTriIdx,
#endif//USE_360_Sample
		const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,vrInt idx_i,vrInt idx_j,vrInt idx_I)
	{
		vrFloat doubleLayer_Term = 0.0, singleLayer_Term = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();

		const vrInt n_gpts = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_SubTri * GlobalConf::g_n_Sample_GaussPointSize_eta_In_Rho_SubTri;
#if USE_360_Sample
		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal;
		const MyMatrix& cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal;
#else//USE_360_Sample
		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal[nSubTriIdx];
		const MyMatrix& cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal[nSubTriIdx];
#endif//USE_360_Sample

		if (n_gpts != cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows())
		{
			printf("compute_T_ij_I_SST_DisContinuous_Sigmoidal : n_gpts[%d] == cur_gaussQuadrature_xi_eta_polar_SubTri.rows()[%d] \n",n_gpts, cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows());
		}
		Q_ASSERT(n_gpts == cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows());

		for (int index_theta=0,idx_g=0;index_theta<GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_SubTri;++index_theta)
		{
			const vrFloat cur_theta_singlelayer_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_theta_singleLayer];


#if 1 // SST
			const vrFloat A = refDataSST3D.A_theta_SubTri(cur_theta_singlelayer_Sigmoidal);
			const vrFloat N_I_0 = refDataSST3D.N_I_0_eta_SubTri(idx_I);

			const MyVec3& n_x = refDataSST3D.unitNormal_fieldPt;
#if USE_Jacobi_Weight
			const vrInt subTriId = refDataSST3D.search_Theta_in_eta_belong_SubTri_Index(cur_theta_singlelayer_Sigmoidal);
			const vrFloat jacob_eta = refDataSST3D.Jacobi_eta_SubTri[subTriId];
#else//USE_Jacobi_Weight
			const vrFloat jacob_eta = refDataSST3D.Jacobi_eta_SubTri;
#endif//USE_Jacobi_Weight
			
			MyVec3 sst_dr;
			vrFloat sst_drdn = 0.0;
			for (int m=0;m<MyDim;++m)
			{
				sst_dr[m] = refDataSST3D.r_i_SubTri(m,cur_theta_singlelayer_Sigmoidal); 
				sst_drdn += (sst_dr[m]*n_x[m]);
			}
#if 0
			const vrFloat M0 = (sst_drdn)*( (1.0-2.0*mu)*delta_ij + 3.0* sst_dr[idx_i] * sst_dr[idx_j]) - (1.0-2.0*mu)*(sst_dr[idx_i]*n_x[idx_j]-sst_dr[idx_j]*n_x[idx_i]);
			const vrFloat M1 = ((-1.0) / (8.0*numbers::MyPI*(1-mu))) * M0 * jacob_eta ;
#else
			const vrFloat nouse_r = 1.0;
			const vrFloat M0 = get_Tij_SST_3D_Aliabadi(idx_i, idx_j, nouse_r, sst_dr, sst_drdn, n_x);
			const vrFloat M1 = M0 * jacob_eta ;
#endif


			const vrFloat F_1_ij_I = (M1*N_I_0)/(A*A);
#endif // SST

			const vrFloat curWeight_singleLayer_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_weight_singleLayer];
			const vrFloat cur_rho_bar_singleLayer_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_rho_bar_singleLayer];
			//const vrFloat cur_jacobi_singleLayer_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_Jacobi_singleLayer];
			for (int index_rho=0;index_rho<GlobalConf::g_n_Sample_GaussPointSize_eta_In_Rho_SubTri;++index_rho,++idx_g)
			{
				MyNotice;/*the order of theta and rho,(theta,rho)*/
				auto curRows = cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g);
				MyVec2ParamSpace theta_rho;
				theta_rho[TriangleElem::idx_theta_doubleLayer] = curRows[TriangleElem::idx_theta_doubleLayer];
				theta_rho[TriangleElem::idx_rho_doubleLayer]=curRows[TriangleElem::idx_rho_doubleLayer];
				const vrFloat cur_rho = theta_rho[TriangleElem::idx_rho_doubleLayer];
				const vrFloat curWeight_doubleLayer = curRows[TriangleElem::idx_weight_doubleLayer];

				Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_theta_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g).x()));
				Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_rho_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.row(idx_g).y()));
				Q_ASSERT(numbers::isEqual(cur_theta_singlelayer_Sigmoidal,theta_rho[TriangleElem::idx_theta_doubleLayer]));

				const MyVec2ParamSpace currentSrcPtInParam /*in eta sub triangle space*/ = refDataSST3D.m_SrcPt_in_eta_SubTri;
				Q_ASSERT( ((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))) );
				if (!((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))))
				{
					printf("compute_T_ij_I_SST_DisContinuous : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
				}
				const MyVec2ParamSpace cur_eta = refDataSST3D.pc2eta(currentSrcPtInParam /*MyVec2ParamSpace(0.0,0.0)*/MYNOTICE, theta_rho);
				const MyVec2ParamSpace cur_xi = refDataSST3D.eta2xi_SubTri(cur_eta);
				const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

				MyFloat jacob_eta_nouse;/*jacob_eta = jacob_xi * mat_T_Inv*/
				MyVec3 normals_fieldpoint;
				MyFloat r;
				MyVec3 dr;
				MyFloat drdn;
				getKernelParameters_3D_SST_SubTri(srcPos,fieldPoint,refDataSST3D,jacob_eta_nouse,normals_fieldpoint,r,dr,drdn);

				//Q_ASSERT(numbers::isEqual(jacob_eta, jacob_eta_nouse));

				//printf("[%d,%d] r(%f)  rho*A(%f*%f=%f) \n",index_theta,index_rho,r,cur_rho,A,cur_rho*A);vrPause;
				Q_ASSERT(numbers::isEqual(r,cur_rho*A));

				const vrFloat Tij = get_Tij_SST_3D_Aliabadi(idx_i, idx_j, r, dr,drdn,normals_fieldpoint);
				const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);

				const vrFloat SingularTerm_Tij_I = Tij * N_I * jacob_eta * theta_rho[TriangleElem::idx_rho_doubleLayer] ; 


				const vrFloat SingularTerm_F_1_ij_I = (1.0/cur_rho)*(F_1_ij_I);

				doubleLayer_Term += (SingularTerm_Tij_I - SingularTerm_F_1_ij_I) * curWeight_doubleLayer ;
			}

			const vrFloat beta = 1.0 / A;

			singleLayer_Term += F_1_ij_I * log( abs(cur_rho_bar_singleLayer_Sigmoidal/beta) ) * curWeight_singleLayer_Sigmoidal ;
		}
		return (doubleLayer_Term + singleLayer_Term);
	}
	vrFloat vrBEM3D::compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(
#if !USE_360_Sample
		const vrInt nSubTriIdx,
#endif//USE_360_Sample
		const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,const vrInt idx_i,const vrInt idx_j,const vrInt idx_I,const vrInt idx_k)
	{
#if 1
		vrFloat retVal = 0.0;
		vrFloat doubleLayer_Term_k = 0.0, singleLayer_Term_k = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();

		const vrInt nGaussPointSize_eta_In_Theta_SubTri_Regular = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_SubTri;
		const vrInt nGaussPointSize_eta_In_Rho_SubTri_Regular = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Rho_SubTri;
		const vrInt n_gpts = nGaussPointSize_eta_In_Theta_SubTri_Regular * nGaussPointSize_eta_In_Rho_SubTri_Regular;

#if USE_360_Sample
		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri_Regular = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal;
		const MyMatrix& cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Regular = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal;
#else//USE_360_Sample
		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri_Regular = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal[nSubTriIdx];
		const MyMatrix& cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Regular = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal[nSubTriIdx];
#endif//USE_360_Sample

		if (n_gpts != cur_gaussQuadrature_xi_eta_polar_SubTri_Regular.rows())
		{
			printf("compute_K_ij_I_SST_DisContinuous_Sigmoidal : n_gpts[%d] == cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows()[%d] \n",n_gpts, cur_gaussQuadrature_xi_eta_polar_SubTri_Regular.rows());
		}
		Q_ASSERT(n_gpts == cur_gaussQuadrature_xi_eta_polar_SubTri_Regular.rows());
		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);

		//for (int idx_k=0; idx_k < MyDim; ++idx_k)
		{
			for (int index_theta=0,idx_g=0; index_theta < nGaussPointSize_eta_In_Theta_SubTri_Regular; ++index_theta)
			{
				const vrFloat cur_theta_singlelayer_Regular = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Regular.row(index_theta)[TriangleElem::idx_theta_singleLayer];

#if 1 //SST
				const vrFloat A = refDataSST3D.A_theta_SubTri(cur_theta_singlelayer_Regular);
				const vrFloat N_I_0 = refDataSST3D.N_I_0_eta_SubTri(idx_I);
				//const vrFloat N_I_1 = refDataSST3D.N_I_1_eta(idx_I,cur_theta_singlelayer);
				const MyVec3& n_x = refDataSST3D.unitNormal_fieldPt;

#if USE_Jacobi_Weight
				const vrInt subTriId = refDataSST3D.search_Theta_in_eta_belong_SubTri_Index(cur_theta_singlelayer_Regular);
				const vrFloat jacob_eta = refDataSST3D.Jacobi_eta_SubTri[subTriId];
#else//USE_Jacobi_Weight
				const vrFloat jacob_eta = refDataSST3D.Jacobi_eta_SubTri;
#endif//USE_Jacobi_Weight

				MyVec3 sst_dr;
				vrFloat sst_drdn = 0.0;
				for (int m=0;m<MyDim;++m)
				{
					sst_dr[m] = refDataSST3D.r_i_SubTri(m,cur_theta_singlelayer_Regular); 
					sst_drdn += (sst_dr[m]*n_x[m]);
				}
#if 0
				const vrFloat M0 = (
					(1.0-2.0*mu)*(delta_ij * sst_dr[idx_k] + delta_jk * sst_dr[idx_i] MyNotice - delta_ik * sst_dr[idx_j]) + 
					3.0 * sst_dr[idx_i] * sst_dr[idx_j] * sst_dr[idx_k]
				) * unitNormal_srcPt[idx_k];

				const vrFloat M1 = ((1.0) / (8.0*numbers::MyPI*(1-mu))) * M0 * jacob_eta;
#else
				const vrFloat nouse_r = 1.0;

				/*vrFloat M0 = 0.0;
				for (int idx_k=0; idx_k < MyDim; ++idx_k)
				{
					M0 += get_Kij_SST_3D_k_Aliabadi_Peng(idx_i, idx_j, idx_k, nouse_r, sst_dr, sst_drdn, n_s);
				}*/
				const vrFloat M0 = get_Kij_SST_3D_k_Aliabadi(idx_i, idx_j, idx_k, nouse_r, sst_dr, sst_drdn);
				//printf("get_Kij_SST_3D_k_Aliabadi [nouse_r] = %f\n",M0);
				const vrFloat M1 = M0 * jacob_eta ;
#endif

				const vrFloat F_1_ij_I_k = (M1*N_I_0)/(A*A);
#endif//SST
				const vrFloat curWeight_singleLayer_Regular = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Regular.row(index_theta)[TriangleElem::idx_weight_singleLayer];
				const vrFloat cur_rho_bar_Regular = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Regular.row(index_theta)[TriangleElem::idx_rho_bar_singleLayer];
				//const vrFloat cur_jacobi_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_Jacobi_singleLayer];
				for (int index_rho=0; index_rho < nGaussPointSize_eta_In_Rho_SubTri_Regular;++index_rho,++idx_g)
				{
					auto curRows_Regular = cur_gaussQuadrature_xi_eta_polar_SubTri_Regular.row(idx_g);
					MyVec2ParamSpace theta_rho_Regular;
					theta_rho_Regular[TriangleElem::idx_theta_doubleLayer] = curRows_Regular[TriangleElem::idx_theta_doubleLayer];
					theta_rho_Regular[TriangleElem::idx_rho_doubleLayer]=curRows_Regular[TriangleElem::idx_rho_doubleLayer];
					const vrFloat cur_rho = theta_rho_Regular[TriangleElem::idx_rho_doubleLayer];
					const vrFloat curWeight_doubleLayer = curRows_Regular[TriangleElem::idx_weight_doubleLayer];

					Q_ASSERT(numbers::isEqual(theta_rho_Regular[TriangleElem::idx_theta_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Regular.row(idx_g).x()));
					Q_ASSERT(numbers::isEqual(theta_rho_Regular[TriangleElem::idx_rho_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Regular.row(idx_g).y()));
					Q_ASSERT(numbers::isEqual(cur_theta_singlelayer_Regular,theta_rho_Regular[TriangleElem::idx_theta_doubleLayer]));

					const MyVec2ParamSpace currentSrcPtInParam /*in eta sub triangle space*/ = refDataSST3D.m_SrcPt_in_eta_SubTri;
					Q_ASSERT( ((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))) );
					if (!((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))))
					{
						printf("compute_K_ij_I_SST_DisContinuous_Sigmoidal : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
					}

					const MyVec2ParamSpace cur_eta = refDataSST3D.pc2eta(currentSrcPtInParam MYNOTICE, theta_rho_Regular);
					const MyVec2ParamSpace cur_xi = refDataSST3D.eta2xi_SubTri(cur_eta);
					const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

					MyFloat jacob_eta_nouse;/*jacob_eta = jacob_xi * mat_T_Inv*/
					MyVec3 normals_fieldpoint;
					MyFloat r;
					MyVec3 dr;
					MyFloat drdn;
					getKernelParameters_3D_SST_SubTri(srcPos,fieldPoint,refDataSST3D,jacob_eta_nouse,normals_fieldpoint,r,dr,drdn);

					//Q_ASSERT(numbers::isEqual(jacob_eta, jacob_eta_nouse));
					Q_ASSERT(numbers::isEqual(r,cur_rho*A));

					/*vrFloat Kij_k = 0.0;
					for (int idx_k=0; idx_k < MyDim; ++idx_k)
					{
						Kij_k += get_Kij_SST_3D_k_Aliabadi_Peng(idx_i, idx_j, idx_k, r, dr, drdn,n_s);
					}*/
					const vrFloat Kij_k = get_Kij_SST_3D_k_Aliabadi(idx_i, idx_j, idx_k, r, dr, drdn);
					//printf("get_Kij_SST_3D_k_Aliabadi = %f\n",Kij_k);
					const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);

					//////////////////////////////////////////////////////////////////////////

					const vrFloat SingularTerm_Kij_I_k = Kij_k * N_I * jacob_eta * theta_rho_Regular[TriangleElem::idx_rho_doubleLayer];
					const vrFloat SingularTerm_F_1_ij_I_k = (1.0/cur_rho)*(F_1_ij_I_k);
					doubleLayer_Term_k += (SingularTerm_Kij_I_k - SingularTerm_F_1_ij_I_k) * curWeight_doubleLayer;

				}//for (int index_rho=0;index_rho<TriangleElem::GaussPointSize_eta_In_Rho_SubTri;++index_rho,++idx_g)

				const vrFloat beta = 1.0 / A;

				singleLayer_Term_k += F_1_ij_I_k * log( abs(cur_rho_bar_Regular/beta) ) * curWeight_singleLayer_Regular;
			}//for (int index_theta=0,idx_g=0;index_theta<TriangleElem::GaussPointSize_eta_In_Theta_SubTri;++index_theta)

			retVal += (doubleLayer_Term_k + singleLayer_Term_k);

		}//for (int idx_k=0; idx_k < MyDim; ++idx_k)

		return retVal;
#endif
	}

	vrFloat vrBEM3D::compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(
#if !USE_360_Sample
		const vrInt nSubTriIdx,
#endif//USE_360_Sample
		const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,const vrInt idx_i,const vrInt idx_j,const vrInt idx_I,const vrInt idx_k)
	{
#if 1
		vrFloat retVal = 0.0;
		vrFloat doubleLayer_Term_k = 0.0, singleLayer_Term_k = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();
		const MyVec3& n_s = curSourcePtr->getVertexNormal();MyNotice;

		const vrInt nGaussPointSize_eta_In_Theta_SubTri_Regular = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_SubTri;
		const vrInt nGaussPointSize_eta_In_Rho_SubTri_Regular = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Rho_SubTri;
		const vrInt n_gpts = GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_SubTri * GlobalConf::g_n_Sample_GaussPointSize_eta_In_Rho_SubTri;

#if USE_360_Sample
		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri_Regular = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal;
		const MyMatrix& cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Regular = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal;
#else//USE_360_Sample
		const MyMatrix& cur_gaussQuadrature_xi_eta_polar_SubTri_Regular = refDataSST3D.m_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal[nSubTriIdx];
		const MyMatrix& cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Regular = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal[nSubTriIdx];
#endif//USE_360_Sample

		if (n_gpts != cur_gaussQuadrature_xi_eta_polar_SubTri_Regular.rows())
		{
			printf("compute_S_ij_I_SST_DisContinuous_Sigmoidal : n_gpts[%d] == cur_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal.rows()[%d] \n",n_gpts, cur_gaussQuadrature_xi_eta_polar_SubTri_Regular.rows());
		}
		Q_ASSERT(n_gpts == cur_gaussQuadrature_xi_eta_polar_SubTri_Regular.rows());
		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
		//for (int idx_k=0; idx_k < MyDim; ++idx_k)
		{
			for (int index_theta=0,idx_g=0;index_theta<nGaussPointSize_eta_In_Theta_SubTri_Regular;++index_theta)
			{
				const vrFloat cur_theta_singlelayer_Regular = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Regular.row(index_theta)[TriangleElem::idx_theta_singleLayer];

#if 1 // SST
				const vrFloat A = refDataSST3D.A_theta_SubTri(cur_theta_singlelayer_Regular);
				const vrFloat N_I_0 = refDataSST3D.N_I_0_eta_SubTri(idx_I);
				const vrFloat N_I_1 = refDataSST3D.N_I_1_eta_SubTri(idx_I,cur_theta_singlelayer_Regular);
				const MyVec3& n_x = refDataSST3D.unitNormal_fieldPt;
				
#if USE_Jacobi_Weight
				const vrInt subTriId = refDataSST3D.search_Theta_in_eta_belong_SubTri_Index(cur_theta_singlelayer_Regular);
				const vrFloat jacob_eta = refDataSST3D.Jacobi_eta_SubTri[subTriId];
#else//USE_Jacobi_Weight
				const vrFloat jacob_eta = refDataSST3D.Jacobi_eta_SubTri;
#endif//USE_Jacobi_Weight

				MyVec3 sst_dr;
				vrFloat sst_drdn = 0.0;
				for (int m=0;m<MyDim;++m)
				{
					sst_dr[m] = refDataSST3D.r_i_SubTri(m,cur_theta_singlelayer_Regular); 
					sst_drdn += (sst_dr[m]*n_x[m]);
				}
#if 0
				const vrFloat M0 = (
					3.0 * sst_drdn * ( (1.0-2.0*mu)*delta_ik*sst_dr[idx_j] + mu*(delta_ij*sst_dr[idx_k]+delta_jk*sst_dr[idx_i]) MyNotice - 5.0*sst_dr[idx_i]*sst_dr[idx_j]*sst_dr[idx_k] ) +
					3.0 * mu * (n_x[idx_i]*sst_dr[idx_j]*sst_dr[idx_k]+n_x[idx_k]*sst_dr[idx_i]*sst_dr[idx_j]) MyNotice - 
					(1.0-4.0*mu) * delta_ik * n_x[idx_j] + 
					(1.0-2.0*mu) * (3.0 * n_x[idx_j] * sst_dr[idx_i] * sst_dr[idx_k] + delta_ij * n_x[idx_k] + delta_jk * n_x[idx_i])
					) * unitNormal_srcPt[idx_k];
				const vrFloat M1 = ( (shearMod)/(4.0*numbers::MyPI*(1.0-mu)) ) * M0 * jacob_eta;
#else
				const vrFloat nouse_r = 1.0;
				/*vrFloat M0 = 0.0;
				for (int idx_k=0; idx_k < MyDim; ++idx_k)
				{
					M0 += get_Sij_SST_3D_k_Aliabadi_Peng(idx_i, idx_j, idx_k, nouse_r, sst_dr, sst_drdn, n_x, n_s);
				}*/
				const vrFloat M0 = get_Sij_SST_3D_k_Aliabadi(idx_i, idx_j, idx_k, nouse_r, sst_dr, sst_drdn, n_x);
				const vrFloat M1 = M0 * jacob_eta ;
#endif

				const vrFloat F_2_ij_I_k = (M1*N_I_0)/(A*A*A);
				const vrFloat F_1_ij_I_k = (M1*N_I_1)/(A*A);
#endif//SST
				const vrFloat curWeight_singleLayer_Regular = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Regular.row(index_theta)[TriangleElem::idx_weight_singleLayer];
				const vrFloat cur_rho_bar_Regular = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Regular.row(index_theta)[TriangleElem::idx_rho_bar_singleLayer];
				//const vrFloat cur_jacobi_Sigmoidal = cur_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal.row(index_theta)[TriangleElem::idx_Jacobi_singleLayer];
				for (int index_rho=0; index_rho<nGaussPointSize_eta_In_Rho_SubTri_Regular; ++index_rho,++idx_g)
				{
					MyNotice;/*the order of theta and rho,(theta,rho)*/
					auto curRows_Regular = cur_gaussQuadrature_xi_eta_polar_SubTri_Regular.row(idx_g);
					MyVec2ParamSpace theta_rho_Regular;
					theta_rho_Regular[TriangleElem::idx_theta_doubleLayer] = curRows_Regular[TriangleElem::idx_theta_doubleLayer];
					theta_rho_Regular[TriangleElem::idx_rho_doubleLayer]=curRows_Regular[TriangleElem::idx_rho_doubleLayer];
					const vrFloat cur_rho = theta_rho_Regular[TriangleElem::idx_rho_doubleLayer];
					const vrFloat curWeight_doubleLayer = curRows_Regular[TriangleElem::idx_weight_doubleLayer];

					Q_ASSERT(numbers::isEqual(theta_rho_Regular[TriangleElem::idx_theta_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Regular.row(idx_g).x()));
					Q_ASSERT(numbers::isEqual(theta_rho_Regular[TriangleElem::idx_rho_doubleLayer], cur_gaussQuadrature_xi_eta_polar_SubTri_Regular.row(idx_g).y()));
					Q_ASSERT(numbers::isEqual(cur_theta_singlelayer_Regular,theta_rho_Regular[TriangleElem::idx_theta_doubleLayer]));

					const MyVec2ParamSpace currentSrcPtInParam /*in eta sub triangle space*/ = refDataSST3D.m_SrcPt_in_eta_SubTri;
					Q_ASSERT( ((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))) );
					if (!((!numbers::isEqual(currentSrcPtInParam[0],0.0)) && (!numbers::isEqual(currentSrcPtInParam[1],0.0))))
					{
						printf("compute_S_ij_I_SST_DisContinuous_Sigmoidal : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
					}
					const MyVec2ParamSpace cur_eta = refDataSST3D.pc2eta(currentSrcPtInParam /*MyVec2ParamSpace(0.0,0.0)*/MYNOTICE, theta_rho_Regular);
					const MyVec2ParamSpace cur_xi = refDataSST3D.eta2xi_SubTri(cur_eta);
					const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

					MyFloat jacob_eta_nouse;/*jacob_eta = jacob_xi * mat_T_Inv*/
					MyVec3 normals_fieldpoint;
					MyFloat r;
					MyVec3 dr;
					MyFloat drdn;
					getKernelParameters_3D_SST_SubTri(srcPos,fieldPoint,refDataSST3D,jacob_eta_nouse,normals_fieldpoint,r,dr,drdn);

					//Q_ASSERT(numbers::isEqual(jacob_eta, jacob_eta_nouse));

					//printf("[%d,%d] r(%f)  rho*A(%f) \n",index_theta,index_rho,r,cur_rho*A);
					Q_ASSERT(numbers::isEqual(r,cur_rho*A));

					/*vrFloat Sij_k = 0.0;
					for (int idx_k=0; idx_k < MyDim; ++idx_k)
					{
						Sij_k += get_Sij_SST_3D_k_Aliabadi_Peng(idx_i, idx_j, idx_k, r, dr, drdn, normals_fieldpoint, n_s);
					}*/
					const vrFloat Sij_k = get_Sij_SST_3D_k_Aliabadi(idx_i, idx_j, idx_k, r, dr, drdn, normals_fieldpoint);
					const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);

					const vrFloat SingularTerm_Sij_I_k = Sij_k * N_I * jacob_eta * theta_rho_Regular[TriangleElem::idx_rho_doubleLayer] ;

					const vrFloat SingularTerm_F_1_ij_I_k = (1.0/(cur_rho))*(F_1_ij_I_k);
					const vrFloat SingularTerm_F_2_ij_I_k = (1.0/(cur_rho*cur_rho)) * (F_2_ij_I_k);

					doubleLayer_Term_k += (SingularTerm_Sij_I_k - (SingularTerm_F_1_ij_I_k + SingularTerm_F_2_ij_I_k)) * curWeight_doubleLayer;

				}//for (int index_rho=0;index_rho<TriangleElem::GaussPointSize_eta_In_Rho_SubTri;++index_rho,++idx_g)

				const vrFloat beta = 1.0 / A;

				singleLayer_Term_k += ( (F_1_ij_I_k * log( abs(cur_rho_bar_Regular/beta) )) MyNotice - (F_2_ij_I_k * (1.0/(cur_rho_bar_Regular))) )* curWeight_singleLayer_Regular;

			}//for (int index_theta=0,idx_g=0;index_theta<TriangleElem::GaussPointSize_eta_In_Theta_SubTri;++index_theta)

			retVal += (doubleLayer_Term_k + singleLayer_Term_k);

		}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
		return retVal;
#endif
	}
#endif//USE_Aliabadi_RegularSample
	
	
	
	

	vrFloat vrBEM3D::compute_S_ij_I_Aliabadi(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D, const vrInt idx_i, const vrInt idx_j, const  vrInt idx_I, const vrInt idx_k)
	{
		/*MyError("vrBEM3D::compute_S_ij_I.");
		return 0.0;*/
#if 1
		//MyError("vrBEM3D::compute_S_ij_I.");
		vrFloat retVal = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();

		const MyVec3& n_s = curSourcePtr->getVertexNormal();MyNotice;
#if SPEEDUP_5_31
		int tmp_GaussPointSize_xi_In_Theta = 0;
		int tmp_GaussPointSize_xi_In_Rho = 0;
		if (dis_regular == refDataSST3D.m_DisContinuousType)
		{
			tmp_GaussPointSize_xi_In_Theta = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Theta;
			tmp_GaussPointSize_xi_In_Rho = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Rho;
		}
		else
		{
			tmp_GaussPointSize_xi_In_Theta = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Theta_DisContinuous;
			tmp_GaussPointSize_xi_In_Rho = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Rho_DisContinuous;
		}
		const vrInt nGaussPointSize_xi_In_Theta = tmp_GaussPointSize_xi_In_Theta;
		const vrInt nGaussPointSize_xi_In_Rho = tmp_GaussPointSize_xi_In_Rho;
#endif
		const vrInt n_gpts = nGaussPointSize_xi_In_Theta * nGaussPointSize_xi_In_Rho;
		Q_ASSERT(n_gpts == refDataSST3D.m_gaussQuadrature_xi_polar.rows());

		//for (int idx_k=0; idx_k < MyDim; ++idx_k)
		{
			for (int idx_g=0;idx_g < n_gpts;++idx_g)
			{
				MyNotice;/*the order of theta and rho,(theta,rho)*/
				auto curRows = refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g);
				MyVec2ParamSpace theta_rho;
				theta_rho[TriangleElem::idx_theta_doubleLayer] = curRows[TriangleElem::idx_theta_doubleLayer];
				theta_rho[TriangleElem::idx_rho_doubleLayer]=curRows[TriangleElem::idx_rho_doubleLayer];
				const vrFloat curWeight = curRows[TriangleElem::idx_weight_doubleLayer];
				Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_theta_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g).x()));
				Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_rho_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g).y()));

				const MyVec2ParamSpace currentSrcPtInParam /*in xi space*/ = refDataSST3D.m_SrcPt_in_xi;
				Q_ASSERT( ((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) /*compute_T_ij_I*/);
				if (!((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) )
				{
					printf("compute_S_ij_I : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
				}
				const MyVec2ParamSpace cur_xi = refDataSST3D.pc2xi(currentSrcPtInParam MYNOTICE, theta_rho);
				const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

				MyFloat jacob_xi;
				MyVec3 unitNormals_fieldPt;
				MyFloat r;
				MyVec3 dr;
				MyFloat drdn;
				getKernelParameters_3D(srcPos,fieldPoint,refDataSST3D,jacob_xi,unitNormals_fieldPt,r,dr,drdn);

#if USE_Peng_Kernel
				vrFloat Sij_k = 0.0;
				for (int idx_k=0; idx_k < MyDim; ++idx_k)
				{
					Sij_k += get_Sij_SST_3D_k_Aliabadi_Peng(idx_i, idx_j, idx_k, r, dr, drdn, unitNormals_fieldPt, n_s);
				}
#else//USE_Peng_Kernel
				const vrFloat Sij_k = get_Sij_SST_3D_k_Aliabadi(idx_i, idx_j, idx_k, r, dr, drdn, unitNormals_fieldPt);
#endif//USE_Peng_Kernel
				
				
				const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);
				retVal += Sij_k * N_I * jacob_xi * theta_rho[TriangleElem::idx_rho_doubleLayer] * curWeight;

			}//for (int idx_g=0;idx_g < n_gpts;++idx_g)
		}//for (int idx_k=0; idx_k < MyDim; ++idx_k)
		return retVal;
#endif
	}

	

	vrFloat vrBEM3D::compute_K_ij_I_Aliabadi(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D, const vrInt idx_i, const vrInt idx_j, const  vrInt idx_I, const vrInt idx_k)
	{
		/*MyError("vrBEM3D::compute_K_ij_I.");
		return 0.0;*/
#if 1


		//MyError("vrBEM3D::compute_K_ij_I.");
		vrFloat retVal = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();

		const MyVec3& n_s = curSourcePtr->getVertexNormal();MyNotice;
#if SPEEDUP_5_31
		int tmp_GaussPointSize_xi_In_Theta = 0;
		int tmp_GaussPointSize_xi_In_Rho = 0;
		if (dis_regular == refDataSST3D.m_DisContinuousType)
		{
			tmp_GaussPointSize_xi_In_Theta = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Theta;
			tmp_GaussPointSize_xi_In_Rho = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Rho;
		}
		else
		{
			tmp_GaussPointSize_xi_In_Theta = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Theta_DisContinuous;
			tmp_GaussPointSize_xi_In_Rho = GlobalConf::g_n_Sample_GaussPointSize_xi_In_Rho_DisContinuous;
		}
		const vrInt nGaussPointSize_xi_In_Theta = tmp_GaussPointSize_xi_In_Theta;
		const vrInt nGaussPointSize_xi_In_Rho = tmp_GaussPointSize_xi_In_Rho;
#endif
		const vrInt n_gpts = nGaussPointSize_xi_In_Theta * nGaussPointSize_xi_In_Rho;
		Q_ASSERT(n_gpts == refDataSST3D.m_gaussQuadrature_xi_polar.rows());

		//for (int idx_k=0; idx_k < MyDim; ++idx_k)
		{
			for (int idx_g=0;idx_g < n_gpts;++idx_g)
			{
				MyNotice;/*the order of theta and rho,(theta,rho)*/
				auto curRows = refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g);
				MyVec2ParamSpace theta_rho;
				theta_rho[TriangleElem::idx_theta_doubleLayer] = curRows[TriangleElem::idx_theta_doubleLayer];
				theta_rho[TriangleElem::idx_rho_doubleLayer]=curRows[TriangleElem::idx_rho_doubleLayer];
				const vrFloat curWeight = curRows[TriangleElem::idx_weight_doubleLayer];
				Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_theta_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g).x()));
				Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_rho_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g).y()));

				const MyVec2ParamSpace currentSrcPtInParam /*in xi space*/ = refDataSST3D.m_SrcPt_in_xi;
				Q_ASSERT( ((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) /*compute_T_ij_I*/);
				if (!((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) )
				{
					printf("compute_T_ij_I : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
				}
				const MyVec2ParamSpace cur_xi = refDataSST3D.pc2xi( currentSrcPtInParam MYNOTICE,theta_rho);
				const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

				MyFloat jacob_xi;
				MyVec3 unitNormals_fieldPt;
				MyFloat r;
				MyVec3 dr;
				MyFloat drdn;
				getKernelParameters_3D(srcPos,fieldPoint,refDataSST3D,jacob_xi,unitNormals_fieldPt,r,dr,drdn);

#if USE_Peng_Kernel
				vrFloat Kij = 0.0;
				for (int idx_k=0; idx_k < MyDim; ++idx_k)
				{
					Kij += get_Kij_SST_3D_k_Aliabadi_Peng(idx_i, idx_j, idx_k, r, dr, drdn,n_s);
				}
#else//USE_Peng_Kernel
				const vrFloat Kij = get_Kij_SST_3D_k_Aliabadi(idx_i, idx_j, idx_k, r, dr, drdn);
#endif//USE_Peng_Kernel
				
				
				const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);
				retVal += Kij * N_I * jacob_xi * theta_rho[TriangleElem::idx_rho_doubleLayer] * curWeight;
			}
		}

		return retVal;
#endif
	}
#else//USE_Aliabadi

vrFloat vrBEM3D::compute_T_ij_I(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,vrInt idx_i,vrInt idx_j,vrInt idx_I)
{
	vrFloat retVal = 0.0;
	const MyVec3 srcPos = curSourcePtr->getPos();

#if SPEEDUP_5_31
	int tmp_GaussPointSize_xi_In_Theta = 0;
	int tmp_GaussPointSize_xi_In_Rho = 0;
	if (dis_regular == refDataSST3D.m_DisContinuousType)
	{
		tmp_GaussPointSize_xi_In_Theta = TriangleElem::GaussPointSize_xi_In_Theta;
		tmp_GaussPointSize_xi_In_Rho = TriangleElem::GaussPointSize_xi_In_Rho;
	}
	else
	{
		tmp_GaussPointSize_xi_In_Theta = TriangleElem::GaussPointSize_xi_In_Theta_DisContinuous;
		tmp_GaussPointSize_xi_In_Rho = TriangleElem::GaussPointSize_xi_In_Rho_DisContinuous;
	}
	const vrInt nGaussPointSize_xi_In_Theta = tmp_GaussPointSize_xi_In_Theta;
	const vrInt nGaussPointSize_xi_In_Rho = tmp_GaussPointSize_xi_In_Rho;
#endif
	const vrInt n_gpts = nGaussPointSize_xi_In_Theta * nGaussPointSize_xi_In_Rho;
	Q_ASSERT(n_gpts == refDataSST3D.m_gaussQuadrature_xi_polar.rows());

	for (int idx_g=0;idx_g < n_gpts;++idx_g)
	{
		MyNotice;/*the order of theta and rho,(theta,rho)*/
		auto curRows = refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g);
		MyVec2ParamSpace theta_rho;
		theta_rho[TriangleElem::idx_theta_doubleLayer] = curRows[TriangleElem::idx_theta_doubleLayer];
		theta_rho[TriangleElem::idx_rho_doubleLayer]=curRows[TriangleElem::idx_rho_doubleLayer];
		const vrFloat curWeight = curRows[TriangleElem::idx_weight_doubleLayer];
		Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_theta_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g).x()));
		Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_rho_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g).y()));

		const MyVec2ParamSpace currentSrcPtInParam /*in xi space*/ = refDataSST3D.m_SrcPt_in_xi;
		Q_ASSERT( ((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) /*compute_T_ij_I*/);
		if (!((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) )
		{
			printf("compute_T_ij_I : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
		}
		const MyVec2ParamSpace cur_xi = refDataSST3D.pc2xi(currentSrcPtInParam /*MyVec2ParamSpace(0.0,0.0)*/ MYNOTICE, theta_rho);
		const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

		MyFloat jacob_xi;
		MyVec3 normals_fieldpoint;
		MyFloat r;
		MyVec3 dr;
		MyFloat drdn;
		getKernelParameters_3D(srcPos,fieldPoint,refDataSST3D,jacob_xi,normals_fieldpoint,r,dr,drdn);
		//Q_ASSERT(numbers::isEqual(r,theta_rho[TriangleElemData::idx_rho]));


		const vrFloat Tij = get_Tij_SST_3D(idx_i, idx_j, r, dr,drdn,normals_fieldpoint);
		const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);
		retVal += Tij * N_I * jacob_xi * theta_rho[TriangleElem::idx_rho_doubleLayer] * curWeight;
	}
	return retVal;
}

	vrFloat vrBEM3D::compute_T_ij_I_SST(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,vrInt idx_i,vrInt idx_j,vrInt idx_I)
	{
		/*MyError("vrBEM3D::compute_T_ij_I_SST.");
		return 0.0;*/
#if 1

		MyNotice;//using ETA space
		//
		vrFloat doubleLayer_Term = 0.0, singleLayer_Term = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();

#if SPEEDUP_5_31

		int tmp_GaussPointSize_eta_In_Theta = 0;
		int tmp_GaussPointSize_eta_In_Rho = 0;
		if (dis_regular == refDataSST3D.m_DisContinuousType)
		{
			tmp_GaussPointSize_eta_In_Theta = TriangleElem::GaussPointSize_eta_In_Theta;
			tmp_GaussPointSize_eta_In_Rho = TriangleElem::GaussPointSize_eta_In_Rho;
		}
		else
		{
			tmp_GaussPointSize_eta_In_Theta = TriangleElem::GaussPointSize_eta_In_Theta_DisContinuous;
			tmp_GaussPointSize_eta_In_Rho = TriangleElem::GaussPointSize_eta_In_Rho_DisContinuous;
		}
		const vrInt nGaussPointSize_eta_In_Theta = tmp_GaussPointSize_eta_In_Theta;
		const vrInt nGaussPointSize_eta_In_Rho = tmp_GaussPointSize_eta_In_Rho;
#endif

		const vrInt n_gpts = nGaussPointSize_eta_In_Theta * nGaussPointSize_eta_In_Rho;
		Q_ASSERT(n_gpts == refDataSST3D.m_gaussQuadrature_xi_eta_polar.rows());

		for (int index_theta=0,idx_g=0;index_theta<nGaussPointSize_eta_In_Theta;++index_theta)
		{
			const vrFloat cur_theta_singlelayer = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer.row(index_theta)[TriangleElem::idx_theta_singleLayer];
			const vrFloat curWeight_singleLayer = refDataSST3D.m_gaussQuadrature_eta_theta_singleLayer.row(index_theta)[TriangleElem::idx_weight_singleLayer];
#if 1 // SST
			const vrFloat A = refDataSST3D.A_theta(cur_theta_singlelayer);
			const vrFloat N_I_0 = refDataSST3D.N_I_0_eta(idx_I);
			//const vrFloat N_I_1 = refDataSST3D.N_I_1_eta(idx_I,cur_theta_singlelayer);
			const MyVec3& n_x = refDataSST3D.unitNormal_fieldPt;
			const vrFloat jacob_eta = refDataSST3D.Jacobi_eta;
			MyVec3 sst_dr;
			vrFloat sst_drdn = 0.0;
			for (int m=0;m<MyDim;++m)
			{
				sst_dr[m] = refDataSST3D.r_i(m,cur_theta_singlelayer); 
				sst_drdn += (sst_dr[m]*n_x[m]);
			}
			const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
			const vrFloat M0 = (sst_drdn)*( (1.0-2.0*mu)*delta_ij + 3.0* sst_dr[idx_i] * sst_dr[idx_j]) - (1.0-2.0*mu)*(sst_dr[idx_i]*n_x[idx_j]-sst_dr[idx_j]*n_x[idx_i]);
			const vrFloat M1 = ((-1.0) / (8.0*numbers::MyPI*(1-mu))) * M0 * jacob_eta;
			const vrFloat F_1_ij_I = (M1*N_I_0)/(A*A);
#endif
			for (int index_rho=0;index_rho<nGaussPointSize_eta_In_Rho;++index_rho,++idx_g)
			{
				MyNotice;/*the order of theta and rho,(theta,rho)*/
				auto curRows = refDataSST3D.m_gaussQuadrature_xi_eta_polar.row(idx_g);
				MyVec2ParamSpace theta_rho;
				theta_rho[TriangleElem::idx_theta_doubleLayer] = curRows[TriangleElem::idx_theta_doubleLayer];
				theta_rho[TriangleElem::idx_rho_doubleLayer]=curRows[TriangleElem::idx_rho_doubleLayer];
				const vrFloat cur_rho = theta_rho[TriangleElem::idx_rho_doubleLayer];
				const vrFloat curWeight_doubleLayer = curRows[TriangleElem::idx_weight_doubleLayer];
				Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_theta_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_eta_polar.row(idx_g).x()));
				Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_rho_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_eta_polar.row(idx_g).y()));

				Q_ASSERT(numbers::isEqual(cur_theta_singlelayer,theta_rho[TriangleElem::idx_theta_doubleLayer]));

				const MyVec2ParamSpace currentSrcPtInParam/* in eta space*/ = refDataSST3D.m_SrcPt_in_eta;
				Q_ASSERT( ((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) /*compute_T_ij_I_SST*/);
				if (!((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))))
				{
					printf("compute_T_ij_I_SST : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
				}
				const MyVec2ParamSpace cur_eta = refDataSST3D.pc2eta(currentSrcPtInParam /*MyVec2ParamSpace(0.0,0.0)*/ MYNOTICE,theta_rho);
				const MyVec2ParamSpace cur_xi = refDataSST3D.eta2xi(cur_eta);
				const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

				MyFloat jacob_eta_nouse;/*jacob_eta = jacob_xi * mat_T_Inv*/
				MyVec3 normals_fieldpoint;
				MyFloat r;
				MyVec3 dr;
				MyFloat drdn;
				getKernelParameters_3D_SST(srcPos,fieldPoint,refDataSST3D,jacob_eta_nouse,normals_fieldpoint,r,dr,drdn);

				Q_ASSERT(numbers::isEqual(jacob_eta, jacob_eta_nouse));

				//printf("[%d,%d] r(%f)  rho*A(%f) \n",index_theta,index_rho,r,cur_rho*A);
				Q_ASSERT(numbers::isEqual(r,cur_rho*A));

				const vrFloat Tij = get_Tij_SST_3D(idx_i, idx_j, r, dr,drdn,normals_fieldpoint);
				const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);

				const vrFloat SingularTerm_Tij_I = Tij * N_I * jacob_eta * theta_rho[TriangleElem::idx_rho_doubleLayer];


				const vrFloat SingularTerm_F_1_ij_I = (1.0/cur_rho)*(F_1_ij_I);

				doubleLayer_Term += (SingularTerm_Tij_I - SingularTerm_F_1_ij_I) * curWeight_doubleLayer;
			}

			const vrFloat beta = 1.0 / A;
			const vrFloat cur_Rho_hat = refDataSST3D.rho_hat(cur_theta_singlelayer);
			singleLayer_Term += F_1_ij_I * log( abs(cur_Rho_hat/beta) ) * curWeight_singleLayer;
		}

		return (doubleLayer_Term + singleLayer_Term);
#endif
	}

	vrFloat vrBEM3D::compute_U_ij_I(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,vrInt idx_i,vrInt idx_j,vrInt idx_I)
	{
		vrFloat retVal = 0.0;
		const MyVec3 srcPos = curSourcePtr->getPos();
		const bool isDisContinuousVtx = curSourcePtr->isDisContinuousVertex();

#if SPEEDUP_5_31
		int tmp_GaussPointSize_xi_In_Theta = 0;
		int tmp_GaussPointSize_xi_In_Rho = 0;
		if (dis_regular == refDataSST3D.m_DisContinuousType)
		{
			tmp_GaussPointSize_xi_In_Theta = TriangleElem::GaussPointSize_xi_In_Theta;
			tmp_GaussPointSize_xi_In_Rho = TriangleElem::GaussPointSize_xi_In_Rho;
		}
		else
		{
			tmp_GaussPointSize_xi_In_Theta = TriangleElem::GaussPointSize_xi_In_Theta_DisContinuous;
			tmp_GaussPointSize_xi_In_Rho = TriangleElem::GaussPointSize_xi_In_Rho_DisContinuous;
		}
		const vrInt nGaussPointSize_xi_In_Theta = tmp_GaussPointSize_xi_In_Theta;
		const vrInt nGaussPointSize_xi_In_Rho = tmp_GaussPointSize_xi_In_Rho;
#endif
		const vrInt n_gpts = nGaussPointSize_xi_In_Theta * nGaussPointSize_xi_In_Rho;
		Q_ASSERT(n_gpts == refDataSST3D.m_gaussQuadrature_xi_polar.rows());
		for (int idx_g=0;idx_g < n_gpts;++idx_g)
		{
			MyNotice/*the order of theta and rho,(theta,rho)*/
				auto curRows = refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g);
			MyVec2ParamSpace theta_rho;
			theta_rho[TriangleElem::idx_theta_doubleLayer] = curRows[TriangleElem::idx_theta_doubleLayer];
			theta_rho[TriangleElem::idx_rho_doubleLayer]=curRows[TriangleElem::idx_rho_doubleLayer];
			const vrFloat curWeight = curRows[TriangleElem::idx_weight_doubleLayer];
			/*std::cout << theta_rho << std::endl;
			std::cout << refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g) << std::endl;*/
			Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_theta_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g).x()));
			Q_ASSERT(numbers::isEqual(theta_rho[TriangleElem::idx_rho_doubleLayer], refDataSST3D.m_gaussQuadrature_xi_polar.row(idx_g).y()));


			const MyVec2ParamSpace currentSrcPtInParam /*in xi space*/ = refDataSST3D.m_SrcPt_in_xi;
			Q_ASSERT( ((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) /*compute_U_ij_I*/ );
			if (!((numbers::isEqual(currentSrcPtInParam[0],0.0)) && (numbers::isEqual(currentSrcPtInParam[1],0.0))) )
			{
				printf("compute_U_ij_I : currentSrcPtInParam(%f, %f)\n",currentSrcPtInParam[0], currentSrcPtInParam[1]);
			}
			const MyVec2ParamSpace cur_xi = refDataSST3D.pc2xi(currentSrcPtInParam /*MyVec2ParamSpace(0.0,0.0)*/ MYNOTICE, theta_rho);
			const MyVec3 fieldPoint = refDataSST3D.xi2global(cur_xi);

			MyFloat jacob_xi;
			MyVec3 normals_fieldpoint;
			MyFloat r;
			MyVec3 dr;
			MyFloat drdn;
			getKernelParameters_3D(srcPos,fieldPoint,refDataSST3D,jacob_xi,normals_fieldpoint,r,dr,drdn);
			/*printf("r %f, rho %f\n",r,theta_rho[TriangleElemData::idx_rho]*refDataSST3D.Jacobi_xi);
			printf("r %f, rho %f\n",r*refDataSST3D.Jacobi_xi,theta_rho[TriangleElemData::idx_rho]);*/
			MyNotice;
			//Q_ASSERT(numbers::isEqual(r,theta_rho[TriangleElemData::idx_rho]));
			const vrFloat Uij = get_Uij_SST_3D(idx_i, idx_j, r, dr);

			const vrFloat N_I = refDataSST3D.shapefunction_xi(idx_I,cur_xi);

			retVal += Uij * N_I * jacob_xi * theta_rho[TriangleElem::idx_rho_doubleLayer] * curWeight;
		}


		return retVal;
	}

	vrFloat vrBEM3D::get_Kij_SST_3D_k(vrInt idx_i,vrInt idx_j,vrInt idx_k, vrFloat r,const MyVector& dr, const vrFloat drdn, const MyVec3& unitNormal_fieldPoint, const MyVec3& unitNormal_srcPt)
	{
		vrFloat retVal = 0.0;
		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
		//for (int idx_k=0;idx_k<MyDim;++idx_k)
		{
			const vrFloat delta_jk = TriangleElemData::delta_ij(idx_j,idx_k);
			const vrFloat delta_ik = TriangleElemData::delta_ij(idx_i,idx_k);
			retVal += ((1.0) / (8.0*numbers::MyPI*(1-mu)*r*r)) * 
				(
				(1.0-2.0*mu)*(delta_ij*dr[idx_k]+delta_jk*dr[idx_i]/*mynotice*/-delta_ik*dr[idx_j]) + 
				3.0*dr[idx_i]*dr[idx_j]*dr[idx_k]
			)*
				unitNormal_srcPt[idx_k];
		}

		return retVal;
	}

	vrFloat vrBEM3D::get_Hij_SST_3D_k(vrInt idx_i,vrInt idx_j,vrInt idx_k,vrFloat r,const MyVector& dr, const vrFloat drdn, const MyVec3& unitNormal_fieldPoint, const MyVec3& unitNormal_srcPt)
	{
		return get_Sij_SST_3D_k( idx_i, idx_j, idx_k, r, dr, drdn, unitNormal_fieldPoint, unitNormal_srcPt);
	}

	vrFloat vrBEM3D::get_Sij_SST_3D_k(vrInt idx_i,vrInt idx_j,vrInt idx_k,vrFloat r,const MyVector& dr, const vrFloat drdn, const MyVec3& unitNormal_fieldPoint, const MyVec3& unitNormal_srcPt)
	{
		vrFloat retVal = 0.0;
		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);

		const MyVec3& n_x = unitNormal_fieldPoint;
		const MyVec3& n_s = unitNormal_srcPt;
		//for (int idx_k=0;idx_k<MyDim;++idx_k)
		{
			const vrFloat delta_jk = TriangleElemData::delta_ij(idx_j,idx_k);
			const vrFloat delta_ik = TriangleElemData::delta_ij(idx_i,idx_k);
			retVal += (shearMod / (4.0*numbers::MyPI*(1.0-mu)*r*r*r)) * 
				(
				3.0*drdn*( (1.0-2.0*mu)*delta_ik*dr[idx_j] + mu * (delta_ij*dr[idx_k] + delta_jk*dr[idx_i]) MyNotice - 5.0 * dr[idx_i] * dr[idx_j] * dr[idx_k]) + 
				3.0 * mu * (n_x[idx_i] * dr[idx_j] * dr[idx_k] + n_x[idx_k] * dr[idx_i] * dr[idx_j]) MyNotice - 
				(1.0 - 4.0*mu) * delta_ik * n_x[idx_j] + 
				(1.0 - 2.0*mu) * ( (3.0 * n_x[idx_j] * dr[idx_i] * dr[idx_k]) + delta_ij * n_x[idx_k] + delta_jk * n_x[idx_i])
				) * unitNormal_srcPt[idx_k];
		}
		return retVal;
	}

	vrFloat vrBEM3D::get_Tij_SST_3D(vrInt idx_i,vrInt idx_j,vrFloat r,const MyVector& dr, const vrFloat drdn, const MyVec3& normal_fieldPoint)
	{
		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
		return ((-1.0) / (8.0*numbers::MyPI*(1-mu)*r*r)) * 
			(
			(drdn*( (1.0-2.0*mu)*delta_ij+3.0*dr[idx_i]*dr[idx_j] )) -
			(1.0-2.0*mu)*(dr[idx_i]*normal_fieldPoint[idx_j]-dr[idx_j]*normal_fieldPoint[idx_i])
			);
	}

	vrFloat vrBEM3D::get_Uij_SST_3D(vrInt idx_i,vrInt idx_j,vrFloat r,const MyVector& dr)
	{
		const vrFloat delta_ij = TriangleElemData::delta_ij(idx_i,idx_j);
		return ((1.0) / (16.0 * numbers::MyPI * shearMod*(1-mu)*r)) * 	( (3.0-4.0*mu)*delta_ij + dr[idx_i] * dr[idx_j] );
	}
#if USE_NEW_DUAL_6_3
	void vrBEM3D::AssembleSystem_DisContinuous_DualEquation_New(const vrInt v, const vrInt idx_i, const vrInt idx_j)
	{
#define DualEquation_New_RegularSection (1)
#define DualEquation_New_PositiveSection (1)
#define DualEquation_New_NegativeSection (1)
		const MyInt ne = TriangleElem::getTriangleSize();		
		VertexPtr curSourcePtr = Vertex::getVertex(v);	
		const vrMat3& CMatrix = curSourcePtr->getCMatrix();
		const MyVec3I& srcDofs = curSourcePtr->getDofs();
		const bool isDisContinuousVtx = curSourcePtr->isDisContinuousVertex();

		const VertexTypeInDual srcPtType = curSourcePtr->getVertexTypeInDual();//Regular=0,Mirror_Positive=1, Mirror_Negative=2,CrackTip=3

		if (VertexTypeInDual::Regular == srcPtType)
		{
#if DualEquation_New_RegularSection
			const vrFloat Cij = CMatrix.coeff(idx_i,idx_j);
			vrFloat Tij_I,Uij_I;
			for (int idx_e=0;idx_e<ne;++idx_e)
			{
				TriangleElemPtr curTriElemPtr =  TriangleElem::getTriangle(idx_e);
				const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();//dis_1_1=1, dis_1_2=2, dis_1_3=3, dis_2_3=4, dis_2_2=5, dis_2_1=6, dis_3_3=7, dis_regular=8

				vrInt srcPtIdx;
				bool ptInElem = isVertexInElement(curSourcePtr, curTriElemPtr, srcPtIdx);
				MyVec3I srcPt_SST_LookUp;			
				TriangleElemData data4SST_with_DisContinuous;
				if (ptInElem)
				{
					const DisContinuousType tmpTriElemDisContinuousType = TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,srcPtIdx);
					MyVec3 vtx_globalCoord[Geometry::vertexs_per_tri];

					for (int v=0;v<Geometry::vertexs_per_tri;++v)
					{
						srcPt_SST_LookUp[v] = srcPtIdx;
						vtx_globalCoord[v] = curTriElemPtr->getVertex(srcPtIdx)->getPos();
						srcPtIdx = (srcPtIdx+1) % Geometry::vertexs_per_tri;
					}
					data4SST_with_DisContinuous.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord);
				}	

				for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				{
					if (ptInElem)
					{
						//source point in triangle element.
						if (isDisContinuousVtx)
						{
							Q_ASSERT( (DisContinuousType::dis_regular != curTriElemDisContinuousType) );

							Tij_I = 0.0;
							Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal(0, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
							Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal(1, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);
							Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal(2, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;

							/////////////////////////////////////////////////////////////////////////////////////////////////////
							Uij_I = compute_U_ij_I( curSourcePtr, curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
							m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
							
						}
						else
						{
							Tij_I = compute_T_ij_I_SST(curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;

							/////////////////////////////////////////////////////////////////////////////////////////////////////
							Uij_I = compute_U_ij_I(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);							
							m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
							
						}
					}
					else
					{
						//source point do not locate in triangle element.
						Tij_I = compute_T_ij_I(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;

						/////////////////////////////////////////////////////////////////////////////////////////////////////
						Uij_I = compute_U_ij_I(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);							
						m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
					}
				}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)

			}//for (int idx_e=0;idx_e<ne;++idx_e)

			m_Hsubmatrix.coeffRef(srcDofs[idx_i],srcDofs[idx_j]) += Cij;
#endif//DualEquation_New_RegularSection
		}
		else if (VertexTypeInDual::Mirror_Positive == srcPtType)
		{
#if DualEquation_New_PositiveSection
			Q_ASSERT(curSourcePtr->isMirrorVertex());
			VertexPtr curSourcePtr_Positive_Ptr = curSourcePtr;
			VertexPtr curSource_Mirror_Negative_Ptr = curSourcePtr->getMirrorVertex();
			const vrMat3& CMatrix_Mirror_Negative = curSource_Mirror_Negative_Ptr->getCMatrix();

			Q_ASSERT(isDisContinuousVtx);
			Q_ASSERT(1 == (curSourcePtr->getShareElement().size()));
			const vrInt Se0_Plus_Id = (*(curSourcePtr->getShareElement()).begin())->getID();
			Q_ASSERT(1 == (curSource_Mirror_Negative_Ptr->getShareElement().size()));
			const vrInt Se0_Minus_Id = (*(curSource_Mirror_Negative_Ptr->getShareElement()).begin())->getID();

			const MyVec3I& src_Mirror_Negative_Dofs = curSource_Mirror_Negative_Ptr->getDofs();
			const vrFloat Cij_Positive = CMatrix.coeff(idx_i,idx_j);
			const vrFloat Cij_Mirror_Negative = CMatrix_Mirror_Negative.coeff(idx_i,idx_j);

			vrFloat Tij_I,Uij_I;
			for (int idx_e=0;idx_e<ne;++idx_e)
			{
				TriangleElemPtr curTriElemPtr =  TriangleElem::getTriangle(idx_e);
				const TriangleSetType curTriSetType = curTriElemPtr->getTriSetType();//Regular = 0, Positive = 1, Negative

				const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();


				if (Se0_Plus_Id != idx_e && Se0_Minus_Id != idx_e)
				{
					vrInt srcPtIdx;
					Q_ASSERT(!isVertexInElement(curSourcePtr_Positive_Ptr, curTriElemPtr, srcPtIdx));

					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						Tij_I = compute_T_ij_I(curSourcePtr_Positive_Ptr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;

						/////////////////////////////////////////////////////////////////////////////////////////////////////
						Uij_I = compute_U_ij_I(curSourcePtr_Positive_Ptr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);							
						m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;

						//printf("Tij_I [%f]   Uij_I[%f]\n",Tij_I, Uij_I);
					}
				}
				else if (Se0_Plus_Id == idx_e)
				{
					Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
					vrInt srcPtIdx;					
					bool ptInElem = isVertexInElement(curSourcePtr_Positive_Ptr, curTriElemPtr, srcPtIdx);
					Q_ASSERT(ptInElem);

					MyVec3I srcPt_SST_LookUp;			
					TriangleElemData data4SST_with_DisContinuous;
					if (ptInElem)
					{
						Q_ASSERT(Positive == curTriSetType);
						Q_ASSERT(isDisContinuousVtx);
						const DisContinuousType tmpTriElemDisContinuousType = TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,srcPtIdx);

						MyVec3 vtx_globalCoord[Geometry::vertexs_per_tri];

						for (int v=0;v<Geometry::vertexs_per_tri;++v)
						{
							srcPt_SST_LookUp[v] = srcPtIdx;
							vtx_globalCoord[v] = curTriElemPtr->getVertex(srcPtIdx)->getPos();
							srcPtIdx = (srcPtIdx+1) % Geometry::vertexs_per_tri;
						}
						data4SST_with_DisContinuous.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord);
					}	

					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						Tij_I = 0.0;									
						Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal(0, curSourcePtr_Positive_Ptr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);
						Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal(1, curSourcePtr_Positive_Ptr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
						Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal(2, curSourcePtr_Positive_Ptr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;

						//printf("Tij_I [%f]\n",Tij_I);
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else if (Se0_Minus_Id == idx_e)
				{
					Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
					vrInt src_Mirror_Negative_PtIdx;
					bool pt_Mirror_Negative_InElem = isVertexInElement(curSource_Mirror_Negative_Ptr, curTriElemPtr, src_Mirror_Negative_PtIdx);
					Q_ASSERT(pt_Mirror_Negative_InElem);
					const bool isDisContinuousVtx_Mirror_Negative = curSource_Mirror_Negative_Ptr->isDisContinuousVertex();
					Q_ASSERT(isDisContinuousVtx_Mirror_Negative);

					MyVec3 srcPt_Mirror_Negative_SST_LookUp;
					TriangleElemData data4SST_Mirror_Negative_DisContinuous;

					if (pt_Mirror_Negative_InElem)
					{
						const DisContinuousType tmpTriElemDisContinuousType = 
							TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,src_Mirror_Negative_PtIdx);
						MyVec3 vtx_globalCoord[Geometry::vertexs_per_tri];

						for (int v=0;v<Geometry::vertexs_per_tri;++v)
						{
							srcPt_Mirror_Negative_SST_LookUp[v] = src_Mirror_Negative_PtIdx;
							vtx_globalCoord[v] = curTriElemPtr->getVertex(src_Mirror_Negative_PtIdx)->getPos();
							src_Mirror_Negative_PtIdx = (src_Mirror_Negative_PtIdx + 1) % Geometry::vertexs_per_tri;
						}
						data4SST_Mirror_Negative_DisContinuous.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord);
					}

					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						Tij_I = 0.0;
						Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal(0, curSource_Mirror_Negative_Ptr,data4SST_Mirror_Negative_DisContinuous MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);	
						Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal(1, curSource_Mirror_Negative_Ptr,data4SST_Mirror_Negative_DisContinuous MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);
						Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal(2, curSource_Mirror_Negative_Ptr,data4SST_Mirror_Negative_DisContinuous MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);	
						m_Hsubmatrix.coeffRef(src_Mirror_Negative_Dofs[idx_i],curTriElemPtr->getVertex(srcPt_Mirror_Negative_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;

						//printf("Tij_I [%f]\n",Tij_I);
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else
				{
					MyError("impossible element id.");
				}
			}//for (int idx_e=0;idx_e<ne;++idx_e)

			m_Hsubmatrix.coeffRef(srcDofs[idx_i],srcDofs[idx_j]) += Cij_Positive;
			m_Hsubmatrix.coeffRef(src_Mirror_Negative_Dofs[idx_i],src_Mirror_Negative_Dofs[idx_j]) += Cij_Mirror_Negative;
#endif//DualEquation_New_PositiveSection
		}
		else if (VertexTypeInDual::Mirror_Negative == srcPtType)
		{
#if DualEquation_New_NegativeSection
			Q_ASSERT(curSourcePtr->isMirrorVertex());

			VertexPtr curSourcePtr_negative = curSourcePtr; MYNOTICE;//current source point locate on the negative element
			const MyVec3I& srcDofs_negative = curSourcePtr_negative->getDofs();
			VertexPtr curSource_Mirror_Ptr_Positive = curSourcePtr_negative->getMirrorVertex();
			const MyVec3I& src_Mirror_Dofs_Positive = curSource_Mirror_Ptr_Positive->getDofs();

			Q_ASSERT(isDisContinuousVtx);
			Q_ASSERT(1 == (curSourcePtr_negative->getShareElement().size()));
			const vrInt Se0_Minus_Id = (*(curSourcePtr_negative->getShareElement()).begin())->getID();
			Q_ASSERT(1 == (curSource_Mirror_Ptr_Positive->getShareElement().size()));
			const vrInt Se0_Miror_Plus_Id = (*(curSource_Mirror_Ptr_Positive->getShareElement()).begin())->getID();
			
			const vrMat3& CMatrix_Negative = curSourcePtr_negative->getCMatrix();
			const vrMat3& CMatrix_Mirror_Positive = curSource_Mirror_Ptr_Positive->getCMatrix();

			const vrFloat Cij_Mirror_Positive = CMatrix_Mirror_Positive.coeff(idx_i,idx_j);
			const vrFloat Cij_Negative = CMatrix_Negative.coeff(idx_i,idx_j);

			const bool isDisContinuousVtx_source_negative = curSourcePtr_negative->isDisContinuousVertex();
			vrFloat Kij_I,Sij_I;
			for (int idx_e=0;idx_e<ne;++idx_e)
			{
				TriangleElemPtr curTriElemPtr =  TriangleElem::getTriangle(idx_e);
				const TriangleSetType curTriSetType = curTriElemPtr->getTriSetType();//Regular = 0, Positive = 1, Negative

				const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();


				if (Se0_Miror_Plus_Id != idx_e && Se0_Minus_Id != idx_e)
				{
					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						Sij_I = compute_S_ij_I(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);									
						m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Sij_I;

						Kij_I = compute_K_ij_I(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
						m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Kij_I;

						//printf("Sij_I [%f]   Kij_I [%f]\n",Sij_I, Kij_I);
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else if (Se0_Minus_Id == idx_e)
				{
					Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
					vrInt srcPtIdx_negative;
					bool ptInElem_negative = isVertexInElement(curSourcePtr_negative, curTriElemPtr, srcPtIdx_negative);
					Q_ASSERT(ptInElem_negative);

					MyVec3 srcPt_SST_LookUp_negative;
					TriangleElemData data4SST_negative;

					if (ptInElem_negative)
					{
						Q_ASSERT(TriangleSetType::Negative == curTriSetType);
						Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
						const DisContinuousType tmpTriElemDisContinuousType = TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,srcPtIdx_negative);
						Q_ASSERT(dis_3_3 == tmpTriElemDisContinuousType);
						MyVec3 vtx_globalCoord_negative[Geometry::vertexs_per_tri];

						for (int v=0;v<Geometry::vertexs_per_tri;++v)
						{
							srcPt_SST_LookUp_negative[v] = srcPtIdx_negative;
							vtx_globalCoord_negative[v] = curTriElemPtr->getVertex(srcPtIdx_negative)->getPos();
							srcPtIdx_negative = (srcPtIdx_negative+1) %Geometry::vertexs_per_tri;
						}
						data4SST_negative.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord_negative);
					}

					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						Sij_I = 0.0;
						Sij_I += compute_S_ij_I_SST_DisContinuous_Sigmoidal(0, curSourcePtr_negative,data4SST_negative MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);	
						Sij_I += compute_S_ij_I_SST_DisContinuous_Sigmoidal(1, curSourcePtr_negative,data4SST_negative MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);
						Sij_I += compute_S_ij_I_SST_DisContinuous_Sigmoidal(2, curSourcePtr_negative,data4SST_negative MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);	
						m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_j)) += Sij_I;

						Kij_I = 0.0;
						Kij_I += compute_K_ij_I_SST_DisContinuous_Sigmoidal(0, curSourcePtr_negative,data4SST_negative MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);
						Kij_I += compute_K_ij_I_SST_DisContinuous_Sigmoidal(1, curSourcePtr_negative,data4SST_negative MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);
						Kij_I += compute_K_ij_I_SST_DisContinuous_Sigmoidal(2, curSourcePtr_negative,data4SST_negative MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);
						m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_j)) += Kij_I;

						//printf("Sij_I [%f]   Kij_I [%f]\n",Sij_I, Kij_I);
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else if (Se0_Miror_Plus_Id == idx_e)
				{
					const bool isDisContinuousVtx_source_Mirror_Positive = curSource_Mirror_Ptr_Positive->isDisContinuousVertex();
					Q_ASSERT(isDisContinuousVtx_source_Mirror_Positive);

					vrInt src_Mirror_PtIdx_Positive;
					bool pt_Mirror_InElem_Positive = isVertexInElement(curSource_Mirror_Ptr_Positive, curTriElemPtr, src_Mirror_PtIdx_Positive);
					Q_ASSERT(pt_Mirror_InElem_Positive);

					MyVec3 srcPt_Mirror_SST_LookUp_Positive;
					TriangleElemData data4SST_Mirror_Positive;

					if (pt_Mirror_InElem_Positive)
					{
						Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
						const DisContinuousType tmpTriElemDisContinuousType = TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,src_Mirror_PtIdx_Positive);
						MyVec3 vtx_globalCoord_Positive[Geometry::vertexs_per_tri];

						for (int v=0;v<Geometry::vertexs_per_tri;++v)
						{
							srcPt_Mirror_SST_LookUp_Positive[v] = src_Mirror_PtIdx_Positive;
							vtx_globalCoord_Positive[v] = curTriElemPtr->getVertex(src_Mirror_PtIdx_Positive)->getPos();
							src_Mirror_PtIdx_Positive = (src_Mirror_PtIdx_Positive + 1) % Geometry::vertexs_per_tri;
						}
						data4SST_Mirror_Positive.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord_Positive);
					}	

					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						Sij_I = 0.0;
						Sij_I += compute_S_ij_I_SST_DisContinuous_Sigmoidal(0, curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);	
						Sij_I += compute_S_ij_I_SST_DisContinuous_Sigmoidal(1, curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);
						Sij_I += compute_S_ij_I_SST_DisContinuous_Sigmoidal(2, curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);	

						m_Hsubmatrix.coeffRef(src_Mirror_Dofs_Positive[idx_i],curTriElemPtr->getVertex(srcPt_Mirror_SST_LookUp_Positive[idx_I])->getDof(idx_j)) += Sij_I;


						Kij_I = 0.0;
						Kij_I += compute_K_ij_I_SST_DisContinuous_Sigmoidal(0, curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);
						Kij_I += compute_K_ij_I_SST_DisContinuous_Sigmoidal(1, curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);
						Kij_I += compute_K_ij_I_SST_DisContinuous_Sigmoidal(2, curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);

						m_Gsubmatrix.coeffRef(src_Mirror_Dofs_Positive[idx_i],curTriElemPtr->getVertex(srcPt_Mirror_SST_LookUp_Positive[idx_I])->getDof(idx_j)) += Kij_I;

						//printf("Sij_I [%f]   Kij_I [%f]\n",Sij_I, Kij_I);
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else
				{
					MyError("impossible element id.");
				}
			}//for (int idx_e=0;idx_e<ne;++idx_e)

			/* G matrix */m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],srcDofs_negative[idx_j]) += Cij_Negative;
			/* G matrix */m_Gsubmatrix.coeffRef(src_Mirror_Dofs_Positive[idx_i],src_Mirror_Dofs_Positive[idx_j]) += 1.0 MyNoticeMsg("USE_MI_METHOD") *Cij_Mirror_Positive;
#endif//DualEquation_New_NegativeSection
		}
	}
#endif//USE_NEW_DUAL_6_3

	void vrBEM3D::AssembleSystem_DisContinuous_DualEquation(const vrInt v, const vrInt idx_i, const vrInt idx_j)
	{
		//printf("[%d] [%d] [%d]\n",v,idx_i,idx_j);
		const MyInt ne = TriangleElem::getTriangleSize();		
		VertexPtr curSourcePtr = Vertex::getVertex(v);		
		const vrMat3& CMatrix = curSourcePtr->getCMatrix();
		const MyVec3I& srcDofs = curSourcePtr->getDofs();
		const bool isDisContinuousVtx = curSourcePtr->isDisContinuousVertex();

		const VertexTypeInDual srcPtType = curSourcePtr->getVertexTypeInDual();//Regular=0,Mirror_Positive=1, Mirror_Negative=2,CrackTip=3

		if ( (VertexTypeInDual::Regular == srcPtType) )
		{
#if 1
			const vrFloat Cij = CMatrix.coeff(idx_i,idx_j);
			vrFloat Tij_I,Uij_I;
			for (int idx_e=0;idx_e<ne;++idx_e)
			{
				TriangleElemPtr curTriElemPtr =  TriangleElem::getTriangle(idx_e);
				const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();//dis_1_1=1, dis_1_2=2, dis_1_3=3, dis_2_3=4, dis_2_2=5, dis_2_1=6, dis_3_3=7, dis_regular=8

				vrInt srcPtIdx;
				bool ptInElem = isVertexInElement(curSourcePtr, curTriElemPtr, srcPtIdx);
				MyVec3I srcPt_SST_LookUp;			
				TriangleElemData data4SST_with_DisContinuous;
				if (ptInElem)
				{

					const DisContinuousType tmpTriElemDisContinuousType = TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,srcPtIdx);


					MyVec3 vtx_globalCoord[Geometry::vertexs_per_tri];

					for (int v=0;v<Geometry::vertexs_per_tri;++v)
					{
						srcPt_SST_LookUp[v] = srcPtIdx;
						vtx_globalCoord[v] = curTriElemPtr->getVertex(srcPtIdx)->getPos();
						srcPtIdx = (srcPtIdx+1) %Geometry::vertexs_per_tri;
					}
					data4SST_with_DisContinuous.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord);
				}	

				for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				{
					if (ptInElem)
					{
						//source point in triangle element.
						if (isDisContinuousVtx)
						{
							Q_ASSERT( (DisContinuousType::dis_regular != curTriElemDisContinuousType) );
							//							
#if USE_SUBTRI_INTE
							Tij_I = 0.0;
							Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal(0, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
							Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal(1, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);
							Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal(2, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
#else
							Tij_I = compute_T_ij_I_SST_DisContinuous(999, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
#endif	
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;

							if (isOutofRange(Tij_I))
							{
								printf("Tij_I [%f] isDisContinuousVtx [%d][%d][%d][%d]\n",Tij_I,999,idx_i,idx_j,idx_I);
								//compute_T_ij_I_SST_DisContinuous_nouse_debug(idx, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);
								vrPause;
							}
							/////////////////////////////////////////////////////////////////////////////////////////////////////
#if !USE_SUBTRI_INTE
							Uij_I = 0.0;
							Uij_I += compute_U_ij_I_DisContinuous(0, curSourcePtr, curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
							Uij_I += compute_U_ij_I_DisContinuous(1, curSourcePtr, curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
							Uij_I += compute_U_ij_I_DisContinuous(2, curSourcePtr, curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
#else
							Uij_I = compute_U_ij_I( curSourcePtr, curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
#endif					
							//Uij_I = compute_U_ij_I(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);							
							m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
							if (isOutofRange(Uij_I))
							{
								printf("Uij_I [%f]\n",Uij_I);vrPause;
							}
						}
						else
						{
							Tij_I = compute_T_ij_I_SST(curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;

							if (isOutofRange(Tij_I))
							{
								printf("Tij_I [%f]  ContinuousVtx\n",Tij_I);vrPause;
							}

							/////////////////////////////////////////////////////////////////////////////////////////////////////
							Uij_I = compute_U_ij_I(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);							
							m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
							if (isOutofRange(Uij_I))
							{
								printf("Uij_I [%f]\n",Uij_I);vrPause;
							}
						}
					}
					else
					{
						//source point do not locate in triangle element.
						Tij_I = compute_T_ij_I(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;

						/////////////////////////////////////////////////////////////////////////////////////////////////////
						Uij_I = compute_U_ij_I(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);							
						m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
						if (isOutofRange(Uij_I))
						{
							printf("Uij_I [%f]\n",Uij_I);vrPause;
						}
					}

				}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)

			}//for (int idx_e=0;idx_e<ne;++idx_e)

			m_Hsubmatrix.coeffRef(srcDofs[idx_i],srcDofs[idx_j]) += Cij;
#endif
		}//if ( (VertexTypeInDual::Regular == srcPtType) )
		else if (VertexTypeInDual::Mirror_Positive == srcPtType)
		{
#if 1
			Q_ASSERT(curSourcePtr->isMirrorVertex());
			VertexPtr curSourcePtr_Positive_Ptr = curSourcePtr;
			VertexPtr curSource_Mirror_Negative_Ptr = curSourcePtr_Positive_Ptr->getMirrorVertex();
			const vrMat3& CMatrix_Mirror_Negative = curSource_Mirror_Negative_Ptr->getCMatrix();

			const MyVec3I& src_Mirror_Negative_Dofs = curSource_Mirror_Negative_Ptr->getDofs();

			const vrFloat Cij_Positive = CMatrix.coeff(idx_i,idx_j);
			const vrFloat Cij_Mirror_Negative = CMatrix_Mirror_Negative.coeff(idx_i,idx_j);

			vrFloat Tij_I,Uij_I;
			for (int idx_e=0;idx_e<ne;++idx_e)
			{
				TriangleElemPtr curTriElemPtr =  TriangleElem::getTriangle(idx_e);
				const TriangleSetType curTriSetType = curTriElemPtr->getTriSetType();//Regular = 0, Positive = 1, Negative

				const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();
				vrInt srcPtIdx;
				bool ptInElem = isVertexInElement(curSourcePtr_Positive_Ptr, curTriElemPtr, srcPtIdx);
				MyVec3I srcPt_SST_LookUp;			
				TriangleElemData data4SST_with_DisContinuous;
				if (ptInElem)
				{
					Q_ASSERT(Positive == curTriSetType);
					Q_ASSERT(isDisContinuousVtx);
					const DisContinuousType tmpTriElemDisContinuousType = TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,srcPtIdx);

					MyVec3 vtx_globalCoord[Geometry::vertexs_per_tri];

					for (int v=0;v<Geometry::vertexs_per_tri;++v)
					{
						srcPt_SST_LookUp[v] = srcPtIdx;
						vtx_globalCoord[v] = curTriElemPtr->getVertex(srcPtIdx)->getPos();
						srcPtIdx = (srcPtIdx+1) %Geometry::vertexs_per_tri;
					}
					data4SST_with_DisContinuous.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord);
				}	

				if (TriangleSetType::Regular == curTriSetType)
				{
					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						if (ptInElem)
						{
							MyError("current source point is Mirror_Positive, current element is regular, so source point is impossible in element.");
						}
						else
						{
							Tij_I = compute_T_ij_I(curSourcePtr_Positive_Ptr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;

							/////////////////////////////////////////////////////////////////////////////////////////////////////
							Uij_I = compute_U_ij_I(curSourcePtr_Positive_Ptr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);							
							m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
							if (isOutofRange(Uij_I))
							{
								printf("Uij_I [%f]\n",Uij_I);vrPause;
							}
						}
					}
				}
				else if (TriangleSetType::Positive == curTriSetType)
				{
					Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						if (ptInElem)
						{
							//source point is singular point using SST method
							if (isDisContinuousVtx)
							{
								//
#if USE_SUBTRI_INTE
								Tij_I = 0.0;									
								Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal(0, curSourcePtr_Positive_Ptr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);
								Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal(1, curSourcePtr_Positive_Ptr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
								Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal(2, curSourcePtr_Positive_Ptr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
#else
								Tij_I = compute_T_ij_I_SST_DisContinuous(999, curSourcePtr_Positive_Ptr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);
#endif

								if (isOutofRange(Tij_I))
								{
									printf("Tij_I [%f] isDisContinuousVtx [%d][%d][%d][%d]\n",Tij_I,3,idx_i,idx_j,idx_I);
									//compute_T_ij_I_SST_DisContinuous_nouse_debug(idx, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);
									vrPause;
								}
								m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
								//printf("Tij_I [%f]\n",Tij_I);
								//Q_ASSERT(!isOutofRange(Tij_I));

								MyNotice;/*do not compute U .*/
							}
							else
							{
								MyError("current source point is Mirror_Positive, current element is Positive and source point is in this element, so source point must be discontinuous.");
							}
						} 
						else
						{
							//using standard gauss method
							Tij_I = compute_T_ij_I(curSourcePtr_Positive_Ptr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;
							MyNotice;/*do not compute U .*/
						}
					}
				}
				else if (TriangleSetType::Negative == curTriSetType)
				{
					Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
					/*
					VertexPtr curSource_Mirror_Negative_Ptr = curSourcePtr_Positive_Ptr->getMirrorVertex();
					const vrMat3& CMatrix_Mirror_Negative = curSource_Mirror_Negative_Ptr->getCMatrix();
					const MyVec3I& src_Mirror_Negative_Dofs = curSource_Mirror_Negative_Ptr->getDofs();
					*/
					vrInt src_Mirror_Negative_PtIdx;
					bool pt_Mirror_Negative_InElem = isVertexInElement(curSource_Mirror_Negative_Ptr, curTriElemPtr, src_Mirror_Negative_PtIdx);
					const bool isDisContinuousVtx_Mirror_Negative = curSource_Mirror_Negative_Ptr->isDisContinuousVertex();
					Q_ASSERT(isDisContinuousVtx_Mirror_Negative);

					MyVec3 srcPt_Mirror_Negative_SST_LookUp;
					TriangleElemData data4SST_Mirror_Negative_DisContinuous;

					if (pt_Mirror_Negative_InElem)
					{
						const DisContinuousType tmpTriElemDisContinuousType = 
							TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,src_Mirror_Negative_PtIdx);
						MyVec3 vtx_globalCoord[Geometry::vertexs_per_tri];

						for (int v=0;v<Geometry::vertexs_per_tri;++v)
						{
							srcPt_Mirror_Negative_SST_LookUp[v] = src_Mirror_Negative_PtIdx;
							vtx_globalCoord[v] = curTriElemPtr->getVertex(src_Mirror_Negative_PtIdx)->getPos();
							src_Mirror_Negative_PtIdx = (src_Mirror_Negative_PtIdx + 1) % Geometry::vertexs_per_tri;
						}
						data4SST_Mirror_Negative_DisContinuous.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord);
					}

					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						if (pt_Mirror_Negative_InElem)
						{
							if (isDisContinuousVtx_Mirror_Negative)
							{
#if USE_SUBTRI_INTE
								Tij_I = 0.0;
								Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal(0, curSource_Mirror_Negative_Ptr,data4SST_Mirror_Negative_DisContinuous MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);	
								Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal(1, curSource_Mirror_Negative_Ptr,data4SST_Mirror_Negative_DisContinuous MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);
								Tij_I += compute_T_ij_I_SST_DisContinuous_Sigmoidal(2, curSource_Mirror_Negative_Ptr,data4SST_Mirror_Negative_DisContinuous MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);	
#else
								Tij_I = compute_T_ij_I_SST_DisContinuous(999, curSource_Mirror_Negative_Ptr,data4SST_Mirror_Negative_DisContinuous MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);	
#endif

								if (isOutofRange(Tij_I))
								{
									printf("Tij_I [%f] isDisContinuousVtx [%d][%d][%d][%d]\n",Tij_I,4,idx_i,idx_j,idx_I);
									//compute_T_ij_I_SST_DisContinuous_nouse_debug(idx, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);
									vrPause;
								}
#if USE_MI_METHOD
								m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_Mirror_Negative_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
#else//USE_MI_METHOD   000000000000000000000000000000000000000
								m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_Mirror_Negative_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
#endif//USE_MI_METHOD
								

								MyNotice;/*do not compute U .*/
							}
							else
							{
								MyError("current mirror source point is Mirror_negative, current element is negative and mirror source point is in this negative element, so source point must be discontinuous.");
							}
						}
						else
						{
							//000000000000000000000000000Tij_I = compute_T_ij_I(curSource_Mirror_Negative_Ptr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
							Tij_I = compute_T_ij_I(curSourcePtr_Positive_Ptr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
#if USE_MI_METHOD
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;
#else//USE_MI_METHOD
							m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;
#endif//USE_MI_METHOD
							
							MyNotice;/*do not compute U .*/
						}
					}
				}
				else
				{
					MyError("Unsupport Triangle element type.");
				}

			}//for (int idx_e=0;idx_e<ne;++idx_e)

			m_Hsubmatrix.coeffRef(srcDofs[idx_i],srcDofs[idx_j]) += Cij_Positive;
#if USE_MI_METHOD
			const MyVec3I& src_Mirror_Negative_Dofs = curSource_Mirror_Negative_Ptr->getDofs();
			m_Hsubmatrix.coeffRef(srcDofs[idx_i],src_Mirror_Negative_Dofs[idx_j]) += Cij_Mirror_Negative;
#else//USE_MI_METHOD
			m_Hsubmatrix.coeffRef(srcDofs[idx_i],src_Mirror_Negative_Dofs[idx_j]) += Cij_Mirror_Negative;
#endif//USE_MI_METHOD
			
#endif
		}//if (VertexTypeInDual::Mirror_Positive == srcPtType)
		else if (VertexTypeInDual::Mirror_Negative == srcPtType)
		{
#if 1
			Q_ASSERT(curSourcePtr->isMirrorVertex());
			VertexPtr curSourcePtr_negative = curSourcePtr; MYNOTICE;//current source point locate on the negative element
			const MyVec3I& srcDofs_negative = curSourcePtr_negative->getDofs();

			VertexPtr curSource_Mirror_Ptr_Positive = curSourcePtr_negative->getMirrorVertex();
#if !USE_MI_METHOD
			const MyVec3I& src_Mirror_Dofs_Positive = curSource_Mirror_Ptr_Positive->getDofs();
#endif//USE_MI_METHOD
			

			const vrMat3& CMatrix_Negative = curSourcePtr_negative->getCMatrix();
			const vrMat3& CMatrix_Mirror_Positive = curSource_Mirror_Ptr_Positive->getCMatrix();

			const vrFloat Cij_Mirror_Positive = CMatrix_Mirror_Positive.coeff(idx_i,idx_j);
			const vrFloat Cij_Negative = CMatrix_Negative.coeff(idx_i,idx_j);

			const bool isDisContinuousVtx_source_negative = curSourcePtr_negative->isDisContinuousVertex();
			vrFloat Kij_I,Sij_I;

			for (int idx_e=0;idx_e<ne;++idx_e)
			{
				TriangleElemPtr curTriElemPtr =  TriangleElem::getTriangle(idx_e);						
				const TriangleSetType curTriSetType = curTriElemPtr->getTriSetType();//Regular = 0, Positive = 1, Negative

				const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();

				vrInt srcPtIdx_negative;
				bool ptInElem_negative = isVertexInElement(curSourcePtr_negative, curTriElemPtr, srcPtIdx_negative);

				MyVec3 srcPt_SST_LookUp_negative;
				TriangleElemData data4SST_negative;

				if (ptInElem_negative)
				{
					Q_ASSERT(TriangleSetType::Negative == curTriSetType);
					Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
					const DisContinuousType tmpTriElemDisContinuousType = TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,srcPtIdx_negative);
					Q_ASSERT(dis_3_3 == tmpTriElemDisContinuousType);
					MyVec3 vtx_globalCoord_negative[Geometry::vertexs_per_tri];

					for (int v=0;v<Geometry::vertexs_per_tri;++v)
					{
						srcPt_SST_LookUp_negative[v] = srcPtIdx_negative;
						vtx_globalCoord_negative[v] = curTriElemPtr->getVertex(srcPtIdx_negative)->getPos();
						srcPtIdx_negative = (srcPtIdx_negative+1) %Geometry::vertexs_per_tri;
					}
					data4SST_negative.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord_negative);
				}

				if (TriangleSetType::Regular == curTriSetType)
				{
					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						if (ptInElem_negative)
						{
							MyError("current source point is Mirror Negative, current element is regular, so source point is impossible in element.");
						}
						else
						{
							Sij_I = compute_S_ij_I(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);									
							m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Sij_I;

							Kij_I = compute_K_ij_I(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
							m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Kij_I;
						}
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else if (TriangleSetType::Positive == curTriSetType)
				{
					//VertexPtr curSource_Mirror_Ptr_Positive = curSourcePtr_negative->getMirrorVertex();
					//const MyVec3I& src_Mirror_Dofs_Positive = curSource_Mirror_Ptr_Positive->getDofs();
					const bool isDisContinuousVtx_source_Mirror_Positive = curSource_Mirror_Ptr_Positive->isDisContinuousVertex();


					vrInt src_Mirror_PtIdx_Positive;
					bool pt_Mirror_InElem_Positive = isVertexInElement(curSource_Mirror_Ptr_Positive, curTriElemPtr, src_Mirror_PtIdx_Positive);

					MyVec3 srcPt_Mirror_SST_LookUp_Positive;
					TriangleElemData data4SST_Mirror_Positive;

					if (pt_Mirror_InElem_Positive)
					{
						Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
						const DisContinuousType tmpTriElemDisContinuousType = TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,src_Mirror_PtIdx_Positive);
						MyVec3 vtx_globalCoord_Positive[Geometry::vertexs_per_tri];

						for (int v=0;v<Geometry::vertexs_per_tri;++v)
						{
							srcPt_Mirror_SST_LookUp_Positive[v] = src_Mirror_PtIdx_Positive;
							vtx_globalCoord_Positive[v] = curTriElemPtr->getVertex(src_Mirror_PtIdx_Positive)->getPos();
							src_Mirror_PtIdx_Positive = (src_Mirror_PtIdx_Positive + 1) %Geometry::vertexs_per_tri;
						}
						data4SST_Mirror_Positive.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord_Positive);
					}	

					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						if (pt_Mirror_InElem_Positive)
						{	
							if (isDisContinuousVtx_source_Mirror_Positive)
							{
#if USE_SUBTRI_INTE
								Sij_I = 0.0;
								Sij_I += compute_S_ij_I_SST_DisContinuous_Sigmoidal(0, curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);	
								Sij_I += compute_S_ij_I_SST_DisContinuous_Sigmoidal(1, curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);
								Sij_I += compute_S_ij_I_SST_DisContinuous_Sigmoidal(2, curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);	
#else
								Sij_I = compute_S_ij_I_SST_DisContinuous(999, curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);	
#endif

								if (isOutofRange(Sij_I))
								{
									printf("Sij_I [%f] isDisContinuousVtx [%d][%d][%d][%d]\n",Sij_I,4,idx_i,idx_j,idx_I);
									//compute_T_ij_I_SST_DisContinuous_nouse_debug(idx, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);
									vrPause;
								}
#if USE_MI_METHOD
								m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_Mirror_SST_LookUp_Positive[idx_I])->getDof(idx_j)) += Sij_I;
#else//USE_MI_METHOD  000000000000000000000
								m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_Mirror_SST_LookUp_Positive[idx_I])->getDof(idx_j)) += Sij_I;
#endif//USE_MI_METHOD
								
								
								Kij_I = 0.0;
								Kij_I += compute_K_ij_I_SST_DisContinuous_Sigmoidal(0, curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);
								Kij_I += compute_K_ij_I_SST_DisContinuous_Sigmoidal(1, curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);
								Kij_I += compute_K_ij_I_SST_DisContinuous_Sigmoidal(2, curSource_Mirror_Ptr_Positive,data4SST_Mirror_Positive MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);
								
#if USE_MI_METHOD
								m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_Mirror_SST_LookUp_Positive[idx_I])->getDof(idx_j)) += Kij_I;
#else//USE_MI_METHOD   000000000000000000000000
								m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_Mirror_SST_LookUp_Positive[idx_I])->getDof(idx_j)) += Kij_I;
#endif//USE_MI_METHOD
								
							}
							else
							{
								MyError("current mirror source point is on the positive surface element, which is impossible be a continuous point.");
							}
						}
						else
						{
							//1. source point is on the negative crack. and source MIRROR point is on the positive crack.
							//2. triangle element is positive crack.
							//3. source MIRROR point is not in the triangle element.

							//using standard gauss method

							//00000000000000000000000000000000000000000000000000000000
							Sij_I = compute_S_ij_I(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
#if USE_MI_METHOD
							m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Sij_I;
#else//USE_MI_METHOD   0000000000000000000000000000000000
							m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Sij_I;
#endif//USE_MI_METHOD
							
							//000000000000000000000000000000000000000000000
							Kij_I = compute_K_ij_I(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
#if USE_MI_METHOD
							m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Kij_I;
#else//USE_MI_METHOD   0000000000000000000000000000000000
							m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Kij_I;
#endif//USE_MI_METHOD				
							
						}
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else if (TriangleSetType::Negative == curTriSetType)
				{
					Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);
					for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
					{
						if (ptInElem_negative)
						{
							if (isDisContinuousVtx_source_negative)
							{
#if USE_SUBTRI_INTE
								Sij_I = 0.0;
								Sij_I += compute_S_ij_I_SST_DisContinuous_Sigmoidal(0, curSourcePtr_negative,data4SST_negative MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);	
								Sij_I += compute_S_ij_I_SST_DisContinuous_Sigmoidal(1, curSourcePtr_negative,data4SST_negative MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);
								Sij_I += compute_S_ij_I_SST_DisContinuous_Sigmoidal(2, curSourcePtr_negative,data4SST_negative MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);	
#else
								Sij_I = compute_S_ij_I_SST_DisContinuous(999, curSourcePtr_negative,data4SST_negative MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);	
#endif

								if (isOutofRange(Sij_I))
								{
									printf("Sij_I [%f] isDisContinuousVtx [%d][%d][%d][%d]\n",Sij_I,4,idx_i,idx_j,idx_I);
									//compute_T_ij_I_SST_DisContinuous_nouse_debug(idx, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);
									vrPause;
								}
								m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_j)) += Sij_I;

								Kij_I = 0.0;
								Kij_I += compute_K_ij_I_SST_DisContinuous_Sigmoidal(0, curSourcePtr_negative,data4SST_negative MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);
								Kij_I += compute_K_ij_I_SST_DisContinuous_Sigmoidal(1, curSourcePtr_negative,data4SST_negative MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);
								Kij_I += compute_K_ij_I_SST_DisContinuous_Sigmoidal(2, curSourcePtr_negative,data4SST_negative MYNOTICE /*Mirror*/,idx_i,idx_j,idx_I);
								m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp_negative[idx_I])->getDof(idx_j)) += Kij_I;
								
							} 
							else
							{
								MyError("current source point is on the negative, curent element is also on the negative, when source point is in the element, the source point must be discontinuous.");
							}
						}
						else
						{
							//1. source point is on the negative crack.
							//2. triangle element is negative crack.
							//3. source point is not in the triangle element.

							//using standard gauss method
							Sij_I = compute_S_ij_I(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);									
							m_Hsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Sij_I;

							Kij_I = compute_K_ij_I(curSourcePtr_negative,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
							m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Kij_I;
						}
					}//for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
				}
				else
				{
					MyError("Unsupport Triangle element type.");
				}

			}//for (int idx_e=0;idx_e<ne;++idx_e)

			/* G matrix */m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],srcDofs_negative[idx_j]) += Cij_Negative;
#if USE_MI_METHOD
			const MyVec3I& src_Mirror_Dofs_Positive = curSource_Mirror_Ptr_Positive->getDofs();
			/* G matrix */m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],src_Mirror_Dofs_Positive[idx_j]) += -1.0 MyNoticeMsg("USE_MI_METHOD") *Cij_Mirror_Positive;
#else//USE_MI_METHOD   000000000000000000000000000000
			/* G matrix */m_Gsubmatrix.coeffRef(srcDofs_negative[idx_i],src_Mirror_Dofs_Positive[idx_j]) += -1.0 MyNoticeMsg("USE_MI_METHOD") *Cij_Mirror_Positive;
#endif//USE_MI_METHOD
			
#endif
		}//if (VertexTypeInDual::Mirror_Negative == srcPtType)
		else
		{
			MyError("Unsupport source point type. Maybe is CrackTip.");
		}
	}

	void vrBEM3D::AssembleSystem_DisContinuous(const vrInt v, const vrInt idx_i, const vrInt idx_j)
	{
#if 0
		
		printf("[%d] [%d] [%d]\n",v,idx_i,idx_j);
		const MyInt ne = TriangleElem::getTriangleSize();		
		VertexPtr curSourcePtr = Vertex::getVertex(v);		
		const vrMat3& CMatrix = curSourcePtr->getCMatrix();
		const MyVec3I& srcDofs = curSourcePtr->getDofs();
		const bool isDisContinuousVtx = curSourcePtr->isDisContinuousVertex();
		//		Q_ASSERT(isDisContinuousVtx);

		MyNotice;//const MyVec2ParamSpace xi_s(0.0,0.0);
		//1.compute free term
		const vrFloat Cij = CMatrix.coeff(idx_i,idx_j);
		/*if (idx_i == idx_j)
		{
		Q_ASSERT(numbers::isEqual(0.5,Cij));
		}
		else
		{
		Q_ASSERT(numbers::isEqual(0.0,Cij));
		}*/


		//2.compute Tij_I
		vrFloat Tij_I,Uij_I;
		for (int idx_e=0;idx_e<ne;++idx_e)
		{
			TriangleElemPtr curTriElemPtr =  TriangleElem::getTriangle(idx_e);
			const DisContinuousType curTriElemDisContinuousType = curTriElemPtr->getTriContinuousType();
			//			Q_ASSERT(dis_3_3 == curTriElemDisContinuousType);

			vrInt srcPtIdx;
			bool ptInElem = isVertexInElement(curSourcePtr, curTriElemPtr, srcPtIdx);
			MyVec3I srcPt_SST_LookUp;			
			TriangleElemData data4SST_with_DisContinuous;
			if (ptInElem)
			{
				const DisContinuousType tmpTriElemDisContinuousType = TriangleElemData::computeTmpDisContinuousTypePlus(curTriElemDisContinuousType,srcPtIdx);
				//				Q_ASSERT(dis_3_3 == tmpTriElemDisContinuousType);
				MyVec3 vtx_globalCoord[Geometry::vertexs_per_tri];

				for (int v=0;v<Geometry::vertexs_per_tri;++v)
				{
					srcPt_SST_LookUp[v] = srcPtIdx;
					vtx_globalCoord[v] = curTriElemPtr->getVertex(srcPtIdx)->getPos();
					srcPtIdx = (srcPtIdx+1) %Geometry::vertexs_per_tri;
				}
				data4SST_with_DisContinuous.compute_Shape_Deris_Jacobi_SST_3D(tmpTriElemDisContinuousType, vtx_globalCoord);
				//data4SST_with_DisContinuous.compute_Shape_Deris_Jacobi_SST_3D_SubTri(tmpTriElemDisContinuousType, vtx_globalCoord);

			}			

			for (int idx_I=0;idx_I<Geometry::vertexs_per_tri;++idx_I)
			{
				if (ptInElem)
				{
					if (isDisContinuousVtx)
					{
						Tij_I = 0.0;
						int idx=0;
						Tij_I += compute_T_ij_I_SST_DisContinuous(idx, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	

						Tij_I += compute_T_ij_I_SST_DisContinuous(++idx, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	

						Tij_I += compute_T_ij_I_SST_DisContinuous(++idx, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
						if (isOutofRange(Tij_I))
						{
							printf("Tij_I [%f] isDisContinuousVtx [%d][%d][%d][%d]\n",Tij_I,idx,idx_i,idx_j,idx_I);
							//compute_T_ij_I_SST_DisContinuous_nouse_debug(idx, curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);
							vrPause;
						}
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
						//printf("Tij_I [%f]\n",Tij_I);
						//Q_ASSERT(!isOutofRange(Tij_I));

						/////////////////////////////////////////////////////////////////////////////////////////////////////
						Uij_I = compute_U_ij_I(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);							
						m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
						if (isOutofRange(Uij_I))
						{
							printf("Uij_I [%f]\n",Uij_I);vrPause;
						}
					}
					else
					{
						//source point is singular point using SST method
						//MyError("Tij_I = compute_T_ij_I_SST.");
						Tij_I = compute_T_ij_I_SST(curSourcePtr,data4SST_with_DisContinuous,idx_i,idx_j,idx_I);	
						m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(srcPt_SST_LookUp[idx_I])->getDof(idx_j)) += Tij_I;
						/*printf("Tij_I [%f]\n",Tij_I);
						Q_ASSERT(!isOutofRange(Tij_I));*/
						//Q_ASSERT(!isOutofRange(Tij_I));
						if (isOutofRange(Tij_I))
						{
							printf("Tij_I [%f]  ContinuousVtx\n",Tij_I);vrPause;
						}

						/////////////////////////////////////////////////////////////////////////////////////////////////////
						Uij_I = compute_U_ij_I(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);							
						m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
						if (isOutofRange(Uij_I))
						{
							printf("Uij_I [%f]\n",Uij_I);vrPause;
						}
					}
				} 
				else
				{
					//using standard gauss method
					Tij_I = compute_T_ij_I(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);
					m_Hsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Tij_I;

					/////////////////////////////////////////////////////////////////////////////////////////////////////
					Uij_I = compute_U_ij_I(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);							
					m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
					if (isOutofRange(Uij_I))
					{
						printf("Uij_I [%f]\n",Uij_I);vrPause;
					}
				}


				/*Uij_I = compute_U_ij_I(curSourcePtr,curTriElemPtr->m_data_SST_3D,idx_i,idx_j,idx_I);							
				m_Gsubmatrix.coeffRef(srcDofs[idx_i],curTriElemPtr->getVertex(idx_I)->getDof(idx_j)) += Uij_I;
				if (isOutofRange(Uij_I))
				{
				printf("Uij_I [%f]\n",Uij_I);vrPause;
				}*/
			}

		}

		m_Hsubmatrix.coeffRef(srcDofs[idx_i],srcDofs[idx_j]) += Cij;
#endif
	}


	void vrBEM3D::AssembleSystem(const vrInt v)
	{
#if !USE_DIS

		const MyInt ne = TriangleElem::getTriangleSize();
		infoLog << "vertex " << v << std::endl;
		VertexPtr curSourcePtr = Vertex::getVertex(v);
		//infoLog << "source point : " << curSourcePtr->getPos().transpose() << std::endl;
		//const vrMat3& CMatrix = curSourcePtr->getCMatrix();
		//const MyVec3I& srcDofs = curSourcePtr->getDofs();

		if (curSourcePtr->isContinuousVertex())
		{
			for (int idx_i=0;idx_i < MyDim;++idx_i)
			{
				for (int idx_j=0;idx_j < MyDim;++idx_j)
				{
					AssembleSystem(v,idx_i,idx_j);
				}
			}
		}
		else
		{
			for (int idx_i=0;idx_i < MyDim;++idx_i)
			{
				for (int idx_j=0;idx_j < MyDim;++idx_j)
				{
					AssembleSystem_DisContinuous(v,idx_i,idx_j);
				}
			}
		}

		return ;
		//MyMatrix mat_FreeTerm_3x3;mat_FreeTerm_3x3.resize(Geometry::vertexs_per_tri,MyDim);mat_FreeTerm_3x3.setZero();
		for (int idx_i=0;idx_i < MyDim;++idx_i)
		{
			for (int idx_j=0;idx_j < MyDim;++idx_j)
			{
				AssembleSystem(v,idx_i,idx_j);
			}
		}
#endif
	}
#endif//USE_Aliabadi

#endif//USE_Fracture
}//namespace VR