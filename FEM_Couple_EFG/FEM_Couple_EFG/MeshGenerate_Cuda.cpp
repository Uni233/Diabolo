#include "stdafx.h"
#include "MeshGenerate.h"

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <SOIL.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "CellStructOnCuda.h"
#include <queue>

extern int getCurrentGPUMemoryInfo();
extern bool showTex;

extern float rotate_x;
extern float rotate_y;
extern float translate_z;
extern float scaled;

typedef float EigenValueType;
typedef EigenValueType * EigenValueTypePtr;
typedef float ValueType;
typedef ValueType * ValueTypePtr;
typedef int    IndexType;
typedef IndexType * IndexTypePtr;

void do_loop(int nCount,float4* pos_Lines, uchar4 *colorPos, float4* pos_Triangles, float3* vertexNormal);
extern void initBoundaryCondition();
void initVBODataStruct(IndexTypePtr line_vertex_pair,IndexType lineCount,IndexTypePtr vertex_dofs,ValueTypePtr vertex_pos,IndexType vertexCount);
void initVBODataStruct_triangleMesh(IndexType nTriangleMeshVertexCount,
									IndexType nTriangleMeshVertexNormalCount,
									IndexType nTriangleMeshFaceCount,
									IndexType nTriangleMeshVNCount,
									ValueTypePtr elemVertexPos,IndexType nVertexPosLen,
									ValueTypePtr elemVertexTriLinearWeight,IndexType nVertexTriLinearWeightLen,
									IndexTypePtr elemVertexRelatedDofs,IndexType nVertexRelatedDofsLen,
									ValueTypePtr elemVertexNormal,IndexType nVertexNormalLen,
									IndexTypePtr elemFaceVertexIdx,IndexType nFaceLen,
									IndexTypePtr elemVertexNormalIdx,IndexType nVertexNormalIdxLen);
extern void initial_Cusp(const IndexType nDofs,

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

	VR_FEM::MyFloat dbNewMarkConstant[8],
	IndexTypePtr boundaryconditionPtr,
	IndexType bcCount,
	IndexTypePtr forceConditionPtr,
	IndexType forceCount/*,
	ValueTypePtr nativeLocationPtr,
	IndexType nativeLocationCount,

	IndexTypePtr lineDofPtr,
	IndexType lineDofSize*/);
void intCellElementOnCuda_EFG(int nCount,EFGCellOnCuda * cellElementOnCpu,int linesCount,float * linesPtrOnCpu);
void initVertexOnCuda(int nCount,VertexOnCuda * VertexOnCudaPtr);
void initLinePair();
void initial_Cuda(  const IndexType nDofs,
					EigenValueType dbNewMarkConstant[8],
					IndexTypePtr boundaryconditionPtr,
					IndexType bcCount,
					IndexTypePtr boundaryconditionDisplacementPtr,
					IndexType bcCountDisplace
					);
void makeInfluncePointList(const float , const int );
void debugCopyCellStruct(int & nCellSize,EFGCellOnCuda * CellOnCudaPtr,EFGCellOnCuda_Ext *);
void callCudaHostAlloc();
void makeGlobalIndexPara(float YoungModulus, float PossionRatio, float Density, float *externForce);
void assembleSystemOnCuda_EFG_RealTime();
void initBlade(ValueTypePtr elementBlade,IndexType nElementCount, ValueTypePtr bladeNormal);
void initBlade_MeshCutting( int nUpBladeVertexSize,int nDownBladeVertexSize,int nUpBladeEdgeSize,
	int nDownBladeEdgeSize,int nUpBladeSurfaceSize,int nDownBladeSurfaceSize,
	MC_Vertex_Cuda*	UpBlade_MC_Vertex_Cuda,MC_Edge_Cuda* UpBlade_MC_Edge_Cuda,MC_Surface_Cuda*	UpBlade_MC_Surface_Cuda,
	MC_Vertex_Cuda*	DownBlade_MC_Vertex_Cuda,MC_Edge_Cuda* DownBlade_MC_Edge_Cuda,MC_Surface_Cuda*	DownBlade_MC_Surface_Cuda);
int g_bcMinCount = 30;
int g_bcMaxCount = 38;
int g_nTimeStep = 0;
float g_externalForceFactor = 0*30.50f;
extern int g_nLineCount4Display;
extern int g_nMCSurfaceSize;
extern int g_nUpBladeSurfaceSize;
extern int g_nDownBladeSurfaceSize;

extern void drawAxis();

namespace VR_FEM
{
	GLuint MeshGenerate::vbo_triangles;
	GLuint MeshGenerate::vbo_lines;
	GLuint MeshGenerate::vbo_triangles_color;
	GLuint MeshGenerate::vbo_triangles_vertexNormal;
	GLuint MeshGenerate::m_nVBOTexCoords;

	class CTexCoord													// Texture Coordinate Class
	{
	public:
		float u;													// U Component
		float v;													// V Component
	};

	void MeshGenerate::makeLineSet_Steak()
	{
		static int linePair[12][2] = {{0,1},{2,3},{0,2},{1,3},{2,6},{3,7},{0,4},{1,5},{4,6},{5,7},{6,7},{4,5}};
		const int nCellSize = Cell::getCellSize();

		for (unsigned c=0; c<nCellSize;++c)
		{
			CellPtr curCellPtr = Cell::getCell(c);
			for (unsigned l=0;l<12;++l)
			{
				const int leftId = curCellPtr->getVertex(linePair[l][0])->getId();
				const int rightId = curCellPtr->getVertex(linePair[l][1])->getId();
				m_map_lineId4Steak[leftId][rightId] = true;
			}
		}
	}

	void MeshGenerate::initVBOStructure()
	{
		int * line_vertex_pair;
		int lineCount;
		int * vertex_dofs;
		float* vertex_pos;
		int vertexCount;
		makeCudaMemory_Steak(&line_vertex_pair,lineCount,&vertex_dofs,&vertex_pos,vertexCount);
		printf("lineCount is %d ; vertexCount is %d \n",lineCount,vertexCount);
		initVBODataStruct(line_vertex_pair,lineCount,vertex_dofs,vertex_pos,vertexCount);
		freeCudaMemory(&line_vertex_pair,&vertex_dofs,&vertex_pos);
		printf("initVBODataStruct return \n");
#if 0
		int nTriangleMeshVertexCount;
		int nTriangleMeshVertexNormalCount;
		int nTriangleMeshFaceCount;
		int nTriangleMeshVNCount;
		MyFloat * elemVertexPos;int nVertexPosLen;
		MyFloat * elemVertexTriLinearWeight;int nVertexTriLinearWeightLen;
		int     * elemVertexRelatedDofs;int nVertexRelatedDofsLen;
		MyFloat * elemVertexNormal;int nVertexNormalLen;
		int     * elemFaceVertexIdx;int nFaceLen;
		int     * elemVertexNormalIdx;int  nVertexNormalIdxLen;
		makeTriangleMeshMemory(nTriangleMeshVertexCount,nTriangleMeshVertexNormalCount,nTriangleMeshFaceCount,nTriangleMeshVNCount,
			&elemVertexPos,nVertexPosLen,
			&elemVertexTriLinearWeight,nVertexTriLinearWeightLen,
			&elemVertexRelatedDofs,nVertexRelatedDofsLen,
			&elemVertexNormal,nVertexNormalLen,
			&elemFaceVertexIdx,nFaceLen,
			&elemVertexNormalIdx,nVertexNormalIdxLen);
		printf("triangle mesh 5 TriangleMesh[vertexes:%d vertex normal:%d Face:%d vn:%d][nVertexPosLen is %d]\n",nTriangleMeshVertexCount,nTriangleMeshVertexNormalCount,nTriangleMeshFaceCount,	nTriangleMeshVNCount,nVertexPosLen);
		initVBODataStruct_triangleMesh(nTriangleMeshVertexCount,
			nTriangleMeshVertexNormalCount,
			nTriangleMeshFaceCount,
			nTriangleMeshVNCount,
			elemVertexPos,nVertexPosLen,
			elemVertexTriLinearWeight,nVertexTriLinearWeightLen,
			elemVertexRelatedDofs,nVertexRelatedDofsLen,
			elemVertexNormal,nVertexNormalLen,
			elemFaceVertexIdx,nFaceLen,
			elemVertexNormalIdx,nVertexNormalIdxLen);
		printf("triangle mesh 6 \n");
		freeTriangleMeshMemory(&elemVertexPos,
			&elemVertexTriLinearWeight,
			&elemVertexRelatedDofs,
			&elemVertexNormal,
			&elemFaceVertexIdx,
			&elemVertexNormalIdx);
		printf("triangle mesh 7 \n");
#else
		makeMeshStructureOnCuda_Steak();
		int nTriangleMeshFaceCount = face_Position_indicies.size() * 3;
#endif
		initVBOScene(nTriangleMeshFaceCount,lineCount);
	}

	void MeshGenerate::makeCudaMemory_Steak(int ** line_vertex_pair,int& lineCount,int ** vertex_dofs,float** vertex_pos,int& vertexCount)
	{
		static int linePair[12][2] = {{0,1},{2,3},{0,2},{1,3},{2,6},{3,7},{0,4},{1,5},{4,6},{5,7},{6,7},{4,5}};
		const int nCellSize = Cell::getCellSize();
		lineCount = 0;
		for (unsigned c=0; c<nCellSize;++c)
		{
			CellPtr curCellPtr = Cell::getCell(c);
			for (unsigned l=0;l<12;++l)
			{
				const int leftId = curCellPtr->getVertex(linePair[l][0])->getId();
				const int rightId = curCellPtr->getVertex(linePair[l][1])->getId();
				m_map_lineId4Steak[leftId][rightId] = true;
			}
		}

		std::map< int, std::map<int,bool> >::const_iterator ci = m_map_lineId4Steak.begin();
		for (;ci != m_map_lineId4Steak.end(); ++ci)
		{
			lineCount += (*ci).second.size();
		}

		m_nLineCount_In_map_lineID = lineCount;
		vertexCount = Vertex::getVertexSize();

		*line_vertex_pair = new int[lineCount * 2];
		memset(*line_vertex_pair,'\0',sizeof(int) * lineCount * 2 );
		*vertex_dofs = new int[vertexCount * 3];
		memset(*vertex_dofs,'\0',sizeof(int) * vertexCount * 3 );
		*vertex_pos = new float[vertexCount * 3];
		memset(*vertex_pos,'\0',sizeof(float) * vertexCount * 3);

		ci = m_map_lineId4Steak.begin();
		for (int idx = 0;ci != m_map_lineId4Steak.end(); ++ci)
		{
			const int leftVertexId = (*ci).first;
			const std::map<int,bool>& refMap = (*ci).second;
			std::map<int,bool>::const_iterator ciLine = refMap.begin();
			for (;ciLine != refMap.end();++ciLine,idx += 2)
			{
				(*line_vertex_pair)[idx] = leftVertexId;
				(*line_vertex_pair)[idx+1] = (*ciLine).first;
			}
		}


		for (int v=0,idx = 0;v < vertexCount;++v,idx +=3)
		{
			const MyVectorI& refDofs = Vertex::getVertex(v)->getDofs();
			const MyPoint& refPos = Vertex::getVertex(v)->getPos();

			(*vertex_dofs)[idx] = refDofs.x();
			(*vertex_dofs)[idx+1] = refDofs.y();
			(*vertex_dofs)[idx+2] = refDofs.z();

			(*vertex_pos)[idx] = refPos.x();
			(*vertex_pos)[idx+1] = refPos.y();
			(*vertex_pos)[idx+2] = refPos.z();
		}
	}

	void MeshGenerate::freeCudaMemory(int ** line_vertex_pair,int ** vertex_dofs,float** vertex_pos)
	{
		delete [] *line_vertex_pair;
		delete [] *vertex_dofs;
		delete [] *vertex_pos;
	}

	void MeshGenerate::freeTriangleMeshMemory(MyFloat ** elemVertexPos,
		MyFloat ** elemVertexTriLinearWeight,
		int     ** elemVertexRelatedDofs,
		MyFloat ** elemVertexNormal,
		int     ** elemFaceVertexIdx,
		int     ** elemVertexNormalIdx)
	{
		delete [](* elemVertexPos);
		delete [](* elemVertexTriLinearWeight);
		delete [](* elemVertexRelatedDofs);
		delete [](* elemVertexNormal);
		delete [](* elemFaceVertexIdx);
		delete [](* elemVertexNormalIdx);
	}

	void MeshGenerate::makeTriangleMeshMemory(int& nTriangleMeshVertexCount,
		int& nTriangleMeshVertexNormalCount,
		int& nTriangleMeshFaceCount,
		int& nTriangleMeshVNCount,
		MyFloat ** elemVertexPos,int& nVertexPosLen,
		MyFloat ** elemVertexTriLinearWeight,int & nVertexTriLinearWeightLen,
		int     ** elemVertexRelatedDofs,int & nVertexRelatedDofsLen,
		MyFloat ** elemVertexNormal,int & nVertexNormalLen,
		int     ** elemFaceVertexIdx,int & nFaceLen,
		int     ** elemVertexNormalIdx,int & nVertexNormalIdxLen)
	{
		nTriangleMeshVertexCount = vertices.size();
		nTriangleMeshVertexNormalCount = verticeNormals.size();
		nTriangleMeshFaceCount = face_Position_indicies.size();
		nTriangleMeshVNCount = face_VertexNormal_indicies.size();
		int v;
		nVertexPosLen = nTriangleMeshVertexCount * 3;	
		nVertexTriLinearWeightLen =  nTriangleMeshVertexCount * 8;	
		nVertexRelatedDofsLen = nTriangleMeshVertexCount * 24;	

		printf("triangle mesh 1 \n");
		*elemVertexPos = new MyFloat[nVertexPosLen];
		*elemVertexTriLinearWeight = new MyFloat[nVertexTriLinearWeightLen];
		*elemVertexRelatedDofs = new int [nVertexRelatedDofsLen];

		if (m_vecVertexIdx2NodeIdxInside.size() != nTriangleMeshVertexCount)
		{
			printf("assert(e.vec_vertexIdx2NodeIdxInside.size()[%d] == VR_FEM::ObjParser::vertices.size()[%d]  );\n",m_vecVertexIdx2NodeIdxInside.size(),nTriangleMeshVertexCount);
			MyExit;
		}

		for (v=0;v< nTriangleMeshVertexCount;++v)
		{
			TriangleMeshNode& ref_TriangleMesh = m_vecVertexIdx2NodeIdxInside[v];

			for (int i=0;i<3;++i)
			{
				(*elemVertexPos)[v*3 + i] = vertices[v][i];
			}

			for (int i = 0; i < 8; ++i)
			{
				(*elemVertexTriLinearWeight)[v*8+i] = ref_TriangleMesh.m_TriLinearWeight[i];
			}

			for (int i=0;i<24;++i)
			{
				(*elemVertexRelatedDofs)[v*24 + i] = ref_TriangleMesh.m_VertexDofs[i];
			}

		}
		printf("triangle mesh 2 \n");
		nVertexNormalLen = verticeNormals.size() * 3;
		(* elemVertexNormal) = new MyFloat[nVertexNormalLen];
		for (int v = 0;v < verticeNormals.size();++v)
		{
			(* elemVertexNormal)[v * 3 +0] = verticeNormals[v][0];
			(* elemVertexNormal)[v * 3 +1] = verticeNormals[v][1];
			(* elemVertexNormal)[v * 3 +2] = verticeNormals[v][2];
		}

		nFaceLen = face_Position_indicies.size() * 3;
		nVertexNormalIdxLen = face_VertexNormal_indicies.size() * 3;
		(* elemFaceVertexIdx) = new int [nFaceLen];
		(* elemVertexNormalIdx) = new int [nVertexNormalIdxLen];

		printf("triangle mesh 3 \n");
		nFaceLen = 0;nVertexNormalIdxLen = 0;
		for (int v = 0;v < face_Position_indicies.size();++v )
		{
			if (face_Position_indicies[v][0] < 0 || face_Position_indicies[v][1] < 0 || face_Position_indicies[v][2] < 0  )
			{
				continue;
			}
			for (int i=0;i<3;++i)
			{
				(* elemFaceVertexIdx)[v * 3 + i] = face_Position_indicies[v][i];
				(* elemVertexNormalIdx)[v*3 + i] = face_VertexNormal_indicies[v][i];
				++nVertexNormalIdxLen;
				++nFaceLen;
			}

		}
		nTriangleMeshFaceCount = nFaceLen / 3;
		nTriangleMeshVNCount = nVertexNormalIdxLen / 3;
		printf("triangle mesh 4  [nTriangleMeshFaceCount is %d]  [nTriangleMeshVNCount is %d] \n",nTriangleMeshFaceCount,nTriangleMeshVNCount);
	}

	void MeshGenerate::initVBOScene(const int nTrianglCount, const int nLineCount)
	{

#if ShowTriangle
		createVBO(&vbo_triangles, sizeof(float4),nTrianglCount*3);
		createVBO(&vbo_triangles_vertexNormal, sizeof(float3),nTrianglCount*3);
		createVBO(&vbo_triangles_color,sizeof(uchar4),nTrianglCount*3);
#endif

#if ShowLines
		//createVBO(&vbo_lines, sizeof(float4),nLineCount*2);
		createVBO(&vbo_lines, sizeof(float4),3136*12*2*2);
#endif
		const unsigned nTrianglesSize = face_Position_indicies.size();
		for (unsigned f=0;f<nTrianglesSize;++f)
		{
			const MyVectorI faceIndicies = face_Position_indicies[f];
			const MyVectorI texcIndicies = face_Texcood_indicies[f];
			const MyVectorI vtxnIndicies = face_VertexNormal_indicies[f];
			glTexCoord2f(1*verticeTexcood[texcIndicies[0]].first,-1*verticeTexcood[texcIndicies[0]].second); glNormal3f(verticeNormals[vtxnIndicies[0]][0],verticeNormals[vtxnIndicies[0]][1],verticeNormals[vtxnIndicies[0]][2]); glVertex3f(vertices[faceIndicies[0]][0], vertices[faceIndicies[0]][1],  vertices[faceIndicies[0]][2]);
			glTexCoord2f(1*verticeTexcood[texcIndicies[1]].first,-1*verticeTexcood[texcIndicies[1]].second); glNormal3f(verticeNormals[vtxnIndicies[1]][0],verticeNormals[vtxnIndicies[1]][1],verticeNormals[vtxnIndicies[1]][2]); glVertex3f(vertices[faceIndicies[1]][0], vertices[faceIndicies[1]][1],  vertices[faceIndicies[1]][2]);
			glTexCoord2f(1*verticeTexcood[texcIndicies[2]].first,-1*verticeTexcood[texcIndicies[2]].second); glNormal3f(verticeNormals[vtxnIndicies[2]][0],verticeNormals[vtxnIndicies[2]][1],verticeNormals[vtxnIndicies[2]][2]); glVertex3f(vertices[faceIndicies[2]][0], vertices[faceIndicies[2]][1],  vertices[faceIndicies[2]][2]);
		}

		const int nVtSize = face_Texcood_indicies.size();
		CTexCoord * m_pTexCoords = new CTexCoord[nVtSize*3];
		for (int vt=0;vt < nVtSize;++vt)
		{
			m_pTexCoords[vt*3+0].u = verticeTexcood[face_Texcood_indicies[vt][0]].first;m_pTexCoords[vt*3+0].v = -1*verticeTexcood[face_Texcood_indicies[vt][0]].second;
			m_pTexCoords[vt*3+1].u = verticeTexcood[face_Texcood_indicies[vt][1]].first;m_pTexCoords[vt*3+1].v = -1*verticeTexcood[face_Texcood_indicies[vt][1]].second;
			m_pTexCoords[vt*3+2].u = verticeTexcood[face_Texcood_indicies[vt][2]].first;m_pTexCoords[vt*3+2].v = -1*verticeTexcood[face_Texcood_indicies[vt][2]].second;
		}
		

		glGenBuffersARB( 1, &m_nVBOTexCoords );							// Get A Valid Name
		glBindBufferARB( GL_ARRAY_BUFFER_ARB, m_nVBOTexCoords );		// Bind The Buffer
		// Load The Data
		glBufferDataARB( GL_ARRAY_BUFFER_ARB, nVtSize*3*2*sizeof(float), m_pTexCoords, GL_STATIC_DRAW_ARB );

		//
		/*createVBO(&axisVBO,sizeof(float4),17*2);
		createVBO(&axisColorVBO,sizeof(uchar4),17*2);*/
		// make certain the VBO gets cleaned up on program exit
		atexit(cleanupCuda);

	}

	void MeshGenerate::createVBO(GLuint* vbo, unsigned int typeSize,unsigned nCount)
	{
		// create buffer object
		glGenBuffers(1, vbo);
		glBindBuffer(GL_ARRAY_BUFFER, *vbo);

		// initialize buffer object
		unsigned int size = nCount * typeSize;
		glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		// register buffer object with CUDA
		cudaGLRegisterBufferObject(*vbo);
	}

	////////////////////////////////////////////////////////////////////////////////
	//! Delete VBO
	////////////////////////////////////////////////////////////////////////////////
	void MeshGenerate::deleteVBO(GLuint* vbo)
	{
		glBindBuffer(1, *vbo);
		glDeleteBuffers(1, vbo);

		cudaGLUnregisterBufferObject(*vbo);

		*vbo = NULL;
	}

	void MeshGenerate::cleanupCuda()
	{

#if ShowTriangle
		deleteVBO(&vbo_triangles);
		deleteVBO(&vbo_triangles_vertexNormal);
		deleteVBO(&vbo_triangles_color);
#endif

#if ShowLines
		deleteVBO(&vbo_lines);
#endif

		
	}
	void MeshGenerate::loadObjDataBunny(const char* lpszFilePath)
	{
		std::ifstream infile(lpszFilePath);
		std::cout << "begin parser obj file : " << lpszFilePath << std::endl;
		if (!infile.is_open())
		{
			printf("obj file open fail.\n");
		}
		
        Q_ASSERT(infile.is_open());

        std::stringstream ss;
        MyFloat db1,db2,db3;
        int i1,i2,i3;
        char * lpszLineBuffer = new char[BufferSize];
		int nFirstIdx;
		Eigen::Vector3i tmpFaceIdx,tmpTexcoodIdx,tmpNormalIdx;
		MyDenseVector   tmpVertexNormal,tmpVertex;
		MyFloat nTmpVtIdx_0,nTmpVtIdx_1;
        while(!infile.eof())
        {
            std::stringstream ss;
            memset(lpszLineBuffer,'\0',BufferSize);
            infile.getline(lpszLineBuffer,BufferSize);
            
				
			if ('v' == lpszLineBuffer[0] && 'n' == lpszLineBuffer[1])
			{
				nFirstIdx = 2;
				while (' ' == lpszLineBuffer[nFirstIdx])
				{
					++nFirstIdx;
				}

				std::string str(&lpszLineBuffer[nFirstIdx]);
				boost::algorithm::split_iterator< std::string::iterator > iStr( 
					str,
					boost::algorithm::token_finder(
					boost::algorithm::is_any_of( " /\t\n\r" ),
					boost::algorithm::token_compress_on 
					) 
					);
				boost::algorithm::split_iterator< std::string::iterator> end;

				tmpVertexNormal[0] = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() );
				++iStr;
				tmpVertexNormal[1] = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() );
				++iStr;
				tmpVertexNormal[2] = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() );
				++iStr;
				verticeNormals.push_back(tmpVertexNormal);
				
				//std::cout << "tmpVertexNormal : " << tmpVertexNormal << std::endl;
			}
            else if ('f' == lpszLineBuffer[0])
            {
				nFirstIdx = 1;
				while (' ' == lpszLineBuffer[nFirstIdx])
				{
					++nFirstIdx;
				}

				
				std::string str(&lpszLineBuffer[nFirstIdx]);
				boost::algorithm::split_iterator< std::string::iterator > iStr( 
					str,
					boost::algorithm::token_finder(
													boost::algorithm::is_any_of( " /\t\n\r" ),
													boost::algorithm::token_compress_on 
												  ) 
					);
				boost::algorithm::split_iterator< std::string::iterator> end;

				tmpFaceIdx[0] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
				++iStr;
				/*tmpTexcoodIdx[0] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
				++iStr;*/
				tmpNormalIdx[0] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
				++iStr;

				tmpFaceIdx[1] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
				++iStr;
				/*tmpTexcoodIdx[1] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
				++iStr;*/
				tmpNormalIdx[1] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
				++iStr;

				tmpFaceIdx[2] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
				++iStr;
				/*tmpTexcoodIdx[2] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
				++iStr;*/
				tmpNormalIdx[2] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
				++iStr;

				face_Position_indicies.push_back(tmpFaceIdx);
				//face_Texcood_indicies.push_back(tmpTexcoodIdx);
				face_VertexNormal_indicies.push_back(tmpNormalIdx);
				/*std::cout << "face : " << tmpFaceIdx << tmpTexcoodIdx << tmpNormalIdx << std::endl;
				MyPause;*/
            }
			else if ('v' == lpszLineBuffer[0] && 't' == lpszLineBuffer[1])
			{
				nFirstIdx = 2;
				while (' ' == lpszLineBuffer[nFirstIdx])
				{
					++nFirstIdx;
				}

				std::string str(&lpszLineBuffer[nFirstIdx]);
				boost::algorithm::split_iterator< std::string::iterator > iStr( 
					str,
					boost::algorithm::token_finder(
					boost::algorithm::is_any_of( " /\t\n\r" ),
					boost::algorithm::token_compress_on 
					) 
					);
				boost::algorithm::split_iterator< std::string::iterator> end;

				nTmpVtIdx_0 = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() );
				++iStr;
				nTmpVtIdx_1 = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() );
				++iStr;

				verticeTexcood.push_back(std::make_pair(nTmpVtIdx_0,nTmpVtIdx_1));
				/*std::cout << "vt : " << nTmpVtIdx_0 << nTmpVtIdx_1 << std::endl;
				MyPause;*/
			}
			else if ('v' == lpszLineBuffer[0] && 'n' != lpszLineBuffer[1] && 't' != lpszLineBuffer[1])
            {
				nFirstIdx = 1;
				while (' ' == lpszLineBuffer[nFirstIdx])
				{
					++nFirstIdx;
				}

				std::string str(&lpszLineBuffer[nFirstIdx]);
				boost::algorithm::split_iterator< std::string::iterator > iStr( 
					str,
					boost::algorithm::token_finder(
					boost::algorithm::is_any_of( " /\t\n\r" ),
					boost::algorithm::token_compress_on 
					) 
					);
				boost::algorithm::split_iterator< std::string::iterator> end;

				tmpVertex[0] = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() ) + Steak_Translation;
				++iStr;
				tmpVertex[1] = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() ) + Steak_Translation;
				++iStr;
				tmpVertex[2] = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() ) + Steak_Translation;
				++iStr;
				vertices.push_back(tmpVertex);
				//std::cout << "vertive : " << tmpVertex << std::endl;
				
			}
            else if ('#' == lpszLineBuffer[0])
            {
                //printf(lpszLineBuffer);
            }
			
        }

        printf("[vertices is %d] [verticeNormals is %d] [verticeTexcood is %d] [face is %d %d %d]\n",
				vertices.size(),verticeNormals.size(),verticeTexcood.size(),
				face_Position_indicies.size(),face_Texcood_indicies.size(),face_VertexNormal_indicies.size());
        delete []lpszLineBuffer;
	}

	void MeshGenerate::loadObjDataSteak(const char* lpszFilePath)
	{		

		std::ifstream infile(lpszFilePath);
		std::cout << "begin parser obj file : " << lpszFilePath << std::endl;
		if (!infile.is_open())
		{
			printf("obj file open fail.\n");
		}
		
        Q_ASSERT(infile.is_open());

        std::stringstream ss;
        MyFloat db1,db2,db3;
        int i1,i2,i3;
        char * lpszLineBuffer = new char[BufferSize];
		int nFirstIdx;
		Eigen::Vector3i tmpFaceIdx,tmpTexcoodIdx,tmpNormalIdx;
		MyDenseVector   tmpVertexNormal,tmpVertex;
		MyFloat nTmpVtIdx_0,nTmpVtIdx_1;
        while(!infile.eof())
        {
            std::stringstream ss;
            memset(lpszLineBuffer,'\0',BufferSize);
            infile.getline(lpszLineBuffer,BufferSize);
            
				
			if ('v' == lpszLineBuffer[0] && 'n' == lpszLineBuffer[1])
			{
				nFirstIdx = 2;
				while (' ' == lpszLineBuffer[nFirstIdx])
				{
					++nFirstIdx;
				}

				std::string str(&lpszLineBuffer[nFirstIdx]);
				boost::algorithm::split_iterator< std::string::iterator > iStr( 
					str,
					boost::algorithm::token_finder(
					boost::algorithm::is_any_of( " /\t\n\r" ),
					boost::algorithm::token_compress_on 
					) 
					);
				boost::algorithm::split_iterator< std::string::iterator> end;

				tmpVertexNormal[0] = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() );
				++iStr;
				tmpVertexNormal[1] = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() );
				++iStr;
				tmpVertexNormal[2] = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() );
				++iStr;
				verticeNormals.push_back(tmpVertexNormal);
				
				//std::cout << "tmpVertexNormal : " << tmpVertexNormal << std::endl;
			}
            else if ('f' == lpszLineBuffer[0])
            {
				nFirstIdx = 1;
				while (' ' == lpszLineBuffer[nFirstIdx])
				{
					++nFirstIdx;
				}

				
				std::string str(&lpszLineBuffer[nFirstIdx]);
				boost::algorithm::split_iterator< std::string::iterator > iStr( 
					str,
					boost::algorithm::token_finder(
													boost::algorithm::is_any_of( " /\t\n\r" ),
													boost::algorithm::token_compress_on 
												  ) 
					);
				boost::algorithm::split_iterator< std::string::iterator> end;

				tmpFaceIdx[0] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
				++iStr;
				tmpTexcoodIdx[0] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
				++iStr;
				tmpNormalIdx[0] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
				++iStr;

				tmpFaceIdx[1] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
				++iStr;
				tmpTexcoodIdx[1] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
				++iStr;
				tmpNormalIdx[1] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
				++iStr;

				tmpFaceIdx[2] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
				++iStr;
				tmpTexcoodIdx[2] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
				++iStr;
				tmpNormalIdx[2] = boost::lexical_cast<int>( boost::lexical_cast<std::string>((*iStr)).c_str() ) - 1;
				++iStr;

				face_Position_indicies.push_back(tmpFaceIdx);
				face_Texcood_indicies.push_back(tmpTexcoodIdx);
				face_VertexNormal_indicies.push_back(tmpNormalIdx);
				/*std::cout << "face : " << tmpFaceIdx << tmpTexcoodIdx << tmpNormalIdx << std::endl;
				MyPause;*/
            }
			else if ('v' == lpszLineBuffer[0] && 't' == lpszLineBuffer[1])
			{
				nFirstIdx = 2;
				while (' ' == lpszLineBuffer[nFirstIdx])
				{
					++nFirstIdx;
				}

				std::string str(&lpszLineBuffer[nFirstIdx]);
				boost::algorithm::split_iterator< std::string::iterator > iStr( 
					str,
					boost::algorithm::token_finder(
					boost::algorithm::is_any_of( " /\t\n\r" ),
					boost::algorithm::token_compress_on 
					) 
					);
				boost::algorithm::split_iterator< std::string::iterator> end;

				nTmpVtIdx_0 = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() );
				++iStr;
				nTmpVtIdx_1 = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() );
				++iStr;

				verticeTexcood.push_back(std::make_pair(nTmpVtIdx_0,nTmpVtIdx_1));
				/*std::cout << "vt : " << nTmpVtIdx_0 << nTmpVtIdx_1 << std::endl;
				MyPause;*/
			}
			else if ('v' == lpszLineBuffer[0] && 'n' != lpszLineBuffer[1] && 't' != lpszLineBuffer[1])
            {
				nFirstIdx = 1;
				while (' ' == lpszLineBuffer[nFirstIdx])
				{
					++nFirstIdx;
				}

				std::string str(&lpszLineBuffer[nFirstIdx]);
				boost::algorithm::split_iterator< std::string::iterator > iStr( 
					str,
					boost::algorithm::token_finder(
					boost::algorithm::is_any_of( " /\t\n\r" ),
					boost::algorithm::token_compress_on 
					) 
					);
				boost::algorithm::split_iterator< std::string::iterator> end;

				tmpVertex[0] = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() ) + Steak_Translation;
				++iStr;
				tmpVertex[1] = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() ) + Steak_Translation;
				++iStr;
				tmpVertex[2] = boost::lexical_cast<MyFloat>( boost::lexical_cast<std::string>((*iStr)).c_str() ) + Steak_Translation;
				++iStr;
				vertices.push_back(tmpVertex);
				//std::cout << "vertive : " << tmpVertex << std::endl;
				
			}
            else if ('#' == lpszLineBuffer[0])
            {
                //printf(lpszLineBuffer);
            }
			
        }

        printf("[vertices is %d] [verticeNormals is %d] [verticeTexcood is %d] [face is %d %d %d]\n",
				vertices.size(),verticeNormals.size(),verticeTexcood.size(),
				face_Position_indicies.size(),face_Texcood_indicies.size(),face_VertexNormal_indicies.size());
        delete []lpszLineBuffer;
		//MyPause;
	}

	bool MeshGenerate::LoadGLTextures(const char* lpszTexturePath)
	{
		glGenTextures(1, &texture_steak);					// Create The Texture

		// Typical Texture Generation Using Data From The Bitmap
		glBindTexture(GL_TEXTURE_2D, texture_steak);
		unsigned char* image;
		int width, height;
		image = SOIL_load_image( lpszTexturePath, &width, &height, 0, SOIL_LOAD_RGB );
		if (image)
		{
			glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image );
			//glTexImage2D(GL_TEXTURE_2D, 0, 3, TextureImage[0]->sizeX, TextureImage[0]->sizeY, 0, GL_RGB, GL_UNSIGNED_BYTE, TextureImage[0]->data);
			SOIL_free_image_data( image );
			glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
			return true;
		}
		return false;
	}

	void MeshGenerate::solve_timestep_steak(unsigned nStep)
	{
		//printf("begin runCuda \n");
		// map OpenGL buffer object for writing from CUDA
		float4 *dptrLines;
		float4 *dptrTriangles;
		float3 *vertexNormal;
		uchar4 *cptr;
		//unsigned int *iptr;
		
		cudaGLMapBufferObject((void**)&dptrLines, vbo_lines);
		cudaGLMapBufferObject((void**)&dptrTriangles, vbo_triangles);
		cudaGLMapBufferObject((void**)&vertexNormal, vbo_triangles_vertexNormal);
		cudaGLMapBufferObject((void**)&cptr, vbo_triangles_color);
		//printf("do_loop \n");

		do_loop(nStep,dptrLines,cptr,dptrTriangles,vertexNormal);

		/*static int nCount = 0;
		if (g_nTimeStep > 100 && nCount < 4)
		{
			g_nTimeStep = 0;
			nCount++;
		}*/
		cudaGLUnmapBufferObject(vbo_lines);
		cudaGLUnmapBufferObject(vbo_triangles);
		cudaGLUnmapBufferObject(vbo_triangles_vertexNormal);
		cudaGLUnmapBufferObject(vbo_triangles_color);
	}

	void drawBlade()
	{
		glColor3f(.5f,1.f,0.5f);
		glBegin(GL_TRIANGLES);
		glVertex3f(BLADE_A);
		glVertex3f(BLADE_B);
		glVertex3f(BLADE_C);

		glVertex3f(BLADE_A);
		glVertex3f(BLADE_C);
		glVertex3f(BLADE_B);

		glVertex3f(BLADE_E);
		glVertex3f(BLADE_F);
		glVertex3f(BLADE_G);

		glVertex3f(BLADE_E);
		glVertex3f(BLADE_G);
		glVertex3f(BLADE_F);
		glEnd();
	}

	void MeshGenerate::print_steak()
	{
		// maxDiameter is 8.260920 translation_x(4.048582,4.075137,4.127901)
		static GLuint torusList=0;

		if(!torusList)
		{
			torusList=glGenLists(1);
			glNewList(torusList, GL_COMPILE);
			{
				glColor3f(1.0f, 0.0f, 0.0f);
				glPushMatrix();

				glTranslatef(0.5f, 0.5f, 0.0f);
				//glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
				glutSolidTorus(0.2, 0.5, 24, 48);

				glPopMatrix();
			}
			glEndList();
		}

		
		

		::glPushMatrix();
			//glRotatef(90,0,0,1);
			
			glRotatef(rotate_x, 1.0, 0.0, 0.0);
			glRotatef(rotate_y, 0.0, 1.0f, 0.0);
			//glTranslatef(-0.5f, -0.5f, translate_z);
			//glRotatef(-90, 1.0, 0.0, 0.0);
			//glTranslatef(0.f, -1., 0.f);
			glScalef(scaled,scaled,scaled);
			//glTranslatef(-Steak_Translation, -Steak_Translation,-Steak_Translation);
			drawAxis();
			::glPushMatrix();
			/*glTranslatef(-0.5f, -0.5f, 0.);
			glCallList(torusList);
			*/
			glTranslatef(-0.5f, -0.5f, -0.5);
#if 0
			
			glBegin(GL_TRIANGLES);
			const unsigned nTrianglesSize = face_Position_indicies.size();
			for (unsigned f=0;f<nTrianglesSize;++f)
			{
				const MyVectorI faceIndicies = face_Position_indicies[f];
				const MyVectorI texcIndicies = face_Texcood_indicies[f];
				const MyVectorI vtxnIndicies = face_VertexNormal_indicies[f];
				glTexCoord2f(1*verticeTexcood[texcIndicies[0]].first,-1*verticeTexcood[texcIndicies[0]].second); glNormal3f(verticeNormals[vtxnIndicies[0]][0],verticeNormals[vtxnIndicies[0]][1],verticeNormals[vtxnIndicies[0]][2]); glVertex3f(vertices[faceIndicies[0]][0], vertices[faceIndicies[0]][1],  vertices[faceIndicies[0]][2]);
				glTexCoord2f(1*verticeTexcood[texcIndicies[1]].first,-1*verticeTexcood[texcIndicies[1]].second); glNormal3f(verticeNormals[vtxnIndicies[1]][0],verticeNormals[vtxnIndicies[1]][1],verticeNormals[vtxnIndicies[1]][2]); glVertex3f(vertices[faceIndicies[1]][0], vertices[faceIndicies[1]][1],  vertices[faceIndicies[1]][2]);
				glTexCoord2f(1*verticeTexcood[texcIndicies[2]].first,-1*verticeTexcood[texcIndicies[2]].second); glNormal3f(verticeNormals[vtxnIndicies[2]][0],verticeNormals[vtxnIndicies[2]][1],verticeNormals[vtxnIndicies[2]][2]); glVertex3f(vertices[faceIndicies[2]][0], vertices[faceIndicies[2]][1],  vertices[faceIndicies[2]][2]);
			}
			glEnd();

			glBegin(GL_LINES);
			std::map< int, std::map<int,bool> >::const_iterator ci = m_map_lineId4Steak.begin();
			for (int idx = 0;ci != m_map_lineId4Steak.end(); ++ci)
			{
				const int leftVertexId = (*ci).first;
				const MyPoint & leftPos = Vertex::getVertex(leftVertexId)->getPos();
				const std::map<int,bool>& refMap = (*ci).second;
				std::map<int,bool>::const_iterator ciLine = refMap.begin();
				for (;ciLine != refMap.end();++ciLine,idx += 2)
				{
					const MyPoint & rightPos = Vertex::getVertex((*ciLine).first)->getPos();
					glVertex3f(leftPos[0],leftPos[1],leftPos[2]);
					glVertex3f(rightPos[0],rightPos[1],rightPos[2]);
				}
			}
			glEnd();

#endif
if (!showTex)
{
#if 1
			glBindBuffer(GL_ARRAY_BUFFER, vbo_lines);
			glVertexPointer(4, GL_FLOAT, 0, 0);
			glEnableClientState(GL_VERTEX_ARRAY);


			glColor3f(0.0, 1.0, 0.0);
			static int linesCount;
			linesCount = g_nLineCount4Display*2;
			//printf("linesCount is %d \n",linesCount);
			glDrawArrays(GL_LINES, 0, linesCount);
			/*for(int i=0 ; i < linesCount; i+= 2)
			{
				glDrawArrays(GL_LINES, i, 2);
			}*/

			glDisableClientState(GL_VERTEX_ARRAY);
#endif
}



		  glBindBuffer(GL_ARRAY_BUFFER, vbo_triangles);
		  glVertexPointer(4, GL_FLOAT, 0, 0);
		  glEnableClientState(GL_VERTEX_ARRAY);
		  if (!showTex)
		  {
			  glBindBuffer(GL_ARRAY_BUFFER, vbo_triangles_color);
			  glColorPointer(4, GL_UNSIGNED_BYTE, 0, 0);
			  glEnableClientState(GL_COLOR_ARRAY);
		  }
		  else 
		  {
			  glBindTexture(GL_TEXTURE_2D, texture_steak);
			  glBindBufferARB( GL_ARRAY_BUFFER_ARB, m_nVBOTexCoords );
			  glTexCoordPointer( 2, GL_FLOAT, 0, (char *) NULL );		
			  glEnableClientState( GL_TEXTURE_COORD_ARRAY );
		  } 
		  

		  glBindBuffer(GL_ARRAY_BUFFER, vbo_triangles_vertexNormal);
		  glNormalPointer(GL_FLOAT,  0, 0);
		  glEnableClientState(GL_NORMAL_ARRAY);

		  //printf("faceCount is %d \n",g_nMCSurfaceSize+g_nUpBladeSurfaceSize+g_nDownBladeSurfaceSize);
	  
		  glShadeModel(GL_SMOOTH);
		  //glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
		  //glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
		  glDrawArrays(GL_TRIANGLES, 0, (g_nMCSurfaceSize+g_nUpBladeSurfaceSize+g_nDownBladeSurfaceSize)*3);
		  

		  glDisableClientState(GL_VERTEX_ARRAY);
		 /* glDisableClientState(GL_COLOR_ARRAY);*/
		if (!showTex)
		{
			glDisableClientState(GL_COLOR_ARRAY);
		}
		else
		{
				  glDisableClientState( GL_TEXTURE_COORD_ARRAY );				// Disable Texture Coord Arrays
		}

		  //
		  glDisableClientState(GL_NORMAL_ARRAY);
		::glPopMatrix();
		::glPopMatrix();
	}

	MyFloat MeshGenerate::max3(	MyFloat a,MyFloat b,MyFloat c)
	{
		float d = (b>c)?b:c;
		return ((a>d)?a:d);
	}

	void MeshGenerate::makeTriangleMeshInterpolation_Steak()
	{
		if (true)
		{
			printf("begin triangle mesh... \n");
			std::vector<TriangleMeshNode >& vec_vertexIdx2NodeIdxInside = m_vecVertexIdx2NodeIdxInside;
			vec_vertexIdx2NodeIdxInside.clear();
			MyFloat tmpFloat;
			MyDenseVector tmpVec;
			MyFloat weight[8];

			std::vector< MyDenseVector >& ref_vertices = vertices;
			std::vector< VR_FEM::CellPtr >& refCellPool = Cell::getCellVector();
			std::vector< CellToTriangleMeshVertice >& refCellToTriangleMeshVertice = m_vecCellToTriangleMeshVertice;

			refCellToTriangleMeshVertice.resize(refCellPool.size());
			for (int vi=0;vi < ref_vertices.size();++vi)
			{
				MyFloat minDistance = 1000.f;
				int   curNodeIdx = -1;
				bool  bInside = false;
				for (int vj=0;vj < refCellPool.size();++vj )
				{
					tmpVec = ref_vertices[vi] - refCellPool[vj]->getCenterPoint();
					tmpFloat = tmpVec.squaredNorm();
					if (minDistance > tmpFloat)
					{
						curNodeIdx = vj;
						minDistance = tmpFloat;
						if ( (refCellPool[vj]->getRadius() / 2.0f) < max3(std::fabs(tmpVec[0]),std::fabs(tmpVec[1]),std::fabs(tmpVec[2])) )
						{
							bInside = false;
						}
						else
						{
							bInside = true;
						}
					}
				}

				TriangleMeshNode  refNode;// = vec_vertexIdx2NodeIdxInside[vec_vertexIdx2NodeIdxInside.size()-1];

				MyDenseVector p0 = refCellPool[curNodeIdx]->getVertex(0)->getPos();
				MyDenseVector p7 = refCellPool[curNodeIdx]->getVertex(7)->getPos();
				float detaX = (p7[0] - ref_vertices[vi][0]) / (p7[0] - p0[0]);
				float detaY = (p7[1] - ref_vertices[vi][1]) / (p7[1] - p0[1]);
				float detaZ = (p7[2] - ref_vertices[vi][2]) / (p7[2] - p0[2]);

				refNode.m_TriLinearWeight[0] = detaX * detaY * detaZ;
				refNode.m_TriLinearWeight[1] = (1-detaX) * detaY * detaZ;
				refNode.m_TriLinearWeight[2] = detaX * (1-detaY) * detaZ;
				refNode.m_TriLinearWeight[3] = (1-detaX) * (1-detaY) * detaZ;
				refNode.m_TriLinearWeight[4] = detaX * detaY * (1-detaZ);
				refNode.m_TriLinearWeight[5] = (1-detaX) * detaY * (1-detaZ);
				refNode.m_TriLinearWeight[6] = detaX * (1-detaY) * (1-detaZ);
				refNode.m_TriLinearWeight[7] = (1-detaX) * (1-detaY) * (1-detaZ);

				refNode.m_bInside = bInside;

				MyFloat minLen = FLT_MAX;
				MyFloat currentLen;
				for (int vv = 0; vv < 8; ++vv)
				{
					MyVectorI & refDofs = refCellPool[curNodeIdx]->getVertex(vv)->getDofs();
					refNode.m_VertexDofs[vv*3+0] = refDofs[0];//refCellPool[curNodeIdx]->getVertex(vv)->getDof(0,0);
					refNode.m_VertexDofs[vv*3+1] = refDofs[1];//refCellPool[curNodeIdx]->getVertex(vv)->getDof(0,1);
					refNode.m_VertexDofs[vv*3+2] = refDofs[2];//refCellPool[curNodeIdx]->getVertex(vv)->getDof(0,2);
					//currentLen = (ref_vertices[vi]-refCellPool[curNodeIdx]->getVertex(vv)->getPos()).squaredNorm();
					refNode.m_verticeToCellVertexLength[vv] = (ref_vertices[vi]-refCellPool[curNodeIdx]->getVertex(vv)->getPos()).squaredNorm();
					if (refNode.m_verticeToCellVertexLength[vv] < minLen)
					{
						minLen = refNode.m_verticeToCellVertexLength[vv];
						refNode.m_nMinestLengthOfCellVertex = vv;
					}
				}
				refNode.nBelongToCellIdx = curNodeIdx;
				vec_vertexIdx2NodeIdxInside.push_back(refNode);
				refCellToTriangleMeshVertice[curNodeIdx].m_nCellIdx = curNodeIdx;
				refCellToTriangleMeshVertice[curNodeIdx].m_vecRelatedVerticeIdx.push_back(vi);
			}
		}
	}

	void MeshGenerate::assembleOnCuda()
	{
		const unsigned int   n_q_points    = Geometry::vertexs_per_cell;
		const unsigned int   nCellSize     = Cell::getCellSize();

		EFGCellOnCuda * CellOnCpuPtr = new EFGCellOnCuda[nCellSize];
		memset(CellOnCpuPtr,'\0',nCellSize * sizeof(EFGCellOnCuda));

		std::vector< CellPtr >& refCellVector = Cell::getCellVector();
		std::vector< std::pair<MyDenseVector,MyDenseVector> > vec_Lines;
		static int linePair[12][2] = {{0,2},{4,6},{0,4},{2,6},{1,3},{5,7},{1,5},{3,7},{0,1},{4,5},{2,3},{6,7}};
		for (int cellOnCpuIdx = 0;cellOnCpuIdx < nCellSize;++cellOnCpuIdx)
		{
			CellPtr curCellPtr = Cell::getCell(cellOnCpuIdx);
			const MyPoint &currentPos = curCellPtr->getCenterPoint();

			CellOnCpuPtr[cellOnCpuIdx].cellType = curCellPtr->getCellType();
			
			CellOnCpuPtr[cellOnCpuIdx].m_nStiffnessMatrixIdx = curCellPtr->getFEMCellStiffnessMatrixIdx();
			CellOnCpuPtr[cellOnCpuIdx].m_nMassMatrixIdx = curCellPtr->getFEMCellMassMatrixIdx();
			CellOnCpuPtr[cellOnCpuIdx].m_nRhsIdx = curCellPtr->getFEMCellRhsVectorIdx();
#if USE_CO_RATION
			CellOnCpuPtr[cellOnCpuIdx].radiusx2 = curCellPtr->getRadius()*2;
#endif
			for (unsigned p=0;p<8 && (COUPLE == CellOnCpuPtr[cellOnCpuIdx].cellType);++p)
			{
				CellOnCpuPtr[cellOnCpuIdx].m_ShapeFunction_R[p] = curCellPtr->m_vec_RInGaussPt_8[p];
				CellOnCpuPtr[cellOnCpuIdx].m_ShapeDeriv_R[p][0] = curCellPtr->m_vec_RDerivInGaussPt_8_3[p][0];
				CellOnCpuPtr[cellOnCpuIdx].m_ShapeDeriv_R[p][1] = curCellPtr->m_vec_RDerivInGaussPt_8_3[p][1];
				CellOnCpuPtr[cellOnCpuIdx].m_ShapeDeriv_R[p][2] = curCellPtr->m_vec_RDerivInGaussPt_8_3[p][2];
			}
			
			/*printf("m_nStiffnessMatrixIdx[%d] m_nMassMatrixIdx[%d] m_nRhsIdx[%d]\n",CellOnCpuPtr[cellOnCpuIdx].m_nStiffnessMatrixIdx,CellOnCpuPtr[cellOnCpuIdx].m_nMassMatrixIdx,CellOnCpuPtr[cellOnCpuIdx].m_nRhsIdx);
			MyPause;*/
			CellOnCpuPtr[cellOnCpuIdx].m_nLevel;
			CellOnCpuPtr[cellOnCpuIdx].m_bNewOctreeNodeList;//initialize and modify on cuda
			CellOnCpuPtr[cellOnCpuIdx].m_bTopLevelOctreeNodeList;//initialize and modify on cuda

			CellOnCpuPtr[cellOnCpuIdx].m_bLeaf = true;


			CellOnCpuPtr[cellOnCpuIdx].m_nGhostCellCount = 0;
			CellOnCpuPtr[cellOnCpuIdx].m_nGhostCellIdxInVec = -1;
			for (int k=0;k<n_q_points;++k)
			{
				CellOnCpuPtr[cellOnCpuIdx].vertexId[k] = curCellPtr->getVertex(k)->getId();
				const MyPoint& refPos = curCellPtr->getGlobalGaussPoint(k);
				CellOnCpuPtr[cellOnCpuIdx].m_EFGGobalGaussPoint[k][0] = refPos[0];
				CellOnCpuPtr[cellOnCpuIdx].m_EFGGobalGaussPoint[k][1] = refPos[1];
				CellOnCpuPtr[cellOnCpuIdx].m_EFGGobalGaussPoint[k][2] = refPos[2];
			}
			CellOnCpuPtr[cellOnCpuIdx].m_nLinesBaseIdx = vec_Lines.size()*6;
			CellOnCpuPtr[cellOnCpuIdx].m_nLinesCount = 3 + Geometry::lines_per_cell;
			CellOnCpuPtr[cellOnCpuIdx].m_nJxW = curCellPtr->getEFGJxW();
			//
			MyDenseVector  x_step(CellRaidus,0,0),y_step(0,CellRaidus,0),z_step(0,0,CellRaidus);
			vec_Lines.push_back(std::make_pair(currentPos + x_step,currentPos + -1.f*x_step));
			vec_Lines.push_back(std::make_pair(currentPos + y_step,currentPos + -1.f*y_step));
			vec_Lines.push_back(std::make_pair(currentPos + z_step,currentPos + -1.f*z_step));

			for (int lv = 0;lv < Geometry::lines_per_cell;++lv)
			{
				vec_Lines.push_back(std::make_pair(curCellPtr->getVertex(linePair[lv][0])->getPos(),curCellPtr->getVertex(linePair[lv][1])->getPos()));
			}
		}


		MyFloat * linesOnCpuPtr = new MyFloat[vec_Lines.size() * 6];

		for (int v=0,innerIdx = 0;v<vec_Lines.size();++v)
		{

			linesOnCpuPtr[innerIdx++] = vec_Lines[v].first(0);
			linesOnCpuPtr[innerIdx++] = vec_Lines[v].first(1);
			linesOnCpuPtr[innerIdx++] = vec_Lines[v].first(2);

			linesOnCpuPtr[innerIdx++] = vec_Lines[v].second(0);
			linesOnCpuPtr[innerIdx++] = vec_Lines[v].second(1);
			linesOnCpuPtr[innerIdx++] = vec_Lines[v].second(2);
		}

		intCellElementOnCuda_EFG(nCellSize,CellOnCpuPtr,vec_Lines.size() * 6,linesOnCpuPtr);


		const int VertexCount = Vertex::getVertexSize();
		VertexOnCuda* VertexOnCudaPtr = new VertexOnCuda[VertexCount];
		memset(VertexOnCudaPtr,'\0',VertexCount * sizeof(VertexOnCuda));


		for (int v=0;v<VertexCount;++v)
		{
			VertexPtr curVertexPtr = Vertex::getVertex(v);
			MyDenseVector &refLocal = curVertexPtr->getPos();
			MyVectorI & refDofs = curVertexPtr->getDofs();

			VertexOnCudaPtr[v].m_createTimeStamp = 0;
			VertexOnCudaPtr[v].m_nId = v;
			VertexOnCudaPtr[v].local[0] =  refLocal[0];
			VertexOnCudaPtr[v].local[1] =  refLocal[1];
			VertexOnCudaPtr[v].local[2] =  refLocal[2];
			VertexOnCudaPtr[v].m_nDof[0] = refDofs[0];
			VertexOnCudaPtr[v].m_nDof[1] = refDofs[1];
			VertexOnCudaPtr[v].m_nDof[2] = refDofs[2];
			VertexOnCudaPtr[v].m_fromDomainId = curVertexPtr->getFromDomainId();

		}
		if (VertexCount >= MaxVertexCount)
		{
			printf("VertexCount >= MaxVertexCount\n");
		}
		if (nCellSize >= MaxCellCount)
		{
			printf("nCellSize >= MaxCellCount\n");
		}
		initVertexOnCuda(VertexCount, VertexOnCudaPtr);
		makeInfluncePointList(SupportSize, ValidEFGDomainId);
		initLinePair();

		delete [] CellOnCpuPtr;
		delete [] VertexOnCudaPtr;
		delete [] linesOnCpuPtr;

		const int boundaryconditionSize = m_vec_boundary.size();
		int *elem_boundaryCondition = new int [boundaryconditionSize+1];

		for (unsigned c=0;c<boundaryconditionSize;++c)
		{
			elem_boundaryCondition[c] = m_vec_boundary[c];

		}

		const int forceConditionSize = m_vecForceBoundary.size();
		int *elem_forceCondition = new int[forceConditionSize+1];
		for (unsigned c=0;c<forceConditionSize;++c)
		{
			elem_forceCondition[c] = m_vecForceBoundary[c];
		}

		initial_Cuda(m_nDof,m_db_NewMarkConstant,elem_boundaryCondition,boundaryconditionSize,elem_forceCondition,forceConditionSize);
		
		printf("initial_Cuda\n");
		//MyPause;

		float externForce[3] = {Cell::externalForce[0],Cell::externalForce[1],Cell::externalForce[2]};
		makeGlobalIndexPara(Material::YoungModulus, Material::PossionRatio, Material::Density, &externForce[0]);

		//getCurrentGPUMemoryInfo();
		assembleSystemOnCuda_EFG_RealTime();
		/*getCurrentGPUMemoryInfo();
		MyPause;*/
		initBoundaryCondition();
		//getCurrentGPUMemoryInfo();
		//printf("assembleSystemOnCuda_EFG_RealTime\n");
		//MyExit;

	}

	void getNormal4Plane(MyFloat* m_elementBlade,int m_nBladeBase,std::vector< MyDenseVector>& m_vecBladePlaneNormal)
	{
		MyDenseVector v0(m_elementBlade[m_nBladeBase + 0] - m_elementBlade[m_nBladeBase + 6],
			m_elementBlade[m_nBladeBase + 1] - m_elementBlade[m_nBladeBase + 7],
			m_elementBlade[m_nBladeBase + 2] - m_elementBlade[m_nBladeBase + 8]),
			v1(m_elementBlade[m_nBladeBase + 3] - m_elementBlade[m_nBladeBase + 6],
			m_elementBlade[m_nBladeBase + 4] - m_elementBlade[m_nBladeBase + 7],
			m_elementBlade[m_nBladeBase + 5] - m_elementBlade[m_nBladeBase + 8]);
		m_vecBladePlaneNormal.push_back(v0.cross(v1)) ;
		std::cout << "normal vector : " << m_vecBladePlaneNormal[0] << std::endl;
		//system("pause");
		m_nBladeBase += 9;
		MyDenseVector v2(m_elementBlade[m_nBladeBase + 0] - m_elementBlade[m_nBladeBase + 6],
			m_elementBlade[m_nBladeBase + 1] - m_elementBlade[m_nBladeBase + 7],
			m_elementBlade[m_nBladeBase + 2] - m_elementBlade[m_nBladeBase + 8]),
			v3(m_elementBlade[m_nBladeBase + 3] - m_elementBlade[m_nBladeBase + 6],
			m_elementBlade[m_nBladeBase + 4] - m_elementBlade[m_nBladeBase + 7],
			m_elementBlade[m_nBladeBase + 5] - m_elementBlade[m_nBladeBase + 8]);
		m_vecBladePlaneNormal.push_back(v2.cross(v3));
		m_nBladeBase -= 9;
	}

	void MeshGenerate::initCuttingStructure()
	{


		MyFloat tmpElement[] = {BLADE_A,BLADE_B,BLADE_C,BLADE_E,BLADE_F,BLADE_G};
//#define CUDA_BLADE_STEAK_0  0.417f, 1.4517f, -1.01f
//#define CUDA_BLADE_STEAK_1  0.417f, 1.4517f, 2.01f
//#define CUDA_BLADE_STEAK_2  0.417f, -1.51f,  0.f
//		MyFloat tmpElement[] = {CUDA_BLADE_STEAK_0,CUDA_BLADE_STEAK_1,CUDA_BLADE_STEAK_2,BLADE_E,BLADE_F,BLADE_G};

		std::vector< MyDenseVector> m_vecBladePlaneNormal;

		getNormal4Plane(&tmpElement[0],0,m_vecBladePlaneNormal);
		MyFloat* bladeNormal = new MyFloat[6];
		bladeNormal[0] = m_vecBladePlaneNormal[0][0];
		bladeNormal[1] = m_vecBladePlaneNormal[0][1];
		bladeNormal[2] = m_vecBladePlaneNormal[0][2];
		bladeNormal[3] = m_vecBladePlaneNormal[1][0];
		bladeNormal[4] = m_vecBladePlaneNormal[1][1];
		bladeNormal[5] = m_vecBladePlaneNormal[1][2];
		initBlade(&tmpElement[0],sizeof(tmpElement),bladeNormal);

		VR_FEM::MyPoint pt0(BLADE_A);
		VR_FEM::MyPoint pt1(BLADE_B);
		VR_FEM::MyPoint pt2(BLADE_C);
		makeBladeToMultiTriangle(pt0,pt1,pt2,5);
	}

	class PointCompare
	{
	public:
		PointCompare(MyFloat x,MyFloat y,MyFloat z):m_point(x,y,z){}
		PointCompare(const MyPoint& p):m_point(p){}
		bool operator()(MyPoint& p)
		{
			return  (numbers::isZero(m_point(0) - p(0))) && 
				(numbers::isZero(m_point(1) - p(1))) && 
				(numbers::isZero(m_point(2) - p(2)));
		}
	private:
		MyDenseVector m_point;
	};

	struct BladeTriangle
	{
		BladeTriangle(const Vector3i& t,int l):tri(t),nLevel(l){}
		Vector3i tri;
		int nLevel;
	};

	int getPointId(std::vector< MyPoint >& pointSet,const MyPoint& p01)
	{		
		std::vector< MyPoint >::const_iterator itr;
		itr = std::find_if(pointSet.begin(),pointSet.end(),PointCompare(p01));

		if (itr == pointSet.end())
		{
			pointSet.push_back(p01);
			return pointSet.size()-1;
		}
		else
		{
			return itr - pointSet.begin();
		}
	}

	void MeshGenerate::makeBladeToMultiTriangle(const MyPoint& pt0,const MyPoint& pt1,const MyPoint& pt2,const int nMaxLevel)
	{
		
		std::vector< MyPoint > pointSet;
		pointSet.push_back(pt0);
		pointSet.push_back(pt1);
		pointSet.push_back(pt2);
		std::vector< Vector3i > triSet;
		std::queue< BladeTriangle > q;
		q.push(BladeTriangle(Vector3i(0,1,2),0));
		triSet.push_back(Vector3i(0,1,2));
		while (!q.empty())
		{
			BladeTriangle curNode = q.front();
			q.pop();
			if (curNode.nLevel<nMaxLevel)
			{
				const MyPoint& p0 = pointSet[curNode.tri[0]];
				const MyPoint& p1 = pointSet[curNode.tri[1]];
				const MyPoint& p2 = pointSet[curNode.tri[2]];

				MyPoint p01 = (p0 + p1) / 2.0f;
				MyPoint p12 = (p1 + p2) / 2.0f;
				MyPoint p20 = (p2 + p0) / 2.0f;

				int p01_Id=-1,p12_Id=-1,p20_Id=-1;
				int p0_Id = curNode.tri[0],p1_Id = curNode.tri[1],p2_Id = curNode.tri[2];

				p01_Id = getPointId(pointSet,p01);
				p12_Id = getPointId(pointSet,p12);
				p20_Id = getPointId(pointSet,p20);

				q.push(BladeTriangle(Vector3i(p01_Id,p1_Id,p12_Id),curNode.nLevel+1));

				p01_Id = getPointId(pointSet,p01);
				p12_Id = getPointId(pointSet,p12);
				p20_Id = getPointId(pointSet,p20);
				q.push(BladeTriangle(Vector3i(p0_Id,p01_Id,p20_Id),curNode.nLevel+1));

				p01_Id = getPointId(pointSet,p01);
				p12_Id = getPointId(pointSet,p12);
				p20_Id = getPointId(pointSet,p20);
				q.push(BladeTriangle(Vector3i(p01_Id,p12_Id,p20_Id),curNode.nLevel+1));

				p01_Id = getPointId(pointSet,p01);
				p12_Id = getPointId(pointSet,p12);
				p20_Id = getPointId(pointSet,p20);
				q.push(BladeTriangle(Vector3i(p20_Id,p12_Id,p2_Id),curNode.nLevel+1));
			}
			else
			{
				triSet.push_back(curNode.tri);
			}
		}

		///////////////////////////////////////////////////////
		int nMaxVertexCount = 0;
		int nMaxLineCount = 0;
		int nMaxFaceCount = 0;
		//const int nMaxVertexSize = 2*vertices.size();
		std::vector< MyPoint >& vertices = pointSet;
		const int nVertexSize = vertices.size();

		MC_Vertex_Cuda* tmp_Vertex = new MC_Vertex_Cuda[nVertexSize];
		memset(tmp_Vertex,'\0',nVertexSize * sizeof(MC_Vertex_Cuda));
		for (unsigned v=0;v<nVertexSize;++v)
		{
			MC_Vertex_Cuda& ptV = tmp_Vertex[v];
			ptV.m_isValid = false;//£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡
			ptV.m_isJoint = true;ptV.m_isSplit = false;ptV.m_nVertexId = v;
			ptV.m_VertexPos[0] = vertices[v][0];
			ptV.m_VertexPos[1] = vertices[v][1];
			ptV.m_VertexPos[2] = vertices[v][2];
			ptV.m_CloneVertexIdx[0]=ptV.m_CloneVertexIdx[1]=Invalid_Id;
			ptV.m_distanceToBlade = 0.f;
			ptV.m_state = 0;
			ptV.m_MeshVertex2CellId = Invalid_Id;
		}

		std::vector< Vector3i >& face_Position_indicies = triSet;
		m_map_lineSet.clear();
		for (unsigned s=0,lineId=0;s<face_Position_indicies.size();++s)
		{
			const Vector3i& refVec3 = face_Position_indicies[s];
			std::pair<int,int> curPair = std::make_pair(refVec3[0],refVec3[1]);
			if (m_map_lineSet.find(curPair) == m_map_lineSet.end())
			{
				std::pair<int,int> curPair = std::make_pair(refVec3[1],refVec3[0]);
				if (m_map_lineSet.find(curPair) == m_map_lineSet.end())
				{
					m_map_lineSet[curPair] = lineId;
					lineId++;
				}
			}
			else
			{
				
			}

			curPair = std::make_pair(refVec3[1],refVec3[2]);
			if (m_map_lineSet.find(curPair) == m_map_lineSet.end())
			{
				curPair = std::make_pair(refVec3[2],refVec3[1]);
				if (m_map_lineSet.find(curPair) == m_map_lineSet.end())
				{
					m_map_lineSet[curPair] = lineId;
					lineId++;
				}
			}

			curPair = std::make_pair(refVec3[2],refVec3[0]);
			if (m_map_lineSet.find(curPair) == m_map_lineSet.end())
			{
				curPair = std::make_pair(refVec3[0],refVec3[2]);
				if (m_map_lineSet.find(curPair) == m_map_lineSet.end())
				{
					m_map_lineSet[curPair] = lineId;
					lineId++;
				}
			}
		}

		std::vector< Vector3i> triLine;
		int linePair[3][2] = {{1,2},{2,0},{0,1}};
		for (unsigned t=0;t<face_Position_indicies.size();++t)
		{
			const Vector3i& refVec3 = face_Position_indicies[t];
			Vector3i curTriLineOrder;
			//1,2  2,0  0,1
			for (unsigned l=0;l<3;++l)
			{
				std::pair<int,int> line_0 = std::make_pair(refVec3[linePair[l][0]],refVec3[linePair[l][1]]);
				std::pair<int,int> line_1 = std::make_pair(refVec3[linePair[l][1]],refVec3[linePair[l][0]]);
				if (m_map_lineSet.find(line_0) != m_map_lineSet.end())
				{
					curTriLineOrder[l] = m_map_lineSet.at(line_0);
				}
				else if (m_map_lineSet.find(line_1) != m_map_lineSet.end())
				{
					curTriLineOrder[l] = m_map_lineSet.at(line_1);
				}
				else
				{
					printf("error 1435\n");
					MyPause;
				}
				
			}
			triLine.push_back(curTriLineOrder);
		}
		Q_ASSERT(triLine.size() == face_Position_indicies.size());


		const int nEdgeSize = m_map_lineSet.size();
		MC_Edge_Cuda* tmp_Edge = new MC_Edge_Cuda[nEdgeSize];
		memset(tmp_Edge,'\0',nEdgeSize * sizeof(MC_Edge_Cuda));

		std::map< std::pair<int,int>,int >::const_iterator ci = m_map_lineSet.begin();
		std::map< std::pair<int,int>,int >::const_iterator endc = m_map_lineSet.end();
		for (;ci != endc;++ci)
		{
			const int l = ci->second;
			const std::pair<int,int>& lineSet = ci->first;

			MC_Edge_Cuda& curEdge = tmp_Edge[l];

			curEdge.m_hasClone = false;curEdge.m_isValid=true;curEdge.m_isJoint=true;curEdge.m_isCut=false;
			curEdge.m_state=0;curEdge.m_nLineId = l;curEdge.m_Vertex[0] = lineSet.first;curEdge.m_Vertex[1] = lineSet.second;
			curEdge.m_belongToTri[0] = curEdge.m_belongToTri[1] = curEdge.m_belongToTri[2] = Invalid_Id;
			curEdge.m_CloneIntersectVertexIdx[0] = curEdge.m_CloneIntersectVertexIdx[1] = Invalid_Id;
			curEdge.m_CloneEdgeIdx[0] = curEdge.m_CloneEdgeIdx[1] = Invalid_Id;
		}

		const int nFaceSize = triLine.size();
		MC_Surface_Cuda* tmp_Surface = new MC_Surface_Cuda[nFaceSize];
		memset(tmp_Surface,'\0',nFaceSize*sizeof(MC_Surface_Cuda));
		std::vector< Vector3i >& tri = face_Position_indicies;
		std::vector< Vector3i >& triVN = face_VertexNormal_indicies;
		for (unsigned s=0;s<nFaceSize;++s)
		{
			MC_Surface_Cuda& curFace = tmp_Surface[s];
			curFace.m_isValid = false/*£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡£¡*/;
			curFace.m_isJoint = true;curFace.m_nSurfaceId=s;
			curFace.m_Vertex[0] = tri[s][0];curFace.m_Vertex[1] = tri[s][1];curFace.m_Vertex[2] = tri[s][2];
			curFace.m_Lines[0] = triLine[s][0];curFace.m_Lines[1] = triLine[s][1];curFace.m_Lines[2] = triLine[s][2];
			curFace.m_VertexNormal[0] = 0;curFace.m_VertexNormal[1] = 0;curFace.m_VertexNormal[2] = 0;

			float* p0 = &tmp_Vertex[curFace.m_Vertex[0]].m_VertexPos[0];
			float* p1 = &tmp_Vertex[curFace.m_Vertex[1]].m_VertexPos[0];
			float* p2 = &tmp_Vertex[curFace.m_Vertex[2]].m_VertexPos[0];

			{
				MyPoint center;
				MyFloat radius;
				Circumcircle(MyPoint(p0[0],p0[1],p0[2]),
							 MyPoint(p1[0],p1[1],p1[2]),
							 MyPoint(p2[0],p2[1],p2[2]),
							 center,
							 radius);
				/*SetCircumCircle(p0.m_VertexPos[0],p0.m_VertexPos[1],p1.m_VertexPos[0],p1.m_VertexPos[1],p2.m_VertexPos[0],p2.m_VertexPos[1],
					vSuper.m_center[0],vSuper.m_center[1],vSuper.m_R,vSuper.m_R2);*/
				curFace.m_center[0] = center[0];
				curFace.m_center[1] = center[1];
				curFace.m_center[2] = center[2];
				curFace.m_R = radius;
				curFace.m_R2 = radius*radius;
			}
			/*curFace.m_center[0] = (p0[0] + p1[0] + p2[0]) / 3.f;
			curFace.m_center[1] = (p0[1] + p1[1] + p2[1]) / 3.f;
			curFace.m_center[2] = (p0[2] + p1[2] + p2[2]) / 3.f;*/
			curFace.m_state = 0;
			curFace.m_nParentId4MC = -1;

			for (unsigned l=0;l<3;++l)
			{
				const unsigned lineId = curFace.m_Lines[l];

				int order[2];
				for (unsigned v=0;v<3;++v)
				{
					if (curFace.m_Vertex[v] == tmp_Edge[lineId].m_Vertex[0])
					{
						order[0] = v;
					}
					if (curFace.m_Vertex[v] == tmp_Edge[lineId].m_Vertex[1])
					{
						order[1] = v;
					}
				}

				tmp_Edge[lineId].m_belongToTri[l] = s;
				tmp_Edge[lineId].m_belongToTriVertexIdx[l][0] = order[0];
				tmp_Edge[lineId].m_belongToTriVertexIdx[l][1] = order[1];
			}
		}
		if (1025 != nFaceSize)
		{
			printf("vertex(%d) edge(%d) face(%d)\n",nVertexSize,nEdgeSize,nFaceSize);
			MyPause;
		}
		printf("vertex(%d) edge(%d) face(%d)\n",nVertexSize,nEdgeSize,nFaceSize);
		//MyPause;
#if 0
		m_vecTest_Vertex.clear();
		for (unsigned v=0;v<nVertexSize;++v)
		{
			m_vecTest_Vertex.push_back(tmp_Vertex[v]);
		}
		m_vecTest_Edge.clear();
		for (unsigned e=0;e<nEdgeSize;++e)
		{
			m_vecTest_Edge.push_back(tmp_Edge[e]);
		}
		m_vecTest_Surface.clear();
		for (unsigned t=0;t<nFaceSize;++t)
		{
			m_vecTest_Surface.push_back(tmp_Surface[t]);
		}
#else
		initBlade_MeshCutting(nVertexSize,nVertexSize,nEdgeSize,nEdgeSize,nFaceSize,nFaceSize,tmp_Vertex,tmp_Edge,tmp_Surface,tmp_Vertex,tmp_Edge,tmp_Surface);
		//getCurrentGPUMemoryInfo();
#endif
		/*getCurrentGPUMemoryInfo();
		MyPause;*/
		delete [] tmp_Vertex;
		delete [] tmp_Edge;
		delete [] tmp_Surface;
	}
}