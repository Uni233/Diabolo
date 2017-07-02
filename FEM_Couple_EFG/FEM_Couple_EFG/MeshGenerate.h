#ifndef _MESHGENERATE_H
#define _MESHGENERATE_H

#include "VR_Global_Define.h"
#include "plane.h"
#include "Cell.h"
#include "TripletNode.h"
#include <map>
#include "triangleMeshStruct.h"
#include "CellToTriangleMeshVertice.h"
#include "MeshCuttingStructure.h"
#include "Delaunay/Delaunay.h"
#include "MeshCuttingStructureOnCuda.h"

namespace VR_FEM
{
	class MeshGenerate
	{
		
	public:
		MeshGenerate(MyFloat cellRadius, const int X_Count, const int Y_Count, const int Z_Count, std::vector< Plane >& vecPlanes);
		void generate();
		void generate_steak(const char* lpszSteakFile,const char* lpszTexturePath);
		void solve_timestep(unsigned nTimeStep);
		void print();
		void printCuttingCell();
	private:
		void distributeDof();
		void create_boundary();
		void create_boundary_steak();
		void apply_boundary_values();
		void createGlobalMassAndStiffnessAndDampingMatrix();
		void createNewMarkMatrix();
		void setMatrixRowZeroWithoutDiag(MySpMat& matrix, const int  rowIdx );
		
		void update_rhs(int nStep);
		void update_u_v_a();
		void solve_linear_problem();

		void distribute_local_to_global (const MyMatrix& local_matrix, const std::vector< int>& local_dof_indices, const VR_FEM::MySpMat&);
		void distribute_local_to_global (const MyVector& local_vector,const std::vector<int> &local_dof_indices, MyVector& global_vector) const;
		bool is_constrained(int)const{return false;}
		void make_sorted_row_list (const std::vector< int> &local_dof_indices,GlobalRowsFromLocal  &global_rows) const;
		void  set_matrix_diagonals (const GlobalRowsFromLocal &global_rows,
			const std::vector< int>              &local_dof_indices,
			const MyDenseMatrix                       &local_matrix,
			const VR_FEM::MySpMat                        &global_matrix);
		void   resolve_matrix_row (const GlobalRowsFromLocal&global_rows,
			const unsigned int        i,/*1-24*/
			const unsigned int        column_start/*0*/,
			const unsigned int        column_end/*24*/,
			const MyDenseMatrix           &local_matrix,
			const MySpMat                    &sparse_matrix);

		void create_mass_matrix(VR_FEM::MySpMat &matrix);
		void copy_local_to_global(const std::vector< int> &indices,const MyDenseMatrix  &vals,const VR_FEM::MySpMat &matrix);
		std::map<long,std::map<long,TripletNode > > m_TripletNode;

		void compareMatrix(MySpMat& leftMatrix,MySpMat& rightMatrix);

	private:
		void createGlobalMassAndStiffnessAndDampingMatrixEFG();
		void ApproxAtPoint(const MyPoint& evPoint, unsigned& nInfluncePoint, std::vector<VertexPtr>& vecInflunceVertex, MyVector& vecShapeFunction );
		unsigned InfluentPoints(const MyPoint& gaussPointInGlobalCoordinate,std::vector<VertexPtr>& vecInflunceVertex );
	private:
		MyFloat m_radius;
		MyVectorI m_axis_count;
		std::vector< CellPtr > m_vec_cell;
		std::vector< Plane > m_vecPlanes;
		unsigned int m_nDof;
		std::vector< unsigned > m_vec_boundary,m_vecForceBoundary;
		std::vector< VertexPtr > m_vec_boundaryVertex;
		MySpMat m_computeMatrix,m_global_MassMatrix,m_global_StiffnessMatrix,m_global_DampingMatrix;
		MyVector m_computeRhs;
		MyFloat m_db_NewMarkConstant[8];

		MyVector R_rhs,mass_rhs,damping_rhs,displacement,velocity,acceleration,old_acceleration,old_displacement;
		MyVector incremental_displacement;
		CellType m_BoundaryType;
		
	private:
		void assembleRotationSystemMatrix();
		MyVector m_RotationRHS;

	public:
		void solve_timestep_steak(unsigned nStep);
		void print_steak();
	private:///////////////////////////////   CUDA Structure  ////////////////////////
		bool LoadGLTextures(const char* lpszTexturePath);
		void loadObjDataSteak(const char* lpszFilePath);
		void makeScalarObjDataSteak();
		void loadObjDataBunny(const char* lpszFilePath);
		void makeLineSet_Steak();
		void makeTriangleMeshInterpolation_Steak();
		MyFloat max3(	MyFloat a,MyFloat b,MyFloat c);
		void assembleOnCuda();
		void initVBOStructure();
		void initVBOScene(const int nTrianglCount, const int nLineCount);
		void freeVBOScene();
		void makeCudaMemory_Steak(int ** line_vertex_pair,int& lineCount,int ** vertex_dofs,float** vertex_pos,int& vertexCount);
		void freeCudaMemory(int ** line_vertex_pair,int ** vertex_dofs,float** vertex_pos);
		void makeTriangleMeshMemory(int& ,int& ,int& ,int& ,MyFloat ** ,int& ,MyFloat ** ,int & ,int ** ,int & ,MyFloat ** ,int & ,int ** ,int & ,int ** ,int & );
		void freeTriangleMeshMemory(MyFloat ** ,MyFloat ** ,int ** ,MyFloat ** ,int ** ,int  ** );
		std::map< int, std::map<int,bool> > m_map_lineId4Steak;
		static void createVBO(GLuint* vbo, unsigned int typeSize,unsigned nCount);
		static void deleteVBO(GLuint* vbo);
		static void cleanupCuda();
		static GLuint vbo_triangles;
		static GLuint vbo_lines;
		static GLuint vbo_triangles_color;
		static GLuint vbo_triangles_vertexNormal;
		static GLuint m_nVBOTexCoords;
		int m_nLineCount_In_map_lineID;

		std::vector< MyDenseVector > vertices;
		std::vector< MyDenseVector > verticeNormals;
		std::vector< std::pair<MyFloat,MyFloat> > verticeTexcood;
		std::vector< Vector3i > face_Position_indicies;
		std::vector< Vector3i > face_Texcood_indicies;
		std::vector< Vector3i > face_VertexNormal_indicies;
		MyFloat m_xMin,m_xMax,m_yMin,m_yMax,m_zMin,m_zMax;
		MyFloat m_maxDiameter,m_translation_x,m_translation_y,m_translation_z;

		GLuint texture_steak;
		std::vector<TriangleMeshNode > m_vecVertexIdx2NodeIdxInside;
		std::vector< CellToTriangleMeshVertice > m_vecCellToTriangleMeshVertice;

	private:
		void assembleFEMonCPU();
		void assembleFEMPreComputeMatrix();
		void initCuttingStructure();

		/************************************************************************/
		/* Mesh Cutting                                                         */
		/************************************************************************/
	public:

		std::vector<  MC_Vertex >  m_vec_MCVertex;
		std::vector< MC_Edge > m_vec_MCEdge;
		std::vector< MC_Surface > m_vec_MCSurface;
		std::map< std::pair<int,int>,int > m_map_lineSet;
		int m_nMCVertexCount;
		int m_nMCEdgeCount;
		int m_nMCSurfaceCount;

		int m_nLastVertexIdx;
		int m_nLastEdgeIdx;
		int m_nLastSurfaceIdx;
		int m_nNewAddVertex;

		MyFloat m_dbDistance;
		//tool set
		bool CheckLineTri( const MyPoint &L1, const MyPoint &L2, const MyPoint &PV1, const MyPoint &PV2, const MyPoint &PV3, MyPoint &HitP );
		MyFloat computePt2LineDistance(const MyPoint& pt,const MyPoint& linePtA, const MyPoint& linePtB);
		MyDenseVector computeSurfaceNormal(const MyPoint& A,const MyPoint& B,const MyPoint& C);
		MyFloat computeTriangleArea(const MyPoint& A,const MyPoint& B,const MyPoint& C);
		bool isPointInLine(const MyPoint& pt,const MyPoint& linePtA, const MyPoint& linePtB);
		bool isPointInTri(const MyPoint& pt,const MyPoint& TriPtA, const MyPoint& TriPtB, const MyPoint& TriPtC);
		int computePt2Blade(const MyPoint& pt,const MyPoint& A,const MyPoint& B,const MyPoint& C);
		MyFloat checkPointPlane(const MyPoint& pt,const MyPoint& A,const MyPoint& B,const MyPoint& C,const MyDenseVector& normal)const;
		MyPoint makeScalarPoint(const MyPoint& src, const MyPoint& dst,MyFloat scalarFactor=0.9123f);
		//function
		void testBladeValid(MyFloat ptA0,MyFloat ptA1,MyFloat ptA2,MyFloat ptB0,MyFloat ptB1,MyFloat ptB2,MyFloat ptC0,MyFloat ptC1,MyFloat ptC2);
		void makeVertex2BladeRelationShip(MyFloat ptA0,MyFloat ptA1,MyFloat ptA2,MyFloat ptB0,MyFloat ptB1,MyFloat ptB2,MyFloat ptC0,MyFloat ptC1,MyFloat ptC2);
		void makeSplitEdge();
		void spliteLine();
		void spliteTriangle();
		void spliteVertex();
		void spliteTriangleByVertex();
		void computeTriangleCenter();
		void makeMesh();
		void printMesh();
		void printCuttedLine();
		void printCuttedFace();
		void printVertex();
		void printLines();
		void cuttingMesh();
		void printFace();
		void printNewFace();
		void makeMesh_Steak(const char * objPath);
		void computeEdgeUpDown(MyFloat ptA0,MyFloat ptA1,MyFloat ptA2,MyFloat ptB0,MyFloat ptB1,MyFloat ptB2,MyFloat ptC0,MyFloat ptC1,MyFloat ptC2);
		void computeTriCenterUpDown(MyFloat ptA0,MyFloat ptA1,MyFloat ptA2,MyFloat ptB0,MyFloat ptB1,MyFloat ptB2,MyFloat ptC0,MyFloat ptC1,MyFloat ptC2);
		int m_nMaxVertexCount;
		int m_nMaxLineCount;
		int m_nMaxFaceCount;
		std::vector< MC_Surface > m_vec_MCSurface_new;

	public:
		int m_nVertices;
		vertexSet m_Vertices;
		triangleSet m_Triangles;
		Delaunay m_Delaunay;
		void makeDelaunay();
		void printDelaunay();

		void cuttingMesh_Steak();
		void makeMeshStructureOnCuda_Steak();
		
		int m_nTestCudaVertexSize;
		int m_nTestCudaEdgeSize;
		int m_nTestCudaSurfaceSize;
		std::vector< MC_Surface_Cuda > m_vecTest_Surface;
		std::vector< MC_Edge_Cuda > m_vecTest_Edge;
		std::vector< MC_Vertex_Cuda > m_vecTest_Vertex;
		void printTestFace();

		void makeBladeToMultiTriangle(const MyPoint& pt0,const MyPoint& pt1,const MyPoint& pt2,const int nLevel);
		
#if 1
		std::vector< MyPoint > m_vec_VertexAngle;
		void makeDelaunayMySelf();
		void printDelaunayMySelf();
		std::vector<  MC_Vertex_Cuda >  m_vec_CudaMCVertex;
		std::vector< MC_Edge_Cuda > m_vec_CudaMCEdge;
		std::vector< MC_Surface_Cuda > m_vec_CudaMCSurface;
		int m_nVertexSize;
		int m_nEdgeSize;
		int m_nTriangleSize;
		void Circumcircle(const MyDenseVector& A,const MyDenseVector& B,const MyDenseVector& C,MyDenseVector& Center,MyFloat& radius);
#endif
	};
}
#endif//_MESHGENERATE_H