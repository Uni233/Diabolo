#ifndef _vrBEM3D_H_
#define _vrBEM3D_H_
#include "vrGeometry/VR_Geometry_MeshDataStruct.h"
#include "bemDefines.h"
#include "bemTriangleElem.h"
#include "VDBWrapper.h"
#include<iostream>
#include<algorithm>
#include<iterator>
#include <fstream>
#include <set>
#include "PostProcessor.h"
using namespace std;
namespace VR
{
	class vrBEM3D
	{
	public:
		typedef TriangleElemData_DisContinuous TriangleElemData;
	public:
		vrBEM3D(const MyFloat _E /*= 1e5*/, const MyFloat _mu /*= 0.3*/,
			const MyFloat _shearMod /*= E / (2 * (1 + mu))*/);
		~vrBEM3D();
	public:
		int initPhysicalSystem(vrLpsz lpszObjName, vrFloat resolution);
		void renderScene()const;
		MyInt getDofs()const{ return m_nGlobalDof; }
	private:
		void distributeDof3d();
		void createForceBoundaryCondition3d();
		bool isDCCondition(const MyVec3& pos);
		//bool isForceCondition(const MyVec3& pos);
		bool isForceCondition_up(const MyVec3& pos);
		bool isForceCondition_down(const MyVec3& pos);
		void createGMatrixAndHMatrixBEM3d();
		void makeRigidH();
		MyVector GaussElimination(const MyMatrix& K, MyVector& b);
		bool isVertexInElement(const VertexPtr curVtxPtr, const TriangleElemPtr curElementPtr, vrInt& index);
		bool isOutofRange(const vrFloat val)const;
#if USE_Mantic_CMat
	public:
		void createGMatrixAndHMatrixBEM3d_SST();
	private:
#if USE_TBB
		void Parallel_AssembleSystem();
#if USE_DUAL
		void Parallel_AssembleSystem_DualEquation();
#endif//USE_DUAL
	public:
#if USE_Aliabadi
		
		static vrFloat compute_K_ij_I_Aliabadi(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D, const vrInt idx_i, const vrInt idx_j, const  vrInt idx_I, const vrInt idx_k);
		
		static vrFloat compute_S_ij_I_Aliabadi(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D, const vrInt idx_i, const vrInt idx_j, const  vrInt idx_I, const vrInt idx_k);
		
		static vrFloat compute_T_ij_I_Aliabadi(const VertexPtr curSourcePtr, const TriangleElemData& curTriElemData,vrInt idx_i,vrInt idx_j,vrInt idx_I);
		static vrFloat compute_T_ij_I_SST_Aliabadi(const VertexPtr curSourcePtr, const TriangleElemData& curTriElemData,vrInt idx_i,vrInt idx_j,vrInt idx_I);
		static vrFloat compute_U_ij_I_Aliabadi(const VertexPtr curSourcePtr, const TriangleElemData& curTriElemData,vrInt idx_i,vrInt idx_j,vrInt idx_I);
		static vrFloat compute_U_ij_I_SST_Aliabadi(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,vrInt idx_i,vrInt idx_j,vrInt idx_I);
#if USE_360_Sample
		static vrFloat compute_U_ij_I_SST_DisContinuous_Regular_Aliabadi(const VertexPtr curSourcePtr, const TriangleElemData& curTriElemData,vrInt idx_i,vrInt idx_j,vrInt idx_I);
		static vrFloat compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(const VertexPtr curSourcePtr, const TriangleElemData& curTriElemData,vrInt idx_i,vrInt idx_j,vrInt idx_I);
		static vrFloat compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,const vrInt idx_i,const vrInt idx_j,const vrInt idx_I,const vrInt idx_k);
		static vrFloat compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,const vrInt idx_i,const vrInt idx_j,const vrInt idx_I,const vrInt idx_k);
#else//USE_360_Sample

#if USE_Aliabadi_RegularSample
		static vrFloat compute_U_ij_I_SST_DisContinuous_Regular_Aliabadi(const vrInt nSubTriIdx, const VertexPtr curSourcePtr, const TriangleElemData& curTriElemData,vrInt idx_i,vrInt idx_j,vrInt idx_I);
		static vrFloat compute_T_ij_I_SST_DisContinuous_Regular_Aliabadi(const vrInt nSubTriIdx, const VertexPtr curSourcePtr, const TriangleElemData& curTriElemData,vrInt idx_i,vrInt idx_j,vrInt idx_I);
		static vrFloat compute_K_ij_I_SST_DisContinuous_Regular_Aliabadi(const vrInt nSubTriIdx, const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,const vrInt idx_i,const vrInt idx_j,const vrInt idx_I,const vrInt idx_k);
		static vrFloat compute_S_ij_I_SST_DisContinuous_Regular_Aliabadi(const vrInt nSubTriIdx, const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,const vrInt idx_i,const vrInt idx_j,const vrInt idx_I,const vrInt idx_k);
#endif//USE_Aliabadi_RegularSample

#endif//USE_360_Sample

		static vrFloat get_Sij_SST_3D_k_Aliabadi(vrInt idx_i,vrInt idx_j,vrInt idx_k,vrFloat r,const MyVector& dr, const vrFloat drdn, const MyVec3& unitNormal_fieldPoint);
		static vrFloat get_Kij_SST_3D_k_Aliabadi(vrInt idx_i,vrInt idx_j,vrInt idx_k,vrFloat r,const MyVector& dr, const vrFloat drdn);

		static vrFloat get_Uij_SST_3D_Aliabadi(vrInt idx_i,vrInt idx_j,vrFloat r,const MyVector& dr);
		static vrFloat get_Tij_SST_3D_Aliabadi(vrInt idx_i,vrInt idx_j,vrFloat r,const MyVector& dr, const vrFloat drdn, const MyVec3& normal_fieldPoint);

		void AssembleSystem_DisContinuous_DualEquation_Aliabadi(const vrInt v, const vrInt idx_i, const vrInt idx_j);
#if USE_UniformSampling
		void AssembleSystem_DisContinuous_DualEquation_Aliabadi_Nouse(const vrInt v, const vrInt idx_i, const vrInt idx_j);
		void AssembleSystem_DisContinuous_DualEquation_Aliabadi_Peng(const vrInt v, const vrInt idx_i, const vrInt idx_j);
		void DualEquation_Regular_Peng(const vrInt idx_i, const vrInt idx_j, const int ne, const vrFloat Cij, VertexPtr curSourcePtr);
		void DualEquation_Positive_Peng(const vrInt idx_i, const vrInt idx_j, const int ne, const vrFloat Cij, VertexPtr curSourcePtr);
		void DualEquation_Negative_Peng(const vrInt idx_i, const vrInt idx_j, const int ne, const vrFloat Cij, VertexPtr curSourcePtr);
		static vrFloat compute_S_ij_I_Peng(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D, const vrInt idx_i, const vrInt idx_j, const  vrInt idx_I);
		static vrFloat compute_K_ij_I_Peng(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D, const vrInt idx_i, const vrInt idx_j, const  vrInt idx_I);
		static vrFloat get_Kij_SST_3D_k_Aliabadi_Peng(vrInt idx_i,vrInt idx_j,vrInt idx_k,vrFloat r,const MyVector& dr, const vrFloat drdn, const MyVec3& n_s);
		static vrFloat get_Sij_SST_3D_k_Aliabadi_Peng(vrInt idx_i,vrInt idx_j,vrInt idx_k,vrFloat r,const MyVector& dr, const vrFloat drdn, const MyVec3& n_x, const MyVec3& n_s);
		static vrFloat compute_S_ij_I_SST_DisContinuous_Peng(const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,const vrInt idx_i,const vrInt idx_j,const vrInt idx_I);
#endif//USE_UniformSampling
		
#endif//USE_Aliabadi
		
#if USE_DUAL
		
		vrFloat compute_K_ij_I_k_SST(const VertexPtr curSourcePtr, const TriangleElemData& curTriElemData, const vrInt idx_i, const vrInt idx_j, const  vrInt idx_I);
		vrFloat compute_K_ij_I(const VertexPtr curSourcePtr, const TriangleElemData& curTriElemData, const vrInt idx_i, const vrInt idx_j, const  vrInt idx_I);
		

		vrFloat compute_H_ij_I_k_SST(const VertexPtr curSourcePtr, const TriangleElemData& curTriElemData, const vrInt idx_i, const vrInt idx_j, const  vrInt idx_I);
		vrFloat compute_S_ij_I_k_SST(const VertexPtr curSourcePtr, const TriangleElemData& curTriElemData, const vrInt idx_i, const vrInt idx_j, const  vrInt idx_I);
		vrFloat compute_H_ij_I(const VertexPtr curSourcePtr, const TriangleElemData& curTriElemData, const vrInt idx_i, const vrInt idx_j, const  vrInt idx_I);
		vrFloat compute_S_ij_I(const VertexPtr curSourcePtr, const TriangleElemData& curTriElemData, const vrInt idx_i, const vrInt idx_j, const  vrInt idx_I);
		
#endif//USE_DUAL
	private:
#endif//USE_TBB
		
		
#if USE_Sigmoidal && USE_Aliabadi_RegularSample
		
		static vrFloat compute_T_ij_I_SST_DisContinuous_Sigmoidal(const vrInt nSubTriIdx, const VertexPtr curSourcePtr, const TriangleElemData& curTriElemData,vrInt idx_i,vrInt idx_j,vrInt idx_I);
		static vrFloat compute_S_ij_I_SST_DisContinuous_Sigmoidal(const vrInt nSubTriIdx, const VertexPtr curSourcePtr, const TriangleElemData& curTriElemData,vrInt idx_i,vrInt idx_j,vrInt idx_I);
		static vrFloat compute_K_ij_I_SST_DisContinuous_Sigmoidal(const vrInt nSubTriIdx, const VertexPtr curSourcePtr, const TriangleElemData& curTriElemData,vrInt idx_i,vrInt idx_j,vrInt idx_I);
#endif//USE_Sigmoidal
		
		static vrFloat compute_U_ij_I_DisContinuous(const vrInt nSubTriIdx, const VertexPtr curSourcePtr, const TriangleElemData& refDataSST3D,vrInt idx_i,vrInt idx_j,vrInt idx_I);
		static void getKernelParameters_3D(const MyVec3& srcPos, const MyVec3& fieldPoint,const TriangleElemData& curTriElemData,
			MyFloat& jacob_xi,MyVec3& normals_fieldpoint,MyFloat& r,MyVec3& dr,MyFloat& drdn);
#if DEBUG_3_3
		static void getKernelParameters_3D_SST(const MyVec3& srcPos, const MyVec3& fieldPoint,const TriangleElemData& curTriElemData,
			MyFloat& jacob_eta,MyVec3& normals_fieldpoint,MyFloat& r,MyVec3& dr,MyFloat& drdn);
#endif
		static void getKernelParameters_3D_SST_SubTri(const MyVec3& srcPos, const MyVec3& fieldPoint,const TriangleElemData& curTriElemData,
			MyFloat& jacob_eta,MyVec3& normals_fieldpoint,MyFloat& r,MyVec3& dr,MyFloat& drdn);
		
		vrInt sgn(vrFloat val);
		vrMat3 tensor_product(const MyVec3 vec_a, const MyVec3& vec_b);
		void sortVertexConnectedVertexSurface();

		void HG2A(const vrInt nDofs);
		void resizeSystemMatrix(const vrInt nDofs);
		void applyBoundaryCondition(const vrInt nDofs);
		void applyForceCondition(const vrInt nDofs, const vrFloat scale = 1.0);
		void solve();
		void onlyUpdateExternalForceAndSolve();


		void compute_Guiggiani_CMatrix_for_Test();
		MyInt index2to1(const MyInt idx0, const MyInt idx1, const MyInt sum = Geometry::vertexs_per_tri)
		{
			MyInt nRet = sum - idx0 - idx1;
			Q_ASSERT(nRet >= 0 && nRet < Geometry::vertexs_per_tri);
			Q_ASSERT(idx0 != idx1);
			return sum - idx0 - idx1;
		}
#endif
	private:
		Geometry::MeshDataStructPtr m_ObjMesh_ptr;
		MyInt m_nGlobalDof;
		//MyVec3 m_trace3d;

		//std::vector< TriangleElemPtr > m_vec_triElem;
#if USE_Mantic_CMat
		std::vector< VertexPtr > m_vec_vertex_boundary;
#endif
		
#if USE_VDB
		std::map< vrInt/*region ID*/, std::vector< VertexPtr > > map_regionId_vecVertex;
#else
		std::vector< VertexPtr > m_vec_vertex3d_trace[2];
#endif
		

		MyMatrix m_Hsubmatrix, m_Gsubmatrix, m_A_matrix;
		MyVector m_rhs, m_displacement;
	private:
		static MyFloat E;
		static MyFloat mu;
		static MyFloat shearMod;
		

	private:
		std::string m_strMeshFile;//"D:/myDualBEM3D/Release/mesh/BUAA_BEAM_bunnypos"
		vrString m_str_currentDateTime;

#if USE_Fracture
	public:
		
		
		
		//typedef std::map<unsigned int, CRACK_STATE> state_map;
		//typedef std::map<unsigned int, Eigen::Vector3d> vect3d_map;
	private:
		vrFloat crackMeshSize;
		bool bemInitialized;
		bool vdbInitialized;
		bool fractureInitialized;
		vrString lastVDBfile;

		
		std::string m_strNodeFile;//"D:/myDualBEM3D/Release/mesh/BUAA_BEAM_bunnypos.nodes"
		std::string m_strElemFile;//"D:/myDualBEM3D/Release/mesh/BUAA_BEAM_bunnypos.elements"
		std::string m_strRegionFile;//"D:/myDualBEM3D/Release/mesh/BUAA_BEAM_bunnypos.regions"

		elemMap m_reader_Elems;/* sim.getElems() */
		idMap   m_reader_Regions;/* sim.getRegions() */
		elemMap m_reader_CrackTips;/* sim.getCrackTips() */
		elemMap m_reader_CrackTipParents;/* sim.getCrackTipParents() */
		nodeMap m_reader_Nodes;/* sim.getNodes() */

		idSet m_reader_Cracks; //set of region-IDs which are cracks (as opposed to boundaries of the object)		
		state_map m_reader_CrackTipStates; // map a crack tip (line) element to it's state
		
		std::set<unsigned int> m_reader_elemBodies;/* region_ids */
		std::set<unsigned int> m_reader_bndryBodies;/* crack_tip_ids */

#if USE_SST_DEBUG
		boost::shared_ptr<FractureSim::VDBWrapper>  levelSet;  // maintains VDB grids representing the geometry implicitly
#endif
		
	private:
		inline nodeMap& getNodes(){ return m_reader_Nodes; }
		inline elemMap& getElems(){ return m_reader_Elems; }
		inline idMap& getRegions(){ return m_reader_Regions; }
		inline idSet& getCracks(){ return m_reader_Cracks; }
		inline elemMap& getCrackTips(){ return m_reader_CrackTips; }
		inline elemMap& getCrackTipParents(){ return m_reader_CrackTipParents; }
		inline state_map& getCrackTipStates(){ return m_reader_CrackTipStates; }
		//inline vector_type& getSIFs(){ return sifs; } // old version
		// NEW FOR FractureRB:
		//inline id_set& getFracturedNodes(){ return fracturedNodes; }
		//inline id_set& getFracturedElems(){ return fracturedElems; }

		int readModel(elemMap& elems/* sim.getElems() */, idMap& bodyIDs/* sim.getRegions() */,
			elemMap& bndrys/* sim.getCrackTips() */, elemMap& bndryParents/* sim.getCrackTipParents() */, 
			nodeMap& nodes/* sim.getNodes() */,
			const ELEM_TYPE elemType/* FractureSim::TRI */,  const std::set<unsigned int> elemBodies/* region_ids */,
			const ELEM_TYPE bndryType/* FractureSim::LINE */, const std::set<unsigned int> bndryBodies/* crack_tip_ids */);
		int readElems(elemMap& elems, ELEM_TYPE elemType, std::set<unsigned int> bodies,
			int typeColumn = ELEMENTS_FILE, idMap* bodyIDs=NULL, elemMap* parentIDs=NULL, bool strictlyListedBodies=false);
		int readNodes(nodeMap& nodes);
#if USE_SST_DEBUG
		int initVDB(double voxelSize=1.0, bool noSI=false, double nbHWidth=3.0);
		int remesh(int nTris, double adaptive=0.1, double offsetVoxels=0.0);
#endif
		

	public:
		void outputMesh2Obj(const std::string& filename, const elem_map& elems, const node_map& nodes);
		void outputMeshInfo(const std::string& filename, const elem_map& elems, const id_map& bodyIDs, const elem_map& bndrys, const elem_map& bndryParents, const node_map& nodes);
		void output_elem_map(ostream& out, std::string name, const elem_map& elems);
		void output_id_map(ostream& out, std::string name, const id_map& ids);
		void output_node_map(ostream& out, std::string name, const node_map& nodes);
		void output_id_set(ofstream& out, std::string name, const id_set& ids);
		void output_state_map(ofstream& out, std::string name, const state_map& elems);
		
		int computeSurfaceStresses(
            node_map& retValues, const vector_type& displacements,
			const vector_type& crackBaseDisplacements,
            double E=1.0, double nu=0.0
        );

		// computes SVD of the deformation gradient F = U*S*Vt
		// and principal stresses P = 2 mu (S-I) + lambda tr(S-I);
		int computeElementPrincipalStresses(
			Eigen::Matrix3d& U, Eigen::Matrix3d& S, Eigen::Matrix3d& Vt, Eigen::Matrix3d& P,
			const Eigen::Vector3d&  a, const Eigen::Vector3d&  b, const Eigen::Vector3d&  c,
			const Eigen::Vector3d& ua, const Eigen::Vector3d& ub, const Eigen::Vector3d& uc,
			double mu, double lambda
			);
		void generateSurfaceStresses();
#endif

#if USE_VDB
	public:
		void setDoFractrue(bool flag){m_doFracture = flag;}
		bool isDoFractrue()const{return m_doFracture;}
		void setFractureStep(vrInt maxStep){m_maxFractureStep = maxStep;m_fractureStep=0;}
	
	private:
		bool m_doFracture;
		int m_fractureStep;
		int m_maxFractureStep;
#if USE_SST_DEBUG
	private:
			vrInt seedCracksAndPropagate(int maxSeed);
		PostProcessor *postPro;
#endif//USE_SST_DEBUG
		
		vect3d_map nodeSIFs, crackTipFaceNormals, crackTipTangents;
#endif

#if USE_DebugGMatrix
		std::set<int> m_nouse_set_dof;
		std::set<int> m_vtx_id;
		static std::set<int> m_set_colIdx;
#endif//USE_DebugGMatrix
	};
}
#endif//_vrBEM3D_H_