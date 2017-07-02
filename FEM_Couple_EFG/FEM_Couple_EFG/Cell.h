#ifndef _CELL_H
#define _CELL_H
#include "VR_Global_Define.h"
#include "Vertex.h"
#include "Polynomial.h"
#include <boost/smart_ptr.hpp>
#include <iostream>
#include "globalrowsfromlocal.h"
#include "TripletNode.h"
#include "plane.h"
#include <map>
#include "CellStructOnCuda.h"

namespace VR_FEM
{
	struct tuple_matrix
	{
		int m_nDomainId;
		long m_Radius;
		MyDenseMatrix matrix;
	};

	struct tuple_vector
	{
		int m_nDomainId;
		long m_Radius;
		MyVector vec;
	};

	class Cell;
	typedef boost::shared_ptr< Cell > CellPtr;

	class Cell
	{
	public:
		enum{CoupleDomainId=99};
		enum{LocalDomainCount = 2};
		enum{CoupleDomainCount = 1};

		class CellCompare
		{
		public:
			CellCompare(MyFloat x,MyFloat y,MyFloat z):m_point(x,y,z){}

			bool operator()(CellPtr& p)
			{
				return  (numbers::isZero(m_point(0) - (*p).m_center(0))) && 
					(numbers::isZero(m_point(1) - (*p).m_center(1))) && 
					(numbers::isZero(m_point(2) - (*p).m_center(2)));
			}
		private:
			MyDenseVector m_point;
		};
	public:
		int getID()const{return m_nID;}
		void setId(int id){m_nID = id;}
		void get_dof_indices(std::vector<int> &vecDofs);
		void get_dof_indices(unsigned gaussIdx,std::vector<int> &vecDofs);
		MyFloat getRadius()const{return m_radius;}
		MyPoint getCenterPoint()const{return m_center;}
		CellType getCellType()const{return m_CellType;}
		void setCellType(CellType type){m_CellType = type;}
		VertexPtr getVertex(unsigned idx){return m_elem_vertex[idx];}
		void initialize();
		void clear();
		void print(std::ostream& out);
		const MyMatrix& getMassMatrix()const{return MassMatrix_24_24;}
		const MyMatrix& getStiffnessMatrix()const{return StiffnessMatrix_24_24;}
		const MyVector& getRhsVector()const{return RightHandValue_24_1;}

		const MyMatrix& getMassMatrix(unsigned gaussIdx)const{return MassMatrix_81_81[gaussIdx];}
		const MyMatrix& getStiffnessMatrix(unsigned gaussIdx)const{return StiffnessMatrix_81_81[gaussIdx];}
		const MyVector& getRhsVector(unsigned gaussIdx)const{return RightHandValue_81_1[gaussIdx];}
	private:
		static void computeGaussPoint();
		static void makeGaussPoint(double gauss[2],double w);
		static void makeMaterialMatrix(const int nDomainId);
		//static void makeMaterialMatrix_Couple_EFG(const int nDomainId);
		static void makeSymmetry(MyMatrix& objMatrix);
		static void compressMassMatrix(const MyMatrix& objMatrix,std::vector<int> &);
		static void compressStiffnessMatrix(const MyMatrix& objMatrix,std::vector<int> &);
		static void compressMatrix(const MyMatrix& objMatrix,std::vector<int> &,std::map<long,std::map<long,TripletNode > >& TripletNodeMap);
		static void compressRHS(const MyVector& rhs,std::vector<int> &);

		void computeShapeFunction();
		void computeJxW();
		void computeShapeGrad();		
		void makeShapeFunctionMatrix();
		
		void makeStiffnessMatrix();
		void makeMassMatrix();
		void makeLocalRhs();
		
		

		void compute(const MyPoint &p, std::vector< MyFloat> &values, std::vector< MyDenseVector > &grads, std::vector< MyMatrix_3X3 > &grad_grads);
		void compute_index (const unsigned int i, unsigned int  (&indices)[3]) const;
		MyFloat determinant (const MyMatrix_3X3 &t);
		MyMatrix_3X3 invert (const MyMatrix_3X3 &t);
		void contract (MyDenseVector &dest, const MyDenseVector &src1, const MyMatrix_3X3 &src2);
		
		

		//*********************** EFG *****************************
		void jacobi(const MyPoint& gaussPoint,MyPoint& gaussPointInGlobalCoordinate, MyFloat &weight);
		
		void InfluentPoints(unsigned gaussIdx, const MyPoint& gaussPointInGlobalCoordinate );
		void ApproxAtPoint(unsigned gaussIdx, const MyPoint& gaussPointInGlobalCoordinate,bool isCouple);
		
		std::vector< VertexPtr > InflPts_8_27[Geometry::gauss_Sample_Point];
		MyVector   shapeFunctionInEFG_8_27[Geometry::gauss_Sample_Point];
		MyMatrix   shapeDerivativeValue_8_27_3[Geometry::gauss_Sample_Point];
		MyFloat            gaussPointInfluncePointCount_8[Geometry::gauss_Sample_Point];
		MyMatrix	StiffnessMatrix_81_81[Geometry::gauss_Sample_Point];
		MyMatrix	MassMatrix_81_81[Geometry::gauss_Sample_Point];
		MyVector	RightHandValue_81_1[Geometry::gauss_Sample_Point];

	private:
		Cell(MyPoint center, MyFloat radius, VertexPtr vertexes[]);
		MyPoint m_center;
		MyFloat m_radius;
		VertexPtr m_elem_vertex[Geometry::vertexs_per_cell];
		CellType m_CellType;
		int m_nID;

	private:
		//MyPoint vertex[Geometry::vertexs_per_cell];
		static MyPoint GaussVertex[Geometry::vertexs_per_cell];
		static MyFloat   GaussVertexWeight[Geometry::vertexs_per_cell];
		static MyMatrix MaterialMatrix_6_5[Cell::LocalDomainCount];
		static bool     bMaterialMatrixInitial[Cell::LocalDomainCount];

		
		MyDenseVector shapeDerivativeValue_8_8_3[Geometry::vertexs_per_cell][Geometry::vertexs_per_cell];
		MyDenseVector shapeDerivativeValue_mapping_8_8_3[Geometry::vertexs_per_cell][Geometry::vertexs_per_cell];
		MyMatrix_3X3 shapeSecondDerivativeValue_8_8_3_3[Geometry::vertexs_per_cell][Geometry::vertexs_per_cell];
		MyMatrix_3X3 contravariant[Geometry::vertexs_per_cell];
		MyMatrix_3X3 covariant[Geometry::vertexs_per_cell];
		MyFloat JxW_values[Geometry::vertexs_per_cell];
		
		MyMatrix StrainMatrix_6_24;
		MyMatrix StiffnessMatrix_24_24;
		MyMatrix MassMatrix_24_24;
		MyVector RightHandValue_24_1;
		//MyMatrix DampingMatrix_24_24;

		std::vector< Polynomial > m_Polynomials;
	private:
		static std::vector< CellPtr > s_Cell_Cache;
	public:
		static std::map<long,std::map<long,TripletNode > > m_TripletNode_Mass,m_TripletNode_Stiffness;
		static std::map<long,TripletNode >				   m_TripletNode_Rhs;
		
	public:
		static CellPtr makeCell(MyPoint center, MyFloat radius);
		static int getCellSize(){return s_Cell_Cache.size();}
		static std::vector< CellPtr >& getCellVector(){return s_Cell_Cache;}
		static CellPtr getCell(int idx){return s_Cell_Cache[idx];}
		static void basis(const MyPoint& pX, const MyPoint& pXI, MyVector& pP);
		static void dbasis(const MyPoint& pX, const MyPoint& pXI, MyMatrix& DP_27_3);
		static MyFloat WeightFun(const MyPoint& pCenterPt, const MyDenseVector& Radius, const MyPoint& pEvPoint, int DerOrder);

		static std::map< unsigned,bool > s_map_InfluncePointSize;

		MyFloat materialDistance(const MyPoint& gaussPoint, const MyPoint& samplePoint);
		static bool CheckLineTri( const MyPoint &L1, const MyPoint &L2, const MyPoint &PV1, const MyPoint &PV2, const MyPoint &PV3, MyPoint &HitP );

		//////////////////////////////////////////   Belytschko Method  //////////////////////////////
	public:
		void computeCellType(const std::vector< Plane >& vecPlane);
		void computeCellType_Steak();
		void computeCellType_Beam();
		int getDomainId()const{return m_nDomainId;}
		void computeShapeFunctionCouple();
		void resortInfluncePtsVec(std::vector< VertexPtr >& ref_InflunceVec);
		void InfluentPoints4Couple(unsigned gaussIdx, const MyPoint& gaussPointInGlobalCoordinate );
		int m_nDomainId;
		static MyDenseVector externalForce;
	public://couple
		std::vector< MyFloat > m_vec_RInGaussPt_8;
		std::vector< MyDenseVector > m_vec_RDerivInGaussPt_8_3;
	public:
		std::vector< std::pair<int,int> >     m_vec_VertexIdInEFG_Boundary, m_vec_VertexIdInFEM_Boundary;

		/*MyMatrix shapeFunctionValueInCouple_8_8;
		MyDenseVector shapeDerivativeValueInCouple_8_8_3[Geometry::vertexs_per_cell][Geometry::vertexs_per_cell];*/
		MyVector   shapeFunctionInCouple_8_31[Geometry::gauss_Sample_Point];
		MyMatrix   shapeDerivativeValueInCouple_8_31_3[Geometry::gauss_Sample_Point];

		std::map<unsigned,bool> m_mapSupportPtsIdMap;

	private:
		static int s_vertexOutgoingEdgePoints[8][3]; // 8 vertex ,every vertex has 3 outgoing edge poings
		MyMatrix_3X3 RotationMatrix;
		MyMatrix Cell_Corotation_Matrix;
		MyVector Pj_FEM,Pj_EFG[8],Pj_Couple[8];
		MyVector incremental_displacement;
	public:
		void assembleRotationMatrix();
		void computeRotationMatrix(MyVector & global_incremental_displacement);

	public:
		const MyPoint& getGlobalGaussPoint(int idx)const{return m_globalGaussVertex[idx];}
		void setGlobalGaussPoint(int idx,const MyPoint& pos){m_globalGaussVertex[idx]=pos;}
		void setEFGJxW(MyFloat jxw){m_dbEFGJxW = jxw;}
		MyFloat getEFGJxW()const{return m_dbEFGJxW;}
	private:
		MyPoint m_globalGaussVertex[Geometry::vertexs_per_cell];
		MyFloat m_dbEFGJxW;

		
	public:
		static std::vector< tuple_matrix > vec_cell_stiffness_matrix;
		static std::vector< tuple_matrix > vec_cell_mass_matrix;
		static std::vector< tuple_vector > vec_cell_rhs_matrix;
		static std::vector< FEMShapeValue > vec_FEM_ShapeValue;
		
		static int isHasInitialized(const int nDomainId, const long radius, const std::vector< tuple_matrix >& mapp );
		static int isHasInitialized(const int nDomainId, const long radius, const std::vector< tuple_vector >& mapp );
		static int appendMatrix(std::vector< tuple_matrix >& mapp,const int nDomainId, const long radius,const MyDenseMatrix& matrix);
		static int appendVector(std::vector< tuple_vector >& mapp,const int nDomainId, const long radius,const MyVector& matrix);
		static void appendFEMShapeValue(Cell * pThis);

		int m_nRhsIdx;
		int m_nMassMatrixIdx;
		int m_nStiffnessMatrixIdx;

		int getFEMCellStiffnessMatrixIdx()const{return m_nStiffnessMatrixIdx;}
		int getFEMCellMassMatrixIdx()const{return m_nMassMatrixIdx;}
		int getFEMCellRhsVectorIdx()const{return m_nRhsIdx;}
	/*private:
		MyMatrix_3X3 RotationMatrix;
		MyMatrix RotationMatrix_24_24;
		MyMatrix StiffnessMatrix_24_24_Corotation;
		MyVector RightHandValue_24_1_Corotation,Pj;*/
		static int s_nFEM_Cell_Count;
		static int s_nEFG_Cell_Count;
		static int s_nCOUPLE_Cell_Count;

		
		MyInt m_nCoupleId;
		MyInt getCoupleDomainId()const{return m_nCoupleId;}
		void initialize_Couple();
		void assembleSystemMatrix();

		void initialize_Couple_Joint();
		void assembleSystemMatrix_Joint();

		void makeLocalStiffnessMatrix_FEM();
		void makeLocalMassMatrix_FEM();
		void makeLocalRhs_FEM();

		void get_dof_indices_Local_FEM(MyInt nLocalDomainId,std::vector<int> &vecDofs);

		void computeShapeFunction_Couple_EFG();
		void InfluentPoints_Couple_EFG(unsigned gaussIdx, const MyPoint& gaussPointInGlobalCoordinate );
		void ApproxAtPoint_Couple_EFG(unsigned gaussIdx, const MyPoint& gaussPointInGlobalCoordinate);

		void get_dof_indices_Couple_EFG(unsigned gaussIdx,std::vector<int> &vecDofs);

		static void compressLocalFEMStiffnessMatrix(MyInt id,const MyMatrix& objMatrix,std::vector<int> &);
		static void compressLocalFEMMassMatrix(MyInt id,const MyMatrix& objMatrix,std::vector<int> &);		
		static void compressLocalFEMRHS(MyInt id,const MyVector& rhs,std::vector<int> &);

		static void compressCoupleEFGStiffnessMatrix(MyInt id,const MyMatrix& objMatrix,std::vector<int> &);
		static void compressCoupleEFGMassMatrix(MyInt id,const MyMatrix& objMatrix,std::vector<int> &);		
		static void compressCoupleEFGRHS(MyInt id,const MyVector& rhs,std::vector<int> &);

		static std::map<long,std::map<long,TripletNode > > m_TripletNode_LocalMass[LocalDomainCount],m_TripletNode_LocalStiffness[LocalDomainCount];
		static std::map<long,TripletNode >				   m_TripletNode_LocalRhs[LocalDomainCount];

		static std::map<long,std::map<long,TripletNode > > m_TripletNode_LocalMass_EFG[CoupleDomainCount],m_TripletNode_LocalStiffness_EFG[CoupleDomainCount];
		static std::map<long,TripletNode >				   m_TripletNode_LocalRhs_EFG[CoupleDomainCount];

	public://2013-10-25
		MyFloat getJxW(unsigned q)const{return JxW_values[q];}
		MyMatrix shapeFunctionValue_8_8;
	};
}
#endif//_CELL_H