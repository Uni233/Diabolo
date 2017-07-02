#pragma once

#include "VR_Global_Define.h"
#include "constant_numbers.h"
#include "Vertex.h"
#include "Polynomial.h"
#include "MaterialMatrix.h"
#include "Frame/Axis_YC.h"
#include <map>
namespace YC
{
	struct FEMShapeValue
	{
		float  shapeFunctionValue_8_8[8][8];
		float  shapeDerivativeValue_8_8_3[8][8][MyDIM];
	};

	class Cell;
	typedef boost::shared_ptr< Cell > CellPtr;

	class Cell
	{
	public:
		
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
		struct TripletNode
		{
			TripletNode():val(0.0){}
			MyFloat val;
		};
		struct Assemble_State
		{
			std::map<long,std::map<long,TripletNode > > m_TripletNode_Mass,m_TripletNode_Stiffness;
			std::map<long,TripletNode >				   m_TripletNode_Rhs;
			std::map<long,std::map<long,TripletNode > > m_TripletNode_ModalWarp;
		};
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
		Cell(MyPoint center, MyFloat radius, VertexPtr vertexes[]);
		~Cell(void);

	public:
		int getID()const{return m_nID;}
		void setId(int id){m_nID = id;}
		void computeCellType_Global();
		VertexPtr getVertex(unsigned idx){return m_elem_vertex[idx];}
		void initialize_Global();
		void assembleSystemMatrix_Global();
		void get_dof_indices_Global(std::vector<int> &vecDofs);
		/*static int getCellSize(){ return s_Cell_Cache.size(); }
		static std::vector< CellPtr >& getCellVector(){ return s_Cell_Cache; }
		static CellPtr getCell(int idx){ return s_Cell_Cache[idx]; }*/
		const MyPoint& getCellCenter()const{ return m_center; }
#if USE_MODAL_WARP
		void createW_24_24();
		void get_dof_indices_ShareCellCountWeight_Global(std::vector<float> &vecDofsWeights);
		static void compress_W_Matrix_Global(const MyMatrix& objMatrix,std::vector<int> &,std::vector<float> &);
		static MyMatrix m_W_24_24;
		void TestCellRotationMatrix(const MyVector& globlaDisp, MyMatrix_3X3& RotationMatrix, MyDenseVector& m_FrameTranslationVector);
		void TestModalWrapMatrix(const MyVector& globlaDisp, MyVector& w, MyVector& translation, MyVector& cellDisplace);
		
		std::vector< Axis::Quaternion > m_vec_LocalQuat;
#endif
	private:
		static void computeGaussPoint();
		void computeShapeFunction();
		void compute(const MyPoint &p, std::vector< MyFloat> &values, std::vector< MyDenseVector > &grads, std::vector< MyMatrix_3X3 > &grad_grads);
		void compute_index (const unsigned int i, unsigned int  (&indices)[3]) const;
		MyFloat determinant (const MyMatrix_3X3 &t);
		MyMatrix_3X3 invert (const MyMatrix_3X3 &t);
		void contract (MyDenseVector &dest, const MyDenseVector &src1, const MyMatrix_3X3 &src2);
		void computeJxW();
		void computeShapeGrad();
		void Cell::makeStiffnessMatrix();
		void Cell::makeMassMatrix_Lumping();
		void Cell::makeLocalRhs();
		int Cell::isHasInitialized( const long radius, const std::vector< tuple_vector >& mapp );
		int Cell::isHasInitialized( const long radius, const std::vector< tuple_matrix >& mapp );
		MyFloat getJxW(unsigned q)const{return JxW_values[q];}
		static void makeGaussPoint(double gauss[2],double w);
		static void compressMassMatrix_Global(const MyMatrix& objMatrix,std::vector<int> &);
		static void compressStiffnessMatrix_Global(const MyMatrix& objMatrix,std::vector<int> &);
		static void compressMatrix(const MyMatrix& objMatrix,std::vector<int> &,std::map<long,std::map<long,TripletNode > >& TripletNodeMap);
		static void compressRHS_Global(const MyVector& rhs,std::vector<int> &);
		static int appendMatrix(std::vector< tuple_matrix >& mapp, const long radius,const MyDenseMatrix& matrix);
		static int appendVector(std::vector< tuple_vector >& mapp, const long radius,const MyVector& Vector);

		static MyPoint GaussVertex[Geometry::vertexs_per_cell];
		static MyFloat   GaussVertexWeight[Geometry::vertexs_per_cell];
		static MaterialMatrix s_MaterialMatrix;
		static MyDenseVector externalForce;
		static MyFloat scaleExternalForce;
		static std::vector< tuple_matrix > vec_cell_stiffness_matrix;
		static std::vector< tuple_matrix > vec_cell_mass_matrix;
		static std::vector< tuple_vector > vec_cell_rhs_matrix;
		MyMatrix shapeFunctionValue_8_8;
		MyDenseVector shapeDerivativeValue_8_8_3[Geometry::vertexs_per_cell][Geometry::vertexs_per_cell];
		MyDenseVector shapeDerivativeValue_mapping_8_8_3[Geometry::vertexs_per_cell][Geometry::vertexs_per_cell];
		MyMatrix_3X3 shapeSecondDerivativeValue_8_8_3_3[Geometry::vertexs_per_cell][Geometry::vertexs_per_cell];
		MyMatrix_3X3 contravariant[Geometry::vertexs_per_cell];
		MyMatrix_3X3 covariant[Geometry::vertexs_per_cell];
		MyFloat JxW_values[Geometry::vertexs_per_cell];
		MyVector Pj_FEM;
		int m_nFEMShapeIdx;
		MyMatrix StrainMatrix_6_24;
		MyMatrix StiffnessMatrix_24_24;
		MyMatrix MassMatrix_24_24;
		MyVector RightHandValue_24_1;
	public:
		static CellPtr makeCell(MyPoint point, MyFloat radius);
		static void makeMaterialMatrix(const int nDomainId);
		static void makeSymmetry(MyMatrix& objMatrix);
		static MyMatrix MaterialMatrix_6_5[LocalDomainCount];
		static bool     bMaterialMatrixInitial[LocalDomainCount];
		void makeLocalStiffnessMatrix_FEM();
		void makeLocalMassMatrix_FEM();
		void makeLocalRhs_FEM();
		void makeLocalMassMatrix_FEM_Lumping();

		static int isHasInitialized(const int nDomainId, const long radius, const std::vector< tuple_matrix >& mapp);
		static int isHasInitialized(const int nDomainId, const long radius, const std::vector< tuple_vector >& mapp);
		static int appendMatrix(std::vector< tuple_matrix >& mapp, const int nDomainId, const long radius, const MyDenseMatrix& matrix);
		static int appendVector(std::vector< tuple_vector >& mapp, const int nDomainId, const long radius, const MyVector& matrix);
		static void appendFEMShapeValue(Cell * pThis);

		
		static std::vector< FEMShapeValue > vec_FEM_ShapeValue;
		static MyMatrix s_shapeFunction;
	private:
		int m_nID;
		MyPoint m_center;
		MyFloat m_radius;
		VertexPtr m_elem_vertex[Geometry::vertexs_per_cell];
		CellType m_CellType;

		int m_nRhsIdx;
		int m_nMassMatrixIdx;
		int m_nStiffnessMatrixIdx;
		std::vector< Polynomial > m_Polynomials;
	private:
		static std::vector< CellPtr > s_Cell_Cache;
	public:
		static int s_nFEM_Cell_Count;
		static int s_nEFG_Cell_Count;
		static int s_nCOUPLE_Cell_Count;
		static Assemble_State m_global_state;

#if USE_MultiDomain
	public:
		void TestModalWrapMatrix_ALM(const MyVector& globlaDisp, MyVector& w, MyVector& translation, MyVector& cellDisplace);
		void computeCellType_ALM(const int did);
		static CellPtr makeCell_ALM(MyPoint point, MyFloat radius, const int did);
		void setLM_Boundary(MyInt idx){ m_vec_index.push_back(idx); }
		int getDomainId()const{ return m_nDomainId; }
		void initialize_ALM();
		void computeJxW_Quads();
		void computeQuadsFunction();
		void computeQuads(const MyPoint &p, std::vector< MyFloat> &values, std::vector< MyDenseVector > &grads, std::vector< MyMatrix_3X3 > &grad_grads);
		void computeGaussPoint_Quads();
		MyMatrix makeLagrangeMultiplierMatrix(const int faceId);
		MyMatrix makeLagrangeMultiplierMatrixV(const int faceId);

		void assembleSystemMatrix_ALM();
		void get_dof_indices_Local_ALM(std::vector<int> &vecDofs);
		static void compressLocalFEMStiffnessMatrix_ALM(const MyMatrix& objMatrix, std::vector<int> &);
		static void compressLocalFEMMassMatrix_ALM(const MyMatrix& objMatrix, std::vector<int> &);
		static void compressLocalFEMRHS_ALM(const MyVector& rhs, std::vector<int> &);

		MyMatrix get_LM_ShapeFunction_Q1(){ return LM_Q1_24_12; }
		MyMatrix get_LM_ShapeFunction_Q2(){ return LM_Q2_24_12; }
		MyMatrix get_LM_ShapeFunction_V1(){ return LM_V1_12_12; }
		MyMatrix get_LM_ShapeFunction_V2(){ return LM_V2_12_12; }

		void get_dof_indices_Local_ALM_Q_New(std::vector<int> &vecDofs, const int faceId);
	private:
		
		
		

		static MyPoint GaussVertex_Quads[Geometry::faces_per_cell][Geometry::vertexs_per_face];
		static MyPoint GaussVertex_Quads4ShapeFunction[Geometry::faces_per_cell][Geometry::vertexs_per_face];
		static MyFloat   GaussVertexWeight_Quads[Geometry::faces_per_cell][Geometry::vertexs_per_face];
		MyMatrix shapeFunctionValueQuads_8_8[Geometry::faces_per_cell];
		MyDenseVector shapeDerivativeValueQuads_8_8_3[Geometry::faces_per_cell][Geometry::vertexs_per_cell][Geometry::vertexs_per_cell];
		MyFloat JxW_values_Quads[Geometry::faces_per_cell][Geometry::vertexs_per_face];
		MyMatrix_3X3 contravariant_Quads[Geometry::faces_per_cell][Geometry::vertexs_per_face];

		MyMatrix LM_Q1_24_12, LM_Q2_24_12, LM_V1_12_12, LM_V2_12_12;
		
	public:
		static std::map<long, std::map<long, TripletNode > > m_TripletNode_LocalMass_ALM;
		static std::map<long, std::map<long, TripletNode > > m_TripletNode_LocalStiffness_ALM;
		static std::map<long, TripletNode >				   m_TripletNode_LocalRhs_ALM;
		std::vector< MyInt > m_vec_index;
#endif//USE_MultiDomain

	private:
		MyInt m_nDomainId;
	};
}

