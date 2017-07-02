#pragma once

#include "VR_Global_Define.h"
#include "constant_numbers.h"
#include "Vertex.h"
#include "Polynomial.h"
#include "MaterialMatrix.h"
#include "Frame/Axis_YC.h"
namespace YC
{
	class MyBOOL
	{
	public:
		MyBOOL(){ m_b = false; }
		bool getBool(){ return m_b; }
		void setBool(bool flag){ m_b = flag; }
	private:
		bool m_b;
	};
	class Cell;
	typedef boost::shared_ptr< Cell > CellPtr;

	class Cell
	{
	public:
		
		struct tuple_matrix
		{
			long m_Radius;
			MyDenseMatrix matrix;
		};

		struct tuple_vector
		{
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
#if USE_MultidomainIndependent
	public:
		void computeCellType_DMI(const int did);
		static CellPtr makeCell_DMI(MyPoint point, MyFloat radius, const int did);
		void setLM_Boundary(MyInt idx){ m_vec_index.push_back(idx); }
		void initialize_DMI();
		MyInt getDomainId()const { return m_nDomainId; }
		void TestModalWrapMatrix_DMI(const MyVector& globlaDisp, MyVector& w, MyVector& translation, MyVector& incremental_displacement);

		void assembleSystemMatrix_DMI();
		void get_dof_indices_Local_DMI(std::vector<int> &vecDofs);
		void compressLocalFEMStiffnessMatrix_DMI(const MyMatrix& objMatrix, std::vector<int> &);
		void compressLocalFEMMassMatrix_DMI(const MyMatrix& objMatrix, std::vector<int> &);
		void compressLocalFEMRHS_DMI(const MyVector& rhs, std::vector<int> &);

		static void makeMaterialMatrix(const int nDomainId);
		static void makeSymmetry(MyMatrix& objMatrix);
	private:
		MyInt m_nDomainId;
		
		static MyMatrix MaterialMatrix_6_5[LocalDomainCount];
		static MyBOOL   bMaterialMatrixInitial[LocalDomainCount];

	public:
		std::vector< MyInt > m_vec_index;
		static std::map<long, std::map<long, TripletNode > > s_TripletNode_LocalMass_DMI[LocalDomainCount];
		static std::map<long, std::map<long, TripletNode > > s_TripletNode_LocalStiffness_DMI[LocalDomainCount];
		static std::map<long, TripletNode >				   s_TripletNode_LocalRhs_DMI[LocalDomainCount];
		/*static std::map<long, std::map<long, TripletNode > > s_TripletNode_ModalWarp_DIM[LocalDomainCount];*/
#endif//USE_MultidomainIndependent

	public:
		int getID()const{return m_nID;}
		void setId(int id){m_nID = id;}
		void computeCellType_Global();
		VertexPtr getVertex(unsigned idx){return m_elem_vertex[idx];}
		void initialize_Global();
		void assembleSystemMatrix_Global();
		void get_dof_indices_Global(std::vector<int> &vecDofs);

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
	public:
		static CellPtr makeCell(MyPoint point, MyFloat radius);

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
	};
}

