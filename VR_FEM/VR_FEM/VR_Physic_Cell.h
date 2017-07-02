#ifndef _VR_PHYSIC_CELL_H
#define _VR_PHYSIC_CELL_H
#include <boost/smart_ptr.hpp>
#include "VR_MACRO.h"
#include "VR_Global_Define.h"
#include "constant_numbers.h"
#include "VR_Physic_Vertex.h"

#include "VR_Geometry.h"
#include "Polynomial.h"


namespace YC
{
	
	namespace Physics
	{
		class MaterialMatrix
		{
			struct MatrixInfo
			{
				MatrixInfo(MyFloat y, MyFloat p, const MyMatrix& refM):YoungModule(y),PossionRatio(p),tMatrix(refM){}
				MyFloat YoungModule;
				MyFloat PossionRatio;
				MyMatrix tMatrix;
			};

			class MaterialMatrixCompare
			{
			public:
				MaterialMatrixCompare(MyFloat y,MyFloat p):YoungModule(y),PossionRatio(p){}

				bool operator()(MatrixInfo& p)
				{
					return  (numbers::IsEqual(YoungModule,p.YoungModule)) && (numbers::IsEqual(PossionRatio,p.PossionRatio));
				}
			private:
				MyFloat YoungModule,PossionRatio;
			};


		public:
			static MyMatrix getMaterialMatrix(const MyFloat YoungModule, const MyFloat PossionRatio);
		private:
			static void MaterialMatrix::makeSymmetry(MyMatrix& objMatrix);
			static MyMatrix MaterialMatrix::makeMaterialMatrix(const MyFloat y, const MyFloat p);
			static std::vector<MatrixInfo> s_vecMatrixMatrix;
		};
		namespace GPU
		{
			const int linePair[12][2] = {{0,2},{4,6},{0,4},{2,6},{1,3},{5,7},{1,5},{3,7},{0,1},{4,5},{2,3},{6,7}};
			class Cell;
			typedef boost::shared_ptr< Cell > CellPtr;

			class Cell
			{
//data struct definition
#if 1
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
#endif
//data interface
#if 1
			public:
				int getID()const{return m_nID;}
				void setId(int id){m_nID = id;}
				void get_dof_indices(std::vector<int> &vecDofs);
				void get_postion(MyVector& Pj_FEM);
				MyFloat getRadius()const{return m_radius;}
				MyFloat getRadiusX2()const{return m_radius*2;}
				MyPoint getCenterPoint()const{return m_center;}
				MACRO::CellType getCellType()const{return m_CellType;}
				void setCellType(MACRO::CellType type){m_CellType = type;}
				VertexPtr getVertex(unsigned idx){return m_elem_vertex[idx];}
				void setLevel(MyInt level){m_nLevel = level;}
				MyInt getLevel()const{return m_nLevel;}
				const MyMatrix& getMassMatrix()const{return vec_cell_mass_matrix[m_nMassMatrixIdx].matrix;}
				const MyMatrix& getStiffnessMatrix()const{return vec_cell_stiffness_matrix[m_nStiffnessMatrixIdx].matrix;}
				const MyVector& getRhsVector()const{return vec_cell_rhs_matrix[m_nRhsIdx].vec;}
				static int getCellSize(){return s_Cell_Cache.size();}
				static std::vector< CellPtr >& getCellVector(){return s_Cell_Cache;}
				static CellPtr getCell(int idx){return s_Cell_Cache[idx];}
				int getFEMCellStiffnessMatrixIdx()const{return m_nStiffnessMatrixIdx;}
				int getFEMCellMassMatrixIdx()const{return m_nMassMatrixIdx;}
				int getFEMCellRhsVectorIdx()const{return m_nRhsIdx;}
				int getFEMCellShapeIdx()const{return m_nFEMShapeIdx;}
				MyFloat getJxW(unsigned q)const{return JxW_values[q];}

				void computeCell_NeedBeCutting(){m_needBeCutting = true;}
				void computeCellType(){m_CellType = MACRO::FEM;}
#endif
//function		
#if 1
			public:
				Cell(MyPoint center, MyFloat radius, VertexPtr vertexes[]);
				void initialize();
				void clear();
				void print(std::ostream& out);
				void assembleSystemMatrix();
				void computeRotationMatrix(MyVector & global_incremental_displacement);
				void assembleRotationMatrix();
			private:
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
				
				
#endif
//Utility function
#if 1
			public:
				static CellPtr makeCell(MyPoint center, MyFloat radius);
				static int isHasInitialized(const long radius, const std::vector< tuple_matrix >& mapp );
				static int isHasInitialized(const long radius, const std::vector< tuple_vector >& mapp );
				static const std::vector< tuple_matrix >& getStiffnessMatrixList(){return vec_cell_stiffness_matrix;}
				static const std::vector< tuple_matrix >& getMassMatrixList(){return vec_cell_mass_matrix;}
				static const std::vector< tuple_vector >& getRhsList(){return vec_cell_rhs_matrix;}
				
				static MyDenseVector getExternalForce(){return externalForce;}
			private:
				
				static void computeGaussPoint();
				static void makeGaussPoint(double gauss[2],double w);
				static void compressMassMatrix(const MyMatrix& objMatrix,std::vector<int> &);
				static void compressStiffnessMatrix(const MyMatrix& objMatrix,std::vector<int> &);
				static void compressMatrix(const MyMatrix& objMatrix,std::vector<int> &,std::map<long,std::map<long,TripletNode > >& TripletNodeMap);
				static void compressRHS(const MyVector& rhs,std::vector<int> &);
				static int appendMatrix(std::vector< tuple_matrix >& mapp, const long radius,const MyDenseMatrix& matrix);
				static int appendVector(std::vector< tuple_vector >& mapp, const long radius,const MyVector& Vector);
				
				static const MyDenseMatrix& getStiffMatrix(const int id);
				static const MyDenseMatrix& getMassMatrix(const int id);
				static const MyVector& getRHSVector(const int id);
#endif
//Data Definition
#if 1
			private:
				MyPoint m_center;
				MyFloat m_radius;
				VertexPtr m_elem_vertex[Geometry::vertexs_per_cell];
				MACRO::CellType m_CellType;
				int m_nID;
				MyInt m_nLevel;
				bool m_needBeCutting;

				MyDenseVector shapeDerivativeValue_8_8_3[Geometry::vertexs_per_cell][Geometry::vertexs_per_cell];
				MyDenseVector shapeDerivativeValue_mapping_8_8_3[Geometry::vertexs_per_cell][Geometry::vertexs_per_cell];
				MyMatrix_3X3 shapeSecondDerivativeValue_8_8_3_3[Geometry::vertexs_per_cell][Geometry::vertexs_per_cell];
				MyMatrix_3X3 contravariant[Geometry::vertexs_per_cell];
				MyMatrix_3X3 covariant[Geometry::vertexs_per_cell];
				MyFloat JxW_values[Geometry::vertexs_per_cell];

				MyMatrix StrainMatrix_6_24;
				//MyMatrix StiffnessMatrix_24_24;
				//MyMatrix MassMatrix_24_24;
				//MyVector RightHandValue_24_1;
				MyMatrix_3X3 RotationMatrix;
				MyMatrix Cell_Corotation_Matrix;
				MyVector Pj_FEM;
				MyVector incremental_displacement;
				int m_nRhsIdx;
				int m_nMassMatrixIdx;
				int m_nStiffnessMatrixIdx;
				int m_nFEMShapeIdx;
				MyMatrix shapeFunctionValue_8_8;
				std::vector< Polynomial > m_Polynomials;

			private:
				static MyPoint GaussVertex[Geometry::vertexs_per_cell];
				static MyFloat   GaussVertexWeight[Geometry::vertexs_per_cell];
				static Physics::MaterialMatrix s_MaterialMatrix;
				static std::vector< CellPtr > s_Cell_Cache;
				static MyDenseVector externalForce;
				static MyFloat scaleExternalForce;
				static std::vector< tuple_matrix > vec_cell_stiffness_matrix;
				static std::vector< tuple_matrix > vec_cell_mass_matrix;
				
				static std::vector< tuple_vector > vec_cell_rhs_matrix;
				
			public:
				static std::map<long,std::map<long,TripletNode > > m_TripletNode_Mass,m_TripletNode_Stiffness;
				static std::map<long,TripletNode >				   m_TripletNode_Rhs;
#endif
			};
		}//namespace GPU
	}//namespace Physics
}//namespace YC
#endif//_VR_PHYSIC_CELL_H