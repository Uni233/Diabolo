#ifndef _bemTriangleElem_h_
#define _bemTriangleElem_h_
#include <boost/smart_ptr.hpp>
#include "bemVertex.h"

#define DEBUG_3_3 (1)
namespace VR
{
	typedef enum{ Continuous = 1, DisContinuous = 2/*, HalfContinuous = 3*//*use for dc condition */ } TriElemType;
	typedef enum{ dis_1_1=1, dis_1_2=2, dis_1_3=3, dis_2_3=4, dis_2_2=5, dis_2_1=6, dis_3_3=7, dis_regular=8}DisContinuousType;

	MyVec3 compute_UnitNormal(const MyVec3 vtx_globalCoord[]);

	struct TriangleElemData_DisContinuous
	{

	public:
		friend class vrBEM3D;
		enum{ SubTriSize=3 };
		static vrFloat delta_ij(vrInt i, vrInt j);
		static MyVec2ParamSpace pc2xi(const MyVec2ParamSpace& srcImage, const MyVec2ParamSpace& pc);		
		static MyVec2ParamSpace pc2eta(const MyVec2ParamSpace& srcImage, const MyVec2ParamSpace& pc);
		
#if DEBUG_3_3
		MyVec2ParamSpace eta2xi(const MyVec2ParamSpace& eta)const;
		//MyVec3 xi2global(const MyVec2ParamSpace& xi)const;
		MyVec2ParamSpace xi2eta(const MyVec2ParamSpace& xi)const;
#endif
		MyVec2ParamSpace xi2eta_SubTri(const MyVec2ParamSpace& xi)const;
		MyVec2ParamSpace eta2xi_SubTri(const MyVec2ParamSpace& eta)const;
		MyVec3 xi2global(const MyVec2ParamSpace& xi)const;
		static void TestShapeFunction();
		
		vrFloat shapefunction_xi(vrInt idx,const MyVec2ParamSpace& xi/*vrFloat xi_1, vrFloat xi_2*/)const;
		vrFloat derisShapefunction_xi(vrInt idx,vrInt idx_local,const MyVec2ParamSpace& xi/*vrFloat xi_1, vrFloat xi_2*/)const;
		void compute_Shape_Deris_Jacobi_SST_3D(DisContinuousType type, const MyVec3 vtx_globalCoord[/*Geometry::vertexs_per_tri*/]);
		
		static DisContinuousType computeTmpDisContinuousType(DisContinuousType srcType, const vrInt srcIdx);
		static DisContinuousType computeTmpDisContinuousTypePlus(DisContinuousType srcType, const vrInt srcIdx);

		vrVec3 interpolation_displacement(const vrInt idx, vrVec3 srcDisp[]);
	private:
		
		
		static void testAssist(DisContinuousType currentContinuousType, const MyMatrix& tmp_gaussQuadrature_xi_polar, MyVec2ParamSpace paramCoordsInDiscontinuous[]);
		static vrFloat s_shapefunction_xi(DisContinuousType currentContinuousType, vrInt idx,const MyVec2ParamSpace& xi/*vrFloat xi_1, vrFloat xi_2*/);
		static vrFloat s_derisShapefunction_xi(DisContinuousType currentContinuousType, vrInt idx,vrInt idx_local,const MyVec2ParamSpace& xi/*vrFloat xi_1, vrFloat xi_2*/);

		void compute_Shape_Deris_Jacobi_SST_3D_xi();
		
		
		
		DisContinuousType m_DisContinuousType;//dis_1_1=1, dis_1_2=2, dis_1_3=3, dis_2_3=4, dis_2_2=5, dis_2_1=6, dis_3_3=7, dis_regular=8
		MyVec3 m_vtx_globalCoord[Geometry::vertexs_per_tri];
		MyVec3 unitNormal_fieldPt;

		MyMatrix m_gaussQuadrature_xi_polar;
		//Jacobi in xi
		MyVec3 u1_xi;/* 2. Jacobi u1=@x/@xi_1 */
		MyVec3 u2_xi;/* 3. Jacobi u2=@x/@xi_2 */
		MyVec3 u1_xi_cross_u2_xi;/* 4. Jacobi u1 * u2 */
		vrFloat Jacobi_xi;/* 4. Jacobi J_xi */

#if DEBUG_3_3
		vrFloat rho_hat(const vrFloat theta)const;
		void compute_Shape_Deris_Jacobi_SST_3D_eta();
		MyVec2ParamSpace eta_2;
		MyMatrix_2X2 mat_T,mat_T_Inv;
		vrFloat det_Mat_T;
		vrFloat det_Mat_T_Inv;

		MyVec3 u1_eta;/* 2-1. Jacobi u1=@x/@xi_1 */
		MyVec3 u2_eta;/* 3-1. Jacobi u2=@x/@xi_2 */
		MyVec3 u1_eta_cross_u2_eta;/* 4-1. Jacobi u1 * u2 */
		vrFloat Jacobi_eta;/* 4-1. Jacobi J_xi */

		vrFloat theta_eta;
		MyVec2ParamSpace Pedal_in_eta;
		vrFloat h_Pedal;
		vrFloat theta_0_eta; 

		MyMatrix m_gaussQuadrature_xi_eta_polar;//{row number is GaussPointSize_eta_In_Theta*GaussPointSize_eta_In_Rho; column number is 3 x,y,w }
		MyMatrix m_gaussQuadrature_eta_theta_singleLayer;

		MyVec2ParamSpace m_SrcPt_in_eta;
		MyVec2ParamSpace m_SrcPt_in_xi;
#endif

		

		static MyVec2ParamSpace s_paramCoordsInDiscontinuous[9][MyDim];
		////////////////////////////////////////////////////////////
		

#if DEBUG_3_3
		vrFloat A_i_theta(const vrInt idx_i, const vrFloat theta)const;
		vrFloat B_i_theta(const vrInt idx_i, const vrFloat theta)const;
		vrFloat A_theta(const vrFloat theta)const;
		vrFloat B_theta(const vrFloat theta)const;
		vrFloat r_i(const vrInt idx_i, const vrFloat theta)const;
		vrFloat N_I_0_eta(const vrInt idx_I/*, const MyVec2ParamSpace& eta*/)const;
		vrFloat N_I_1_eta(const vrInt idx_I, const vrFloat theta)const;

		/*vrFloat xi_0_eta_0()const{return 1.0;}
		vrFloat xi_0_eta_1()const{return -1.0*(eta_2[0])/(eta_2[1]);}
		vrFloat xi_1_eta_0()const{return 0.0;}
		vrFloat xi_1_eta_1()const{return 1.0/(eta_2[1]);}*/
#endif

		void compute_Shape_Deris_Jacobi_SST_3D_eta_DisContinuousPt();
		
		vrFloat subtri_Theta_eta[SubTriSize][2];// 0-2*pi
		MyVec2ParamSpace Pedal_in_edge_eta[SubTriSize];
		vrFloat theta_0_eta_SubTri[SubTriSize];// 0-2*pi
		vrFloat Pedal_Length_in[SubTriSize];

		vrFloat rho_hat_SubTri(const vrInt subTriId, const vrFloat theta)const;

#if USE_360_Sample
		//{row number is [GaussPointSize_eta_In_Theta_SubTri*GaussPointSize_eta_In_Rho_SubTri]; column number is 3 theta_bar_new, rho, w}
		MyMatrix m_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal;
		//{row number is [GaussPointSize_eta_In_Theta_SubTri]; column number is 2+2 theta_bar_new(s), w(s), s[0-1, gauss], Jacobi(s)}
		MyMatrix m_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal;	
#else//USE_360_Sample

#if USE_Aliabadi_RegularSample
		//{row number is [GaussPointSize_eta_In_Theta_SubTri*GaussPointSize_eta_In_Rho_SubTri]; column number is 3 theta_bar_new, rho, w}
		MyMatrix m_gaussQuadrature_xi_eta_polar_SubTri_Sigmoidal[SubTriSize];
		//{row number is [GaussPointSize_eta_In_Theta_SubTri]; column number is 2+2 theta_bar_new(s), w(s), s[0-1, gauss], Jacobi(s)}
		MyMatrix m_gaussQuadrature_eta_theta_singleLayer_SubTri_Sigmoidal[SubTriSize];	
#endif//USE_Aliabadi_RegularSample

#endif//USE_360_Sample


		MyVec2ParamSpace m_SrcPt_in_eta_SubTri;
		MyVec2ParamSpace m_SrcPt_in_xi_SubTri;

		const vrInt search_Theta_in_eta_belong_SubTri_Index(vrFloat subtri_Theta_eta[][2], const vrFloat curTheta);
		const vrInt search_Theta_in_eta_belong_SubTri_Index(const vrFloat curTheta)const;

		MyVec2ParamSpace eta_2_SubTri;
		MyMatrix_2X2 mat_T_SubTri,mat_T_Inv_SubTri;
		vrFloat det_Mat_T_SubTri;
		vrFloat det_Mat_T_Inv_SubTri;

		MyVec3 u1_eta_SubTri;/* 2-1. Jacobi u1=@x/@xi_1 */
		MyVec3 u2_eta_SubTri;/* 3-1. Jacobi u2=@x/@xi_2 */
		MyVec3 u1_eta_cross_u2_eta_SubTri;/* 4-1. Jacobi u1 * u2 */
		vrFloat Jacobi_eta_SubTri;/* 4-1. Jacobi J_xi */

		vrFloat A_i_theta_SubTri(const vrInt idx_i, const vrFloat theta)const;
		vrFloat B_i_theta_SubTri(const vrInt idx_i, const vrFloat theta)const;
		vrFloat A_theta_SubTri(const vrFloat theta)const;
		vrFloat B_theta_SubTri(const vrFloat theta)const;
		vrFloat r_i_SubTri(const vrInt idx_i, const vrFloat theta)const;
		vrFloat N_I_0_eta_SubTri(const vrInt idx_I/*, const MyVec2ParamSpace& eta*/)const;
		vrFloat N_I_1_eta_SubTri(const vrInt idx_I, const vrFloat theta)const;

		vrFloat xi_0_eta_0_SubTri()const{return 1.0;}
		vrFloat xi_0_eta_1_SubTri()const{return -1.0*(eta_2_SubTri[0])/(eta_2_SubTri[1]);}
		vrFloat xi_1_eta_0_SubTri()const{return 0.0;}
		vrFloat xi_1_eta_1_SubTri()const{return 1.0/(eta_2_SubTri[1]);}

#if USE_Sigmoidal
		enum{ f_z_m = 1 };

		vrFloat w_theta(const vrFloat theta)const
		{
			return (1.0/numbers::MyPI) * (theta + numbers::PI_2);
		}
		MyVec2ParamSpace w_bar_theta[SubTriSize];
		vrFloat z_s(const MyVec2ParamSpace& w_bar_section, const vrFloat s)const
		{
			return (w_bar_section[1] - w_bar_section[0]) * s + w_bar_section[0];
		}
		vrFloat f_z(const vrFloat z)const
		{
			/*vrInt mmmm;
			mmmm=2;
			const vrFloat m_2 = (std::pow(z, mmmm)) / (std::pow(z, mmmm) + std::pow( (1.0-z), mmmm));
			mmmm=3;
			const vrFloat m_3 = (std::pow(z, mmmm)) / (std::pow(z, mmmm) + std::pow( (1.0-z), mmmm));
			mmmm=1;
			const vrFloat m_1 = (std::pow(z, mmmm)) / (std::pow(z, mmmm) + std::pow( (1.0-z), mmmm));
			printf("f_z : [m=1, %f] [m=2, %f] [m=3, %f]\n",m_1, m_2, m_3);*/
			return (std::pow(z, f_z_m)) / (std::pow(z, f_z_m) + std::pow( (1.0-z), f_z_m)); 
		}
		vrFloat deri_f_z(const vrFloat z)const
		{
			/*vrInt mmmm;
			mmmm=3;
			const vrFloat m_3 = (mmmm * std::pow(z, mmmm-1) * std::pow( (1.0-z), mmmm-1)) / (std::pow( (std::pow(z, mmmm) + std::pow( (1.0-z), mmmm)), 2));
			mmmm=2;
			const vrFloat m_2 = (mmmm * std::pow(z, mmmm-1) * std::pow( (1.0-z), mmmm-1)) / (std::pow( (std::pow(z, mmmm) + std::pow( (1.0-z), mmmm)), 2));
			mmmm=1;
			const vrFloat m_1 = (mmmm * std::pow(z, mmmm-1) * std::pow( (1.0-z), mmmm-1)) / (std::pow( (std::pow(z, mmmm) + std::pow( (1.0-z), mmmm)), 2));
			printf("deri_f_z : [m=1, %f] [m=2, %f] [m=3, %f]\n",m_1, m_2, m_3);*/
			//const vrFloat bottom = (std::pow(z, f_z_m) + std::pow( (1.0-z), f_z_m));
			//const vrFloat return_0 = (f_z_m * std::pow(z, f_z_m-1) * bottom - (std::pow(z, f_z_m) * (f_z_m * std::pow(z, f_z_m-1) - f_z_m * std::pow( (1.0-z), f_z_m-1)) )) / (bottom * bottom);
			const vrFloat return_1 = (f_z_m * std::pow(z, f_z_m-1) * std::pow( (1.0-z), f_z_m-1)) / (std::pow( (std::pow(z, f_z_m) + std::pow( (1.0-z), f_z_m)), 2));
			//const vrFloat _f_z =  f_z(z);
			//const vrFloat return_2 = (f_z_m * std::pow(_f_z, f_z_m-1) /** std::pow( (1.0-z), f_z_m-1)*/) / (std::pow( (std::pow(_f_z, f_z_m) + std::pow( (1.0-_f_z), f_z_m)), 2));
			//printf("return_0 [%f] return_1 [%f] return_2 [%f] \n", return_0, return_1, return_2);vrPause;
			return return_1;
			//const vrFloat _f_z =  f_z(z);
			//return (f_z_m * std::pow(_f_z, f_z_m-1) /** std::pow( (1.0-z), f_z_m-1)*/) / (std::pow( (std::pow(_f_z, f_z_m) + std::pow( (1.0-_f_z), f_z_m)), 2));
		}

		vrFloat deri_z_s(const MyVec2ParamSpace& w_bar_section)const
		{
			return (w_bar_section[1] - w_bar_section[0]);
		}

		vrFloat theta_bar_new(const vrFloat _f_z_s)const
		{
			return (numbers::MyPI * _f_z_s - numbers::PI_2);
		}

		vrFloat jacobi_Sigmoidal(const MyVec2ParamSpace& w_bar_section, const vrFloat z)const
		{
			
			return numbers::MyPI * deri_f_z(z) * deri_z_s(w_bar_section);
		}

	public:
		MyVec2ParamSpace Pedal_Vector_In_eta[SubTriSize];
#endif//USE_Sigmoidal
	};

	

	class TriangleElem;
	typedef boost::shared_ptr< TriangleElem > TriangleElemPtr;
	class TriangleElem
	{
	public:
#if USE_Sigmoidal
		enum{idx_theta_doubleLayer=0, idx_rho_doubleLayer=1, idx_weight_doubleLayer=2, doubleLayerSize = 3};
		enum{idx_theta_singleLayer=0, idx_weight_singleLayer=1, idx_rho_bar_singleLayer=2, idx_Jacobi_singleLayer = 3, singleLayerSize=4};
#endif

		class TriangleElemCompare4DualPlus
		{
		public:
			TriangleElemCompare4DualPlus(VertexPtr vtxPtr[])
			{
				vertex_id[0] = vtxPtr[0]->getId();
				vertex_id[1] = vtxPtr[1]->getId();
				vertex_id[2] = vtxPtr[2]->getId();
			}

			bool operator()(TriangleElemPtr& tri)
			{
				const vrInt tri_v0_id = (*tri).getVertex(0)->getId();
				const vrInt tri_v1_id = (*tri).getVertex(1)->getId();
				const vrInt tri_v2_id = (*tri).getVertex(2)->getId();

				return numbers::isEqual(tri_v0_id, vertex_id[0]) && 
					numbers::isEqual(tri_v1_id, vertex_id[1]) && 
					numbers::isEqual(tri_v2_id, vertex_id[2]);
			}
		private:
			MyVec3I vertex_id;
		};
		

	public:
		TriangleElem(VertexPtr vertexes[], const TriangleSetType type);
		~TriangleElem();
	public:
		MyInt getID()const{ return m_nID; }
		void setId(MyInt id){ m_nID = id; }
		void get_dof_indices(MyVector9I &vecDofs);
		void get_postion(MyVector9& Pj_FEM);
		VertexPtr getVertex(unsigned idx){ return m_elem_vertex[idx]; }
		MyFloat getJxW(unsigned q)const{ return JxW_values[q]; }

		MyInt searchVtxIndexByVtxId(const MyInt nId);
		MyInt searchVtxIndexByVtxPos(const MyVec3& pos);


		/*void setTriElemDiscontinuous(){ m_TriElemType = DisContinuous; }
		bool isDiscontinuous()const{ return DisContinuous == m_TriElemType; }*/
		void computeTriElemContinuousType();
		//void setTriElemContinuousType(TriElemType type){ m_TriElemType = type; }		
		bool isContinuous()const{ return Continuous == m_TriElemType; }
		bool isDiscontinuous()const{ return DisContinuous == m_TriElemType; }

		void setElemEndPoint(const MyVec3& endPt0, const MyVec3& endPt1, const MyVec3& endPt2)
		{
			m_elemEndPts.row(0) = endPt0.transpose(); 
			m_elemEndPts.row(1) = endPt1.transpose(); 
			m_elemEndPts.row(2) = endPt2.transpose();
		}
		void setElemVtxPoint(const MyVec3& vtxPt0, const MyVec3& vtxPt1, const MyVec3& vtxPt2)
		{ 
			m_elemVtxPts.row(0) = vtxPt0.transpose(); 
			m_elemVtxPts.row(1) = vtxPt1.transpose(); 
			m_elemVtxPts.row(2) = vtxPt2.transpose();
		}

		MyVec3 getElemEndPoint(MyInt idx)const{ return m_elemEndPts.row(idx).transpose(); }
		MyVec3 getElemVtxPoint(MyInt idx)const{ return m_elemVtxPts.row(idx).transpose(); }
		MyVec3 getElemNormals()const{ return m_TriElemNormal; }
		void setElemNormals(const MyVec3& _normals){ m_TriElemNormal = _normals; }

		void getKernelParameters(const MyInt gpt/*0,1,2*/, const MyVec3& collectPt, MyFloat& JxW, MyVec3& normals, MyFloat& r, MyVec3& dr, MyFloat& drdn);
		void calculateJumpTerm_smooth(MyMatrix_3X3& jumpTerm);
		//void calculateJumpTerm_Guiggiani(const MyVec3& n1, const MyVec3& n2, const MyFloat mu, MyMatrix_3X3& jumpTerm);
		
		

		void compute_Shape_Deris_Jacobi_SST_3D();
		

		void initialize();
		void assembleSystemMatrix();
		void computeShapeFunction();
		void computeJxW();
		void computeShapeGrad();
		static void computeGaussPoint();
		void printElemInfo()const;

		MyVec3 makeDistance_r_positive(const MyVec3 & r);

		MyVec3 getTriCenter()
		{			
			return (m_elemEndPts.row(0).transpose() + m_elemEndPts.row(1).transpose() + m_elemEndPts.row(2).transpose()) / 3;
		}


	public:
		static int getTriangleSize(){ return s_Triangle_Cache.size(); }
		static std::vector< TriangleElemPtr >& getTriangleVector(){ return s_Triangle_Cache; }
		static TriangleElemPtr getTriangle(int idx){ return s_Triangle_Cache[idx]; }
		static void CountSurfaceType();
		static TriangleElemPtr makeTriangleElem4DualPlus(VertexPtr vtxPtr[], VertexPtr endVtxPtr[], const TriangleSetType type);
		

		static MyFloat lineShapeFunction(const MyVec2ParamSpace& localCoords, const MyInt n);
		static MyFloat lineDeris(const MyVec2ParamSpace& localCoords, const MyInt n, const MyInt eta);
		static MyFloat jacobian(const MyVec3& vtxEndPt0, const MyVec3& vtxEndPt1, const MyVec3& vtxEndPt2);
		
		static MyFloat lineShapeFunction_1(const MyFloat x, const MyFloat x0, const MyFloat x1);
		static MyFloat lineShapeFunction_2(const MyFloat x, const MyFloat x0, const MyFloat x1);
		static MyFloat lineDeris_0(const MyFloat /*x*/, const MyFloat x0, const MyFloat x1);
		static MyFloat lineDeris_1(const MyFloat /*x*/, const MyFloat x0, const MyFloat x1);
	private:
		MyInt m_nID;
		VertexPtr m_elem_vertex[Geometry::vertexs_per_tri];
		MyFloat JxW_values[Geometry::vertexs_per_tri];
		MyMatrix_3X3 m_elemEndPts, m_elemVtxPts;
		MyVec3 m_TriElemNormal;

		/*row 0 is ( N1(XSI,ETA) = 1-XSI-ETA; N2(XSI,ETA) = XSI; N3(XSI,ETA) = ETA) of gauss 0 point ;
		  row 1 is ( N1(XSI,ETA) = 1-XSI-ETA; N2(XSI,ETA) = XSI; N3(XSI,ETA) = ETA) of gauss 1 point ;
		  row 2 is ( N1(XSI,ETA) = 1-XSI-ETA; N2(XSI,ETA) = XSI; N3(XSI,ETA) = ETA) of gauss 2 point ; */
		MyMatrix_3X3 shapeFunctionN; //row is gauss point, col is vertex

		//      ^
		//    1 | 2
		//      | |.
		//    Y | | .
		//      | |  .
		//    0 | 0---1
		//      +------->
		//        0 X 1
		static MyVec2ParamSpace s_paramSpace[Geometry::vertexs_per_tri];//[0,+1]

		// 0 : (¦Î1, ¦Ç1) = (1/6, 1/6) , (¦Î2, ¦Ç2) = (2/3 , 1/6) , (¦Î3, ¦Ç3) = (1/6 , 2/3) , w1 = w2 = w3 = 1/3
		static MyVec2ParamSpace s_gaussPtInParamSpace[Geometry::vertexs_per_tri];//[-sqrt(2/3), +sqrt(2/3)]

		// w1 = w2 = w3 = 1/3
		static MyFloat s_gaussPtWeigth[Geometry::vertexs_per_tri];
		
		/*deris_dN[gauss  point id]:[N0, N1, N2] : [XSI, ETA]*/
		/*eg: deris_dN[2][1][0]: dN1 / dXSI of gauss point 2*/
		MyMatrix_3X2 deris_dN[Geometry::vertexs_per_tri];
		MyMatrix_3X3 Q_Discontinuous;
#if USE_NEW_VERTEX
	public:
		TriangleSetType getTriSetType()const{return m_TriSetType;}
		DisContinuousType getTriContinuousType()const{return m_ElemDisContinuousType;}
	private:
		TriangleSetType m_TriSetType;
#else
		
#endif
		TriElemType m_TriElemType;
		DisContinuousType m_ElemDisContinuousType;
		static std::vector< TriangleElemPtr > s_Triangle_Cache;

#if USE_Mantic_CMat
	public:

#if USE_UniformSampling
		TriangleElemData_DisContinuous& get_m_data_SST_3D(const int idx=0){return m_data_SST_3D_Elem[idx];}
		TriangleElemData_DisContinuous m_data_SST_3D_Elem[Geometry::vertexs_per_tri];
#else//USE_UniformSampling
		TriangleElemData_DisContinuous m_data_SST_3D;
#endif//USE_UniformSampling
		


#if !SPEEDUP_5_31
		/*enum{GaussPointSize_xi_In_Theta = 18};
		enum{GaussPointSize_xi_In_Rho = 6};
		enum{GaussPointSize_eta_In_Theta = 18};
		enum{GaussPointSize_eta_In_Rho = 6};

		enum{GaussPointSize_xi_In_Theta_DisContinuous = 180};
		enum{GaussPointSize_xi_In_Rho_DisContinuous = 60};
		enum{GaussPointSize_eta_In_Theta_DisContinuous = 180};
		enum{GaussPointSize_eta_In_Rho_DisContinuous = 60};

		enum{GaussPointSize_eta_In_Theta_SubTri = 180};
		enum{GaussPointSize_eta_In_Rho_SubTri = 60};*/
#else
		/*
		enum{GaussPointSize_xi_In_Theta = 18};
		enum{GaussPointSize_xi_In_Rho = 18};
		enum{GaussPointSize_eta_In_Theta = 18};
		enum{GaussPointSize_eta_In_Rho = 18};

		enum{GaussPointSize_xi_In_Theta_DisContinuous = 18};
		enum{GaussPointSize_xi_In_Rho_DisContinuous = 18};
		enum{GaussPointSize_eta_In_Theta_DisContinuous = 18};
		enum{GaussPointSize_eta_In_Rho_DisContinuous = 18};

#if USE_360_Sample
		enum{GaussPointSize_eta_In_Theta_SubTri = 54};
		enum{GaussPointSize_eta_In_Rho_SubTri = 54};
#else//USE_360_Sample
		enum{GaussPointSize_eta_In_Theta_SubTri = 18};
		enum{GaussPointSize_eta_In_Rho_SubTri = 18};
#endif//USE_360_Sample

*/
#endif
		


	private:
		VertexPtr m_elem_vertex_endVtx[Geometry::vertexs_per_tri];
	public:
		void setEndVtxPtr(VertexPtr endVtxPtr[])
		{
			m_elem_vertex_endVtx[0] = endVtxPtr[0];
			m_elem_vertex_endVtx[1] = endVtxPtr[1];
			m_elem_vertex_endVtx[2] = endVtxPtr[2];
		}
		VertexPtr getEndVtxPtr(const vrInt idx){return m_elem_vertex_endVtx[idx];}
		MyInt searchVtxIndexByEndVtxId(const MyInt nId);
#endif

	public:
		
		vrInt getTriangleRegionId()const{Q_ASSERT(Invalid_Id != m_tri4RegionId);return m_tri4RegionId;}
		void setTriangleRegionId(vrInt id){Q_ASSERT(Invalid_Id == m_tri4RegionId);m_tri4RegionId = id;}
	private:
		vrInt m_tri4RegionId;

	};

	
}
#endif//_bemTriangleElem_h_