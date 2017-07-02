#ifndef _bemVertex_h_
#define _bemVertex_h_
#include "bemDefines.h"
#include <boost/smart_ptr.hpp>
#include "constant_numbers.h"
namespace VR
{
#if USE_DUAL
	typedef enum{Regular=0,Mirror_Positive=1, Mirror_Negative=2,CrackTip=3} VertexTypeInDual;
	typedef enum{Vtx_Continuous = 1, Vtx_DisContinuous = 2} VertexContinuousType;
	struct searchMirrorVertexNode
	{
		MyVec3 pos;
		vrInt vtxId;
	};

	class searchMirrorVertexNode_LessCompare
	{
	public:
		bool operator()(const searchMirrorVertexNode& lhs,const searchMirrorVertexNode& rhs)const
		{

			if( lhs.pos[2] < rhs.pos[2] )
			{   
				return true;
			}
			else if(lhs.pos[2] > rhs.pos[2])
			{   
				return false;
			}
			// Otherwise z is equal
			if( lhs.pos[1] < rhs.pos[1] )
			{   
				return true;
			}
			else if( lhs.pos[1] > rhs.pos[1] )
			{   
				return false;
			}
			// Otherwise z and y are equal
			if( lhs.pos[0] < rhs.pos[0] )
			{   
				return true;
			}
			/* Simple optimization Do not need this test
			If this fails or succeeded the result is false.
			else if( lhs.x > rhs.x )
			{    return false;
			}*/
			// Otherwise z and y and x are all equal
			return false;
		}
	};
#endif//#if USE_DUAL

	class TriangleElem;
	typedef boost::shared_ptr< TriangleElem > TriangleElemPtr;

	class Vertex;
	typedef boost::shared_ptr< Vertex > VertexPtr;

	class Vertex
	{
		class VertexComparePlus
		{
		public:
			VertexComparePlus(MyFloat x, MyFloat y, MyFloat z, TriangleSetType type) :m_point(x, y, z), m_TriangleSetType(type){}

			bool operator()(VertexPtr& p)
			{
				return  (numbers::isZero(m_point(0) - (*p).m_point(0))) &&
					(numbers::isZero(m_point(1) - (*p).m_point(1))) &&
					(numbers::isZero(m_point(2) - (*p).m_point(2))) &&
					(m_TriangleSetType == p->getTriangleSetType());
				/*return  (numbers::IsEqual(m_point(0) , (*p).m_point(0))) &&
				(numbers::IsEqual(m_point(1) , (*p).m_point(1))) &&
				(numbers::IsEqual(m_point(2) , (*p).m_point(2)));*/
			}
		private:
			MyDenseVector m_point;
			TriangleSetType m_TriangleSetType;
		};

		class VertexCompare
		{
		public:
			VertexCompare(MyFloat x, MyFloat y, MyFloat z) :m_point(x, y, z){}

			bool operator()(VertexPtr& p)
			{
				return  (numbers::isZero(m_point(0) - (*p).m_point(0))) &&
					(numbers::isZero(m_point(1) - (*p).m_point(1))) &&
					(numbers::isZero(m_point(2) - (*p).m_point(2)));
				/*return  (numbers::IsEqual(m_point(0) , (*p).m_point(0))) &&
				(numbers::IsEqual(m_point(1) , (*p).m_point(1))) &&
				(numbers::IsEqual(m_point(2) , (*p).m_point(2)));*/
			}
		private:
			MyDenseVector m_point;
		};

	public:
		Vertex(MyFloat x, MyFloat y, MyFloat z, TriangleSetType TriangleSetType) 
			:m_point(x, y, z), m_nID(Invalid_Id),m_dofs(Invalid_Id, Invalid_Id, Invalid_Id)
		{
#if USE_DUAL
			m_VertexTypeInDual = VertexTypeInDual::Regular;
#endif//USE_DUAL
			m_vtxContinuousType = Vtx_Continuous;
			m_TriangleSetType = TriangleSetType;

		}
		void setId(int id){ m_nID = id; }
		void setDof(unsigned dof_0, unsigned dof_1, unsigned dof_2){ vrASSERT(!isValidDofs()); m_dofs = MyVec3I(dof_0, dof_1, dof_2); }
	public:
		MyInt getId()const{ return m_nID; }
		MyVec3 getPos()const{ return m_point; }

		bool isValidDofs(){ return m_dofs[0] != Invalid_Id; }
		MyVec3I getDofs()const{ return m_dofs; }
		unsigned getDof(unsigned idx)const{ return m_dofs[idx]; }

		const std::vector< TriangleElemPtr >& getShareElement()const { return  m_vec_ShareTriangleElem; }
		void addShareTriangleElement(TriangleElemPtr triPtr);

		bool isSharedElementWithId(const MyInt nElemId);

		void setCMatrix(const vrMat3& c){m_CMatrix = c;}
		const vrMat3& getCMatrix()const{return m_CMatrix;}
		vrFloat getCij(vrInt i, vrInt j){return m_CMatrix.coeff(i,j);}

		const std::vector< TriangleElemPtr >& getNearRegion()const{return m_vec_ShareTriangleElem_Sorted_NearRegion;}
		void addNearRegion(TriangleElemPtr tri);
		void clearNearRegion(){m_vec_ShareTriangleElem_Sorted_NearRegion.clear();}
		vrInt getNearRegionSize()const{return m_vec_ShareTriangleElem_Sorted_NearRegion.size();}
	public:

#if USE_DUAL
		void computeVertexNormal();
		MyVec3 getVertexNormal()const{/*Q_ASSERT(isContinuousVertex());*/return m_vertexNormal;}		
		void setMirrorVertex(const vrInt mirrorId);
		void parserCrackTip();
		bool isMirrorVertex()const{return ( (m_VertexTypeInDual == VertexTypeInDual::Mirror_Positive) || (m_VertexTypeInDual == VertexTypeInDual::Mirror_Negative) );}
		VertexPtr getMirrorVertex();
		VertexTypeInDual getVertexTypeInDual()const{return m_VertexTypeInDual;}

#if USE_Nagetive_InDebugBeam
		void setVertexTypeInDual(VertexTypeInDual type){m_VertexTypeInDual = type;}
#endif //USE_Nagetive_InDebugBeam
		
		static void searchMirrorVertex_CrackTipVertex();
		static void TestMirrorInfo();
	private:
		MyVec3 m_vertexNormal;
		VertexTypeInDual m_VertexTypeInDual;
		std::set< vrInt > setMirrorVertexId;
	public:
#endif//USE_DUAL

#if USE_NEW_VERTEX
		TriangleSetType getTriangleSetType()const{return m_TriangleSetType;}
		TriangleSetType m_TriangleSetType;//Regular = 0, Positive = 1, Negative = 2
		static VertexPtr makeVertex4dualPlus(const TriangleSetType triType, const MyVec3& point);
		static VertexPtr makeVertex4dualPlusEndVtx(const MyVec3& point);
#else
		static VertexPtr makeVertex4dual(const vrInt nId, const MyVec3& point);
		static VertexPtr makeVertex(const MyVec3& point);
		static void makeTriangleVertex(const std::vector< MyVec3 >& coords, std::vector< VertexPtr >& vertexes);
#endif
		static VertexPtr searchVertex(const MyVec3& point);		
		static int getVertexSize(){ return s_vertexCache.size(); }
		static VertexPtr getVertex(int idx);
	private:
		MyVec3 m_point;
		MyInt m_nID;
		MyVec3I m_dofs;

		static std::vector< VertexPtr > s_vertexCache;
		static std::vector< VertexPtr > s_vertexCacheEndVtx;
		static VertexPtr s_invalid_vertex;
	private:
		std::vector< TriangleElemPtr > m_vec_ShareTriangleElem;
		std::vector< TriangleElemPtr > m_vec_ShareTriangleElem_Sorted_NearRegion;
		vrMat3 m_CMatrix;

#if USE_VDB
	public:
		vrFloat getStressInfo(){return m_StressVal;}
		void setStress(vrFloat val){m_StressVal += val;}
		static vrFloat getMaxStress(){return s_MaxStress;}
		static vrFloat getMinStress(){return s_MinStress;}
		static void setStressRange(vrFloat maxStress, vrFloat minStress){s_MaxStress = maxStress;s_MinStress = minStress;}
		static void clearStress();
	private:
		VR::MyFloat m_StressVal;
		static vrFloat s_MaxStress, s_MinStress;
#endif

	public:
		bool isContinuousVertex()const{return Vtx_Continuous == m_vtxContinuousType;}
		bool isDisContinuousVertex()const{return Vtx_DisContinuous == m_vtxContinuousType;}
		void setContinuousType(VertexContinuousType type){m_vtxContinuousType = type;}
	private:
		VertexContinuousType m_vtxContinuousType;

	};
}
#endif//_bemVertex_h_