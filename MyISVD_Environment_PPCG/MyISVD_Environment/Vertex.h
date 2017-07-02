#pragma once

#include "VR_Global_Define.h"
#include "constant_numbers.h"
namespace YC
{
	class Cell;
	typedef boost::shared_ptr< Cell > CellPtr;

	class Vertex;
	typedef boost::shared_ptr< Vertex > VertexPtr;

	class Vertex
	{
	public:
		class VertexCompare
		{
		public:
			VertexCompare(MyFloat x,MyFloat y,MyFloat z):m_point(x,y,z){}

			bool operator()(const VertexPtr& p)const
			{
				return  (numbers::isZero(m_point(0) - (*p).m_point(0))) && 
					(numbers::isZero(m_point(1) - (*p).m_point(1))) && 
					(numbers::isZero(m_point(2) - (*p).m_point(2)));
			}
		private:
			MyDenseVector m_point;
		};
	public:
		
		Vertex(MyFloat x,MyFloat y,MyFloat z);
		~Vertex(void);

#if USE_MultidomainIndependent
	public:
		class VertexCompareDMI
		{
		public:
			VertexCompareDMI(MyFloat x, MyFloat y, MyFloat z, int did) :m_point(x, y, z), m_did(did){}

			bool operator()(const VertexPtr& p)const
			{
				return  (numbers::isZero(m_point(0) - (*p).m_point(0))) &&
					(numbers::isZero(m_point(1) - (*p).m_point(1))) &&
					(numbers::isZero(m_point(2) - (*p).m_point(2))) && (m_did == (*p).getFromDomainId());
			}
		private:
			MyDenseVector m_point;
			int m_did;
		};
		void setCreateOrder(int o){ m_nOrder = o; }
		int getCreateOrder()const{ return m_nOrder; }
		void setFromDomainId(int id){ if (!(m_FromDomainId == id || m_FromDomainId == Invalid_Id)){ printf("m_FromDomainId(%d)  id(%d)\n", m_FromDomainId, id); MyPause; } m_FromDomainId = id; }
		int getFromDomainId()const{ return m_FromDomainId; }
		void setALM_Mate(VertexPtr ptr){ m_ALM_Ptr = ptr; }
		bool hasALM_Mate(){ return m_ALM_Ptr != MyNull; }
		bool isValidDof_Q(){ return m_ALM_Dofs_Q[0] != Invalid_Id; }
		MyVectorI getDofs_Q()const{ return m_ALM_Dofs_Q; }
		void setDof_Q(unsigned dof_0, unsigned dof_1, unsigned dof_2)
		{
			Q_ASSERT(!isValidDof_Q()); m_ALM_Dofs_Q = MyVectorI(dof_0, dof_1, dof_2);
		}
		bool isValidDof_DMI(){ return m_dofs_DMI[0] != Invalid_Id; }
		MyVectorI getDofs_DMI()const{ return m_dofs_DMI; }
		void setDof_DMI(unsigned dof_0, unsigned dof_1, unsigned dof_2){ Q_ASSERT(!isValidDof_DMI()); m_dofs_DMI = MyVectorI(dof_0, dof_1, dof_2); }

		bool isValidDof_DMI_Global(){ return m_dofs_DMI_Global[0] != Invalid_Id; }
		MyVectorI getDofs_DMI_Global()const{ return m_dofs_DMI_Global; }
		void setDof_DMI_Global(unsigned dof_0, unsigned dof_1, unsigned dof_2){ Q_ASSERT(!isValidDof_DMI_Global()); m_dofs_DMI_Global = MyVectorI(dof_0, dof_1, dof_2); }
		void setDof_DMI_Global(const MyVectorI& dofs){ Q_ASSERT(!isValidDof_DMI_Global()); m_dofs_DMI_Global = dofs; }

		VertexPtr getALMPtr(){ return m_ALM_Ptr; }
		static VertexPtr makeVertex_DMI(const MyPoint& point, const int did);
		static void makeCellVertex_DMI(MyPoint center, MyFloat radius, VertexPtr vertexes[], const int did);
	private:
		int m_FromDomainId;
		int m_nOrder;
		VertexPtr m_ALM_Ptr;
		MyVectorI m_ALM_Dofs_Q;
		MyVectorI m_dofs_DMI;
		MyVectorI m_dofs_DMI_Global;
	public:
		static std::vector< std::pair< VertexPtr, VertexPtr > > s_vector_pair_ALM;
#endif
	public:
		
		void setId(int id){m_nID = id;}
		int getId()const{return m_nID;}
		MyPoint getPos()const{return m_point;}
		bool isValidGlobalDof(){return m_dofs[0] != Invalid_Id;}
		MyVectorI getGlobalDofs()const{return m_dofs;}
		void setGlobalDof(unsigned dof_0,unsigned dof_1,unsigned dof_2){Q_ASSERT(!isValidGlobalDof());m_dofs = MyVectorI(dof_0,dof_1,dof_2);}
		void setIsBC(bool flag){m_bIsBC=flag;}
		bool isBC()const{return m_bIsBC;}
	public:
		static VertexPtr makeVertex(const MyPoint& point);
		static void makeCellVertex(MyPoint center, MyFloat radius, VertexPtr vertexes[] );
		static int getVertexSize(){return s_vertexCache.size();}
		static VertexPtr Vertex::getVertex(int idx){return s_vertexCache[idx];}
	private:
		static std::vector< VertexPtr > s_vertexCache;
	private:
		MyPoint m_point;
		int m_nID;
		MyVectorI m_dofs;
		bool m_bIsBC;

#if USE_MODAL_WARP
	public:
		int getShareCellCount(){ return m_vec_ShareCell.size(); }

		std::vector< CellPtr > m_vec_ShareCell;
#endif
	};
}

