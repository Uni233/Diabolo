#ifndef _VERTEX_H
#define _VERTEX_H
#include "VR_Global_Define.h"
#include "constant_numbers.h"
#include <boost/smart_ptr.hpp>

namespace VR_FEM
{
	class Vertex;
	typedef boost::shared_ptr< Vertex > VertexPtr;

	

	class Vertex
	{
	public:
		class VertexCompare
		{
		public:
			VertexCompare(MyFloat x,MyFloat y,MyFloat z):m_point(x,y,z){}

			bool operator()(VertexPtr& p)
			{
				return  (numbers::isZero(m_point(0) - (*p).m_point(0))) && 
					(numbers::isZero(m_point(1) - (*p).m_point(1))) && 
					(numbers::isZero(m_point(2) - (*p).m_point(2)));
			}
		private:
			MyDenseVector m_point;
		};
	public:
		friend class VertexCompare;
		Vertex(MyFloat x,MyFloat y,MyFloat z):m_point(x,y,z),m_nID(Invalid_Id),m_dofs(Invalid_Id,Invalid_Id,Invalid_Id),m_FromDomainId(Invalid_Id),m_nTmpLocalDomainId(Invalid_Id),m_nTmpCoupleDomainId(Invalid_Id){}
		void setId(int id){m_nID = id;}
	public:
		int getId()const{return m_nID;}
		MyPoint getPos()const{return m_point;}

		bool isValidGlobalDof(){return m_dofs[0] != Invalid_Id;}
		MyVectorI getDofs()const{return m_dofs;}
		unsigned getDof(unsigned idx)const{return m_dofs[idx];}
		void setDof(unsigned dof_0,unsigned dof_1,unsigned dof_2){Q_ASSERT(!isValidGlobalDof());m_dofs = MyVectorI(dof_0,dof_1,dof_2);}

		MyVectorI getLocalDof(MyInt id){return m_mapLocalDofs.at(id);}
		bool isValidLocalDof(MyInt id){return m_mapLocalDofs.find(id) != m_mapLocalDofs.end();}
		void setLocalDof(MyInt id,MyInt d0,MyInt d1,MyInt d2){Q_ASSERT(!isValidLocalDof(id));m_mapLocalDofs[id] = MyVectorI(d0,d1,d2);}
		const std::map< MyInt,MyVectorI >& getLocalDofsMap(){return m_mapLocalDofs;}

		MyVectorI getCoupleDof(MyInt id){return m_mapCoupleDofs.at(id);}
		bool isValidCoupleDof(MyInt id){return m_mapCoupleDofs.find(id) != m_mapCoupleDofs.end();}
		void setCoupleDof(MyInt id,MyInt d0,MyInt d1,MyInt d2){Q_ASSERT(!isValidCoupleDof(id));m_mapCoupleDofs[id] = MyVectorI(d0,d1,d2);}
		bool isBelongCoupleDomain(){return m_mapCoupleDofs.size() != 0;}
		//MyInt getCoupleDomainId(){if (m_mapCoupleDofs.size()>1){printf("%d  %d\n",m_mapCoupleDofs.begin()->first,(m_mapCoupleDofs.rbegin()->first));}printf("m_mapCoupleDofs.size() %d\n",m_mapCoupleDofs.size());Q_ASSERT(m_mapCoupleDofs.size() == 1);return (m_mapCoupleDofs.begin()->first);}

		//int makeCellVertex(const MyPoint& center,MyFloat dbRadius,std::vector< VertexPtr >& ref_vec_Vertex );
		void setFromDomainId(int id){if(!(m_FromDomainId == id ||m_FromDomainId == Invalid_Id )){printf("m_FromDomainId(%d)  id(%d)\n",m_FromDomainId,id);MyPause;} m_FromDomainId=id;}
		int getFromDomainId()const{return m_FromDomainId;}
	public:
		MyPoint m_point;
		int m_nID;
		MyVectorI m_dofs;
		std::map< MyInt,MyVectorI > m_mapLocalDofs;
		std::map< MyInt,MyVectorI > m_mapCoupleDofs;
		int m_FromDomainId;
	public:
		static VertexPtr makeVertex(const MyPoint& point);
		static VertexPtr searchVertex(const MyPoint& point);
		static void makeCellVertex(MyPoint center, MyFloat radius, VertexPtr vertexes[] );
		static int getVertexSize(){return s_vertexCache.size();}
		static VertexPtr getVertex(int idx);
	private:
		static std::vector< VertexPtr > s_vertexCache;
		static VertexPtr s_invalid_vertex;

		/////////////////// vertex co-rotation  ////////////////////////
	public:
		
		bool isValidOutgoingEdge()const{return !m_vecRotationE_3.empty();}
		const MyMatrix_3X3& getVertexRotationMatrix()const{return m_vertexRotationMatrixInDisplace;}
		void computeRotationMatrix(const MyVector& globalDisplacement);
		void initializeRotationMatrixInRest(const std::vector< VertexPtr >& vecE);
		void printFrame();
	private:
		MyMatrix_3X3 m_vertexRotationMatrixInRest,m_vertexRotationMatrixInDisplace;
		std::vector< VertexPtr > m_vecRotationE_3;
		MyDenseVector m_transformVector;

	public:
		MyInt m_nTmpLocalDomainId;
		MyInt m_nTmpCoupleDomainId;
		void setTmpLocalDomainId(MyInt id){m_nTmpLocalDomainId = id;}
		MyInt getTmpLocalDomainId()const{return m_nTmpLocalDomainId;}

		void setTmpCoupleDomainId(MyInt id){m_nTmpCoupleDomainId = id;}
		MyInt getTmpCoupleDomainId()const{return m_nTmpCoupleDomainId;}

		bool isTmpBoundary()const{return (m_nTmpLocalDomainId != Invalid_Id) && (m_nTmpCoupleDomainId != Invalid_Id);}

		MyInt getVertexShareDomainCount()const{return m_mapLocalDofs.size();}
	};	
}
#endif//_VERTEX_H