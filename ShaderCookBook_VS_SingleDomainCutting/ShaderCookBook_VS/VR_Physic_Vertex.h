#ifndef _VR_PHYSIC_VERTEX_H
#define _VR_PHYSIC_VERTEX_H
#include <boost/smart_ptr.hpp>
#include "constant_numbers.h"
#include "VR_Global_Define.h"
#include "VR_MACRO.h"
namespace YC
{
	namespace Physics
	{
		namespace GPU
		{
			class Vertex;
			typedef boost::shared_ptr< Vertex > VertexPtr;

			class Vertex
			{
				class VertexCompare
				{
				public:
					VertexCompare(MyFloat x,MyFloat y,MyFloat z):m_point(x,y,z){}

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
				Vertex(MyFloat x,MyFloat y,MyFloat z):m_point(x,y,z),m_nID(Invalid_Id),m_dofs(Invalid_Id,Invalid_Id,Invalid_Id){}
				void setId(int id){m_nID = id;}
			public:
				int getId()const{return m_nID;}
				MyPoint getPos()const{return m_point;}

				bool isValidGlobalDof(){return m_dofs[0] != Invalid_Id;}
				MyVectorI getDofs()const{return m_dofs;}
				unsigned getDof(unsigned idx)const{return m_dofs[idx];}
				void setDof(unsigned dof_0,unsigned dof_1,unsigned dof_2){Q_ASSERT(!isValidGlobalDof());m_dofs = MyVectorI(dof_0,dof_1,dof_2);}

				static VertexPtr makeVertex(const MyPoint& point);
				static VertexPtr searchVertex(const MyPoint& point);
				static void makeCellVertex(MyPoint center, MyFloat radius, VertexPtr vertexes[] );
				static int getVertexSize(){return s_vertexCache.size();}
				static VertexPtr getVertex(int idx);
			private:
				MyPoint m_point;
				int m_nID;
				MyVectorI m_dofs;
			private:
				static std::vector< VertexPtr > s_vertexCache;
				static VertexPtr s_invalid_vertex;
			};
		}//namespace GPU
	}//namespace Physics
}//namespace YC
#endif//_VR_PHYSIC_VERTEX_H