#include "Vertex.h"

namespace YC
{
	std::vector< VertexPtr > Vertex::s_vertexCache;

	Vertex::Vertex(MyFloat x,MyFloat y,MyFloat z)
		:m_point(x,y,z),
		 m_nID(Invalid_Id),
		 m_dofs(Invalid_Id,Invalid_Id,Invalid_Id),
#if USE_MultidomainIndependent
		 m_FromDomainId(Invalid_Id),
		 m_nOrder(Invalid_Id),
		 m_ALM_Ptr(MyNull),
		 m_ALM_Dofs_Q(Invalid_Id, Invalid_Id, Invalid_Id),
		 m_dofs_DMI(Invalid_Id, Invalid_Id, Invalid_Id),
		 m_dofs_DMI_Global(Invalid_Id, Invalid_Id, Invalid_Id),
#endif//USE_MultiDomain
		 m_bIsBC(false)
	{
	}


	Vertex::~Vertex(void)
	{
	}

#if USE_MultidomainIndependent
	std::vector< std::pair< VertexPtr, VertexPtr > > Vertex::s_vector_pair_ALM;

	bool searchContainDMI(const std::vector< VertexPtr >& RefVertexCache, const Vertex::VertexCompareDMI & Compare, int& idx)
	{
		idx = Invalid_Id;
		for (int i = 0; i < RefVertexCache.size(); ++i)
		{
			if (Compare(RefVertexCache[i]))
			{
				idx = i;
				return true;
			}
		}
		return false;
	}

	bool searchContain(const std::vector< VertexPtr >& RefVertexCache, const Vertex::VertexCompare & Compare, int& idx)
	{
		idx = Invalid_Id;
		for (int i = 0; i < RefVertexCache.size(); ++i)
		{
			if (Compare(RefVertexCache[i]))
			{
				idx = i;
				return true;
			}
		}
		return false;
	}

	VertexPtr Vertex::makeVertex_DMI(const MyPoint& point, const int did)
	{
		int idxDMI = Invalid_Id;
		if (!searchContainDMI(s_vertexCache, VertexCompareDMI(point[0], point[1], point[2], did), idxDMI)/*s_vertexCache.end() == itr*/)
		{
			//must be wrote here!
			int idx = Invalid_Id;
			VertexPtr tmpPtr = MyNull;
			//std::vector< VertexPtr >::iterator itr_0 = std::find_if(s_vertexCache.begin(),s_vertexCache.end(),VertexCompare(point[0],point[1],point[2]));
			if (searchContain(s_vertexCache, VertexCompare(point[0], point[1], point[2]), idx)/*itr_0 != s_vertexCache.end()*/)
			{
				tmpPtr = s_vertexCache[idx];//(*itr_0);
			}

			s_vertexCache.push_back(VertexPtr(new Vertex(point[0], point[1], point[2])));
			s_vertexCache[s_vertexCache.size() - 1]->setId(s_vertexCache.size() - 1);
			s_vertexCache[s_vertexCache.size() - 1]->setFromDomainId(did);
			s_vertexCache[s_vertexCache.size() - 1]->setCreateOrder(1);


			if (tmpPtr)
			{
				Q_ASSERT(((tmpPtr)->getFromDomainId()) != did);
				tmpPtr->setCreateOrder(0);
				tmpPtr->setALM_Mate(s_vertexCache[s_vertexCache.size() - 1]);
				s_vertexCache[s_vertexCache.size() - 1]->setALM_Mate(tmpPtr);
				s_vector_pair_ALM.push_back(std::make_pair(tmpPtr, s_vertexCache[s_vertexCache.size() - 1]));

				std::cout << "alm pair : " << point.transpose() << std::endl;
			}

			return s_vertexCache[s_vertexCache.size() - 1];
		}
		else
		{
			return s_vertexCache[idxDMI];//(*itr);
		}
	}

	void Vertex::makeCellVertex_DMI(MyPoint center, MyFloat radius, VertexPtr vertexes[], const int did)
	{
		//FEM Ë³Ðò
		static MyDenseVector step[Geometry::vertexs_per_cell] = { MyDenseVector(-1, -1, -1), MyDenseVector(1, -1, -1),
			MyDenseVector(-1, 1, -1), MyDenseVector(1, 1, -1),
			MyDenseVector(-1, -1, 1), MyDenseVector(1, -1, 1),
			MyDenseVector(-1, 1, 1), MyDenseVector(1, 1, 1) };

		for (int v = 0; v < Geometry::vertexs_per_cell; ++v)
		{
			vertexes[v] = makeVertex_DMI((center + radius * step[v]), did);
		}
	}
#endif

	void Vertex::makeCellVertex(MyPoint center, MyFloat radius, VertexPtr vertexes[] )
	{
		
		for (int v=0;v < Geometry::vertexs_per_cell;++v)
		{
			vertexes[v] = makeVertex((center + radius * Geometry::step[v]));

			//printf("{%f,%f,%f}\n",vertexes[v]->getPos()[0],vertexes[v]->getPos()[1],vertexes[v]->getPos()[2]);
		}
		//MyPause;
	}

	VertexPtr Vertex::makeVertex(const MyPoint& point)
	{
		std::vector< VertexPtr >::reverse_iterator itr = std::find_if(s_vertexCache.rbegin(),s_vertexCache.rend(),VertexCompare(point[0],point[1],point[2]));
		if ( s_vertexCache.rend() == itr )
		{
			//no find
			//printf("{%f,%f,%f}\n",point[0],point[1],point[2]);
			s_vertexCache.push_back( VertexPtr(new Vertex(point[0],point[1],point[2])) );
			s_vertexCache[s_vertexCache.size()-1]->setId(s_vertexCache.size()-1);
			return s_vertexCache[s_vertexCache.size()-1];
		}
		else
		{
			//find it
			//Q_ASSERT(false);
			/*printf("{%f,%f,%f}\n",point[0],point[1],point[2]);
			MyPause;*/
			return (*itr);
		}
	}
}
