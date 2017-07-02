#include "VR_Physic_Vertex.h"
#include "VR_Geometry.h"
namespace YC
{
	namespace Physics
	{
		namespace GPU
		{
			std::vector< VertexPtr > Vertex::s_vertexCache;
			VertexPtr Vertex::s_invalid_vertex(new Vertex(-1,-1,-1));

			VertexPtr Vertex::makeVertex(const MyPoint& point)
			{
				std::vector< VertexPtr >::reverse_iterator itr = std::find_if(s_vertexCache.rbegin(),s_vertexCache.rend(),VertexCompare(point[0],point[1],point[2]));
				if ( s_vertexCache.rend() == itr )
				{
					//no find
					//printf("new pos {%f,%f,%f}\n",point[0],point[1],point[2]);
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
			VertexPtr Vertex::searchVertex(const MyPoint& point)
			{
				std::vector< VertexPtr >::reverse_iterator itr = std::find_if(s_vertexCache.rbegin(),s_vertexCache.rend(),VertexCompare(point[0],point[1],point[2]));
				if ( s_vertexCache.rend() == itr )
				{
					printf("vertex not find!");
					system("pause");
					return s_invalid_vertex;
					//no find

				}
				else
				{
					return (*itr);
				}
			}
			void Vertex::makeCellVertex(MyPoint center, MyFloat radius, VertexPtr vertexes[] )
			{
				for (int v=0;v < Geometry::vertexs_per_cell;++v)
				{
					vertexes[v] = makeVertex((center + radius * Geometry::Cell_Vertex_Sequence[v]));
					//printf("{%f,%f,%f}\n",vertexes[v]->getPos()[0],vertexes[v]->getPos()[1],vertexes[v]->getPos()[2]);
				}
			}
			VertexPtr Vertex::getVertex(int idx)
			{
				return s_vertexCache[idx];
			}
		}//namespace GPU
	}//namespace Physics
}//namespace YC