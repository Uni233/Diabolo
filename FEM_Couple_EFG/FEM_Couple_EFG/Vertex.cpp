#include "stdafx.h"
#include "Vertex.h"
#include "Frame/Axis_YC.h"

namespace VR_FEM
{
	std::vector< VertexPtr > Vertex::s_vertexCache;
	VertexPtr Vertex::s_invalid_vertex(new Vertex(-1,-1,-1));

	VertexPtr Vertex::getVertex(int idx)
	{
		return s_vertexCache[idx];
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
		//FEM Ë³Ðò
		static MyDenseVector step[Geometry::vertexs_per_cell] = {MyDenseVector(-1,-1,-1), MyDenseVector(1,-1,-1),
																 MyDenseVector(-1,1,-1)	, MyDenseVector(1,1,-1),
																 MyDenseVector(-1,-1,1)	, MyDenseVector(1,-1,1),
																 MyDenseVector(-1,1,1)	, MyDenseVector(1,1,1)};
		
		for (int v=0;v < Geometry::vertexs_per_cell;++v)
		{
			vertexes[v] = makeVertex((center + radius * step[v]));
			//printf("{%f,%f,%f}\n",vertexes[v]->getPos()[0],vertexes[v]->getPos()[1],vertexes[v]->getPos()[2]);
		}
		//MyPause;
	}

	void Vertex::computeRotationMatrix(const MyVector& globalDisplacement)
	{
		//MyDenseVector vertexDisplaceCenter,vertexDisplaceE[3];
		MyVectorI centerDofs = getDofs();
		m_transformVector = MyDenseVector(m_point[0]+globalDisplacement[centerDofs[0]],m_point[1]+globalDisplacement[centerDofs[1]],m_point[2]+globalDisplacement[centerDofs[2]]);
		

		MyVectorI e1Dofs = m_vecRotationE_3[0]->getDofs();
		MyDenseVector e1Pos = m_vecRotationE_3[0]->getPos() - m_transformVector;
		MyDenseVector e1(e1Pos[0]+globalDisplacement[e1Dofs[0]],e1Pos[1]+globalDisplacement[e1Dofs[1]],e1Pos[2]+globalDisplacement[e1Dofs[2]]);

		MyVectorI e2Dofs = m_vecRotationE_3[1]->getDofs();
		MyDenseVector e2Pos = m_vecRotationE_3[1]->getPos() - m_transformVector;
		MyDenseVector e2(e2Pos[0]+globalDisplacement[e2Dofs[0]],e2Pos[1]+globalDisplacement[e2Dofs[1]],e2Pos[2]+globalDisplacement[e2Dofs[2]]);

		MyVectorI e3Dofs = m_vecRotationE_3[2]->getDofs();
		MyDenseVector e3Pos = m_vecRotationE_3[2]->getPos() - m_transformVector;
		MyDenseVector e3(e3Pos[0]+globalDisplacement[e3Dofs[0]],e3Pos[1]+globalDisplacement[e3Dofs[1]],e3Pos[2]+globalDisplacement[e3Dofs[2]]);

		MyDenseVector n1 = ((e1 + e2 + e3) / 3.0f);
		n1.normalize();

		e1.normalize();
		MyDenseVector n2 = n1.cross(e1);

		MyDenseVector n3 = n1.cross(n2);

		m_vertexRotationMatrixInDisplace.block(0,0,3,1) = n1;
		m_vertexRotationMatrixInDisplace.block(0,1,3,1) = n2;
		m_vertexRotationMatrixInDisplace.block(0,2,3,1) = n3;

		m_vertexRotationMatrixInDisplace = m_vertexRotationMatrixInDisplace * m_vertexRotationMatrixInRest.transpose();
	}

	void Vertex::initializeRotationMatrixInRest(const std::vector< VertexPtr >& vecE)
	{
		Q_ASSERT(3 == vecE.size());
		m_vecRotationE_3 = vecE;
		MyDenseVector e1,e2,e3;

		e1 = vecE[0]->getPos() - m_point;
		e2 = vecE[1]->getPos() - m_point;
		e3 = vecE[2]->getPos() - m_point;

		MyDenseVector n1 = ((e1 + e2 + e3) / 3.0f);
		n1.normalize();

		e1.normalize();
		MyDenseVector n2 = n1.cross(e1);

		MyDenseVector n3 = n1.cross(n2);

		m_vertexRotationMatrixInRest.block(0,0,3,1) = n1;
		m_vertexRotationMatrixInRest.block(0,1,3,1) = n2;
		m_vertexRotationMatrixInRest.block(0,2,3,1) = n3;
	}

	void Vertex::printFrame()
	{
		Matrix3<float> mat;
		mat.matrix[0].elems[0] = (float)m_vertexRotationMatrixInDisplace(0,0);
		mat.matrix[0].elems[1] = (float)m_vertexRotationMatrixInDisplace(0,1);
		mat.matrix[0].elems[2] = (float)m_vertexRotationMatrixInDisplace(0,2);

		mat.matrix[1].elems[0] = (float)m_vertexRotationMatrixInDisplace(1,0);
		mat.matrix[1].elems[1] = (float)m_vertexRotationMatrixInDisplace(1,1);
		mat.matrix[1].elems[2] = (float)m_vertexRotationMatrixInDisplace(1,2);

		mat.matrix[2].elems[0] = (float)m_vertexRotationMatrixInDisplace(2,0);
		mat.matrix[2].elems[1] = (float)m_vertexRotationMatrixInDisplace(2,1);
		mat.matrix[2].elems[2] = (float)m_vertexRotationMatrixInDisplace(2,2);
		Axis::Quaternion quater;
		quater.fromMatrix(mat);
		Axis::draw(Vec3<float>((float)m_transformVector[0],(float)m_transformVector[1],(float)m_transformVector[2]), quater, .01);
	}
}