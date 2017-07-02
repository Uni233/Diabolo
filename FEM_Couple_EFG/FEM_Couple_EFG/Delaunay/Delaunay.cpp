/********************************************************************************
	Copyright (C) 2004-2005 Sjaak Priester	

	This is free software; you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation; either version 2 of the License, or
	(at your option) any later version.

	This file is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Tinter; if not, write to the Free Software
	Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
********************************************************************************/

// Delaunay
// Class to perform Delaunay triangulation on a set of vertices
//
// Version 1.1 (C) 2005, Sjaak Priester, Amsterdam.
// - Removed bug which gave incorrect results for co-circular vertices.
//
// Version 1.0 (C) 2004, Sjaak Priester, Amsterdam.
// mailto:sjaak@sjaakpriester.nl


#include "StdAfx.h"
#include "Delaunay.h"
#include <iterator>

const float sqrt3 = 1.732050808F;

namespace VR_FEM
{
	
void triangle::SetCircumCircle()
{
	REAL x0 = m_Vertices[0]->GetX();
	REAL y0 = m_Vertices[0]->GetY();

	REAL x1 = m_Vertices[1]->GetX();
	REAL y1 = m_Vertices[1]->GetY();

	REAL x2 = m_Vertices[2]->GetX();
	REAL y2 = m_Vertices[2]->GetY();

	REAL y10 = y1 - y0;
	REAL y21 = y2 - y1;

	bool b21zero = y21 > -REAL_EPSILON && y21 < REAL_EPSILON;

	if (y10 > -REAL_EPSILON && y10 < REAL_EPSILON)
	{
		if (b21zero)	// All three vertices are on one horizontal line.
		{
			if (x1 > x0)
			{
				if (x2 > x1) x1 = x2;
			}
			else
			{
				if (x2 < x0) x0 = x2;
			}
			m_Center.X = (x0 + x1) * .5F;
			m_Center.Y = y0;
		}
		else	// m_Vertices[0] and m_Vertices[1] are on one horizontal line.
		{
			REAL m1 = - (x2 - x1) / y21;

			REAL mx1 = (x1 + x2) * .5F;
			REAL my1 = (y1 + y2) * .5F;

			m_Center.X = (x0 + x1) * .5F;
			m_Center.Y = m1 * (m_Center.X - mx1) + my1;
		}
	}
	else if (b21zero)	// m_Vertices[1] and m_Vertices[2] are on one horizontal line.
	{
		REAL m0 = - (x1 - x0) / y10;

		REAL mx0 = (x0 + x1) * .5F;
		REAL my0 = (y0 + y1) * .5F;

		m_Center.X = (x1 + x2) * .5F;
		m_Center.Y = m0 * (m_Center.X - mx0) + my0;
	}
	else	// 'Common' cases, no multiple vertices are on one horizontal line.
	{
		REAL m0 = - (x1 - x0) / y10;
		REAL m1 = - (x2 - x1) / y21;

		REAL mx0 = (x0 + x1) * .5F;
		REAL my0 = (y0 + y1) * .5F;

		REAL mx1 = (x1 + x2) * .5F;
		REAL my1 = (y1 + y2) * .5F;

		m_Center.X = (m0 * mx0 - m1 * mx1 + my1 - my0) / (m0 - m1);
		m_Center.Y = m0 * (m_Center.X - mx0) + my0;
	}

	REAL dx = x0 - m_Center.X;
	REAL dy = y0 - m_Center.Y;

	m_R2 = dx * dx + dy * dy;	// the radius of the circumcircle, squared
	m_R = (REAL) sqrt(m_R2);	// the proper radius

	// Version 1.1: make m_R2 slightly higher to ensure that all edges
	// of co-circular vertices will be caught.
	// Note that this is a compromise. In fact, the algorithm isn't really
	// suited for very many co-circular vertices.
	m_R2 *= 1.000001f;
}

// Function object to check whether a triangle has one of the vertices in SuperTriangle.
// operator() returns true if it does.
class triangleHasVertex
{
public:
	triangleHasVertex(const vertex SuperTriangle[3]) : m_pSuperTriangle(SuperTriangle)	{}
	bool operator()(const triangle& tri) const
	{
		for (int i = 0; i < 3; i++)
		{
			const vertex * p = tri.GetVertex(i);
			if (p >= m_pSuperTriangle && p < (m_pSuperTriangle + 3)) 
			{
				/*printf("triangleHasVertex\n");
				MyPause;*/
				return true;
			}
		}
		
		return false;
	}
protected:
	const vertex * m_pSuperTriangle;
};

// Function object to check whether a triangle is 'completed', i.e. doesn't need to be checked
// again in the algorithm, i.e. it won't be changed anymore.
// Therefore it can be removed from the workset.
// A triangle is completed if the circumcircle is completely to the left of the current vertex.
// If a triangle is completed, it will be inserted in the output set, unless one or more of it's vertices
// belong to the 'super triangle'.
class triangleIsCompleted
{
public:
	triangleIsCompleted(cvIterator itVertex, triangleSet& output, const vertex SuperTriangle[3])
		: m_itVertex(itVertex)
		, m_Output(output)
		, m_pSuperTriangle(SuperTriangle)
	{}
	bool operator()(const triangle& tri) const
	{
#if 1
		bool b = tri.IsLeftOf(m_itVertex);// true : m_itVertex is out of tri; false : m_itVertex is inside tri.

		if (b)
		{
			triangleHasVertex thv(m_pSuperTriangle);//true : vertex of tri is one of m_pSuperTriangle
			if (! thv(tri)) m_Output.insert(tri);
		}
		return b;
#endif
	}

protected:
	cvIterator m_itVertex;
	triangleSet& m_Output;
	const vertex * m_pSuperTriangle;
};

// Function object to check whether vertex is in circumcircle of triangle.
// operator() returns true if it does.
// The edges of a 'hot' triangle are stored in the edgeSet edges.
class vertexIsInCircumCircle
{
public:
	vertexIsInCircumCircle(cvIterator itVertex, edgeSet& edges) : m_itVertex(itVertex), m_Edges(edges)	{}
	bool operator()(const triangle& tri) const
	{
		bool b = tri.CCEncompasses(m_itVertex);

		if (b)
		{
			HandleEdge(tri.GetVertex(0), tri.GetVertex(1));
			HandleEdge(tri.GetVertex(1), tri.GetVertex(2));
			HandleEdge(tri.GetVertex(2), tri.GetVertex(0));
		}
		return b;
	}
protected:
	void HandleEdge(const vertex * p0, const vertex * p1) const
	{
		const vertex * pVertex0(NULL);
		const vertex * pVertex1(NULL);

		// Create a normalized edge, in which the smallest vertex comes first.
		if (* p0 < * p1)
		{
			pVertex0 = p0;
			pVertex1 = p1;
		}
		else
		{
			pVertex0 = p1;
			pVertex1 = p0;
		}

		edge e(pVertex0, pVertex1);

		// Check if this edge is already in the buffer
		edgeIterator found = m_Edges.find(e);

		if (found == m_Edges.end()) m_Edges.insert(e);		// no, it isn't, so insert
		else m_Edges.erase(found);							// yes, it is, so erase it to eliminate double edges
	}

	cvIterator m_itVertex;
	edgeSet& m_Edges;
};

void Delaunay::Triangulate(vertexSet& vertices, triangleSet& output)
{
	if (vertices.size() < 3) return;	// nothing to handle

	// Determine the bounding box.
	vIterator itVertex = vertices.begin();

	REAL xMin = itVertex->GetX();
	REAL yMin = itVertex->GetY();
	REAL xMax = xMin;
	REAL yMax = yMin;

	++itVertex;		// If we're here, we know that vertices is not empty.
	for (; itVertex != vertices.end(); itVertex++)
	{
		xMax = itVertex->GetX();	// Vertices are sorted along the x-axis, so the last one stored will be the biggest.
		REAL y = itVertex->GetY();
		if (y < yMin) yMin = y;
		if (y > yMax) yMax = y;
	}

	REAL dx = xMax - xMin;
	REAL dy = yMax - yMin;

	// Make the bounding box slightly bigger, just to feel safe.
	REAL ddx = dx * 0.01F;
	REAL ddy = dy * 0.01F;

	xMin -= ddx;
	xMax += ddx;
	dx += 2 * ddx;

	yMin -= ddy;
	yMax += ddy;
	dy += 2 * ddy;

	// Create a 'super triangle', encompassing all the vertices. We choose an equilateral triangle with horizontal base.
	// We could have made the 'super triangle' simply very big. However, the algorithm is quite sensitive to
	// rounding errors, so it's better to make the 'super triangle' just big enough, like we do here.
	vertex vSuper[3];

	vSuper[0] = vertex(xMin - dy * sqrt3 / 3.0F, yMin);	// Simple highschool geometry, believe me.
	vSuper[1] = vertex(xMax + dy * sqrt3 / 3.0F, yMin);
	vSuper[2] = vertex((xMin + xMax) * 0.5F, yMax + dx * sqrt3 * 0.5F);

	triangleSet workset;
	workset.insert(triangle(vSuper));


	for (itVertex = vertices.begin(); itVertex != vertices.end(); itVertex++)
	{
		// First, remove all 'completed' triangles from the workset.
		// A triangle is 'completed' if its circumcircle is entirely to the left of the current vertex.
		// Vertices are sorted in x-direction (the set container does this automagically).
		// Unless they are part of the 'super triangle', copy the 'completed' triangles to the output.
		// The algorithm also works without this step, but it is an important optimalization for bigger numbers of vertices.
		// It makes the algorithm about five times faster for 2000 vertices, and for 10000 vertices,
		// it's thirty times faster. For smaller numbers, the difference is negligible.
#if 1
		triangleSet ::iterator ii = workset.begin();
		triangleSet ::iterator endi = workset.end();
		//tIterator itEnd = remove_if(ii,endi , triangleIsCompleted(itVertex, output, vSuper));

		triangleIsCompleted thv(itVertex, output, vSuper);
		for (;ii != endi;)
		{
			if (thv(*ii))
			{
				ii = workset.erase(ii);
			}
			else
			{
				++ii;
			}
		}

		edgeSet edges;

		// A triangle is 'hot' if the current vertex v is inside the circumcircle.
		// Remove all hot triangles, but keep their edges.
		//itEnd = remove_if(workset.begin(), itEnd, vertexIsInCircumCircle(itVertex, edges));
		ii = workset.begin();
		endi = workset.end();
		vertexIsInCircumCircle vth(itVertex, edges);
		for (;ii != endi;)
		{
			if (vth(*ii))
			{
				ii = workset.erase(ii);
			}
			else
			{
				++ii;
			}
		}
		//workset.erase(itEnd, workset.end());	// remove_if doesn't actually remove; we have to do this explicitly.

		// Create new triangles from the edges and the current vertex.
		for (edgeIterator it = edges.begin(); it != edges.end(); it++)
			workset.insert(triangle(it->m_pV0, it->m_pV1, & (* itVertex)));
#endif
	}



	// Finally, remove all the triangles belonging to the 'super triangle' and move the remaining
	// triangles tot the output; remove_copy_if lets us do that in one go.
	tIterator where = output.begin();
	remove_copy_if(workset.begin(), workset.end(), inserter(output, where), triangleHasVertex(vSuper));
}
#if 0
void Delaunay::Triangulate(const bool flag,std::vector< VR_FEM::MC_Vertex >&  vecMCVertex,const int nLastVertexSize, int &nCurrentVertexSize, std::vector< VR_FEM::MC_Edge >& vecMCEdge,int& nCurrentEdgeSize, std::vector< VR_FEM::MC_Surface >& vecMCSurface, int& nCurrentFaceSize)
{
	if ( (nCurrentVertexSize - nLastVertexSize) < 3) return;	// nothing to handle

	MyFloat xMin = 10000.f,yMin = 10000.f,zMin = 10000.f;
	MyFloat xMax = -10000.f,yMax = -10000.f,zMax = -10000.f;

	for (unsigned v=nLastVertexSize;v < nCurrentVertexSize;++v)
	{
		MC_Vertex& curVertex = vecMCVertex[v];
		if (curVertex.m_VertexPos[0] < xMin)
		{
			xMin = curVertex.m_VertexPos[0];
		}
		else if (curVertex.m_VertexPos[0] > xMax)
		{
			xMax = curVertex.m_VertexPos[0];
		}

		if (curVertex.m_VertexPos[1] < yMin)
		{
			yMin = curVertex.m_VertexPos[1];
		}
		else if (curVertex.m_VertexPos[1] > yMax)
		{
			yMax = curVertex.m_VertexPos[1];
		}

		if (curVertex.m_VertexPos[2] < zMin)
		{
			zMin = curVertex.m_VertexPos[2];
		}
		else if (curVertex.m_VertexPos[2] > zMax)
		{
			zMax = curVertex.m_VertexPos[2];
		}
	}

	MyFloat dx = xMax - xMin;
	MyFloat dy = yMax - yMin;
	MyFloat dz = zMax - zMin;

	REAL ddx = dx * 0.01F;
	REAL ddy = dy * 0.01F;

	xMin -= ddx;
	xMax += ddx;
	dx += 2 * ddx;

	yMin -= ddy;
	yMax += ddy;
	dy += 2 * ddy;
	// Determine the bounding box.
	vIterator itVertex = vertices.begin();

	REAL xMin = itVertex->GetX();
	REAL yMin = itVertex->GetY();
	REAL xMax = xMin;
	REAL yMax = yMin;

	++itVertex;		// If we're here, we know that vertices is not empty.
	for (; itVertex != vertices.end(); itVertex++)
	{
		xMax = itVertex->GetX();	// Vertices are sorted along the x-axis, so the last one stored will be the biggest.
		REAL y = itVertex->GetY();
		if (y < yMin) yMin = y;
		if (y > yMax) yMax = y;
	}

	REAL dx = xMax - xMin;
	REAL dy = yMax - yMin;

	// Make the bounding box slightly bigger, just to feel safe.
	REAL ddx = dx * 0.01F;
	REAL ddy = dy * 0.01F;

	xMin -= ddx;
	xMax += ddx;
	dx += 2 * ddx;

	yMin -= ddy;
	yMax += ddy;
	dy += 2 * ddy;

	// Create a 'super triangle', encompassing all the vertices. We choose an equilateral triangle with horizontal base.
	// We could have made the 'super triangle' simply very big. However, the algorithm is quite sensitive to
	// rounding errors, so it's better to make the 'super triangle' just big enough, like we do here.
	vertex vSuper[3];

	vSuper[0] = vertex(xMin - dy * sqrt3 / 3.0F, yMin);	// Simple highschool geometry, believe me.
	vSuper[1] = vertex(xMax + dy * sqrt3 / 3.0F, yMin);
	vSuper[2] = vertex((xMin + xMax) * 0.5F, yMax + dx * sqrt3 * 0.5F);

	triangleSet workset;
	workset.insert(triangle(vSuper));


	for (itVertex = vertices.begin(); itVertex != vertices.end(); itVertex++)
	{
		// First, remove all 'completed' triangles from the workset.
		// A triangle is 'completed' if its circumcircle is entirely to the left of the current vertex.
		// Vertices are sorted in x-direction (the set container does this automagically).
		// Unless they are part of the 'super triangle', copy the 'completed' triangles to the output.
		// The algorithm also works without this step, but it is an important optimalization for bigger numbers of vertices.
		// It makes the algorithm about five times faster for 2000 vertices, and for 10000 vertices,
		// it's thirty times faster. For smaller numbers, the difference is negligible.
#if 1
		triangleSet ::iterator ii = workset.begin();
		triangleSet ::iterator endi = workset.end();
		//tIterator itEnd = remove_if(ii,endi , triangleIsCompleted(itVertex, output, vSuper));

		triangleIsCompleted thv(itVertex, output, vSuper);
		for (;ii != endi;)
		{
			if (thv(*ii))
			{
				ii = workset.erase(ii);
			}
			else
			{
				++ii;
			}
		}

		edgeSet edges;

		// A triangle is 'hot' if the current vertex v is inside the circumcircle.
		// Remove all hot triangles, but keep their edges.
		//itEnd = remove_if(workset.begin(), itEnd, vertexIsInCircumCircle(itVertex, edges));
		ii = workset.begin();
		endi = workset.end();
		vertexIsInCircumCircle vth(itVertex, edges);
		for (;ii != endi;)
		{
			if (vth(*ii))
			{
				ii = workset.erase(ii);
			}
			else
			{
				++ii;
			}
		}
		//workset.erase(itEnd, workset.end());	// remove_if doesn't actually remove; we have to do this explicitly.

		// Create new triangles from the edges and the current vertex.
		for (edgeIterator it = edges.begin(); it != edges.end(); it++)
			workset.insert(triangle(it->m_pV0, it->m_pV1, & (* itVertex)));
#endif
	}



	// Finally, remove all the triangles belonging to the 'super triangle' and move the remaining
	// triangles tot the output; remove_copy_if lets us do that in one go.
	tIterator where = output.begin();
	remove_copy_if(workset.begin(), workset.end(), inserter(output, where), triangleHasVertex(vSuper));
}
#endif
void Delaunay::TrianglesToEdges(const triangleSet& triangles, edgeSet& edges)
{
	for (ctIterator it = triangles.begin(); it != triangles.end(); ++it)
	{
		HandleEdge(it->GetVertex(0), it->GetVertex(1), edges);
		HandleEdge(it->GetVertex(1), it->GetVertex(2), edges);
		HandleEdge(it->GetVertex(2), it->GetVertex(0), edges);
	}
}

void Delaunay::HandleEdge(const vertex * p0, const vertex * p1, edgeSet& edges)
{
	const vertex * pV0(NULL);
	const vertex * pV1(NULL);

	if (* p0 < * p1)
	{
		pV0 = p0;
		pV1 = p1;
	}
	else
	{
		pV0 = p1;
		pV1 = p0;
	}

	// Insert a normalized edge. If it's already in edges, insertion will fail,
	// thus leaving only unique edges.
	edges.insert(edge(pV0, pV1));
}
}
