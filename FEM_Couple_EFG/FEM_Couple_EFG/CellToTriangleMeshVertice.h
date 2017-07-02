#ifndef _CellToTriangleMeshVertice_h
#define _CellToTriangleMeshVertice_h

#include <vector>

class CellToTriangleMeshVertice
{
public:
	CellToTriangleMeshVertice():m_nCellIdx(-1){}
	int m_nCellIdx;
	std::vector< int > m_vecRelatedVerticeIdx;
};
#endif//_CellToTriangleMeshVertice_h