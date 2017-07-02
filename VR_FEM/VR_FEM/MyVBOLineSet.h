#ifndef _MYVBOLINESET_H
#define _MYVBOLINESET_H

#include "vector_types.h"
class MyVBOLineSet
{

private:
	unsigned int vaoHandle;

public:
	MyVBOLineSet(){m_nLinesCount=0;}
	void initialize(const int nLinesCount, float * pos, int * idx);
	void updateLineSet(const int nLinesCount, float3 * pos, int2 * idx);
	void render();

	int getLineCount()const{return m_nLinesCount;}
	void setLineCount(const int nLineSize){m_nLinesCount = nLineSize;}
	unsigned int getVBOHandle(int idx){return handle[idx];}
private:
	int m_nLinesCount;
	unsigned int handle[2];
};
#endif//_MYVBOLINESET_H