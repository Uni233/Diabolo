#include "MyVBOLineSet.h"

#include "cookbookogl.h"
#include "glutils.h"

void MyVBOLineSet::render() {
	glBindVertexArray(vaoHandle);
	glDrawElements(GL_LINES, m_nLinesCount * 2, GL_UNSIGNED_INT, ((GLubyte *)NULL + (0)));
}

void MyVBOLineSet::initialize(const int nLinesCount, float * pos, int * idx)
{
	m_nLinesCount = nLinesCount;
	glGenVertexArrays( 1, &vaoHandle );
	glBindVertexArray(vaoHandle);

	
	glGenBuffers(2, handle);

	glBindBuffer(GL_ARRAY_BUFFER, handle[0]);
	glBufferData(GL_ARRAY_BUFFER, m_nLinesCount * 2 * 3 * sizeof(float), pos, GL_STATIC_DRAW);
	glVertexAttribPointer( (GLuint)0, 3, GL_FLOAT, GL_FALSE, 0, ((GLubyte *)NULL + (0)) );
	glEnableVertexAttribArray(0);  // Vertex position

	/*glBindBuffer(GL_ARRAY_BUFFER, handle[1]);
	glBufferData(GL_ARRAY_BUFFER, 24 * 3 * sizeof(float), n, GL_STATIC_DRAW);
	glVertexAttribPointer( (GLuint)1, 3, GL_FLOAT, GL_FALSE, 0, ((GLubyte *)NULL + (0)) );
	glEnableVertexAttribArray(1);  // Vertex normal

	glBindBuffer(GL_ARRAY_BUFFER, handle[2]);
	glBufferData(GL_ARRAY_BUFFER, 24 * 2 * sizeof(float), tex, GL_STATIC_DRAW);
	glVertexAttribPointer( (GLuint)2, 2, GL_FLOAT, GL_FALSE, 0, ((GLubyte *)NULL + (0)) );
	glEnableVertexAttribArray(2);  // texture coords
	*/

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, handle[1]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_nLinesCount * 2 * sizeof(GLuint), idx, GL_STATIC_DRAW);

	glBindVertexArray(0);
}

void MyVBOLineSet::updateLineSet(const int nLinesCount, float3 * pos, int2 * idx)
{
	m_nLinesCount = nLinesCount;
	//glGenVertexArrays( 1, &vaoHandle );
	glBindVertexArray(vaoHandle);

	
	//glGenBuffers(2, handle);

	glBindBuffer(GL_ARRAY_BUFFER, handle[0]);
	glBufferData(GL_ARRAY_BUFFER, m_nLinesCount * 2 * sizeof(float3), pos, GL_STATIC_DRAW);
	glVertexAttribPointer( (GLuint)0, 3, GL_FLOAT, GL_FALSE, 0, ((GLubyte *)NULL + (0)) );
	glEnableVertexAttribArray(0);  // Vertex position

	/*glBindBuffer(GL_ARRAY_BUFFER, handle[1]);
	glBufferData(GL_ARRAY_BUFFER, 24 * 3 * sizeof(float), n, GL_STATIC_DRAW);
	glVertexAttribPointer( (GLuint)1, 3, GL_FLOAT, GL_FALSE, 0, ((GLubyte *)NULL + (0)) );
	glEnableVertexAttribArray(1);  // Vertex normal

	glBindBuffer(GL_ARRAY_BUFFER, handle[2]);
	glBufferData(GL_ARRAY_BUFFER, 24 * 2 * sizeof(float), tex, GL_STATIC_DRAW);
	glVertexAttribPointer( (GLuint)2, 2, GL_FLOAT, GL_FALSE, 0, ((GLubyte *)NULL + (0)) );
	glEnableVertexAttribArray(2);  // texture coords
	*/

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, handle[1]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_nLinesCount  * sizeof(int2), idx, GL_STATIC_DRAW);

	glBindVertexArray(0);
}