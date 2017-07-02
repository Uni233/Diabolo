#ifndef _VR_GLSL_FONT_H
#define _VR_GLSL_FONT_H

#include <stdlib.h>
#include <string>
#include <vector>
#include <deque>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>

#include <glload/gl_4_0.h>
#include <glload/gl_load.h>
#include <GL/freeglut.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glutil/glutil.h>
#include <glmesh/glmesh.h>

#include "VR_MACRO.h"

#if SHOWFPS

namespace YC
{
	class MyGLSLFont
	{
#define x_scale (0.02f)
#define y_scale (0.95f)
	public:
		typedef std::deque<std::string> StringQueue;

		MyGLSLFont():g_windowSize(windowWidth, windowHeight)
		{
			g_cameraToClipMatrixUnif = 0;
			g_program = 0;
			g_pFont = NULL;
			g_pVertFmt = NULL;
			g_pStreamBuf = NULL;
			g_numGlyphsToDraw = 0;

			m_fps_pos_x = g_windowSize.x * x_scale;
			m_fps_pos_y = g_windowSize.y * y_scale;
			init();
		}
		virtual ~MyGLSLFont(){}

		void printTextOnGLSL(const char * fps);
		void resize(const int w, const int h);
	private:
		void init();
		void InitializeProgram();
		void InitializeVertexData();
		void PushGlyph( std::vector<GLfloat> &vecVertex, int loop, std::vector<glm::vec2> &positions, std::vector<glm::vec2> &texCoords );
		void AddGlyph(std::vector<GLfloat> &vecVertex, const glutil::GlyphQuad &theGlyph);
		void SendVerts(glmesh::Draw &imm, int index, std::vector<glm::vec2> &positions,	std::vector<glm::vec2> &texCoords);
		void DrawGlyph(glmesh::Draw &imm, const glutil::GlyphQuad &theGlyph);
		void DrawTextString(const std::string &text, const glm::vec2 &location);
	private:	

		std::string g_currString;
		StringQueue g_strings;
		GLint g_cameraToClipMatrixUnif;
		GLuint g_program;
		GLuint g_dataBufferObject;
		GLuint g_vao;

		glutil::Font *g_pFont;
		glmesh::VertexFormat *g_pVertFmt;
		glmesh::StreamBuffer *g_pStreamBuf;

		GLuint g_numGlyphsToDraw;
		/* GLUT callback Handlers */
		glm::ivec2 g_windowSize;
		int m_fps_pos_x;
		int m_fps_pos_y;
	};
}
#endif//SHOWFPS
#endif//_VR_GLSL_FONT_H