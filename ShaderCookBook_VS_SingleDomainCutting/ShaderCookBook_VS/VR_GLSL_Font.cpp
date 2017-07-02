#include "VR_GLSL_Font.h"

#if SHOWFPS

namespace YC
{
	void MyGLSLFont::init()
	{
		g_strings.push_front("Type some stuff:");

		g_pFont = glutil::GenerateFont(glutil::FONT_SIZE_LARGE);

		InitializeProgram();
		InitializeVertexData();
	}

	void MyGLSLFont::InitializeProgram()
	{
		const std::string vertexShader(
			"#version 330\n"
			"\n"
			"layout(location = 0) in vec4 position;\n"
			"layout(location = 1) in vec4 texCoord;\n"
			"\n"
			"smooth out vec2 glyphCoord;\n"
			"\n"
			"uniform mat4 cameraToClipMatrix;\n"
			"\n"
			"void main()\n"
			"{\n"
			"	gl_Position = cameraToClipMatrix * position;\n"
			"	glyphCoord = texCoord.st;\n"
			"}\n"
			);

		const std::string fragmentShader(
			"#version 330\n"
			"\n"
			"smooth in vec2 glyphCoord;\n"
			"out vec4 outputColor;\n"
			"uniform sampler2D glyphTex;"
			"\n"
			"void main()\n"
			"{\n"
			"	outputColor = texture(glyphTex, glyphCoord).rrrr;\n"
			"}\n"
			);

		GLuint vertShader = glutil::CompileShader(GL_VERTEX_SHADER, vertexShader);
		GLuint fragShader = glutil::CompileShader(GL_FRAGMENT_SHADER, fragmentShader);

		g_program = glutil::LinkProgram(vertShader, fragShader);

		glDeleteShader(vertShader);
		glDeleteShader(fragShader);

		g_cameraToClipMatrixUnif = glGetUniformLocation(g_program, "cameraToClipMatrix");
		GLint samplerUnif = glGetUniformLocation(g_program, "glyphTex");
		glUseProgram(g_program);
		glUniform1i(samplerUnif, 0);
		glUseProgram(0);
	}

	std::string GetString()
	{
		return "Test string";
	}

	void MyGLSLFont::InitializeVertexData()
	{
		glmesh::AttributeList attribs;
		//First attribute is attribute index 0, a vec4 of floats.
		attribs.push_back(glmesh::AttribDesc(0, 2, glmesh::VDT_SINGLE_FLOAT, glmesh::ADT_FLOAT));
		//Second attribute is attribute index 1, a vec4 of normalized, unsigned bytes.
		attribs.push_back(glmesh::AttribDesc(1, 2, glmesh::VDT_SINGLE_FLOAT, glmesh::ADT_FLOAT));

		g_pVertFmt = new glmesh::VertexFormat(attribs);
		g_pStreamBuf = new glmesh::StreamBuffer(1024 * 1024);



		std::vector<GLfloat> vecVertex;

		std::string theText = GetString();

		std::vector<glutil::GlyphQuad> glyphs = g_pFont->LayoutLine(theText,
			glm::vec2(50.0f, 250.0f), glutil::REF_BASELINE);

		vecVertex.reserve(24 * glyphs.size());

		assert(glyphs.size());

		for(size_t loop = 0; loop < glyphs.size(); ++loop)
			AddGlyph(vecVertex, glyphs[loop]);

		g_numGlyphsToDraw = glyphs.size();

		glGenBuffers(1, &g_dataBufferObject);

		glBindBuffer(GL_ARRAY_BUFFER, g_dataBufferObject);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vecVertex.size(), &vecVertex[0], GL_STATIC_DRAW);

		glGenVertexArrays(1, &g_vao);

		glBindVertexArray(g_vao);
		glBindBuffer(GL_ARRAY_BUFFER, g_dataBufferObject);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)(2 * sizeof(GLfloat)));

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void MyGLSLFont::PushGlyph( std::vector<GLfloat> &vecVertex, int loop, std::vector<glm::vec2> &positions,
		std::vector<glm::vec2> &texCoords )
	{
		vecVertex.push_back(positions[loop].x);
		vecVertex.push_back(positions[loop].y);
		vecVertex.push_back(texCoords[loop].x);
		vecVertex.push_back(texCoords[loop].y);
	}

	void MyGLSLFont::AddGlyph(std::vector<GLfloat> &vecVertex, const glutil::GlyphQuad &theGlyph)
	{
		std::vector<glm::vec2> positions = theGlyph.GetPositions();
		std::vector<glm::vec2> texCoords = theGlyph.GetTexCoords();

		PushGlyph(vecVertex, 0, positions, texCoords);
		PushGlyph(vecVertex, 1, positions, texCoords);
		PushGlyph(vecVertex, 2, positions, texCoords);

		PushGlyph(vecVertex, 1, positions, texCoords);
		PushGlyph(vecVertex, 3, positions, texCoords);
		PushGlyph(vecVertex, 2, positions, texCoords);
	}

	void MyGLSLFont::SendVerts(glmesh::Draw &imm, int index, std::vector<glm::vec2> &positions,
		std::vector<glm::vec2> &texCoords)
	{
		imm.Attrib(positions[index]);
		imm.Attrib(texCoords[index]);
	}

	void MyGLSLFont::DrawGlyph(glmesh::Draw &imm, const glutil::GlyphQuad &theGlyph)
	{
		std::vector<glm::vec2> positions = theGlyph.GetPositions();
		std::vector<glm::vec2> texCoords = theGlyph.GetTexCoords();

		SendVerts(imm, 0, positions, texCoords);
		SendVerts(imm, 1, positions, texCoords);
		SendVerts(imm, 2, positions, texCoords);

		SendVerts(imm, 1, positions, texCoords);
		SendVerts(imm, 3, positions, texCoords);
		SendVerts(imm, 2, positions, texCoords);
	}

	void MyGLSLFont::DrawTextString(const std::string &text, const glm::vec2 &location)
	{
		std::vector<glutil::GlyphQuad> glyphs = g_pFont->LayoutLine(text,
			location, glutil::REF_BOTTOM);

		glmesh::Draw imm(GL_TRIANGLES, glyphs.size() * 6, *g_pVertFmt, *g_pStreamBuf);

		for(size_t loop = 0; loop < glyphs.size(); ++loop)
		{
			DrawGlyph(imm, glyphs[loop]);
		}
	}

	void MyGLSLFont::resize(const int w, const int h)
	{
		g_windowSize.x = w;
		g_windowSize.y = h;
		m_fps_pos_x = g_windowSize.x * x_scale;
		m_fps_pos_y = g_windowSize.y * y_scale;
	}

	void MyGLSLFont::printTextOnGLSL(const char * fps)
	{
		g_currString = std::string(fps);

		glEnable(GL_BLEND);
		glBlendEquation(GL_FUNC_ADD);
		glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);



		glUseProgram(g_program);
		glBindVertexArray(g_vao);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, g_pFont->GetTexture());

		glutil::MatrixStack persMatrix;
		persMatrix.PixelPerfectOrtho(g_windowSize, glm::vec2(-1.0f, 1.0f), false);

		glUniformMatrix4fv(g_cameraToClipMatrixUnif, 1, GL_FALSE, glm::value_ptr(persMatrix.Top()));

		glm::vec2 currPt((float)m_fps_pos_x, (float)m_fps_pos_y);//position
		if(!g_currString.empty())
			DrawTextString(g_currString, currPt);

		/*for(StringQueue::const_iterator startIt = g_strings.begin();
			startIt != g_strings.end();
			++startIt)
		{
			currPt.y += g_pFont->GetLinePixelHeight();

			DrawTextString(*startIt, currPt);
		}*/

		glBindTexture(GL_TEXTURE_2D, 0);
		//glUseProgram(0);
		glDisable(GL_BLEND);
	}
}

#endif