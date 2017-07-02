#include "stdafx.h"
#include "TexRender.h"
#include <GL/freeglut.h>


namespace YC
{
	char TexRender::fps[256]={0};

	TexRender::TexRender()
	{
		
	}

	TexRender::~TexRender()
	{
		KillFont();
	}

	void TexRender::BuildFont(/*GLFWwindow * window*/)
	{
		HFONT	font;										// Windows Font ID
		HFONT	oldfont;									// Used For Good House Keeping

		base = glGenLists(96);								// Storage For 96 Characters

		font = CreateFont(	-24,							// Height Of Font
			0,								// Width Of Font
			0,								// Angle Of Escapement
			0,								// Orientation Angle
			FW_BOLD,						// Font Weight
			FALSE,							// Italic
			FALSE,							// Underline
			FALSE,							// Strikeout
			ANSI_CHARSET,					// Character Set Identifier
			OUT_TT_PRECIS,					// Output Precision
			CLIP_DEFAULT_PRECIS,			// Clipping Precision
			ANTIALIASED_QUALITY,			// Output Quality
			FF_DONTCARE|DEFAULT_PITCH,		// Family And Pitch
			"Courier New");					// Font Name

		HWND hWnd =	FindWindow(NULL,windowTitle);
		//HWND hWnd =	glfwGetWin32Window (window);
		HDC hDC= GetDC(hWnd);
		oldfont = (HFONT)SelectObject(hDC, font);           // Selects The Font We Want
		wglUseFontBitmaps(hDC, 32, 96, base);				// Builds 96 Characters Starting At Character 32
		SelectObject(hDC, oldfont);							// Selects The Font We Want
		DeleteObject(font);									// Delete The Font
	}
	void TexRender::KillFont(void)									// Delete The Font List
	{
		glDeleteLists(base, 96);							// Delete All 96 Characters
	}

	void TexRender::glPrint(const char *text)					// Custom GL "Print" Routine
	{
		glPushAttrib(GL_LIST_BIT);							// Pushes The Display List Bits
		glListBase(base - 32);								// Sets The Base Character to 32
		glCallLists(strlen(text), GL_UNSIGNED_BYTE, text);	// Draws The Display List Text
		glPopAttrib();										// Pops The Display List Bits
	}
	
}