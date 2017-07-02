#ifndef _TexRender_H
#define _TexRender_H
#include "VR_Global_Define.h"

namespace YC
{
	class TexRender
	{
	public:
		TexRender();
		~TexRender();
	public:
		void BuildFont(/*GLFWwindow * window*/);
		void KillFont();// Delete The Font List
		void glPrint(const char *text)	;
	private:
		unsigned	base;				// Base Display List For The Font Set
	public:
		static char fps[256];
	};
}
#endif//_TexRender_H