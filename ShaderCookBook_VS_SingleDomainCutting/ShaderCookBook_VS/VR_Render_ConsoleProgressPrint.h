#ifndef _VR_RENDER_CONSOLEPROGRESSPRINT_H
#define _VR_RENDER_CONSOLEPROGRESSPRINT_H

#include <stdio.h>
#include <windows.h>
namespace YC
{
	namespace Render
	{
		void progressBar( char label[], int step, int total );
	}
}//namespace YC
#endif//_VR_RENDER_CONSOLEPROGRESSPRINT_H