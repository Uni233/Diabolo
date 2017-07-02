#ifndef _VR_COLOR_H
#define _VR_COLOR_H

#include "VR_Global_Define.h"

namespace YC
{
	namespace Colors
	{
		//typedef enum{black=0,blue=1,green=2,indigoBlue=3,red=4,pink=5,yellow=6,white=7} MyColors;
		//static MyFloat colorTemplage[8][3] = {{0.0f,0.0f,0.0f},//black
		//{0.0f,0.0f,1.0f},//blue
		//{0.0f,1.0f,0.0f},//green
		//{0.0f,1.0f,1.0f},//dian blue
		//{1.0f,0.0f,0.0f},//red
		//{1.0f,0.0f,1.0f},//pink
		//{1.0f,1.0f,0.0f},//yellow
		//{1.0f,1.0f,1.0f}};//white

		typedef enum{black=0,blue=1,green=2,indigoBlue=3,red=4,pink=5,yellow=6,white=7} MyColors;
		static MyFloat colorTemplage[8][3] = {{0.0f,0.0f,0.0f},//black
		{0.0f,0.0f,1.0f},//blue
		{0.0f,1.0f,0.0f},//green
		{0.0f,1.0f,1.0f},//dian blue
		{1.0f,0.0f,0.0f},//red
		{1.0f,0.0f,1.0f},//pink
		{1.0f,1.0f,0.0f},//yellow
		{1.0f,1.0f,1.0f}};//white

#define ColorBarSize  (5)
#define ColorBarScope (1.0f/ColorBarSize)

		MyDenseVector weightToColor(const MyFloat weight);
	}
}
#endif//_VR_COLOR_H