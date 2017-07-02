#include "stdafx.h"
//#include <GL/glew.h>
//#include <GL/freeglut.h>

#include <GL/freeglut.h>
//#include <GL/freeglut_ext.h>

#include <helper_timer.h>
//#include "TexRender.h"
#include "SOIL.h"
#include <sstream>

#include <stdio.h> //for time stamp
#include <time.h> //for time stamp
#include <iostream>//for time stamp

#include "VR_GlobalVariable.h"
//#include <boost/date_time/posix_time/posix_time.hpp>
//#define BOOST_DATE_TIME_SOURCE

void computeFPS();

/* GLUT callback Handlers */

namespace CUDA_SIMULATION
{
	namespace CUDA_CUTTING_GRID
	{
		extern void setCurrentBlade(const int nIdx,const glm::vec3& lastHandle, const glm::vec3& lastTip,const glm::vec3& currentHandle, const glm::vec3& currentTip);
		extern void resetCuttingFlagElement(bool needCheckCutting);
	}
}


extern int myRed;
extern int myGreen;
extern int MyBlue;

#if USE_OUTPUT_RENDER_OBJ_MESH
bool g_outputObjInfo;
#endif

void key(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 's':
	case 'S':
		{
			CUDA_SIMULATION::CUDA_CUTTING_GRID::setCurrentBlade(0, glm::vec3(-0.005100,0.0000,-0.100000),glm::vec3(-0.005100,1.5000,-1.50000),glm::vec3(-0.005100,0.0000,-0.100000),glm::vec3(-0.005100,0.0000,1.200000));
			CUDA_SIMULATION::CUDA_CUTTING_GRID::resetCuttingFlagElement(true);
			break;
		}
#if USE_OUTPUT_RENDER_OBJ_MESH
	case 'o':
	case 'O':
		{
			g_outputObjInfo = true;
			break;
		}
#endif
	case 'p':
	case 'P':
		{
			static int nScreenShotCount=0;
			static std::stringstream ss;
			ss.str("");
			ss << YC::GlobalVariable::g_strCurrentPath << "\\image\\awesomenessity_" << nScreenShotCount++ << ".bmp";
			SOIL_save_screenshot(ss.str().c_str(),	SOIL_SAVE_TYPE_BMP,	0, 0, windowWidth, windowHeight);
			break;
		}
	case 27 :
		glutLeaveMainLoop();
		return ;
		break;
	case 'c':
	case 'C':
		{
			/*boost::posix_time::time_facet* facet = new boost::posix_time::time_facet("%Y_%m_%d_%H_%M_%S_%f");
			std::stringstream date_stream;
			date_stream.imbue(std::locale(date_stream.getloc(), facet));
			date_stream << boost::posix_time::microsec_clock::universal_time()<< ".bmp";			
			
			LogInfo("%s  (%d,%d)",date_stream.str().c_str(),glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));	
			SOIL_save_screenshot(date_stream.str().c_str(),	SOIL_SAVE_TYPE_BMP,	0, 0, glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));*/
			break;
		}

	case '1':
		{
			myRed++;
			break;
		}
	case '2':
		{
			myRed--;
			break;
		}
	case '3':
		{
			myGreen++;
			break;
		}
	case '4':
		{
			myGreen--;
			break;
		}
	case '5':
		{
			MyBlue++;
			break;
		}
	case '6':
		{
			MyBlue--;
			break;
		}
	}

	glutPostRedisplay();
}

void idle(void)
{
	glutPostRedisplay();
}

std::string getTimeStamp()
{
	static std::stringstream ss;
	time_t timep;
	struct tm *p;

	time(&timep); /*当前time_t类型UTC时间*/
	//printf("time():%d\n",timep);
	p = localtime(&timep); /*转换为本地的tm结构的时间按*/
	timep = mktime(p); /*重新转换为time_t类型的UTC时间，这里有一个时区的转换*/ //by lizp 错误，没有时区转换， 将struct tm 结构的时间转换为从1970年至p的秒数
	//printf("time()->localtime()->mktime(): %d\n", timep);
	ss.str("");
	ss << (unsigned)timep;
	return ss.str();
}

