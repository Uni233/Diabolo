#include "stdafx.h"
//#include <GL/glew.h>
//#include <GL/freeglut.h>

#include <GL/freeglut.h>
//#include <GL/freeglut_ext.h>

#include <helper_timer.h>
//#include "TexRender.h"
#include "SOIL.h"
#include <sstream>

//#include <boost/date_time/posix_time/posix_time.hpp>
//#define BOOST_DATE_TIME_SOURCE


static int slices = 16;
static int stacks = 16;

void computeFPS();

/* GLUT callback Handlers */


extern int myRed;
extern int myGreen;
extern int MyBlue;

void key(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27 :
		glutLeaveMainLoop();
		break;
	case 'c':
	case 'C':
		{
			
			break;
		}
	case '+':
		slices++;
		stacks++;
		break;

	case '-':
		if (slices>3 && stacks>3)
		{
			slices--;
			stacks--;
		}
		break;

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

