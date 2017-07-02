/*
* GLUT Shapes Demo
*
* Written by Nigel Stewart November 2003
*
* This program is test harness for the sphere, cone
* and torus shapes in GLUT.
*
* Spinning wireframe and smooth shaded shapes are
* displayed until the ESC or q key is pressed.  The
* number of geometry stacks and slices can be adjusted
* using the + and - keys.
*/
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream>

#include "vrBase/vrRotation.h"
#include "vrGlobalConf.h"
#include "vrBase/vrLog.h"
using namespace std;

#include "bemMarco.h"
#if USE_TBB
#include "tbb/task_scheduler_init.h"
#pragma comment(lib,"tbb.lib")
#endif

#if vrLoggingPP
INITIALIZE_EASYLOGGINGPP
#endif

VR::Interactive::vrBallController g_trackball_1(0.5,
	VR::Interactive::unitquaternion(Eigen::AngleAxisd(VR::Cgmath::DegToRad(-90), VR::vrVec3(0, 1, 0)))/**
	VR::Interactive::unitquaternion(Eigen::AngleAxisd(VR::Cgmath::DegToRad(-30 * 0), VR::vrVec3(1, 0, 0)))*/);

namespace VR
{
	void showScene();
}
void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glColor3d(1, 0, 0);

	glPushMatrix();	
	VR::showScene();
	glPopMatrix();	

	glutSwapBuffers();
}




/* Program entry point */
namespace VR
{
	void init_global();
	void initPhysSystem();
}

void initializeMyShader(const char* lpszVtxShader = NULL, const char* lpszFrgShader = NULL);
extern void key(unsigned char key, int x, int y);
extern void idle(void);
extern void resize(int width, int height);
extern void motion(int x, int y);
extern void mouse(int button, int state, int x, int y);
void Specialkeyboard(int key, int x, int y);
const char* loadShaderAsString(const char* file);
void initializeMyShader(const char* lpszVtxShader, const char* lpszFrgShader);
void initOpenGL(int argc, char *argv[]);

int main(int argc, char *argv[])
{
#if USE_TBB
	tbb::task_scheduler_init init;
#endif
	using namespace VR;
	init_global();

	initPhysSystem();

	initOpenGL(argc, argv);	

	glutMainLoop();

	return EXIT_SUCCESS;
}


