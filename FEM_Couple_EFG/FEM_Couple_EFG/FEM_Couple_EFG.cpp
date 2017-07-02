// FEM_Couple_EFG.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <stdlib.h>
#include "Maths/VECTOR3D.h"
#include "Maths/VECTOR4D.h"
#include "Maths/COLOR.h"
#include "Maths/MATRIX4X4.h"

#include "MeshGenerate.h"
#include "plane.h"
#include <boost/format.hpp>


#include "Maths/Maths.h"

#include "Frame/Mat_YC.h"
#include "Frame/Axis_YC.h"

bool showTex = false;


int drawMode=GL_TRIANGLE_FAN; // the default draw mode

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -5.0;
float scaled = 4.5f;
StopWatchInterface *timer = NULL;

//bool initGL(int argc, char **argv);
void motion(int x, int y);
void display();
void fpsDisplay();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void drawAxis();
void drawTest();
void myidle();
std::string GetModuleDir();
void initCuda(int argc, char** argv);
extern "C" int initCuPrintf(int argc, char **argv);
extern "C" int unInitialCuPrintf(int argc, char **argv);
extern void do_something();
extern void display();
extern void global_Initialize();

extern bool initGL(int argc, char **argv);

#define CuttingMesh (0)
int _tmain(int argc, _TCHAR* argv[])
{
	global_Initialize();

	initCuPrintf(argc,(char**)argv);
	printf("cutCreateTimer \n");
	cutilCheckError( sdkCreateTimer( &timer));
	initGL(argc,(char**)argv);
	initCuda(argc,(char**)argv);

	do_something();	
	
	glutDisplayFunc(fpsDisplay);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutIdleFunc(myidle);
	glutMainLoop();
	return 0;
}



void computeFPS()
{
	static int fpsCount=0;
	static int fpsLimit=100;

	fpsCount++;

	if (fpsCount == fpsLimit) {
		char fps[256];
		float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		sprintf(fps, "Cuda GL Interop Wrapper: %3.1f fps ", ifps);  

		glutSetWindowTitle(fps);
		fpsCount = 0; 

		cutilCheckError(sdkResetTimer(&timer));  
	}
}

void fpsDisplay()
{
	cutilCheckError(sdkStartTimer(&timer)); 
	display();
	cutilCheckError(sdkStopTimer(&timer));
	computeFPS();
}

void keyboard(unsigned char key, int x, int y)
{
	switch(key) {
	case(27) :
		exit(0);
		break;
	case 'a':
		showTex = true;
	case 'A':
		showTex = false;
	case 'D':
		switch(drawMode) {
		case GL_POINTS: drawMode = GL_LINE_STRIP; break;
		case GL_LINE_STRIP: drawMode = GL_TRIANGLE_FAN; break;
		default: drawMode=GL_POINTS;
		} break;
	}
	glutPostRedisplay();
}

// Mouse event handlers for GLUT
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN) {
		mouse_buttons |= 1<<button;
	} else if (state == GLUT_UP) {
		mouse_buttons = 0;
	}
	if (button == GLUT_MIDDLE_BUTTON)
	{
		if (state == GLUT_DOWN)
		{
			// scaled -= 0.5f;
		} 
		else if (GLUT_UP == state)
		{
			scaled += 0.5f;
		}

	}


	mouse_old_x = x;
	mouse_old_y = y;
	glutPostRedisplay();
}

void motion(int x, int y)
{
	float dx, dy;
	dx = x - mouse_old_x;
	dy = y - mouse_old_y;

	if (mouse_buttons & 1) {
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	} else if (mouse_buttons & 4) {
		translate_z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}



void drawAxis()
{
	::glLineWidth(1 ) ;
	glBegin(GL_LINES);
	// light red x axis arrow
	glColor3f(1.f,0.5f,.5f);
	glVertex3f(0.0f,0.0f,0.0f);
	glVertex3f(1.0f,0.0f,0.0f);
	// light green y axis arrow
	glColor3f(.5f,1.f,0.5f);
	glVertex3f(0.0f,0.0f,0.0f);
	glVertex3f(0.0f,1.0f,0.0f);
	// light blue z axis arrow
	glColor3f(.5f,.5f,1.f);
	glVertex3f(0.0f,0.0f,0.0f);
	glVertex3f(0.0f,0.0f,1.0f);
	glEnd();
	glBegin(GL_LINES);
	// x letter & arrowhead
	glColor3f(1.f,0.5f,.5f);
	glVertex3f(1.1f,0.1f,0.0f);
	glVertex3f(1.3f,-0.1f,0.0f);
	glVertex3f(1.3f,0.1f,0.0f);
	glVertex3f(1.1f,-0.1f,0.0f);
	glVertex3f(1.0f,0.0f,0.0f);
	glVertex3f(0.9f,0.1f,0.0f);
	glVertex3f(1.0f,0.0f,0.0f);
	glVertex3f(0.9f,-0.1f,0.0f);
	// y letter & arrowhead
	glColor3f(.5f,1.f,0.5f);
	glVertex3f(-0.1f,1.3f,0.0f);
	glVertex3f(0.f,1.2f,0.0f);
	glVertex3f(0.1f,1.3f,0.0f);
	glVertex3f(0.f,1.2f,0.0f);
	glVertex3f(0.f,1.2f,0.0f);
	glVertex3f(0.f,1.1f,0.0f);
	glVertex3f(0.0f,1.0f,0.0f);
	glVertex3f(0.1f,0.9f,0.0f);
	glVertex3f(0.0f,1.0f,0.0f);
	glVertex3f(-0.1f,0.9f,0.0f);
	// z letter & arrowhead
	glColor3f(.5f,.5f,1.f);
	glVertex3f(0.0f,-0.1f,1.3f);
	glVertex3f(0.0f,0.1f,1.3f);
	glVertex3f(0.0f,0.1f,1.3f);
	glVertex3f(0.0f,-0.1f,1.1f);
	glVertex3f(0.0f,-0.1f,1.1f);
	glVertex3f(0.0f,0.1f,1.1f);
	glVertex3f(0.0f,0.0f,1.0f);
	glVertex3f(0.0f,0.1f,0.9f);
	glVertex3f(0.0f,0.0f,1.0f);
	glVertex3f(0.0f,-0.1f,0.9f);
	glEnd();
}

void drawTest()
{
}

std::string GetModuleDir()
{
	char pFileName[256];
	GetModuleFileName( NULL, pFileName, 255 );

	std::string csFullPath(pFileName);
	int nPos = csFullPath.rfind( _T('\\') );
	if( nPos < 0 )
		return std::string("");
	else
		return csFullPath.substr(0, nPos );
}

void drawFrame()
{
	using namespace VR_FEM;
	std::vector< VR_FEM::MyDenseMatrix > rotationMatrix;
	std::vector< VR_FEM::MyDenseVector > transformVector;
	//e.getFramePara(rotationMatrix,transformVector);
	for (unsigned currentFemId=0;currentFemId < rotationMatrix.size();++currentFemId)
	{
		MyDenseMatrix& m_FrameRotationMatrix = rotationMatrix[currentFemId];
		MyDenseVector& m_FrameTranslationVector = transformVector[currentFemId];
		Matrix3<float> mat;
		mat.matrix[0].elems[0] = m_FrameRotationMatrix(0,0);
		mat.matrix[0].elems[1] = m_FrameRotationMatrix(0,1);
		mat.matrix[0].elems[2] = m_FrameRotationMatrix(0,2);

		mat.matrix[1].elems[0] = m_FrameRotationMatrix(1,0);
		mat.matrix[1].elems[1] = m_FrameRotationMatrix(1,1);
		mat.matrix[1].elems[2] = m_FrameRotationMatrix(1,2);

		mat.matrix[2].elems[0] = m_FrameRotationMatrix(2,0);
		mat.matrix[2].elems[1] = m_FrameRotationMatrix(2,1);
		mat.matrix[2].elems[2] = m_FrameRotationMatrix(2,2);
		Axis::Quaternion quater;
		quater.fromMatrix(mat);
		Axis::draw(Vec3<float>(m_FrameTranslationVector[0]-0.5f,m_FrameTranslationVector[1]-0.5f,m_FrameTranslationVector[2]-0.5f), quater, .5);
	}
}