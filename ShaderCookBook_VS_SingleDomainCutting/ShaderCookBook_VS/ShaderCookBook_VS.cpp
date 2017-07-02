// ShaderCookBook_VS.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

//#include <glload/gl_1_0.hpp>
//#include <glload/gl_3_3.hpp>
#include <glload/gl_4_0.h>
#include <glload/gl_load.h>

#include <GL/freeglut.h>

#include <stdlib.h>
#include <fstream>
#include "MyGLM.h" //#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>



#include "VR_Global_Define.h"
#include "MyScene.h"
#include "glutils.h"
#include "VR_GlobalVariable.h"

#if USE_CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>//cudaGLSetGLDevice
#include <helper_string.h>//checkCmdLineFlag
#include <helper_cuda.h>//gpuGetMaxGflopsDeviceId
#endif

extern void resize(int width, int height);
extern void display(void);
extern void key(unsigned char key, int x, int y);
extern void idle(void);
void mouse(int button, int state, int x, int y);
void motion(int ,int);
void newFeatureTest();
extern void OnShutdown();

const GLfloat light_ambient[]  = { 0.0f, 0.0f, 0.0f, 1.0f };
const GLfloat light_diffuse[]  = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_position[] = { 2.0f, 5.0f, 5.0f, 0.0f };

const GLfloat mat_ambient[]    = { 0.7f, 0.7f, 0.7f, 1.0f };
const GLfloat mat_diffuse[]    = { 0.8f, 0.8f, 0.8f, 1.0f };
const GLfloat mat_specular[]   = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat high_shininess[] = { 100.0f };



namespace YC
{
	//Scene
	MySceneBase* m_currentScene;	

	void initialEnv(int argc, char *argv[])
	{
		YC::GlobalVariable::initGlobalVariable(argc,argv);
	}
#if USE_CUDA
	void initialCUDAEnv(int argc, char *argv[])
	{
		cudaError_t cudaRet;
		if( checkCmdLineFlag(argc, (const char**)argv, "device") ) {
			//		gpuGLDeviceInit(argc, argv);
		} else {
			cudaRet = cudaGLSetGLDevice( gpuGetMaxGflopsDeviceId() );
		}

		cudaDeviceProp deviceProp;

		cudaGetDeviceProperties(&deviceProp, gpuGetMaxGflopsDeviceId());

#if CUDART_VERSION >= 2020
		if(!deviceProp.canMapHostMemory)
		{
			fprintf(stderr, "Device %d cannot map host memory!\n", gpuGetMaxGflopsDeviceId());
			MyError("Test PASSED");
		}
		cudaSetDeviceFlags(cudaDeviceMapHost);
#else
		fprintf(stderr, "This CUDART version does not support <cudaDeviceProp.canMapHostMemory> field\n");
		printf("Test PASSED");
		cutilExit(argc, argv);
#endif
	}
#endif
	void initializeGL() {
		//////////////// PLUG IN SCENE HERE /////////////////
		m_currentScene = new MyScene(YC::GlobalVariable::g_showFPS, YC::GlobalVariable::g_FOV, YC::GlobalVariable::g_near, YC::GlobalVariable::g_far
			,YC::GlobalVariable::lightPos
			,YC::GlobalVariable::camerPos
			,YC::GlobalVariable::eyeCenter
			,YC::GlobalVariable::zUp
			,MyScene::Physical_Scene//Geometry_Scene//Physical_Scene//Cutting_Scene
#if SHOW_SHADOWMAP
			,YC::GlobalVariable::shadowMapWidth
			,YC::GlobalVariable::shadowMapHeight
#endif
			);
		////////////////////////////////////////////////////

		GLUtils::dumpGLInfo();

		glClearColor(1.f,1.f,1.f,1.0f);

		m_currentScene->initScene();
	}
}


/* Program entry point */
extern "C" int initCuPrintf(int argc, char **argv);
#if SHOWFPS
extern StopWatchInterface * timer;
#endif
int main(int argc, char *argv[])
{
	initCuPrintf(argc,(char**)argv);
	glutInit(&argc, argv);
#if SHOWFPS
	cutilCheckError( sdkCreateTimer( &timer));
#endif

	glutInitDisplayMode(GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH | GLUT_STENCIL | GLUT_MULTISAMPLE);
	glutInitContextVersion (4, 0);
	glutInitContextProfile(GLUT_CORE_PROFILE);

	glutInitWindowSize ( windowWidth, windowHeight);
	glutInitWindowPosition (0, 0);
	glutCreateWindow (argv[0]);

	ogl_LoadFunctions();
	//glload::LoadFunctions();

	YC::initialEnv(argc, argv);
	YC::initializeGL();
#if USE_CUDA
	YC::initialCUDAEnv(argc, argv);
	
#endif

    glutReshapeFunc(resize);
    glutDisplayFunc(display);
    glutKeyboardFunc(key);
    glutIdleFunc(idle);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutCloseFunc(OnShutdown);

	newFeatureTest();

    glutMainLoop();

    return EXIT_SUCCESS;
}

#if 1
void display(void)
{
	using namespace YC;
	GLUtils::checkForOpenGLError(__FILE__,__LINE__);
	m_currentScene->render();
	glutSwapBuffers();
}

void resize(int width, int height)
{
	YC::m_currentScene->resize(width,height);
}

void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN && (GLUT_ACTIVE_CTRL == glutGetModifiers()|| GLUT_ACTIVE_ALT == glutGetModifiers()))
	{
		YC::m_currentScene->mouse(x,y);
	}
	//glutPostRedisplay();
}

void motion(int x, int y)
{
	if (GLUT_ACTIVE_CTRL == glutGetModifiers() || GLUT_ACTIVE_ALT == glutGetModifiers() )
	{
		YC::m_currentScene->motion(x,y);
	}
}

void OnShutdown()
{
	LogInfo("Opengl Quit!\n");
}
#endif

