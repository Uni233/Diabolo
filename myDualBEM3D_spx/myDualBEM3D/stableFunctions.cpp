#include <GL/glew.h>
#include <GL/freeglut.h>

#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream>

#include "vrBase/vrRotation.h"
#include "bemDefines.h"
#include "vrGlobalConf.h"
#include "vrBase/vrLog.h"
#include "vrPhysics/vrBEM3D.h"
using namespace std;
using namespace VR;

using namespace std;
extern void display(void);
extern VR::Interactive::vrBallController g_trackball_1;

const GLfloat light_ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
const GLfloat light_diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_position[] = { 2.0f, 5.0f, 5.0f, 0.0f };

const GLfloat mat_ambient[] = { 0.7f, 0.7f, 0.7f, 1.0f };
const GLfloat mat_diffuse[] = { 0.8f, 0.8f, 0.8f, 1.0f };
const GLfloat mat_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat high_shininess[] = { 100.0f };

extern void addSceneScale();
extern void subSceneScale();
extern vrBEM3D * g_BEM3D;
 void key(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'D':
	case 'd':
		{
			g_BEM3D->setDoFractrue(true);
			break;
		}
	case '+':
		addSceneScale();
		break;

	case '-':
		subSceneScale();
		break;

	case 27:
	case 'q':
		exit(0);
		break;
	}
	glutPostRedisplay();
}

void idle(void)
{
	glutPostRedisplay();
}

void resize(int width, int height)
{
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, width / (float)height, 2.0, 10.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, -5.0);

	g_trackball_1.ClientAreaResize(VR::Interactive::ERect(0, 0, width, height));
	glutPostRedisplay();

	/*const float ar = (float)width / (float)height;

	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-ar, ar, -1.0, 1.0, 2.0, 100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	g_trackball_1.ClientAreaResize(VR::Interactive::ERect(0, 0, width, height));*/

}

/* GLUT callback Handlers */
void motion(int x, int y)
{
	if (((GetKeyState(VK_CONTROL) & 0x80) > 0)  /*|| GLUT_ACTIVE_ALT == glutGetModifiers()*/)
	{
		g_trackball_1.MouseMove(VR::Interactive::EPoint(x, y));
		//g_trackball_2.MouseMove(VR_FEM::EPoint(x, y));
		//printf("motion (%d,%d)\n",x,y);
	}

	glutPostRedisplay();
}

void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		if (((GetKeyState(VK_CONTROL) & 0x80) > 0))
		{
			//g_trackball.setCuttingTrack(false);
			g_trackball_1.MouseDown(VR::Interactive::EPoint(x, y));
			//g_trackball_2.MouseDown(VR_FEM::EPoint(x, y));
			//printf("mouse down(%d,%d)\n",x,y);
		}
	}
	else if (state == GLUT_UP)
	{
		g_trackball_1.MouseUp(VR::Interactive::EPoint(x, y));
		//g_trackball_2.MouseUp(VR_FEM::EPoint(x, y));
		//printf("mouse up(%d,%d)\n",x,y);
	}
	glutPostRedisplay();
}

void Specialkeyboard(int key, int x, int y)
{
	switch (key) {
	case GLUT_KEY_LEFT:
	{
						  g_trackball_1.Key(39);
						  //g_trackball_2.Key(39);
						  break;
	}
	case GLUT_KEY_RIGHT:
	{
						   g_trackball_1.Key(37);
						   //g_trackball_2.Key(37);
						   break;
	}
	case GLUT_KEY_UP:
	{
						g_trackball_1.Key(38);
						//g_trackball_2.Key(38);
						break;
	}
	case GLUT_KEY_DOWN:
	{
						  g_trackball_1.Key(40);
						  //g_trackball_2.Key(40);
						  break;
	}
	default:
	{
			   printf("Special key %d\n", key);
	}
	}
	glutPostRedisplay();
}

namespace VR
{
	void trimString_conf( vrString & str ) 
	{
		vrLpsz whiteSpace = " \t\n\r";
		vrSizt_t location;
		location = str.find_first_not_of(whiteSpace);
		str.erase(0,location);
		location = str.find_last_not_of(whiteSpace);
		str.erase(location + 1);
	}

	void init_global()
	{
		VR::ConfigureParser::vrPropertyMap propertyMap;
		vrString inifile = FileSystem::get_currentpath() + vrString("/conf/param.conf");

#if 0
		propertyMap[GlobalConf::g_const_Obj_hasCoord];
		propertyMap[GlobalConf::g_const_Obj_hasVerticeNormal];
		propertyMap[GlobalConf::g_const_Obj_normalizeMesh];
		propertyMap[GlobalConf::g_const_Obj_boundaryAxis];
		propertyMap[GlobalConf::g_const_Obj_textureName];
		propertyMap[GlobalConf::g_const_Obj_meshId];
		propertyMap[GlobalConf::g_const_Obj_octreeFineLevel];
		propertyMap[GlobalConf::g_const_Simulation_doSimulation];
		propertyMap[GlobalConf::g_const_Simulation_MinCount];
		propertyMap[GlobalConf::g_const_Simulation_MaxCount];
		propertyMap[GlobalConf::g_const_Simulation_YoungModulus];
		propertyMap[GlobalConf::g_const_Simulation_externalForceFactor];
		propertyMap[GlobalConf::g_const_Simulation_scriptForceFactor];
		propertyMap[GlobalConf::g_const_Simulation_animation_max_fps];
		propertyMap[GlobalConf::g_const_Simulation_camera_zoom];

		
		propertyMap[GlobalConf::g_const_Scene_bkgColor];
		propertyMap[GlobalConf::g_const_Scene_planeXsize];
		propertyMap[GlobalConf::g_const_Scene_planeZsize];
		propertyMap[GlobalConf::g_const_Scene_planeXdivs];
		propertyMap[GlobalConf::g_const_Scene_planeZdivs];
		propertyMap[GlobalConf::g_const_Scene_planeColor];
		propertyMap[GlobalConf::g_const_Scene_modelColor];
		propertyMap[GlobalConf::g_const_Scene_modelColor4Ka];
		propertyMap[GlobalConf::g_const_Scene_CamerPos];
		propertyMap[GlobalConf::g_const_Scene_LightPos];
		propertyMap[GlobalConf::g_const_Scene_EyeCenter];
		propertyMap[GlobalConf::g_const_Scene_ZUp];
		propertyMap[GlobalConf::g_const_Scene_ShowFPS];
		propertyMap[GlobalConf::g_const_Scene_FOV];
		propertyMap[GlobalConf::g_const_Scene_Near];
		propertyMap[GlobalConf::g_const_Scene_Far];
		propertyMap[GlobalConf::g_const_Scene_ShadowMapWidth];
		propertyMap[GlobalConf::g_const_Scene_ShadowMapHeight];
#endif
		
		propertyMap[GlobalConf::g_const_Obj_meshName];
		propertyMap[GlobalConf::g_const_Obj_remesh];
		


		/*vrFloat density=1200;
		std::string outFile("_result/buaa_mat"); 
		vrFloat youngsMod = 3000e6;
		vrFloat toughness=7e5;
		vrFloat poissonsRatio=0.327;
		vrFloat strength=76e6;*/

		propertyMap[GlobalConf::g_const_BEM_density];
		propertyMap[GlobalConf::g_const_BEM_youngsMod];
		propertyMap[GlobalConf::g_const_BEM_toughness];
		propertyMap[GlobalConf::g_const_BEM_poissonsRatio];
		propertyMap[GlobalConf::g_const_BEM_strength];
		propertyMap[GlobalConf::g_const_BEM_compress];
		propertyMap[GlobalConf::g_const_BEM_matMdl];
		propertyMap[GlobalConf::g_const_BEM_boundarycondition_dc];
		propertyMap[GlobalConf::g_const_BEM_boundarycondition_nm];
		
		propertyMap[GlobalConf::g_const_BEM_outFile];
		
		propertyMap[GlobalConf::g_const_Scene_windowWidth];
		propertyMap[GlobalConf::g_const_Scene_windowHeight];

		propertyMap[GlobalConf::g_const_DebugHsubmatrix];
		propertyMap[GlobalConf::g_const_DebugGsubmatrix];
		propertyMap[GlobalConf::g_const_DebugDisplacement];

		propertyMap[GlobalConf::g_const_GaussPointSize_xi_In_Theta];
		propertyMap[GlobalConf::g_const_GaussPointSize_xi_In_Rho];
		propertyMap[GlobalConf::g_const_GaussPointSize_eta_In_Theta];
		propertyMap[GlobalConf::g_const_GaussPointSize_eta_In_Rho];
		propertyMap[GlobalConf::g_const_GaussPointSize_xi_In_Theta_DisContinuous];
		propertyMap[GlobalConf::g_const_GaussPointSize_xi_In_Rho_DisContinuous];
		propertyMap[GlobalConf::g_const_GaussPointSize_eta_In_Theta_DisContinuous];
		propertyMap[GlobalConf::g_const_GaussPointSize_eta_In_Rho_DisContinuous];
		propertyMap[GlobalConf::g_const_GaussPointSize_eta_In_Theta_360];
		propertyMap[GlobalConf::g_const_GaussPointSize_eta_In_Rho_360];
		propertyMap[GlobalConf::g_const_GaussPointSize_eta_In_Theta_SubTri];
		propertyMap[GlobalConf::g_const_GaussPointSize_eta_In_Rho_SubTri];
		


		VR::ConfigureParser::parser_configurefile(inifile, propertyMap);

		VR::ConfigureParser::getConfPropertyValue(propertyMap, GlobalConf::g_const_BEM_density, GlobalConf::density);
		VR::ConfigureParser::getConfPropertyValue(propertyMap, GlobalConf::g_const_BEM_youngsMod, GlobalConf::youngsMod);
		VR::ConfigureParser::getConfPropertyValue(propertyMap, GlobalConf::g_const_BEM_toughness, GlobalConf::toughness);
		VR::ConfigureParser::getConfPropertyValue(propertyMap, GlobalConf::g_const_BEM_poissonsRatio, GlobalConf::poissonsRatio);
		VR::ConfigureParser::getConfPropertyValue(propertyMap, GlobalConf::g_const_BEM_strength, GlobalConf::strength);
		VR::ConfigureParser::getConfPropertyValue(propertyMap, GlobalConf::g_const_BEM_compress, GlobalConf::compress);
		VR::ConfigureParser::getConfPropertyValueStr(propertyMap, GlobalConf::g_const_BEM_matMdl, GlobalConf::matMdl);
		
		std::string strBnd_DC, strBnd_NM;
		VR::ConfigureParser::getConfPropertyValueStr(propertyMap, GlobalConf::g_const_BEM_boundarycondition_dc, strBnd_DC);
		VR::ConfigureParser::getConfPropertyValueStr(propertyMap, GlobalConf::g_const_BEM_boundarycondition_nm, strBnd_NM);
		trimString_conf(strBnd_DC);
		{
			std::string& vertString = strBnd_DC;
			printf("boundary dc string[%s]\n",strBnd_DC.c_str());
			size_t slash1,lastslash1=-1;
			slash1 = vertString.find(",");
			while (slash1 != string::npos)
			{
				int bc_region_id = atoi( vertString.substr(lastslash1+1,slash1).c_str() );
				GlobalConf::boundarycondition_dc.insert(bc_region_id);
				printf("boundary dc region [%d]\n",bc_region_id);
				lastslash1 = slash1;
				slash1 = vertString.find(",", lastslash1 + 1 );
			}

			
			if (lastslash1 < vertString.length())
			{
				int bc_region_id = atoi( vertString.substr(lastslash1+1,vertString.length()).c_str() );
				printf("boundary dc region [%d]\n",bc_region_id);
				GlobalConf::boundarycondition_dc.insert(bc_region_id);
			}
		}
		trimString_conf(strBnd_NM);
		{
			std::string& vertString = strBnd_NM;
			printf("boundary nm string[%s]\n",strBnd_NM.c_str());
			size_t slash1,lastslash1=-1;
			vrInt bc_region_id;
			MyVec3 forceVec;
			slash1 = vertString.find(":");
			while (slash1 != string::npos)
			{
				bc_region_id = atoi( vertString.substr(lastslash1+1,slash1).c_str() );
				lastslash1 = slash1;
				slash1 = vertString.find(",", lastslash1 + 1 );

				//printf("sub [%s]\n",vertString.substr(lastslash1+1,slash1).c_str());
				forceVec[0] = atof( vertString.substr(lastslash1+1,slash1).c_str() );
				lastslash1 = slash1;
				slash1 = vertString.find(",", lastslash1 + 1 );

				//printf("sub [%s]\n",vertString.substr(lastslash1+1,slash1).c_str());
				forceVec[1] = atof( vertString.substr(lastslash1+1,slash1).c_str() );
				lastslash1 = slash1;
				slash1 = vertString.find(",", lastslash1 + 1 );

				//printf("sub [%s]\n",vertString.substr(lastslash1+1,slash1).c_str());
				forceVec[2] = atof( vertString.substr(lastslash1+1,slash1).c_str() );
				lastslash1 = slash1;
				slash1 = vertString.find(":", lastslash1 + 1 );

				GlobalConf::boundarycondition_nm[bc_region_id] = forceVec;

				std::cout << bc_region_id << " : " <<  forceVec.transpose() << std::endl;
			}

		}
		vrPause;
		
		VR::ConfigureParser::getConfPropertyValueStr(propertyMap, GlobalConf::g_const_BEM_outFile, GlobalConf::outFile);

		VR::ConfigureParser::getConfPropertyValue(propertyMap, GlobalConf::g_const_Scene_windowWidth, GlobalConf::g_n_Scene_windowWidth);
		VR::ConfigureParser::getConfPropertyValue(propertyMap, GlobalConf::g_const_Scene_windowHeight, GlobalConf::g_n_Scene_windowHeight);


		VR::ConfigureParser::getConfPropertyValueStr(propertyMap, GlobalConf::g_const_Obj_meshName, GlobalConf::g_str_Obj_meshName);
		VR::ConfigureParser::getConfPropertyValue(propertyMap, GlobalConf::g_const_Obj_remesh, GlobalConf::g_n_Obj_remesh);

		VR::ConfigureParser::getConfPropertyValueStr(propertyMap, GlobalConf::g_const_DebugHsubmatrix, GlobalConf::g_str_Obj_DebugHsubmatrix);
		VR::ConfigureParser::getConfPropertyValueStr(propertyMap, GlobalConf::g_const_DebugGsubmatrix, GlobalConf::g_str_Obj_DebugGsubmatrix);
		VR::ConfigureParser::getConfPropertyValueStr(propertyMap, GlobalConf::g_const_DebugDisplacement, GlobalConf::g_str_Obj_DebugDisplacement);
		
		VR::ConfigureParser::getConfPropertyValue(propertyMap, GlobalConf::g_const_GaussPointSize_xi_In_Theta,GlobalConf::g_n_Sample_GaussPointSize_xi_In_Theta);
		VR::ConfigureParser::getConfPropertyValue(propertyMap, GlobalConf::g_const_GaussPointSize_xi_In_Rho,GlobalConf::g_n_Sample_GaussPointSize_xi_In_Rho);
		VR::ConfigureParser::getConfPropertyValue(propertyMap, GlobalConf::g_const_GaussPointSize_eta_In_Theta,GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta);
		VR::ConfigureParser::getConfPropertyValue(propertyMap, GlobalConf::g_const_GaussPointSize_eta_In_Rho,GlobalConf::g_n_Sample_GaussPointSize_eta_In_Rho);
		VR::ConfigureParser::getConfPropertyValue(propertyMap, GlobalConf::g_const_GaussPointSize_xi_In_Theta_DisContinuous,GlobalConf::g_n_Sample_GaussPointSize_xi_In_Theta_DisContinuous);
		VR::ConfigureParser::getConfPropertyValue(propertyMap, GlobalConf::g_const_GaussPointSize_xi_In_Rho_DisContinuous,GlobalConf::g_n_Sample_GaussPointSize_xi_In_Rho_DisContinuous);
		VR::ConfigureParser::getConfPropertyValue(propertyMap, GlobalConf::g_const_GaussPointSize_eta_In_Theta_DisContinuous,GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_DisContinuous);
		VR::ConfigureParser::getConfPropertyValue(propertyMap, GlobalConf::g_const_GaussPointSize_eta_In_Rho_DisContinuous,GlobalConf::g_n_Sample_GaussPointSize_eta_In_Rho_DisContinuous);
		VR::ConfigureParser::getConfPropertyValue(propertyMap, GlobalConf::g_const_GaussPointSize_eta_In_Theta_360,GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_360);
		VR::ConfigureParser::getConfPropertyValue(propertyMap, GlobalConf::g_const_GaussPointSize_eta_In_Rho_360,GlobalConf::g_n_Sample_GaussPointSize_eta_In_Rho_360);
		VR::ConfigureParser::getConfPropertyValue(propertyMap, GlobalConf::g_const_GaussPointSize_eta_In_Theta_SubTri,GlobalConf::g_n_Sample_GaussPointSize_eta_In_Theta_SubTri);
		VR::ConfigureParser::getConfPropertyValue(propertyMap, GlobalConf::g_const_GaussPointSize_eta_In_Rho_SubTri,GlobalConf::g_n_Sample_GaussPointSize_eta_In_Rho_SubTri);

		GlobalConf::printConf();
	}
}

const char* loadShaderAsString(const char* file)
{
	std::ifstream shader_file(file, std::ifstream::in);
	std::string str((std::istreambuf_iterator<char>(shader_file)), std::istreambuf_iterator<char>());
	return str.c_str();
}


void initializeMyShader(const char* lpszVtxShader, const char* lpszFrgShader)
{
	GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
	if (0 == vertShader)
	{
		fprintf(stderr, "Error creating vertex shader.\n");
		exit(1);
	}

	const GLchar * shaderCode = loadShaderAsString(lpszVtxShader);
	const GLchar* codeArray[] = { shaderCode };
	glShaderSource(vertShader, 1, codeArray, NULL);

	glCompileShader(vertShader);

	GLint result;
	glGetShaderiv(vertShader, GL_COMPILE_STATUS, &result);
	if (GL_FALSE == result)
	{
		fprintf(stderr, "Vertex shader compilation failed!\n");
		GLint logLen;
		glGetShaderiv(vertShader, GL_INFO_LOG_LENGTH, &logLen);
		if (logLen > 0)
		{
			char * log = (char *)malloc(logLen);
			GLsizei written;
			glGetShaderInfoLog(vertShader, logLen, &written, log);
			fprintf(stderr, "Shader log:\n%s", log);
			free(log);
		}
	}
}

void initOpenGL(int argc, char *argv[])
{
	using namespace VR;
	glutInit(&argc, argv);
	glutInitWindowSize(GlobalConf::g_n_Scene_windowWidth, GlobalConf::g_n_Scene_windowHeight);
	glutInitWindowPosition(10, 10);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

	glutCreateWindow(vrAppTitle);

	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		printf("Error initializing GLEW: %s\n", glewGetErrorString(err));
	}
	else
	{
		const GLubyte *renderer = glGetString(GL_RENDERER);
		const GLubyte *vendor = glGetString(GL_VENDOR);
		const GLubyte *version = glGetString(GL_VERSION);
		const GLubyte *glslVersion =
			glGetString(GL_SHADING_LANGUAGE_VERSION);
		GLint major, minor;
		glGetIntegerv(GL_MAJOR_VERSION, &major);
		glGetIntegerv(GL_MINOR_VERSION, &minor);
		printf("GL Vendor : %s\n", vendor);
		printf("GL Renderer : %s\n", renderer);
		printf("GL Version (string) : %s\n", version);
		printf("GL Version (integer) : %d.%d\n", major, minor);
		printf("GLSL Version : %s\n", glslVersion);

		/*GLint nExtensions;
		glGetIntegerv(GL_NUM_EXTENSIONS, &nExtensions);
		for( int i = 0; i < nExtensions; i++ )
		printf("%s\n", glGetStringi( GL_EXTENSIONS, i ) );*/

		if (GL_FALSE == glewGetExtension("GL_SHADING_LANGUAGE_VERSION"))
		{
			printf("GL_SHADING_LANGUAGE unsupport!\n");
		}

		//initializeMyShader((FileSystem::get_currentpath() + vrString("/myshader/basic.vert")).c_str());
	}
	g_trackball_1.SetColor(RGB(130, 80, 30));
	g_trackball_1.SetDrawConstraints();

	glutReshapeFunc(resize);
	glutDisplayFunc(display);
	glutKeyboardFunc(key);
	glutIdleFunc(idle);
	glutSpecialFunc(Specialkeyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	glClearColor(1, 1, 1, 1);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glEnable(GL_LIGHT0);
	glEnable(GL_NORMALIZE);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_LIGHTING);

	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, high_shininess);
}