#include <GL/glew.h>
#include <GL/freeglut.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <boost/math/constants/constants.hpp> // pi
#include "bemDefines.h"
#include "vrBase/vrRotation.h"
#include "vrGlobalConf.h"
#include "vrBase/vrLog.h"
#include "vrGeometry/VR_Geometry_MeshDataStruct.h"
#include "vrPhysics/vrBEM3D.h"
#include <boost/atomic.hpp>
using namespace std;
using namespace VR;

boost::atomic<int> g_atomix_count(0);

extern VR::Interactive::vrBallController g_trackball_1;

vrBEM3D * g_BEM3D=NULL;

MyFloat g_scene_scale = 4.0;

void addSceneScale()
{
	g_scene_scale *= 1.1;
}

void subSceneScale()
{
	g_scene_scale *= 0.9;
}

namespace VR
{
	void showScene()
	{
		g_trackball_1.IssueGLrotation();
		glScalef(g_scene_scale, g_scene_scale, g_scene_scale);
		g_BEM3D->renderScene();
	}

	void initPhysSystem()
	{

		const MyFloat pi = boost::math::constants::pi<MyFloat>();
		const MyFloat E = GlobalConf::youngsMod;
		const MyFloat mu = GlobalConf::poissonsRatio;
		
		const MyFloat shearMod = E / (2 * (1 + mu));
		infoLog << "youngsMod : " << E << "  poissonsRatio : " << mu << "  shearMod : " << shearMod << std::endl;vrPause;
		

		g_BEM3D = new vrBEM3D(E , mu ,shearMod );
		//vrBEM3D g_BEM3D(E , mu ,shearMod, const4 ,const3 , const2 ,const1 , kappa );
		//1.load 3d obj mesh

		g_BEM3D->setFractureStep(GlobalConf::g_n_crackSteps);
		g_BEM3D->initPhysicalSystem(vrCStr(GlobalConf::g_str_Obj_meshName), GlobalConf::g_db_resMesh);
	}
}