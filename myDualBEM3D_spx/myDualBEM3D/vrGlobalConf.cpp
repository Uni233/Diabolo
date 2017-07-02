#include "vrGlobalConf.h"
#include "bemDefines.h"
#include "vrBase/vrLog.h"
namespace VR
{
	namespace GlobalConf
	{
#if 0
		bool g_Obj_hasCoord;
		vrString const g_const_Obj_hasCoord = vrString("Obj.hasCoord");

		bool g_Obj_hasVerticeNormal;
		vrString const g_const_Obj_hasVerticeNormal = vrString("Obj.hasVerticeNormal");

		bool g_Obj_normalizeMesh;
		vrString const g_const_Obj_normalizeMesh = vrString("Obj.normalizeMesh");

		vrInt g_n_Obj_boundaryAxis;
		vrString const g_const_Obj_boundaryAxis = vrString("Obj.boundaryAxis");

		vrString g_str_Obj_textureName;
		vrString const g_const_Obj_textureName = vrString("Obj.textureName");

		vrString g_str_Obj_meshId;
		vrString const g_const_Obj_meshId = vrString("Obj.meshId");

		vrInt g_n_Obj_octreeFineLevel;
		vrString const g_const_Obj_octreeFineLevel = vrString("Obj.octreeFineLevel");

		bool g_Simulation_doSimulation;
		vrString const g_const_Simulation_doSimulation = vrString("Simulation.doSimulation");

		vrInt g_n_Simulation_MinCount;
		vrString const g_const_Simulation_MinCount = vrString("Simulation.MinCount");

		vrInt g_n_Simulation_MaxCount;
		vrString const g_const_Simulation_MaxCount = vrString("Simulation.MaxCount");

		vrFloat g_db_Simulation_YoungModulus;
		vrString const g_const_Simulation_YoungModulus = vrString("Simulation.YoungModulus");

		vrFloat g_db_Simulation_externalForceFactor;
		vrString const g_const_Simulation_externalForceFactor = vrString("Simulation.externalForceFactor");


		vrFloat g_db_Simulation_scriptForceFactor;
		vrString const g_const_Simulation_scriptForceFactor = vrString("Simulation.scriptForceFactor");

		

		vrVec4 g_vec4_Scene_bkgColor;
		vrString const g_const_Scene_bkgColor = vrString("Scene.bkgColor");

		vrFloat g_db_Scene_planeXsize;
		vrString const g_const_Scene_planeXsize = vrString("Scene.planeXsize");

		vrFloat g_db_Scene_planeZsize;
		vrString const g_const_Scene_planeZsize = vrString("Scene.planeZsize");

		vrInt g_n_Scene_planeXdivs;
		vrString const g_const_Scene_planeXdivs = vrString("Scene.planeXdivs");

		vrInt g_n_Scene_planeZdivs;
		vrString const g_const_Scene_planeZdivs = vrString("Scene.planeZdivs");

		vrVec3 g_vec3_Scene_planeColor;
		vrString const g_const_Scene_planeColor = vrString("Scene.planeColor");

		vrVec3 g_vec3_Scene_modelColor;
		vrString const g_const_Scene_modelColor = vrString("Scene.modelColor");

		vrVec3 g_vec3_Scene_modelColor4Ka;
		vrString const g_const_Scene_modelColor4Ka = vrString("Scene.modelColor4Ka");

		vrVec3 g_vec3_Scene_CamerPos;
		vrString const g_const_Scene_CamerPos = vrString("Scene.CamerPos");

		vrVec3 g_vec3_Scene_LightPos;
		vrString const g_const_Scene_LightPos = vrString("Scene.LightPos");

		vrVec3 g_vec3_Scene_EyeCenter;
		vrString const g_const_Scene_EyeCenter = vrString("Scene.EyeCenter");

		vrVec3 g_vec3_Scene_ZUp;
		vrString const g_const_Scene_ZUp = vrString("Scene.ZUp");

		bool g_Scene_ShowFPS;
		vrString const g_const_Scene_ShowFPS = vrString("Scene.ShowFPS");

		vrFloat g_db_Scene_FOV;
		vrString const g_const_Scene_FOV = vrString("Scene.FOV");

		vrFloat g_db_Scene_Near;
		vrString const g_const_Scene_Near = vrString("Scene.Near");

		vrFloat g_db_Scene_Far;
		vrString const g_const_Scene_Far = vrString("Scene.Far");

		vrInt g_n_Scene_ShadowMapWidth;
		vrString const g_const_Scene_ShadowMapWidth = vrString("Scene.ShadowMapWidth");

		vrInt g_n_Scene_ShadowMapHeight;
		vrString const g_const_Scene_ShadowMapHeight = vrString("Scene.ShadowMapHeight");

		vrFloat g_db_Simulation_animation_max_fps;
		vrString const g_const_Simulation_animation_max_fps = vrString("Simulation.animation_max_fps");

		vrFloat g_db_Simulation_camera_zoom;
		vrString const g_const_Simulation_camera_zoom = vrString("Simulation.camera_zoom");
#endif
		

		vrString g_str_Obj_meshName;
		vrString const g_const_Obj_meshName = vrString("Obj.meshName");

		vrInt g_n_Obj_remesh;
		vrString const g_const_Obj_remesh = vrString("Obj.remesh");

		

		vrInt g_n_Obj_offsetMesh=3;
		vrString const g_const_Obj_offsetMesh = vrString("Obj.offsetMesh");


		vrInt g_n_maxCracks = 4;
		vrInt g_n_crackSteps = 30;
		vrFloat g_db_resMesh=0.010;
		vrFloat g_db_resGrid=0.0005;

		//std::vector< std::string> bcStrings;
		

		bool is2D = false;
		bool noSI=true;
		vrInt cpVersion = 0;
		
		bool outVDBsubstep=true;
		vrFloat outMeshQual=2.0;
		bool outMeshNoU=false;
		bool outMeshNoCOD=false;
		bool outMeshClose=false;
		bool outMeshOBJ=false;
		
		vrFloat density;
		std::string outFile; 
		vrFloat youngsMod;
		vrFloat toughness;
		vrFloat poissonsRatio;
		vrFloat strength;
		vrFloat compress;
		std::string matMdl("vdb(mat_grains.vdb,1e5,3e-5,0)");

		std::set< vrInt > boundarycondition_dc;
		std::map<vrInt,MyVec3> boundarycondition_nm;

		vrString const g_const_BEM_density = vrString("BEM.density");
		vrString const g_const_BEM_youngsMod = vrString("BEM.youngsMod");
		vrString const g_const_BEM_toughness = vrString("BEM.toughness");
		vrString const g_const_BEM_poissonsRatio = vrString("BEM.poissonsRatio");
		vrString const g_const_BEM_strength = vrString("BEM.strength");
		vrString const g_const_BEM_compress = vrString("BEM.compress");
		vrString const g_const_BEM_matMdl = vrString("BEM.matMdl");

		vrString const g_const_BEM_boundarycondition_dc = vrString("BEM.bnd_dc");
		vrString const g_const_BEM_boundarycondition_nm = vrString("BEM.bnd_nm");

		vrString const g_const_BEM_outFile = vrString("BEM.outFile");

		vrInt g_n_Scene_windowWidth;
		vrString const g_const_Scene_windowWidth = vrString("Scene.windowWidth");

		vrInt g_n_Scene_windowHeight;
		vrString const g_const_Scene_windowHeight = vrString("Scene.windowHeight");

		vrString const g_const_DebugHsubmatrix = vrString("BEM.DebugHsubmatrix");
		vrString g_str_Obj_DebugHsubmatrix;

		vrString const g_const_DebugGsubmatrix = vrString("BEM.DebugGsubmatrix");
		vrString g_str_Obj_DebugGsubmatrix;

		vrString const g_const_DebugDisplacement = vrString("BEM.DebugDisplacement");
		vrString g_str_Obj_DebugDisplacement;

		vrString g_const_GaussPointSize_xi_In_Theta = vrString("Sample.enum_GaussPointSize_xi_In_Theta");
		vrString g_const_GaussPointSize_xi_In_Rho = vrString("Sample.enum_GaussPointSize_xi_In_Rho");
		vrString g_const_GaussPointSize_eta_In_Theta = vrString("Sample.enum_GaussPointSize_eta_In_Theta");
		vrString g_const_GaussPointSize_eta_In_Rho = vrString("Sample.enum_GaussPointSize_eta_In_Rho");
		vrString g_const_GaussPointSize_xi_In_Theta_DisContinuous = vrString("Sample.enum_GaussPointSize_xi_In_Theta_DisContinuous");
		vrString g_const_GaussPointSize_xi_In_Rho_DisContinuous = vrString("Sample.enum_GaussPointSize_xi_In_Rho_DisContinuous");
		vrString g_const_GaussPointSize_eta_In_Theta_DisContinuous = vrString("Sample.enum_GaussPointSize_eta_In_Theta_DisContinuous");
		vrString g_const_GaussPointSize_eta_In_Rho_DisContinuous = vrString("Sample.enum_GaussPointSize_eta_In_Rho_DisContinuous");
		vrString g_const_GaussPointSize_eta_In_Theta_360 = vrString("Sample.enum_GaussPointSize_eta_In_Theta_360");
		vrString g_const_GaussPointSize_eta_In_Rho_360 = vrString("Sample.enum_GaussPointSize_eta_In_Rho_360");
		vrString g_const_GaussPointSize_eta_In_Theta_SubTri = vrString("Sample.enum_GaussPointSize_eta_In_Theta_SubTri");
		vrString g_const_GaussPointSize_eta_In_Rho_SubTri = vrString("Sample.enum_GaussPointSize_eta_In_Rho_SubTri");
		

		int g_n_Sample_GaussPointSize_xi_In_Theta;
		int g_n_Sample_GaussPointSize_xi_In_Rho;
		int g_n_Sample_GaussPointSize_eta_In_Theta;
		int g_n_Sample_GaussPointSize_eta_In_Rho;
		int g_n_Sample_GaussPointSize_xi_In_Theta_DisContinuous;
		int g_n_Sample_GaussPointSize_xi_In_Rho_DisContinuous;
		int g_n_Sample_GaussPointSize_eta_In_Theta_DisContinuous;
		int g_n_Sample_GaussPointSize_eta_In_Rho_DisContinuous;
		int g_n_Sample_GaussPointSize_eta_In_Theta_360;
		int g_n_Sample_GaussPointSize_eta_In_Rho_360;
		int g_n_Sample_GaussPointSize_eta_In_Theta_SubTri;
		int g_n_Sample_GaussPointSize_eta_In_Rho_SubTri;

		void printConf()
		{
			

			infoLog << g_const_BEM_youngsMod << " = " << youngsMod;
			infoLog << g_const_BEM_toughness << " = " << toughness;
			infoLog << g_const_BEM_outFile << " = " << outFile;
			infoLog << g_const_Obj_meshName << " = " << g_str_Obj_meshName;
			infoLog << g_const_Obj_remesh << " = " << g_n_Obj_remesh;
			
			infoLog << g_const_DebugHsubmatrix << " = " << g_str_Obj_DebugHsubmatrix;
			infoLog << g_const_DebugGsubmatrix << " = " << g_str_Obj_DebugGsubmatrix;
			infoLog << g_const_DebugDisplacement << " = " << g_str_Obj_DebugDisplacement;
			
			infoLog << g_const_GaussPointSize_xi_In_Theta << " = " << g_n_Sample_GaussPointSize_xi_In_Theta;
			infoLog << g_const_GaussPointSize_xi_In_Rho << " = " << g_n_Sample_GaussPointSize_xi_In_Rho;
			infoLog << g_const_GaussPointSize_eta_In_Theta << " = " << g_n_Sample_GaussPointSize_eta_In_Theta;
			infoLog << g_const_GaussPointSize_eta_In_Rho << " = " << g_n_Sample_GaussPointSize_eta_In_Rho;
			infoLog << g_const_GaussPointSize_xi_In_Theta_DisContinuous << " = " << g_n_Sample_GaussPointSize_xi_In_Theta_DisContinuous;
			infoLog << g_const_GaussPointSize_xi_In_Rho_DisContinuous << " = " << g_n_Sample_GaussPointSize_xi_In_Rho_DisContinuous;
			infoLog << g_const_GaussPointSize_eta_In_Theta_DisContinuous << " = " << g_n_Sample_GaussPointSize_eta_In_Theta_DisContinuous;
			infoLog << g_const_GaussPointSize_eta_In_Rho_DisContinuous << " = " << g_n_Sample_GaussPointSize_eta_In_Rho_DisContinuous;
			infoLog << g_const_GaussPointSize_eta_In_Theta_360 << " = " << g_n_Sample_GaussPointSize_eta_In_Theta_360;
			infoLog << g_const_GaussPointSize_eta_In_Rho_360 << " = " << g_n_Sample_GaussPointSize_eta_In_Rho_360;
			infoLog << g_const_GaussPointSize_eta_In_Theta_SubTri << " = " << g_n_Sample_GaussPointSize_eta_In_Theta_SubTri;
			infoLog << g_const_GaussPointSize_eta_In_Rho_SubTri << " = " << g_n_Sample_GaussPointSize_eta_In_Rho_SubTri;
			/*infoLog << g_const_Obj_meshId << " = " << g_str_Obj_meshId;
			infoLog << g_const_Obj_octreeFineLevel << " = " << g_n_Obj_octreeFineLevel;
			infoLog << g_const_Simulation_doSimulation << " = " << g_Simulation_doSimulation;
			infoLog << g_const_Simulation_MinCount << " = " << g_n_Simulation_MinCount;
			infoLog << g_const_Simulation_MaxCount << " = " << g_n_Simulation_MaxCount;
			infoLog << g_const_Simulation_YoungModulus << " = " << g_db_Simulation_YoungModulus;
			infoLog << g_const_Simulation_externalForceFactor << " = " << g_db_Simulation_externalForceFactor;
			infoLog << g_const_Simulation_scriptForceFactor << " = " << g_db_Simulation_scriptForceFactor;
			infoLog << g_const_Simulation_animation_max_fps << " = " << g_db_Simulation_animation_max_fps;
			infoLog << g_const_Simulation_camera_zoom << " = " << g_db_Simulation_camera_zoom;

			infoLog << g_const_Scene_windowWidth << " = " << g_n_Scene_windowWidth;
			infoLog << g_const_Scene_windowHeight << " = " << g_n_Scene_windowHeight;
			infoLog << g_const_Scene_bkgColor << " = " << g_vec4_Scene_bkgColor.transpose();
			infoLog << g_const_Scene_planeXsize << " = " << g_db_Scene_planeXsize;
			infoLog << g_const_Scene_planeZsize << " = " << g_db_Scene_planeZsize;
			infoLog << g_const_Scene_planeXdivs << " = " << g_n_Scene_planeXdivs;
			infoLog << g_const_Scene_planeZdivs << " = " << g_n_Scene_planeZdivs;
			infoLog << g_const_Scene_planeColor << " = " << g_vec3_Scene_planeColor.transpose();
			infoLog << g_const_Scene_modelColor << " = " << g_vec3_Scene_modelColor.transpose();
			infoLog << g_const_Scene_modelColor4Ka << " = " << g_vec3_Scene_modelColor4Ka.transpose();
			infoLog << g_const_Scene_CamerPos << " = " << g_vec3_Scene_CamerPos.transpose();
			infoLog << g_const_Scene_LightPos << " = " << g_vec3_Scene_LightPos.transpose();
			infoLog << g_const_Scene_EyeCenter << " = " << g_vec3_Scene_EyeCenter.transpose();
			infoLog << g_const_Scene_ZUp << " = " << g_vec3_Scene_ZUp.transpose();
			infoLog << g_const_Scene_ShowFPS << " = " << g_Scene_ShowFPS;
			infoLog << g_const_Scene_FOV << " = " << g_db_Scene_FOV;
			infoLog << g_const_Scene_Near << " = " << g_db_Scene_Near;
			infoLog << g_const_Scene_Far << " = " << g_db_Scene_Far;
			infoLog << g_const_Scene_ShadowMapWidth << " = " << g_n_Scene_ShadowMapWidth;
			infoLog << g_const_Scene_ShadowMapHeight << " = " << g_n_Scene_ShadowMapHeight;*/
		}
	}//GlobalConf
}//VR