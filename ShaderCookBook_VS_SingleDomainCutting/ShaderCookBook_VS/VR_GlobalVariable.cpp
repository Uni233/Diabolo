#include "VR_GlobalVariable.h"
#include "Configure/INIReader.h"
#include <time.h>
#include <iostream>

#if PlatformType == Platform_Windows
#include <windows.h>
#elif PlatformType == PlatForm_Linux
#error UnSupport Linux Operating System
#else
#error UnSupport Operating System(Windows, Linux).
#endif

#define PrintCfg(X) do{ std::cout << #X << "-->" << X << std::endl;}while(false)
#define PrintCfg_vec3(X) do{ std::cout << #X << "-->(" << X.x << "," << X.y << "," << X.z << ")" << std::endl;}while(false)

namespace YC
{
	namespace GlobalVariable
	{
		std::string g_strCurrentPath;
		std::string s_qstrCurrentTime;
		std::string s_qstrCurrentTimeStamp;

		bool g_fullscreen=false;
		bool g_hasCoords=false;
		bool g_hasVerticeNormal=false;
		bool g_normalizeMesh=true;
		bool g_showFPS=true;
		bool g_simulation=true;
		float g_FOV = 60.f;
		float g_near = 1.f;
		float g_far = 100.f;
		std::string g_strMeshName;
		std::string g_strMeshId;
		std::string g_strTextureName;
		std::vector< YC::MyVec3 > g_vecMaterialPoint;
		std::vector< float > g_vecMaterialPointStiff;
		float g_bladeForce=0.f,g_tmpBladeForce=0.f;
		float g_hasGravity=0.f;
		float g_domainGravity[3];
		int g_isApplyBladeForceCurrentFrame=0;

		float g_dbYoungModulus=0.f;
		int g_bcMinCount = 30;
		int g_bcMaxCount = 38;
		float g_externalForceFactor = 10.50f;
		float g_scriptForceFactor = 10.05f;

		vec3 camerPos	= vec3(0.0f,0.05f,0.85f);
		vec3 lightPos	= vec3(-0.8f, 0.825f, 0.75f);
		vec3 eyeCenter	= vec3(0.0f,0.1f,0.0f);
		vec3 zUp		= vec3(0.0f,1.f,0.0f);

		bool g_useCrossBlade = false;
		bool g_useScriptBlade = false;
		int shadowMapWidth = 1024;
		int shadowMapHeight = 1024;

		int g_octreeFineLevel;

		std::string GetModuleDir()
		{
#if PlatformType == Platform_Windows
			char pFileName[256];
			GetModuleFileName( NULL, pFileName, 255 );

			std::string csFullPath(pFileName);
			int nPos = csFullPath.rfind( '\\' );
			if( nPos < 0 )
				return std::string("");
			else
				return csFullPath.substr(0, nPos );
#elif PlatformType == PlatForm_Linux
			#error UnSupport Linux Operating System.
#else
#error UnSupport Operating System(Windows, Linux).
#endif
		}

		void printGlobalVariable()
		{
			PrintCfg( g_strCurrentPath);
			PrintCfg( s_qstrCurrentTime);
			PrintCfg( s_qstrCurrentTimeStamp);

			PrintCfg( g_fullscreen);
			PrintCfg( g_hasCoords);
			PrintCfg( g_hasVerticeNormal);
			PrintCfg( g_normalizeMesh);
			PrintCfg( g_showFPS);
			PrintCfg( g_simulation);
			PrintCfg( g_FOV);
			PrintCfg( g_near);
			PrintCfg( g_far);
			PrintCfg( g_strMeshName);
			PrintCfg( g_strMeshId);
			PrintCfg( g_strTextureName);
			//PrintCfg( g_vecMaterialPoint);
			//PrintCfg( g_vecMaterialPointStiff);
			PrintCfg( g_bladeForce);
			PrintCfg( g_tmpBladeForce);
			PrintCfg( g_hasGravity);
			PrintCfg( g_domainGravity);
			PrintCfg( g_isApplyBladeForceCurrentFrame);

			PrintCfg( g_dbYoungModulus);
			PrintCfg( g_bcMinCount);
			PrintCfg( g_bcMaxCount);
			PrintCfg( g_externalForceFactor);
			PrintCfg( g_scriptForceFactor);

			PrintCfg_vec3( camerPos	);
			PrintCfg_vec3( lightPos	);
			PrintCfg_vec3( eyeCenter	);
			PrintCfg_vec3( zUp	);

			PrintCfg( g_useCrossBlade );
			PrintCfg( g_useScriptBlade );
			PrintCfg( shadowMapWidth );
			PrintCfg( shadowMapHeight );
			PrintCfg( g_octreeFineLevel );
			
		}

		void initGlobalVariable(int argc, char *argv[])
		{
			{
				g_strCurrentPath = GetModuleDir();
			}
			{
				std::string s_qstrCurrentData;
				std::stringstream ss;
				time_t now;

				now = time(0);
				tm *tnow = localtime(&now);
				ss << (1900+tnow->tm_year) << "-" << (tnow->tm_mon+1) << "-" << (tnow->tm_mday);
				s_qstrCurrentData = (ss.str());
				ss.str("");
				ss << (tnow->tm_hour) << "-" << (tnow->tm_min) << "-" << (tnow->tm_sec);
				s_qstrCurrentTime = (ss.str());
				s_qstrCurrentTimeStamp = s_qstrCurrentData + std::string("-") + s_qstrCurrentTime;
			}
			{
				INIReader reader(std::string(g_strCurrentPath + "\\Configure\\configure.ini").c_str());

				if (reader.ParseError() < 0) {
					std::cout << "Can't load 'configure.ini'\n";
					MyExit;
					return ;
				}
				std::cout << "Config loaded from 'configure.ini': hasCoord="
					<< reader.GetInteger("Obj", "hasCoord", -1) << ", normalizeMesh="
					<< reader.Get("Obj", "normalizeMesh", "UNKNOWN") << "\n"; 

				g_hasCoords = (0 != reader.GetInteger("Obj", "hasCoord", 0) );
				g_hasVerticeNormal = (0 != reader.GetInteger("Obj", "hasVerticeNormal", 0));
				g_normalizeMesh = (0 != reader.GetInteger("Obj", "normalizeMesh", 0));

				g_octreeFineLevel = reader.GetInteger("Obj","octreeFineLevel",5);
				g_strMeshName = reader.Get("Obj", "meshName", "armadillo_1M_Normal.obj");
				g_strMeshId = reader.Get("Obj", "meshId", "armadillo");
				g_strTextureName = reader.Get("Obj", "textureName", "steak.png");
				g_showFPS = (0 != reader.GetInteger("Scene", "showFPS", 1) );
				g_fullscreen = (0 != reader.GetInteger("Scene", "fullscreen", 1) );

				g_FOV = reader.GetReal("Scene","FOV",60.f);
				g_near = reader.GetReal("Scene","near",0.3f);
				g_far = reader.GetReal("Scene","far",100.f);

				camerPos.x = reader.GetReal("Scene","cameraPositionX",camerPos.x);
				camerPos.y = reader.GetReal("Scene","cameraPositionY",camerPos.y);
				camerPos.z = reader.GetReal("Scene","cameraPositionZ",camerPos.z);
				lightPos.x = reader.GetReal("Scene","lightPositionX",lightPos.x);
				lightPos.y = reader.GetReal("Scene","lightPositionY",lightPos.y);
				lightPos.z = reader.GetReal("Scene","lightPositionZ",lightPos.z);

				eyeCenter.x = reader.GetReal("Scene","eyeCenterX",eyeCenter.x);
				eyeCenter.y = reader.GetReal("Scene","eyeCenterY",eyeCenter.y);
				eyeCenter.z = reader.GetReal("Scene","eyeCenterZ",eyeCenter.z);
				zUp.x = reader.GetReal("Scene","zUpX",zUp.x);
				zUp.y = reader.GetReal("Scene","zUpY",zUp.y);
				zUp.z = reader.GetReal("Scene","zUpZ",zUp.z);
				

				g_simulation = (0 != reader.GetInteger("Simulation", "doSimulation", 1) );
				g_bcMinCount=reader.GetInteger("Simulation", "MinCount", 10000);
				g_bcMaxCount=reader.GetInteger("Simulation", "MaxCount", 10000);
				g_dbYoungModulus=reader.GetReal("Simulation","YoungModulus",600000000.0);
				g_externalForceFactor=reader.GetReal("Simulation","externalForceFactor",0.0);
				g_scriptForceFactor=reader.GetReal("Simulation","scriptForceFactor",0.0);
				g_bladeForce=reader.GetReal("Simulation","bladeForceFactor",0.0);
				g_isApplyBladeForceCurrentFrame=reader.GetInteger("Simulation","ApplyBladeForceCurrentFrame",0);
				g_tmpBladeForce = g_bladeForce;
				g_hasGravity=reader.GetReal("Simulation","gravity",0.0);

				{
					g_domainGravity[0] = reader.GetReal("Simulation","gravity_0",0.0);
					g_domainGravity[1] = reader.GetReal("Simulation","gravity_1",0.0);
					g_domainGravity[2] = reader.GetReal("Simulation","gravity_2",0.0);
				}

				g_useCrossBlade = (0 != reader.GetInteger("Simulation", "useCrossBlade", 1) );
				g_useScriptBlade = (0 != reader.GetInteger("Simulation", "useScriptBlade", 1) );
#if 0
				int nSamplePointSize = reader.GetInteger("Material", "materialPointSize", 0);
				g_vecMaterialPoint.resize(nSamplePointSize);
				g_vecMaterialPointStiff.resize(nSamplePointSize);
				char tmpChar;
				std::stringstream ss;
				std::string ssContent;
				for (int v=0;v<nSamplePointSize;++v)
				{
					ss.str("");
					ss << "materialPoint_" << v;
					ssContent = reader.Get("Material", ss.str().c_str(), "");
					printf("[%s][%s]\n",ss.str().c_str(),ssContent.c_str());
					{
						boost::algorithm::split_iterator< std::string::iterator > iStr( 
							ssContent,
							boost::algorithm::token_finder(
							boost::algorithm::is_any_of( " /\t\n\r" ),
							boost::algorithm::token_compress_on 
							) 
							);
						boost::algorithm::split_iterator< std::string::iterator> end;

						g_vecMaterialPoint[v][0] = boost::lexical_cast<float>( boost::lexical_cast<std::string>((*iStr)).c_str() );
						++iStr;
						g_vecMaterialPoint[v][1] = boost::lexical_cast<float>( boost::lexical_cast<std::string>((*iStr)).c_str() );
						++iStr;
						g_vecMaterialPoint[v][2] = boost::lexical_cast<float>( boost::lexical_cast<std::string>((*iStr)).c_str() );
						++iStr;
					}
					printf("{%f,%f,%f}[%f]\n",g_vecMaterialPoint[v][0],g_vecMaterialPoint[v][1],g_vecMaterialPoint[v][2],g_vecMaterialPointStiff[v]);
					ss.str("");
					ss << "materialStiff_" << v;
					g_vecMaterialPointStiff[v] = (float)reader.GetReal("Material",ss.str().c_str(),0.f);
				}
#endif
			}
			{
				printGlobalVariable();
			}
		}
	}// namespace GlobalVariable
}// namespace YC