#include <igl/colon.h>
#include <igl/harmonic.h>
#include <igl/readOBJ.h>
#include <igl/readDMAT.h>
#include <igl/viewer/Viewer.h>
#include <algorithm>
#include <iostream>
#include "VR_Global_Define.h"
#include "IGL_Mesh_Struct.h"

#if USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/tbb_allocator.h>
#include "tbb_parallel_set_color.h"
#endif

#include "VR_Physics_FEM_Simulation.h"
#include "VR_Physics_FEM_Simulation_MultiDomainIndependent.h"
namespace YC
{
	IGL_Mesh_Struct g_iglMeshStruct;
	VR_Physics_FEM_Simulation g_simulator;
	VR_Physics_FEM_Simulation_MultiDomainIndependent g_simulator_MultidomainIndependent;
	int cuda_bcMinCount = 2;
	int cuda_bcMaxCount = 8;
	int subspaceModeNum = 15;
	int XCount = 5;
	int YCount = 4;
	int ZCount = 4;
	float g_scriptForceFactor = 200.f *0;

	void InitProcess(YC::MyMatrix& matV, YC::MyIntMatrix& matF, YC::MyIntMatrix& matV_dofs, YC::MyIntVector& vecF_ColorId)
	{
#if USE_MultidomainIndependent
		g_simulator_MultidomainIndependent.loadOctreeNode_MultidomainIndependent(XCount, YCount, ZCount);
		g_simulator_MultidomainIndependent.distributeDof_local_Independent();
		g_simulator_MultidomainIndependent.distributeDof_global();
		g_simulator_MultidomainIndependent.createDCBoundaryCondition_Independent();
		g_simulator_MultidomainIndependent.createGlobalMassAndStiffnessAndDampingMatrix_Independent();
		g_simulator_MultidomainIndependent.createNewMarkMatrix_DMI();
		g_simulator_MultidomainIndependent.createConstraintMatrix_DMI();
		g_simulator_MultidomainIndependent.assembleGlobalMatrix_Independent();
		//g_simulator_MultidomainIndependent.TestVtxCellId();
		g_simulator_MultidomainIndependent.creat_Outter_Skin(matV, matF, matV_dofs, vecF_ColorId);
#else//USE_MultidomainIndependent
		g_simulator.loadOctreeNode_Global(XCount, YCount, ZCount);

		g_simulator.createDCBoundaryCondition_Global();
		g_simulator.distributeDof_global();
		g_simulator.createForceBoundaryCondition_Global(XCount, YCount, ZCount);
		g_simulator.createGlobalMassAndStiffnessAndDampingMatrix_FEM_Global();
		g_simulator.createNewMarkMatrix_Global();
#if USE_SUBSPACE
		g_simulator.createModalReduction(subspaceModeNum);
#endif
		g_simulator.TestVtxCellId();

		g_simulator.creat_Outter_Skin(matV, matF, matV_dofs);
#endif//USE_MultidomainIndependent
	}

	void SimulationStep()
	{
		static int sTimeStep = 0;

#if USE_MultidomainIndependent
#if USE_PPCG
		g_simulator_MultidomainIndependent.simulationOnCPU_DMI_Global_PPCG(sTimeStep++);
#else//USE_PPCG
		g_simulator_MultidomainIndependent.simulationOnCPU_DMI_Global(sTimeStep++);
#endif//USE_PPCG
		
		return ;
#else//USE_MultidomainIndependent

#if USE_SUBSPACE
		g_simulator.simulationSubspaceOnCPU(sTimeStep++);
#else

#if USE_MODAL_WARP
		g_simulator.simulationOnCPU_Global_WithModalWrap(sTimeStep++);
#else
		g_simulator.simulationOnCPU_Global(sTimeStep++);
#endif//USE_MODAL_WARP
#endif//USE_SUBSPACE

#endif//USE_MultidomainIndependent
		//sTimeStep = sTimeStep & (64 - 1);
	}
	void MyRenderScene()
	{
		g_simulator.render_Global();
	}

	void Simulation()
	{
		g_simulator.m_bSimulation = true;
	}

	void computeVertexDisp(const YC::MyVector& dispVec, YC::MyMatrix& dispMat, const YC::MyInt nVtxSize)
	{
	}

	void printObjMesh(const char* lpszFileName, const YC::MyIntMatrix& original_face_struct, const MyMatrix& current_vtx_pos, const YC::MyIntVector& current_face_color_id)
	{
		std::ofstream outfile(lpszFileName);
		for (int v = 0; v < current_vtx_pos.rows(); ++v)
		{
			auto& refVec = current_vtx_pos.row(v);
			outfile << "v " << refVec.x() << " " << refVec.y() << " " << refVec.z() << std::endl;
		}

		std::map< MyInt, std::vector<MyInt> > map_groupId2faceSet;
		for (int f = 0; f < current_face_color_id.size(); ++f)
		{
			std::vector<MyInt>& refVec = map_groupId2faceSet[current_face_color_id[f]];
			refVec.push_back(f);
		}

		for (iterAllOf(itr, map_groupId2faceSet))
		{
			std::vector<MyInt>& refVec = (*itr).second;
			outfile << "g sub_" << std::endl;

			for (iterAllOf(ii, refVec))
			{
				auto& refVec = original_face_struct.row(*ii);
				outfile << "f " << refVec.x()+1 << " " << refVec.y()+1 << " " << refVec.z()+1 << std::endl;
			}
		}
		outfile.close();
	}

	bool pre_draw(igl::viewer::Viewer & viewer)
	{
		using namespace Eigen;
		// Determine boundary conditions
		IGL_Mesh_Struct& refMesh = g_iglMeshStruct;

		SimulationStep();
#if USE_PPCG
		g_simulator_MultidomainIndependent.getSkinDisplacement(
			g_simulator_MultidomainIndependent.getGlobalState_DMI_PPCG().m_LocalVector[PPCGI::incremental_displacement],
			refMesh.m_current_vtx_pos,
			refMesh.m_original_vtx_dofs,
			refMesh.m_current_vtx_disp);
#else//USE_PPCG
		g_simulator_MultidomainIndependent.getSkinDisplacement(
			g_simulator_MultidomainIndependent.getGlobalState_DMI().m_LocalVector[PMI::incremental_displacement_Vec],
			refMesh.m_current_vtx_pos,
			refMesh.m_original_vtx_dofs,
			refMesh.m_current_vtx_disp);
#endif//USE_PPCG

		//refMesh.m_current_vtx_disp.setZero();

		viewer.data.set_vertices(/*g_iglMeshStruct.m_current_vtx_pos + */refMesh.m_current_vtx_disp);
		viewer.data.compute_normals();

		static int sCount = 0;
		std::stringstream ss;
		ss << "d:/test_" << sCount++ << ".obj";
		printObjMesh(ss.str().c_str(), refMesh.m_original_face_struct, refMesh.m_current_vtx_disp, refMesh.m_current_face_color_id);
		return false;
	}

	bool key_down(igl::viewer::Viewer &viewer, unsigned char key, int mods)
	{
		switch (key)
		{
		case ' ':
			viewer.core.is_animating = !viewer.core.is_animating;
			return true;
		}
		return false;
	}

	bool init_global_variable(igl::viewer::Viewer & viewer)
	{
		using namespace tbb;
		IGL_Mesh_Struct& refMesh = g_iglMeshStruct;

		InitProcess(refMesh.m_original_vtx_pos, refMesh.m_original_face_struct, refMesh.m_original_vtx_dofs, refMesh.m_current_face_color_id);

		//igl::readOBJ(MyMeshPath "Bunny_3w.obj", refMesh.m_original_vtx_pos, refMesh.m_original_face_struct);
		refMesh.m_current_vtx_pos = refMesh.m_original_vtx_pos;

		refMesh.m_nVtxSize = refMesh.m_original_vtx_pos.rows();
		refMesh.m_nFaceSize = refMesh.m_original_face_struct.rows();
		// Pseudo-color based on selection
		refMesh.m_current_face_color.resize(refMesh.m_nFaceSize, myDim);

#if USE_TBB
		parallel_for(blocked_range<size_t>(0, refMesh.m_nFaceSize), ApplyColorMultiDomain(&refMesh.m_current_face_color, &refMesh.m_current_face_color_id), auto_partitioner());
		//parallel_for(blocked_range<size_t>(0, refMesh.m_nFaceSize), ApplyColor(&refMesh.m_current_face_color), auto_partitioner());
#else
		for (int f = 0; f < refMesh.m_nFaceSize; f++)
		{
			refMesh.m_current_face_color.row(f) = Colors::gold;
		}
#endif

		viewer.data.set_mesh(refMesh.m_current_vtx_pos, refMesh.m_original_face_struct);
		viewer.core.show_lines = true;
		viewer.core.orthographic = true;
		viewer.data.set_colors(refMesh.m_current_face_color);
		//viewer.core.trackball_angle = Eigen::Quaternionf(sqrt(2.0), 0, sqrt(2.0), 0);
		viewer.core.trackball_angle.normalize();
		viewer.callback_pre_draw = &pre_draw;
		viewer.callback_key_down = &key_down;
		viewer.core.is_animating = true;
		viewer.core.animation_max_fps = 30.;
		viewer.core.camera_zoom = 1.5f;

		viewer.core.background_color = Eigen::Vector3f(191.0 / 255.0, 191.0 / 255.0, 191.0 / 255.0);
		//viewer.core.background_color = Eigen::Vector3f(64.0 / 255.0, 64.0 / 255.0, 64.0 / 255.0);
	}

	
}