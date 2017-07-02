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


#if USE_MultiDomain
#include "VR_Physics_FEM_Simulation_MultiDomain.h"
#else
#include "VR_Physics_FEM_Simulation.h"
#endif//USE_MultiDomain

namespace YC
{
	IGL_Mesh_Struct g_iglMeshStruct;
#if USE_MultiDomain
	VR_Physics_FEM_Simulation_MultiDomain g_simulator_multidomain;
	int XCount = 5;
	int YCount = 4;
	int ZCount = 4;
#else
	VR_Physics_FEM_Simulation g_simulator;
	int XCount = 30;
	int YCount = 4;
	int ZCount = 4;
#endif//USE_MultiDomain

	int cuda_bcMinCount = 2;
	int cuda_bcMaxCount = -8;
	int subspaceModeNum = 15;
	
	float g_scriptForceFactor = 200.f *0;

	void InitProcess(YC::MyMatrix& matV, YC::MyIntMatrix& matF, YC::MyIntMatrix& matV_dofs, YC::MyIntVector& vecF_ColorId)
	{
#if USE_MultiDomain
		g_simulator_multidomain.loadOctreeNode_MultiDomain(XCount, YCount, ZCount);
		g_simulator_multidomain.distributeDof_ALM_PPCG();
		g_simulator_multidomain.distributeDof_local();
		g_simulator_multidomain.createDCBoundaryCondition_ALM();

		g_simulator_multidomain.createGlobalMassAndStiffnessAndDampingMatrix_ALM();

		g_simulator_multidomain.createNewMarkMatrix_ALM();
		g_simulator_multidomain.makeALM_PPCG_PerVertex();
		//g_simulator_multidomain.makeALM_PPCG();
		//g_simulator_multidomain.makeALM_New();
		//g_simulator_multidomain.makeALM_PerVertex();
		g_simulator_multidomain.creat_Outter_Skin(matV, matF, matV_dofs, vecF_ColorId);
#else
		g_simulator.loadOctreeNode_Global(XCount, YCount, ZCount);
		g_simulator.createDCBoundaryCondition_Global();
		g_simulator.distributeDof_global();
		g_simulator.createForceBoundaryCondition_Global(XCount, YCount, ZCount);
		g_simulator.createGlobalMassAndStiffnessAndDampingMatrix_FEM_Global();
		g_simulator.createNewMarkMatrix_Global();

		g_simulator.TestVtxCellId();

		g_simulator.creat_Outter_Skin(matV, matF, matV_dofs);

#if USE_SUBSPACE_Whole_Spectrum
		g_simulator.create_Whole_Spectrum(g_simulator.getGlobalState(), g_simulator.getWholeSpectrumState());
#endif//USE_SUBSPACE_Whole_Spectrum

#if USE_SUBSPACE_SVD

#if SingleDomainISVD
		g_simulator.init_iSVD_updater(g_simulator.getGlobalState(), VR_Physics_FEM_Simulation::nSVDModeNum);
#else
		g_simulator.initSVD(g_simulator.getGlobalState(), VR_Physics_FEM_Simulation::nSVDModeNum);
#endif//SingleDomainISVD
		
#endif//USE_SUBSPACE_SVD

#endif//USE_MultiDomain
		
	}

	void SimulationStep()
	{
		static int sTimeStep = 0;
#if USE_MultiDomain
		g_simulator_multidomain.simulation_PPCG();
		//g_simulator_multidomain.simulation_ALM();
		//g_simulator_multidomain.Global_PPCG_State().incremental_displacement_PPCG = g_simulator_multidomain.Global_ALM_State().incremental_displacement;
#else

#if USE_SUBSPACE_SVD

#if SingleDomainISVD
		g_simulator.simulation_iSVDOnCPU(sTimeStep++);
#else
		g_simulator.simulationSVDOnCPU(sTimeStep++);
#endif//SingleDomainISVD
		
		return;
#endif

		//#if USE_SUBSPACE
		//		g_simulator.simulationSubspaceOnCPU(sTimeStep++);
		//#endif

#if USE_MODAL_WARP
		g_simulator.simulationOnCPU_Global_WithModalWrap(sTimeStep++);
		return;
#else
		g_simulator.simulationOnCPU_Global(sTimeStep++);
		return;
#endif

#if USE_SUBSPACE_Whole_Spectrum
		g_simulator.simulationWholeSpectrumOnCPU(sTimeStep++);
		return;
#endif

#endif//USE_MultiDomain


		//sTimeStep = sTimeStep & (64 - 1);
	}
	void MyRenderScene()
	{
	}

	void Simulation()
	{
#if USE_MultiDomain
		g_simulator_multidomain.m_bSimulation = true;
#else
		g_simulator.m_bSimulation = true;
#endif//USE_MultiDomain
		
	}

	void computeVertexDisp(const YC::MyVector& dispVec, YC::MyMatrix& dispMat, const YC::MyInt nVtxSize)
	{
	}

	bool pre_draw(igl::viewer::Viewer & viewer)
	{
		using namespace Eigen;
		// Determine boundary conditions
		IGL_Mesh_Struct& refMesh = g_iglMeshStruct;

		SimulationStep();

#if USE_MultiDomain
		g_simulator_multidomain.getSkinDisplacement(g_simulator_multidomain.Global_PPCG_State().incremental_displacement_PPCG, g_iglMeshStruct.m_current_vtx_pos, refMesh.m_original_vtx_dofs, refMesh.m_current_vtx_disp);
#else

#if USE_SUBSPACE_SVD
		if (g_simulator.isSimulationOnSubspace())
		{
			viewer.data.set_colors(refMesh.m_current_face_color_subspace);
		}
		else
		{
			viewer.data.set_colors(refMesh.m_current_face_color);
		}
#endif
		g_simulator.getSkinDisplacement(g_simulator.getGlobalState(), g_iglMeshStruct.m_current_vtx_pos, refMesh.m_original_vtx_dofs, refMesh.m_current_vtx_disp);

#endif//USE_MultiDomain


		viewer.data.set_vertices(/*g_iglMeshStruct.m_current_vtx_pos + */refMesh.m_current_vtx_disp);
		viewer.data.compute_normals();
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

#if USE_MultiDomain
		parallel_for(blocked_range<size_t>(0, refMesh.m_nFaceSize), ApplyColorMultiDomain(&refMesh.m_current_face_color, &refMesh.m_current_face_color_id), auto_partitioner());
#else
		parallel_for(blocked_range<size_t>(0, refMesh.m_nFaceSize), ApplyColor(&refMesh.m_current_face_color, Colors::gold), auto_partitioner());
#endif//USE_MultiDomain
		refMesh.m_current_face_color_subspace.resize(refMesh.m_nFaceSize, myDim);
		parallel_for(blocked_range<size_t>(0, refMesh.m_nFaceSize), ApplyColor(&refMesh.m_current_face_color_subspace, Colors::purple), auto_partitioner());
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
		return true;
	}
}