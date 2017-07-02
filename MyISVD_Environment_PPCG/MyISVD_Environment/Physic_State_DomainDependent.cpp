#include "Physic_State_DomainDependent.h"

#if USE_MultidomainIndependent

namespace YC
{
	void Physic_State_DomainIndependent::resize()
	{
		const MyInt nDof = m_nLocalDomainDof;

		for (MyInt i = 0; i < PMI::PhysicMatNum; ++i)
		{
			m_LocalMatrix[i].resize(nDof, nDof);
			m_LocalMatrix[i].setZero();
		}

		{
			m_LocalMatrix[PMI::ConstraintMat].resize(m_nGlobal_Dofs_Q, nDof);
			m_LocalMatrix[PMI::ConstraintMat].setZero();
		}

		for (MyInt i = 0; i < PMI::PhysicVecNum; ++i)
		{
			m_LocalVector[i].resize(nDof);
			m_LocalVector[i].setZero();
		}
	}

	void Physic_State_DomainIndependent::getConstraintMat(const MyMatrix& globalCMat)
	{
		m_LocalMatrix[PMI::ConstraintMat] = globalCMat.block(0, m_nGlobalDofsBase, m_nGlobal_Dofs_Q, m_nLocalDomainDof);
	}

#if USE_PPCG
	void Physic_State_DomainIndependent::assembleMatrixLocalToGlobal_PPCG(Physic_PPCG_State_Independent & globalState)
	{
		//G22, A, A_stiff, A_damping, A_mass, B, B1, B1_Inv, B2, C, R_hat;
		globalState.m_LocalMatrix[PPCGI::A].block(m_nGlobalDofsBase, m_nGlobalDofsBase, m_nLocalDomainDof, m_nLocalDomainDof) = m_LocalMatrix[PMI::ComputeMat];
		globalState.m_LocalMatrix[PPCGI::A_stiff].block(m_nGlobalDofsBase, m_nGlobalDofsBase, m_nLocalDomainDof, m_nLocalDomainDof) = m_LocalMatrix[PMI::StiffMat];
		globalState.m_LocalMatrix[PPCGI::A_mass].block(m_nGlobalDofsBase, m_nGlobalDofsBase, m_nLocalDomainDof, m_nLocalDomainDof) = m_LocalMatrix[PMI::MassMat];
		globalState.m_LocalMatrix[PPCGI::A_damping].block(m_nGlobalDofsBase, m_nGlobalDofsBase, m_nLocalDomainDof, m_nLocalDomainDof) = m_LocalMatrix[PMI::DampingMat];
		globalState.m_LocalVector[PPCGI::computeRhs].block(m_nGlobalDofsBase, 0, m_nLocalDomainDof, 1) = m_LocalVector[PMI::ComputeRhs_Vec];
		globalState.m_LocalMatrix[PPCGI::B].block(0, m_nGlobalDofsBase, m_nGlobal_Dofs_Q, m_nLocalDomainDof) = m_LocalMatrix[PMI::ConstraintMat];
	}
#else//USE_PPCG
	void Physic_State_DomainIndependent::assembleMatrixLocalToGlobal(Physic_State_DomainIndependent& globalState)
	{

		globalState.m_LocalMatrix[PMI::ComputeMat].block(m_nGlobalDofsBase, m_nGlobalDofsBase, m_nLocalDomainDof, m_nLocalDomainDof) = m_LocalMatrix[PMI::ComputeMat];
		globalState.m_LocalMatrix[PMI::StiffMat].block(m_nGlobalDofsBase, m_nGlobalDofsBase, m_nLocalDomainDof, m_nLocalDomainDof) = m_LocalMatrix[PMI::StiffMat];
		globalState.m_LocalMatrix[PMI::MassMat].block(m_nGlobalDofsBase, m_nGlobalDofsBase, m_nLocalDomainDof, m_nLocalDomainDof) = m_LocalMatrix[PMI::MassMat];
		globalState.m_LocalMatrix[PMI::DampingMat].block(m_nGlobalDofsBase, m_nGlobalDofsBase, m_nLocalDomainDof, m_nLocalDomainDof) = m_LocalMatrix[PMI::DampingMat];
		globalState.m_LocalVector[PMI::ComputeRhs_Vec].block(m_nGlobalDofsBase, 0, m_nLocalDomainDof, 1) = m_LocalVector[PMI::ComputeRhs_Vec];
		globalState.m_LocalMatrix[PMI::ComputeMat].block(m_nGlobalDofs, m_nGlobalDofsBase, m_nGlobal_Dofs_Q, m_nLocalDomainDof) = m_LocalMatrix[PMI::ConstraintMat];
		globalState.m_LocalMatrix[PMI::ComputeMat].block(m_nGlobalDofsBase, m_nGlobalDofs, m_nLocalDomainDof, m_nGlobal_Dofs_Q) = m_LocalMatrix[PMI::ConstraintMat].transpose();
	}
#endif//USE_PPCG

	void Physic_State_DomainIndependent::printInfo()
	{
		printf("Physic_State_DomainIndependent begin\n");
		printf("m_nGlobalDofsBase %d\n", m_nGlobalDofsBase);
		printf("m_nGlobalDofs %d\n", m_nGlobalDofs);
		printf("m_nLocalDomainDof %d\n", m_nLocalDomainDof);
		printf("m_nGlobal_Dofs_Q %d\n", m_nGlobal_Dofs_Q);
		printf("m_nLocalDC_Dofs %d\n", m_nLocalDC_Dofs);
		printf("m_LocalMatrix [%d, %d] %d\n", m_LocalMatrix[0].rows(), m_LocalMatrix[0].cols());
		printf("Physic_State_DomainIndependent end\n");
	}



}//namespace YC
#endif //USE_MultidomainIndependent