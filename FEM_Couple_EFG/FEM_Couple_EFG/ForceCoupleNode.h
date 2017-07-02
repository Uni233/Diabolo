#ifndef _ForceCoupleNode
#define _ForceCoupleNode

#include "VR_Global_Define.h"

namespace VR_FEM
{
	struct ForceCoupleNode
	{
		MyInt m_nCoupleDomainId;
		MyVectorI m_vecCoupleDomainDofs;
		MyInt m_nLocalDomainId;
		MyVectorI m_vecLocalDomainDofs;
	};
}

#endif//_ForceCoupleNode