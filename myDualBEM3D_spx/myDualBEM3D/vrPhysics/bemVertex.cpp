#include "bemVertex.h"
#include "bemTriangleElem.h"
#include "vrId.h"
namespace VR
{
	std::vector< VertexPtr > Vertex::s_vertexCache;
	std::vector< VertexPtr > Vertex::s_vertexCacheEndVtx;
	
	VertexPtr Vertex::s_invalid_vertex(new Vertex(Invalid_Id, Invalid_Id, Invalid_Id, TriangleSetType::Regular));

#if USE_VDB
	vrFloat Vertex::s_MaxStress = FLT_MIN;
	vrFloat Vertex::s_MinStress = FLT_MAX;

	void Vertex::clearStress()
	{
		for (iterAllOf(ci,s_vertexCache))
		{
			(*ci)->m_StressVal = 0.0;
		}
	}
#endif//USE_VDB

#if USE_NEW_VERTEX
	VertexPtr Vertex::makeVertex4dualPlus(const TriangleSetType triType, const MyVec3& point)
	{
		std::vector< VertexPtr >::reverse_iterator itr = std::find_if(s_vertexCache.rbegin(), s_vertexCache.rend(), VertexComparePlus(point[0], point[1], point[2],triType));
		if (s_vertexCache.rend() == itr)
		{
			//no find
			//printf("new pos {%f,%f,%f}\n",point[0],point[1],point[2]);
			s_vertexCache.push_back(VertexPtr(new Vertex(point[0], point[1], point[2], triType)));
			s_vertexCache[s_vertexCache.size() - 1]->setId(s_vertexCache.size() - 1);
			return s_vertexCache[s_vertexCache.size() - 1];
		}
		else
		{
			//find it
			//Q_ASSERT(false);
			/*printf("{%f,%f,%f}\n",point[0],point[1],point[2]);
			MyPause;*/
			return (*itr);
		}

	}

	VertexPtr Vertex::makeVertex4dualPlusEndVtx(const MyVec3& point)
	{
		std::vector< VertexPtr >::reverse_iterator itr = std::find_if(s_vertexCacheEndVtx.rbegin(), s_vertexCacheEndVtx.rend(), VertexCompare(point[0], point[1], point[2]));
		if (s_vertexCacheEndVtx.rend() == itr)
		{
			//no find
			//printf("new pos {%f,%f,%f}\n",point[0],point[1],point[2]);
			s_vertexCacheEndVtx.push_back(VertexPtr(new Vertex(point[0], point[1], point[2], EndVtx)));
			s_vertexCacheEndVtx[s_vertexCacheEndVtx.size() - 1]->setId(s_vertexCacheEndVtx.size() - 1);
			return s_vertexCacheEndVtx[s_vertexCacheEndVtx.size() - 1];
		}
		else
		{
			//find it
			//Q_ASSERT(false);
			/*printf("{%f,%f,%f}\n",point[0],point[1],point[2]);
			MyPause;*/
			return (*itr);
		}
	}
#else
	VertexPtr Vertex::makeVertex4dual(const vrInt nId, const MyVec3& point)
	{
		s_vertexCache.push_back(VertexPtr(new Vertex(point[0], point[1], point[2])));
		s_vertexCache[s_vertexCache.size() - 1]->setId(s_vertexCache.size() - 1);
		Q_ASSERT(nId == (s_vertexCache[s_vertexCache.size() - 1]->getId()));
		return s_vertexCache[s_vertexCache.size() - 1];
	}
	VertexPtr Vertex::makeVertex(const MyPoint& point)
	{
		std::vector< VertexPtr >::reverse_iterator itr = std::find_if(s_vertexCache.rbegin(), s_vertexCache.rend(), VertexCompare(point[0], point[1], point[2]));
		if (s_vertexCache.rend() == itr)
		{
			//no find
			//printf("new pos {%f,%f,%f}\n",point[0],point[1],point[2]);
			s_vertexCache.push_back(VertexPtr(new Vertex(point[0], point[1], point[2])));
			s_vertexCache[s_vertexCache.size() - 1]->setId(s_vertexCache.size() - 1);
			return s_vertexCache[s_vertexCache.size() - 1];
		}
		else
		{
			//find it
			//Q_ASSERT(false);
			/*printf("{%f,%f,%f}\n",point[0],point[1],point[2]);
			MyPause;*/
			return (*itr);
		}
	}

	void Vertex::makeTriangleVertex(const std::vector< MyPoint >& coords, std::vector< VertexPtr >& vertexes)
	{
		vertexes.resize(coords.size());
		for (int v = 0; v < coords.size();++v)
		{
			vertexes[v] = makeVertex(coords[v]);
		}
	}
#endif

	void Vertex::addShareTriangleElement(TriangleElemPtr triPtr)
	{
		for (int e=0;e<m_vec_ShareTriangleElem.size();++e)
		{
			if ( (triPtr->getID()) == (m_vec_ShareTriangleElem[e]->getID()) )
			{
				return ;
			}
		}
		m_vec_ShareTriangleElem.push_back(triPtr); 
	}

#if USE_DUAL
	void Vertex::computeVertexNormal()
	{
		if (isContinuousVertex())
		{
			m_vertexNormal.setZero();
			for (iterAllOf(ci,m_vec_ShareTriangleElem))
			{
				m_vertexNormal += (*ci)->getElemNormals();
			}
			m_vertexNormal /= m_vec_ShareTriangleElem.size();
		}
		else if (isDisContinuousVertex())
		{
			m_vertexNormal.setZero();
			//printf("m_vec_ShareTriangleElem.size(%d)\n",m_vec_ShareTriangleElem.size());
			Q_ASSERT(1 == m_vec_ShareTriangleElem.size());
			m_vertexNormal = m_vec_ShareTriangleElem[0]->getElemNormals();
		}
		/*
		{
		for each face in model
		{
		if face uses vertices[i]
		{
		tempNorm += face.normal;
		numFaces++;
		}
		}

		vertexNorm[i] = tempNorm/=numFaces;
		}
		*/
	}

	void Vertex::parserCrackTip()
	{
		vrInt nCount[3/*Regular=0, positive=1, negative=2*/]={0,0,0};
		for (iterAllOf(ci,m_vec_ShareTriangleElem))
		{
			nCount[(*ci)->getTriSetType()]++;
		}

		if (0 != (nCount[1]*nCount[2]) )
		{
			MyError("impossible crack tip in dual progress. [Vertex::parserCrackTip]");
			m_VertexTypeInDual = VertexTypeInDual::CrackTip;			
		}
	}

	void Vertex::setMirrorVertex(const vrInt mirrorId)
	{
		Q_ASSERT(mirrorId != getId());
		setMirrorVertexId.insert(mirrorId);
		Q_ASSERT(setMirrorVertexId.size()<2);

		vrInt nCount[3/*Regular=0, positive=1, negative=2*/]={0,0,0};
		for (iterAllOf(ci,m_vec_ShareTriangleElem))
		{
			nCount[(*ci)->getTriSetType()]++;
		}
		
		/*if (56 == getId() || 59 == getId() || 222 == getId() || 142 == getId() || 182 == getId() || 262 == getId() || 302 == getId() || 342 == getId() || 405 == getId() )
		{
			printf("nCount [%d][%d][%d]\n",nCount[0],nCount[1],nCount[2]);
			vrPause;
		}*/
		
		/*if (0 != (nCount[1]*nCount[2]) )
		{
			m_VertexTypeInDual = VertexTypeInDual::CrackTip;
			printf("0 != (nCount[1]*nCount[2]) \n");
			vrPause;
			return ;
		}*/
		Q_ASSERT(0 == (nCount[1]*nCount[2]));
		if (nCount[1] > 0)
		{
			Q_ASSERT(Positive == getTriangleSetType());
			m_VertexTypeInDual = VertexTypeInDual::Mirror_Positive;
		}
		else if (nCount[2] > 0)
		{
			Q_ASSERT(Negative == getTriangleSetType());
			m_VertexTypeInDual = VertexTypeInDual::Mirror_Negative;
		}
		else
		{
			Q_ASSERT(Regular == getTriangleSetType());
			m_VertexTypeInDual = VertexTypeInDual::Regular;
		}
	}

	VertexPtr Vertex::getMirrorVertex()
	{
		Q_ASSERT(isMirrorVertex());
		return getVertex( *setMirrorVertexId.begin() );
	}

	void Vertex::searchMirrorVertex_CrackTipVertex()
	{
		printf("call Vertex::searchMirrorVertex_CrackTipVertex();.\n");
		const vrInt nVtxSize = getVertexSize();
		std::vector< searchMirrorVertexNode > vecSortObj;
		searchMirrorVertexNode curNode;
		vecSortObj.resize(nVtxSize);
		for (int v=0;v<nVtxSize;++v)
		{
			curNode.pos = getVertex(v)->getPos();
			curNode.vtxId = getVertex(v)->getId(); 
			vecSortObj[v] = curNode;
		}
		std::sort(vecSortObj.begin(),vecSortObj.end(),searchMirrorVertexNode_LessCompare());

		Q_ASSERT(nVtxSize >3);

		int v=0;
		
		

		if (/*numbers::IsEqual(vecSortObj[v].pos[0], vecSortObj[v+1].pos[0]) &&
			numbers::IsEqual(vecSortObj[v].pos[1], vecSortObj[v+1].pos[1]) &&
			numbers::IsEqual(vecSortObj[v].pos[2], vecSortObj[v+1].pos[2])*/
			numbers::isEqual(vecSortObj[v].pos, vecSortObj[v+1].pos))
		{
			//std::cout << (vecSortObj[v].pos).transpose() << " = " << (vecSortObj[v+1].pos).transpose() << std::endl;vrPause;
			getVertex(MyNotice (vecSortObj[v].vtxId) )->setMirrorVertex(vecSortObj[v+1].vtxId);
		}
		v++;
		for (;v<(nVtxSize-1);++v)
		{
			if (numbers::isEqual(vecSortObj[v-1].pos,vecSortObj[v].pos))
			{
				//std::cout << (vecSortObj[v-1].pos).transpose() << " = " << (vecSortObj[v].pos).transpose() << std::endl;vrPause;
				getVertex(MyNotice (vecSortObj[v].vtxId))->setMirrorVertex(vecSortObj[v-1].vtxId);
			}

			if (numbers::isEqual(vecSortObj[v].pos,vecSortObj[v+1].pos))
			{
				//std::cout << (vecSortObj[v].pos).transpose() << " = " << (vecSortObj[v+1].pos).transpose() << std::endl;vrPause;
				getVertex(MyNotice (vecSortObj[v].vtxId))->setMirrorVertex(vecSortObj[v+1].vtxId);
			}
		}
		//v = nVtxSize-1
		if (numbers::isEqual(vecSortObj[v-1].pos,vecSortObj[v].pos))
		{
			//std::cout << (vecSortObj[v-1].pos).transpose() << " = " << (vecSortObj[v].pos).transpose() << std::endl;vrPause;
			getVertex(MyNotice (vecSortObj[v].vtxId))->setMirrorVertex(vecSortObj[v-1].vtxId);
		}

		//search Crack Tip Vertex
		for (iterAllOf(ci,s_vertexCache))
		{
			VertexPtr curVtxPtr = *ci;
			if (!(curVtxPtr->isMirrorVertex()))
			{
				curVtxPtr->parserCrackTip();
			}
		}
	}

	void Vertex::TestMirrorInfo()
	{
		printf("call Vertex::TestMirrorInfo(); \n");
		//test
		std::map< int, vrId > vertexTypeCounter;
		for (iterAllOf(ci,s_vertexCache))
		{
			VertexPtr curVtxPtr = (*ci);
			if (curVtxPtr->isMirrorVertex())
			{
				VertexPtr curMirrorVtxPtr = curVtxPtr->getMirrorVertex();
				/*std::cout << (curVtxPtr->getPos()).transpose() << std::endl;
				std::cout << (curMirrorVtxPtr->getPos()).transpose() << std::endl;*/
				Q_ASSERT(curMirrorVtxPtr->isMirrorVertex());
				Q_ASSERT(*(curVtxPtr->setMirrorVertexId.begin()) == curMirrorVtxPtr->getId() );
				Q_ASSERT(*(curMirrorVtxPtr->setMirrorVertexId.begin()) == curVtxPtr->getId() );

				const MyVec3 mustbezero = curVtxPtr->getVertexNormal() + curMirrorVtxPtr->getVertexNormal();

				Q_ASSERT(numbers::isZero(mustbezero[0]) && numbers::isZero(mustbezero[1]) && numbers::isZero(mustbezero[2]));
			}

			vertexTypeCounter[curVtxPtr->m_VertexTypeInDual].increase();
		}

		for (iterAllOf(ci,vertexTypeCounter))
		{
			printf("[%d] [%d]\n", (*ci).first, (*ci).second.value() );
		}
		
	}
#endif//USE_DUAL

	/*VertexPtr Vertex::searchVertex(const MyPoint& point)
	{
		std::vector< VertexPtr >::reverse_iterator itr = std::find_if(s_vertexCache.rbegin(), s_vertexCache.rend(), VertexCompare(point[0], point[1], point[2]));
		if (s_vertexCache.rend() == itr)
		{
			printf("vertex not find!");
			system("pause");
			return s_invalid_vertex;
			//no find

		}
		else
		{
			return (*itr);
		}
	}*/
	

	VertexPtr Vertex::getVertex(int idx)
	{
		return s_vertexCache[idx];
	}

	bool Vertex::isSharedElementWithId(const MyInt nElemId)
	{
		for (iterAllOf(itr,m_vec_ShareTriangleElem))
		{
			const MyInt sharedElemId = (*itr)->getID();
			if (nElemId == sharedElemId)
			{
				return true;
			}
		}
		return false;
	}

	void Vertex::addNearRegion(TriangleElemPtr tri)
	{
		for (iterAllOf(ci,m_vec_ShareTriangleElem_Sorted_NearRegion))
		{
			if ( ((*ci)->getID()) == tri->getID() )
			{
				return ;
			}
		}
		m_vec_ShareTriangleElem_Sorted_NearRegion.push_back(tri);
	}
}//namespace VR