#ifndef _VCGHELPER_INCLUDED
#define _VCGHELPER_INCLUDED
#if USE_SST_DEBUG

#include <openvdb/tools/VolumeToMesh.h>

#include <vcg/complex/complex.h>
#include <vcg/complex/algorithms/smooth.h>
#include <vcg/complex/algorithms/local_optimization.h>
#include <vcg/complex/algorithms/local_optimization/tri_edge_collapse_quadric.h>
#include <vcg/complex/algorithms/update/normal.h>
#include <vcg/wrap/io_trimesh/export_obj.h>

namespace myVCG{
	// from VCG example tridecimator.cpp
	class MyVertex;
	class MyEdge;
	class MyFace;
	struct MyUsedTypes: public vcg::UsedTypes<
		vcg::Use<MyVertex>::AsVertexType,
		vcg::Use<MyEdge>::AsEdgeType,
		vcg::Use<MyFace>::AsFaceType>{};
	class MyVertex  : public vcg::Vertex< MyUsedTypes,
		vcg::vertex::VFAdj,
		vcg::vertex::Coord3f,
		vcg::vertex::Normal3f,
		vcg::vertex::Mark,
		vcg::vertex::BitFlags  >{
	public:
		vcg::math::Quadric<double> &Qd() {return q;}
	private:
		vcg::math::Quadric<double> q;
	};
	class MyEdge : public vcg::Edge< MyUsedTypes> {};
	typedef BasicVertexPair<MyVertex> VertexPair;
	class MyFace    : public vcg::Face< MyUsedTypes,
		vcg::face::VFAdj,
		vcg::face::VertexRef,
		vcg::face::BitFlags > {};
	class VcgMesh    : public vcg::tri::TriMesh<std::vector<MyVertex>, std::vector<MyFace> > {};
	class MyTriEdgeCollapse: public vcg::tri::TriEdgeCollapseQuadric< VcgMesh, VertexPair, MyTriEdgeCollapse, QInfoStandard<MyVertex>  > {
	public:
		typedef  vcg::tri::TriEdgeCollapseQuadric< VcgMesh,  VertexPair, MyTriEdgeCollapse, QInfoStandard<MyVertex>  > TECQ;
		typedef  VcgMesh::VertexType::EdgeType EdgeType;
		inline MyTriEdgeCollapse(  const VertexPair &p, int i, vcg::BaseParameterClass *pp) :TECQ(p,i,pp){}
	};

	template<class vdbVolumeToMesh>
	void copyVDBtoVCG(vdbVolumeToMesh& mesher, VcgMesh& mesh){
		// add vertices
		unsigned int firstVert = mesh.vert.size();
		VcgMesh::VertexIterator vi = vcg::tri::Allocator<VcgMesh>::AddVertices(mesh,mesher.pointListSize());
		for(unsigned int i=0; i<mesher.pointListSize(); ++i,++vi){
			vi->P()[0] = mesher.pointList()[i][0];
			vi->P()[1] = mesher.pointList()[i][1];
			vi->P()[2] = mesher.pointList()[i][2];
		}
		// count how many tri-faces we have
		unsigned int nTrisOri = 0;
		for(int i=0; i<mesher.polygonPoolListSize(); ++i){
			vdb::tools::PolygonPool& pool = mesher.polygonPoolList()[i];
			nTrisOri += pool.numTriangles();
			nTrisOri += 2*pool.numQuads();
		}
		// add triangles
		VcgMesh::FaceIterator fi = vcg::tri::Allocator<VcgMesh>::AddFaces(mesh,nTrisOri);
		for(int i=0; i<mesher.polygonPoolListSize(); ++i){
			vdb::tools::PolygonPool &pool = mesher.polygonPoolList()[i];
			for(int j=0; j<pool.numTriangles(); ++j){
				//swap ordering of nodes to get outside-normals
				fi->V(0)=&(mesh.vert[firstVert+ pool.triangle(j)[0]]);
				fi->V(1)=&(mesh.vert[firstVert+ pool.triangle(j)[2]]);
				fi->V(2)=&(mesh.vert[firstVert+ pool.triangle(j)[1]]);
				++fi;
			}
			for(int j=0; j<pool.numQuads(); ++j){
				//swap ordering of nodes to get outside-normals
				fi->V(0)=&(mesh.vert[firstVert+ pool.quad(j)[0]]);
				fi->V(1)=&(mesh.vert[firstVert+ pool.quad(j)[2]]);
				fi->V(2)=&(mesh.vert[firstVert+ pool.quad(j)[1]]);
				++fi;
				fi->V(0)=&(mesh.vert[firstVert+ pool.quad(j)[0]]);
				fi->V(1)=&(mesh.vert[firstVert+ pool.quad(j)[3]]);
				fi->V(2)=&(mesh.vert[firstVert+ pool.quad(j)[2]]);
				++fi;
			}
		}
	}
}
#endif//USE_SST_DEBUG
#endif //_VCGHELPER_INCLUDED
