#ifndef _PostProcessor_h_
#define _PostProcessor_h_

#include "bemDefines.h"
#if USE_SST_DEBUG

namespace VR
{
	class FractureModel;

	

	class PostProcessor{
    public:
        PostProcessor(
            node_map& nodes_, elem_map& elems_, id_map& regions_, id_set& cracks_,
            elem_map& crackTips_, elem_map& parents_, state_map& crackTipStates_
        ): nodes(nodes_), elems(elems_), regions(regions_), cracks(cracks_),
           crackTips(crackTips_), parents(parents_),
           crackTipStates(crackTipStates_)
		{}
		
		PostProcessor(
			PostProcessor& ori
		): nodes(ori.nodes), elems(ori.elems), regions(ori.regions),
           cracks(ori.cracks), crackTips(ori.crackTips), parents(ori.parents),
           crackTipStates(ori.crackTipStates)
		{}

        virtual ~PostProcessor(){}
        
        /* Compute stress intensity factors in local coordinate system for all
         * crack-tip nodes, using displacement correlation technique
         */
		int computeNodeSIFs(
            const vector_type& displacements, double E, double nu,
			vect3d_map& sifs, vect3d_map& faceNormals, vect3d_map& tangents
        );
    
		

		/* Compute stresses on triangles in the mesh based on nodal displacements
		 * and add to VTK data
         * output-param retValues       (terminology of "node_map" is misused here)
         * is a map<int, vector<double> > that maps element-IDs to 9 double values:
         * the first value is the max. principal stress value,
		 * the next 3 values are the plane-normal across which the principal stress is given;
		 * the following 4 values are magnitude & normal vector for the min. principal stress
		 * the last value is a flag: 0 for regular surface elements, >0 for fracture elements
		 * for fractures, we specify whether max. and min. principal stress reach their largest
		 * magnitude for the positive or negative side of the fracture (sign of applied COD):
		 * 1: both positive; 2: max. negative, min. positive; 3: max. pos., min. neg.; 4: both neg.
		 */
		int computeSurfaceStresses(
            node_map& retValues, const vector_type& displacements,
			const vector_type& crackBaseDisplacements,
            double E=1.0, double nu=0.0, bool addToVTK=false
        );

        

		/* In the triangle given by the node-IDs in el, 
		 * find the node-ID which is neither nd_a nor nd_b
		 */
        inline unsigned int findInteriorNode(
            unsigned int nd_a, unsigned int nd_b,
            const std::vector<unsigned int>& el
        ){
            if(el[0]!=nd_a && el[0]!=nd_b)
                return el[0];
            else if(el[1]!=nd_a && el[1]!=nd_b)
                return el[1];
            else //if(el[2]!=nd_a && el[2]!=nd_b)
                return el[2];
        }

		/* Copy the coordinates of the specified node to coords
		 */
        inline void copyNode(unsigned int node, Eigen::Vector3d& coords){
            coords[0]=nodes[node][0];
            coords[1]=nodes[node][1];
            coords[2]=nodes[node][2];
        }

    protected:
        node_map& nodes;
        elem_map& elems;
        id_map& regions;
        id_set& cracks;
        elem_map& crackTips;
        elem_map& parents;
        state_map& crackTipStates;
        
        std::vector<std::string>    vtkDataNames;
        std::vector<int>            vtkDataDimension;
        std::vector<bool>           vtkDataIsCellData;
        std::vector<vector_type*>	vtkData;
		vector_type s_xx, s_yy, s_zz, s_xy, s_xz, s_yz; // cartesian stresses per node/element

		
		
        
        /* Build a local coordinate frame used to compute SIFs
         * Input are 3 points a,b,c
         * Output are 3 unit vector_types n1,n2,n3, where
         * n1 is the face normal of the triangle (a,b,c),
         * n2 is the edge normal of the edge (a,b) in-plane, and
         * n3 is the tangent unit vector along the edge (a,b)
         */
        inline void getLocalCoordFrame(
            const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c,
            Eigen::Vector3d& n1, Eigen::Vector3d& n2, Eigen::Vector3d& n3
        ){
            // n3 is the edge unit vector_type --> mode III direction (sliding)
            n3 = (b-a); n3.normalize();
            // n1 is the face normal --> mode I direction (opening)
            n1 = n3.cross(c-a); n1.normalize();
            // n2 is the in-plane edge normal --> mode II direction (shear)
            n2 =-n1.cross(n3); n2.normalize(); // normalization should be obsolete
            // now we have a local coordinate system for the edge a-b
        }

		// computes SVD of the deformation gradient F = U*S*Vt
		// and principal stresses P = 2 mu (S-I) + lambda tr(S-I);
		int computeElementPrincipalStresses(
			Eigen::Matrix3d& U, Eigen::Matrix3d& S, Eigen::Matrix3d& Vt, Eigen::Matrix3d& P,
			const Eigen::Vector3d&  a, const Eigen::Vector3d&  b, const Eigen::Vector3d&  c,
			const Eigen::Vector3d& ua, const Eigen::Vector3d& ub, const Eigen::Vector3d& uc,
			double mu, double lambda
		);
    }; 
}
#endif//USE_SST_DEBUG
#endif//_PostProcessor_h_