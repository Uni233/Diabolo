#include "MaterialModel.h"
#if 0

#include <cmath>
#include <cstdio>
using namespace std;

#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
namespace vdb = openvdb::v2_2_0;

namespace VR
{
	MaterialModel* createMaterialModel(vrString spec, vrFloat youngsMod, vrFloat poissonsRatio, vrFloat density, vrFloat strength, vrFloat toughness, vrFloat compress)
	{
		if(spec.compare(0,3,"vdb")==0){ // format: "vdb(filename/toughnessGridName/strengthGridName,scaleFactor,voxelSize,useUDF)"
			std::string file, toughname="toughness", strengthname="strength";
			file=spec.substr(4, spec.find_first_of(',')-4);                     //printf("\n1)     file=\"%s\"",file.c_str());
			if(file.find_first_of('/')!=file.npos){
				toughname=file.substr(file.find_first_of('/')+1);               //printf("\n2)    tough=\"%s\"", toughname.c_str());
				file=file.substr(0,file.find_first_of('/'));                    //printf("\n2)     file=\"%s\"",file.c_str());
				strengthname=toughname;
			}
			if(toughname.find_first_of('/')!=toughname.npos){
				strengthname=toughname.substr(toughname.find_first_of('/')+1);  //printf("\n3) strength=\"%s\"", strengthname.c_str());
				toughname=toughname.substr(0,toughname.find_first_of('/'));     //printf("\n3)    tough=\"%s\"",toughname.c_str());
			}                                                                   //printf("\n4) remainder = \"%s\"", spec.substr(spec.find_first_of(',')+1).c_str());
			double scale, voxelSize; int readUDF; char check=0;
			sscanf( spec.substr(spec.find_first_of(',')+1).c_str(), "%lf,%lf,%d%c", &scale, &voxelSize, &readUDF, &check);
			if(check!=')'){
				printf("\nillegal definition of VDB material: %s, use "
					"\"vdb(filename/toughnessGridName/strengthGridName,scaleFactor,voxelSize,useUDF)\"", spec.c_str());
			}else{
				printf("\n%% VDB material model: %s / %s / %s\n%%   scale %.3lg, voxel size %.3lg, udf %d\n",
					file.c_str(), toughname.c_str(), strengthname.c_str(), scale, voxelSize, (readUDF!=0));
				return new VDBMaterialModel(
					youngsMod, poissonsRatio,density, strength,toughness, compress,
					file,toughname,strengthname,readUDF!=0, scale ,voxelSize);
			}
		}

		return new HomogeneousMaterialModel( youngsMod, poissonsRatio,density, strength,toughness, compress);MYNOTICE;
	}

	// VDBMaterialModel ==================================================================================
	class VDBMaterialModel::vdbData{
	public:
		vdbData() : ux(1.0,0.0,0.0), uy(0.0,1.0,0.0), uz(0.0,0.0,1.0) {};
		vdb::FloatGrid::Ptr toughnessGrid, strengthGrid;
		vdb::math::Transform::Ptr xform;
		vdb::Vec3d ux,uy,uz;
		vdb::math::CoordBBox bbox;
	};
	void VDBMaterialModel::init() const{
		haveToughnessMap=false; haveStrengthMap=false;
		if(data==NULL) data = new vdbData();
		vdb::initialize();
		vdb::io::File inFile(vdbFile);
		inFile.open();
		vdb::GridBase::Ptr baseGrid;
		if( inFile.hasGrid(kName) ){
			baseGrid = inFile.readGrid(kName);
			data->toughnessGrid=vdb::gridPtrCast<vdb::FloatGrid>(baseGrid);
			data->bbox = data->toughnessGrid->evalActiveVoxelBoundingBox();
			haveToughnessMap=true;
		}
		if( kName.compare(sName)) data->strengthGrid = data->toughnessGrid; // share the grid ptr when using the same data for both strength and toughness
		else if( inFile.hasGrid(sName) ){
			baseGrid = inFile.readGrid(sName);
			data->strengthGrid=vdb::gridPtrCast<vdb::FloatGrid>(baseGrid);
			data->bbox.expand( data->strengthGrid->evalActiveVoxelBoundingBox() );
			haveStrengthMap=true;
		}
		inFile.close();
		data->xform = vdb::math::Transform::createLinearTransform(voxelSize);
		printf("\n%% VDB material model initialized%s%s\n",
			haveToughnessMap?", toughness map loaded":"",
			haveStrengthMap ? ", strength map loaded":""
			);
	}
	double VDBMaterialModel::tensileStrength(const Eigen::Vector3d& x) const{
		if(data==NULL) init();
		if(!haveStrengthMap) return Sc;
		vdb::tools::GridSampler<vdb::FloatGrid::ConstAccessor, vdb::tools::BoxSampler> sampler(
			data->strengthGrid->getConstAccessor(), *(data->xform)
			);
		vdb::Vec3d xi = toPeriodicIndexSpace<vdb::Vec3d>(x);
		float gridValue = sampler.isSample(xi) / data->strengthGrid->background();
		if(useUDF) gridValue=std::abs(gridValue);
		return Sc + scale*gridValue;

	}
	double VDBMaterialModel::fractureToughness(const Eigen::Vector3d& x) const{
		if(data==NULL) init();
		if(!haveToughnessMap) return Kc;
		vdb::tools::GridSampler<vdb::FloatGrid::ConstAccessor, vdb::tools::BoxSampler> sampler(
			data->toughnessGrid->getConstAccessor(), *(data->xform)
			);
		vdb::Vec3d xi = toPeriodicIndexSpace<vdb::Vec3d>(x);
		float gridValue = sampler.isSample(xi) / data->toughnessGrid->background();
		if(useUDF) gridValue=std::abs(gridValue);
		//printf("\n eval Kc at (%.3lf,%.3lf,%.3lf) df %.3lf Kc %.3le",x[0],x[1],x[2],gridValue,Kc + scale*gridValue);
		return Kc + scale*gridValue;
	}
	void VDBMaterialModel::fractureToughnessGradient(Eigen::Vector3d& dKc, const Eigen::Vector3d& x) const{
		if(data==NULL) init();
		if(!haveToughnessMap){
			dKc.setZero();
			return;
		}
		vdb::tools::GridSampler<vdb::FloatGrid::ConstAccessor, vdb::tools::BoxSampler> sampler(
			data->toughnessGrid->getConstAccessor(), *(data->xform)
			);
		float x0,x1,y0,y1,z0,z1, bg=data->toughnessGrid->background();
		vdb::Vec3d xi = toPeriodicIndexSpace<vdb::Vec3d>(x);
		x0=sampler.isSample(xi-0.25*data->ux) / bg;
		x1=sampler.isSample(xi+0.25*data->ux) / bg;
		y0=sampler.isSample(xi-0.25*data->uy) / bg;
		y1=sampler.isSample(xi+0.25*data->uy) / bg;
		z0=sampler.isSample(xi-0.25*data->uz) / bg;
		z1=sampler.isSample(xi+0.25*data->uz) / bg;
		if(useUDF){
			x0=std::abs(x0); x1=std::abs(x1);
			y0=std::abs(y0); y1=std::abs(y1);
			z0=std::abs(z0); z1=std::abs(z1);
		}
		dKc[0] = scale*(x1-x0)*2.0/voxelSize;
		dKc[1] = scale*(y1-y0)*2.0/voxelSize;
		dKc[2] = scale*(z1-z0)*2.0/voxelSize;
		//if(dKc.norm() > FLT_EPSILON){
		//	printf("\n x(%.3lf, %.3lf) grad_x %.3le",x0,x1,dKc[0]);
		//	printf("\n y(%.3lf, %.3lf) grad_y %.3le",y0,y1,dKc[1]);
		//	printf("\n z(%.3lf, %.3lf) grad_z %.3le",z0,z1,dKc[2]);
		//}
	}
	template <class Vec3>
	Vec3 VDBMaterialModel::toPeriodicIndexSpace(const Eigen::Vector3d& x) const{
		Vec3 xi = data->xform->worldToIndex(Vec3(x[0],x[1],x[2]));
		//printf("\n pb_map w(%.3lf,%.3lf,%.3lf) -> i(%.1lf,%.1lf,%.1lf)", x[0],x[1],x[2], xi[0],xi[1],xi[2]);
		vdb::math::BBox<Vec3> box ( data->bbox.min_Coord().asVec3d(), data->bbox.max_Coord().asVec3d() );
		if( box.isInside(xi) ) return xi;
		xi-=box.min(); // shift min to (0,0,0)
		//printf("\n--> s(%.1lf,%.1lf,%.1lf)", xi[0],xi[1],xi[2]);
		xi[0] = std::fmod( xi[0], box.extents()[0] ); if(xi[0]<0.0) xi[0]+=box.extents()[0];
		xi[1] = std::fmod( xi[1], box.extents()[1] ); if(xi[1]<0.0) xi[1]+=box.extents()[1];
		xi[2] = std::fmod( xi[2], box.extents()[2] ); if(xi[2]<0.0) xi[2]+=box.extents()[2];
		//printf("\n--> M(%.1lf,%.1lf,%.1lf), extents (%.1lf,%.1lf,%.1lf)", xi[0],xi[1],xi[2], box.extents()[0],box.extents()[1],box.extents()[2]);
		xi+=box.min(); // undo shift
		//printf("\n--> b(%.1lf,%.1lf,%.1lf) in box (%.0lf,%.0lf,%.0lf)-(%.0lf,%.0lf,%.0lf)", xi[0],xi[1],xi[2], box.min()[0],box.min()[1],box.min()[2], box.max()[0],box.max()[1],box.max()[2]);
		return xi;
	}
	// =====================================================================================================
}
#else
#include "bemDefines.h"
namespace VR
{
	MaterialModel* createMaterialModel(vrString , vrFloat , vrFloat , vrFloat , vrFloat , vrFloat , vrFloat )
	{
		return NULL;
	}
}
#endif