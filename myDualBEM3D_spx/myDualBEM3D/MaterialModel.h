#ifndef _MaterialModel_h_
#define _MaterialModel_h_
#if 0

#include "bemDefines.h"
#include <Eigen/Dense>
#include <string>

namespace VR
{
	class MaterialModel;
	MaterialModel* createMaterialModel(vrString spec, vrFloat youngsMod, vrFloat poissonsRatio, vrFloat density, vrFloat strength, vrFloat toughness, vrFloat compressiveFactor);

	//abstract class, use MaterialModelImpl.h for useable subclasses
	class MaterialModel {
	public:
		MaterialModel(vrFloat E=1.0, vrFloat nu=0.3, vrFloat rho=1e3)
			: E(E), nu(nu), rho(rho) {}
		MaterialModel(const MaterialModel& ori)
			: E(ori.E), nu(ori.nu), rho(ori.rho) {}
		virtual MaterialModel* clone() const =0;
		virtual ~MaterialModel(){}

		virtual vrFloat tensileStrength(const MyVec3& x) const =0; // returns the critical tensile stress at which a fracture will start 
		virtual vrFloat compressiveFactor(const MyVec3& x) const =0; // returns a factor specifying how much stronger & tougher the material is under compression as opposed to tension

		virtual vrFloat fractureToughness(const MyVec3& x) const =0; // returns the critical stress intensity at which cracks will propagate
		virtual void   fractureToughnessGradient(MyVec3& dKc, const MyVec3& x) const =0;

		virtual vrFloat getE() const {return E;}
		virtual vrFloat getNu() const {return nu;}
		virtual vrFloat getDensity() const {return rho;}
		virtual void setE(vrFloat value){E=value;}
		virtual void setNu(vrFloat value){nu=value;}
		virtual void setDensity(vrFloat value){rho=value;}
	protected:
		vrFloat E,nu,rho; // store Young's modulus, Poisson's ratio and density
	};

	class HomogeneousMaterialModel : public MaterialModel {
	public:
		HomogeneousMaterialModel(
			vrFloat E=1.0, vrFloat nu=0.3, vrFloat rho=1e3,
			vrFloat strength=1.0, vrFloat toughness=1.0, vrFloat compressiveFactor=3.0
			) : MaterialModel(E,nu,rho), Sc(strength), Kc(toughness), cf(compressiveFactor) {}
		HomogeneousMaterialModel(const HomogeneousMaterialModel& ori)
			: MaterialModel(ori.E,ori.nu,ori.rho), Sc(ori.Sc), Kc(ori.Kc), cf(ori.cf) {}
		virtual HomogeneousMaterialModel* clone() const {return(new HomogeneousMaterialModel(*this)); }
		virtual ~HomogeneousMaterialModel(){}

		inline virtual vrFloat tensileStrength(const MyVec3& x) const {return Sc;}
		inline virtual vrFloat compressiveFactor(const MyVec3& x) const {return cf;}

		inline virtual vrFloat fractureToughness(const MyVec3& x) const { return Kc;}
		inline virtual void   fractureToughnessGradient(MyVec3& dKc, const MyVec3& x) const { dKc.setZero();}
	protected:
		vrFloat Kc,Sc,cf;
	};

	// use an OpenVDB sparse grid data file as the material's toughness and strength maps
	class VDBMaterialModel : public HomogeneousMaterialModel {
	public:
		VDBMaterialModel(
			double E=1.0, double nu=0.3, double rho=1e3,
			double strength=1.0, double toughness=1.0, double compressiveFactor=3.0,
			std::string file="material.vdb", std::string toughnessGrid="toughness", std::string strengthGrid="strength",
			bool udf=false, double scale=1.0, double voxelSize=1.0
			)   : HomogeneousMaterialModel(E,nu,rho,strength,toughness,compressiveFactor),
			data(NULL), vdbFile(file), kName(toughnessGrid), sName(strengthGrid),
			useUDF(udf), scale(scale), voxelSize(voxelSize) {}
		VDBMaterialModel(const VDBMaterialModel& ori)
			: HomogeneousMaterialModel(ori.E,ori.nu,ori.rho,ori.Sc,ori.Kc,ori.cf),
			data(NULL), vdbFile(ori.vdbFile), kName(ori.kName), sName(ori.sName),
			useUDF(ori.useUDF), scale(ori.scale), voxelSize(ori.voxelSize) {}
		virtual VDBMaterialModel* clone() const {return(new VDBMaterialModel(*this)); }
		virtual ~VDBMaterialModel(){ if(data!=NULL) delete data; }

		virtual double tensileStrength(const Eigen::Vector3d& x) const;
		virtual double fractureToughness(const Eigen::Vector3d& x) const;
		virtual void   fractureToughnessGradient(Eigen::Vector3d& dKc, const Eigen::Vector3d& x) const;
	protected:
		std::string vdbFile, kName, sName;
		double scale, voxelSize;
		bool useUDF;
	private:
		void init() const;
		class vdbData; //fwd. decl
		mutable vdbData* data;
		mutable bool haveToughnessMap, haveStrengthMap;
		template <class Vec3> Vec3 toPeriodicIndexSpace(const Eigen::Vector3d& x) const;
	};
}//namespace VR
#else

#include "bemDefines.h"
namespace VR
{
	class MaterialModel
	{
	
	};

	MaterialModel* createMaterialModel(vrString , vrFloat , vrFloat , vrFloat , vrFloat , vrFloat , vrFloat );
}
#endif

#endif//_MaterialModel_h_