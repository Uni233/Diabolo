#pragma once

#include "VR_Global_Define.h"
#include "constant_numbers.h"
namespace YC
{
	class MaterialMatrix
	{
	public:
		MaterialMatrix(void);
		~MaterialMatrix(void);

		struct MatrixInfo
		{
			MatrixInfo(MyFloat y, MyFloat p, const MyMatrix& refM):YoungModule(y),PossionRatio(p),tMatrix(refM){}
			MyFloat YoungModule;
			MyFloat PossionRatio;
			MyMatrix tMatrix;
		};

		class MaterialMatrixCompare
		{
		public:
			MaterialMatrixCompare(MyFloat y,MyFloat p):YoungModule(y),PossionRatio(p){}

			bool operator()(MatrixInfo& p)
			{
				return  (numbers::IsEqual(YoungModule,p.YoungModule)) && (numbers::IsEqual(PossionRatio,p.PossionRatio));
			}
		private:
			MyFloat YoungModule,PossionRatio;
		};


	public:
		static MyMatrix getMaterialMatrix(const MyFloat YoungModule, const MyFloat PossionRatio);
	private:
		static void MaterialMatrix::makeSymmetry(MyMatrix& objMatrix);
		static MyMatrix MaterialMatrix::makeMaterialMatrix(const MyFloat y, const MyFloat p);
		static std::vector<MatrixInfo> s_vecMatrixMatrix;
	};
}

