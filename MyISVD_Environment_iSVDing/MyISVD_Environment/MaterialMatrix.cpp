#include "MaterialMatrix.h"

namespace YC
{

	MaterialMatrix::MaterialMatrix(void)
	{
	}


	MaterialMatrix::~MaterialMatrix(void)
	{
	}

	std::vector<MaterialMatrix::MatrixInfo> MaterialMatrix::s_vecMatrixMatrix;

	void MaterialMatrix::makeSymmetry(MyMatrix& objMatrix)
	{
		for (unsigned r=0;r<objMatrix.rows();++r)
		{
			for (unsigned c=0;c<r;++c)
			{
				objMatrix.coeffRef(c,r) = objMatrix.coeff(r,c);
			}
		}
	}

	MyMatrix MaterialMatrix::makeMaterialMatrix(const MyFloat y, const MyFloat p)
	{		

		MyMatrix MaterialMatrix_6_6;
		MaterialMatrix_6_6.resize(MaterialMatrixSize,MaterialMatrixSize);
		MaterialMatrix_6_6.setZero();

		MyFloat E = y/*Material::YoungModulus*/;
		MyFloat mu = p;

		MyFloat G = E/(2*(1+mu));
		MyFloat lai = mu*E/((1+mu)*(1-2*mu));


		MaterialMatrix_6_6.coeffRef(0,0) = lai + 2*G;

		MaterialMatrix_6_6.coeffRef(1,0) = lai;
		MaterialMatrix_6_6.coeffRef(1,1) = lai + 2*G;

		MaterialMatrix_6_6.coeffRef(2,0) = lai;
		MaterialMatrix_6_6.coeffRef(2,1) = lai;
		MaterialMatrix_6_6.coeffRef(2,2) = lai + 2*G;

		MaterialMatrix_6_6.coeffRef(3,0) = 0.0;
		MaterialMatrix_6_6.coeffRef(3,1) = 0.0;
		MaterialMatrix_6_6.coeffRef(3,2) = 0,0;
		MaterialMatrix_6_6.coeffRef(3,3) = G;

		MaterialMatrix_6_6.coeffRef(4,0) = 0.0;
		MaterialMatrix_6_6.coeffRef(4,1) = 0.0;
		MaterialMatrix_6_6.coeffRef(4,2) = 0.0;
		MaterialMatrix_6_6.coeffRef(4,3) = 0.0;
		MaterialMatrix_6_6.coeffRef(4,4) = G;

		MaterialMatrix_6_6.coeffRef(5,0) = 0.0;
		MaterialMatrix_6_6.coeffRef(5,1) = 0.0;
		MaterialMatrix_6_6.coeffRef(5,2) = 0.0;
		MaterialMatrix_6_6.coeffRef(5,3) = 0.0;
		MaterialMatrix_6_6.coeffRef(5,4) = 0.0;
		MaterialMatrix_6_6.coeffRef(5,5) = G;

		makeSymmetry(MaterialMatrix_6_6);

		return MaterialMatrix_6_6;
	}

	MyMatrix MaterialMatrix::getMaterialMatrix(const MyFloat YoungModule, const MyFloat PossionRatio)
	{
		std::vector< MatrixInfo >::iterator itr = std::find_if(s_vecMatrixMatrix.begin(),s_vecMatrixMatrix.end(),MaterialMatrixCompare(YoungModule,PossionRatio));
		if (itr == s_vecMatrixMatrix.end())
		{
			//no find
			const MyMatrix& tmp = makeMaterialMatrix(YoungModule,PossionRatio);
			s_vecMatrixMatrix.push_back(MatrixInfo(YoungModule,PossionRatio,tmp));
			return s_vecMatrixMatrix[s_vecMatrixMatrix.size()-1].tMatrix;
		}
		else
		{
			//fing
			return (*itr).tMatrix;
		}
	}
}
