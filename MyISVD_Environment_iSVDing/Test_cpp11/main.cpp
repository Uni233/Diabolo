#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include "MyISVD.h"
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <boost/timer.hpp>
#include <fstream>
#include "MyIterMacro.h"
#include <boost/tuple/tuple.hpp>
#include <boost/lexical_cast.hpp>
#include <iomanip>

using namespace boost;
#include <time.h>

typedef Eigen::MatrixXd MyMatrix;
typedef Eigen::VectorXd MyVector;
using namespace std;

/*
A = rnd.rand(100,50)
np.savetxt("d:\\filename.txt",A)
a2 = rnd.rand(100,1)
np.savetxt("d:\\filename_vec.txt",a2)
*/

void loadMatrixFromFile_vec(const char* lpszMatFileName, MyVector& vec)
{
	std::vector< boost::tuple<int, int, double> > vecData;
	//ifstream infile("d:/filename.txt");
	int nRow = 0;
	string line;
	ifstream myfile(lpszMatFileName);
	if (myfile.is_open()) {
		while (!myfile.eof()) {
			getline(myfile, line);
			if (line.size() == 0) {
				break;
			}
			int nCols = 0;
			istringstream iss(line);
			vector<string> tokens;
			copy(istream_iterator<string>(iss), istream_iterator<string>(),
				back_inserter<vector<string> >(tokens));

			for (iterAllOf(itr, tokens))
			{
				double data = boost::lexical_cast<double>((*itr).c_str());
				boost::tuple<int, int, double> tmp = boost::make_tuple(nRow, nCols, data);
				std::cout << tmp.get<0>() << " " << tmp.get<1>() << " " << std::setprecision(20) << tmp.get<2>() << std::endl;
				vecData.push_back(tmp);
				nCols++;
			}
			
			nRow++;
		}
		myfile.close();
		MyPause;
	}
	else {
		cout << "File not opened" << endl;
	}
	std::cout << "vecData.size() = " << vecData.size() << std::endl;
}

void loadMatrixFromFile(const char* lpszMatFileName, MyMatrix& mat)
{
	std::vector< boost::tuple<int, int, double> > vecData;
	//ifstream infile("d:/filename.txt");
	int nRow = 0;
	string line;
	ifstream myfile(lpszMatFileName);
	if (myfile.is_open()) {
		while (!myfile.eof()) {
			getline(myfile, line);
			if (line.size() == 0) {
				break;
			}
			int nCols = 0;
			istringstream iss(line);
			vector<string> tokens;
			copy(istream_iterator<string>(iss), istream_iterator<string>(),
				back_inserter<vector<string> >(tokens));

			for (iterAllOf(itr, tokens))
			{
				double data = boost::lexical_cast<double>((*itr).c_str());
				boost::tuple<int, int, double> tmp = boost::make_tuple(nRow, nCols, data);
				std::cout << tmp.get<0>() << " " << tmp.get<1>() << " " << std::setprecision(20) << tmp.get<2>() << std::endl;
				vecData.push_back(tmp);
				nCols++;
			}
			MyPause;
			nRow++;
		}
		myfile.close();
	}
	else {
		cout << "File not opened" << endl;
	}
	std::cout << "vecData.size() = " << vecData.size() << std::endl;
}

int main()
{
	/*MyMatrix datam;
	loadMatrixFromFile("d:/filename.txt", datam);*/
	MyVector datav;
	loadMatrixFromFile_vec("d:/filename_vec.txt",datav);
	//YC::test_SVD_update_reorth(100, 1, 10, 0.05, 5);
	//YC::test_update_svd(20, 5, 10, 5);
	//YC::test_update_svd(10000, 500, 500, 500);
	//YC::test_root_finder();
	YC::test_SVD_updater();
	//MyPause;
	return 0;
	MyMatrix US(3, 3);
	US << 0, 1, 2, 3, 4, 5, 6, 7, 8;
	Eigen::VectorXd m_vec(3);
	m_vec[0] = 10; m_vec[1] = 20; m_vec[2] = 30;
	US = US * m_vec.asDiagonal();
	//US << 0, 20, 60, 30, 80, 150, 60, 140, 240;
	std::cout << US << std::endl;
	Eigen::JacobiSVD< MyMatrix > svds_thin(US, Eigen::ComputeThinU | Eigen::ComputeThinV);
	cout << "Its singular values are:" << endl << svds_thin.singularValues() << endl;
	cout << "Its left singular vectors are the columns of the thin U matrix:" << endl << svds_thin.matrixU() << endl;
	cout << "Its right singular vectors are the columns of the thin V matrix:" << endl << svds_thin.matrixV().transpose() << endl;
	std::cout << svds_thin.matrixU() * svds_thin.singularValues().asDiagonal() * svds_thin.matrixV().transpose() << std::endl;
	MyMatrix V1(6,3);
	V1 = MyMatrix::Identity(6, 3);
	const int r = V1.cols();	//r = V1.shape[1]
	MyMatrix W_BIG = V1.block(0, 0, r, r);// W = V1[0:r,:]
	MyMatrix w_tmp = V1.row(r); //  w = V1[r,:]; w.shape= (1, w.shape[0]) # row
	std::cout << W_BIG << std::endl << std::endl;
	std::cout << w_tmp << std::endl;

	int orig_sigmas_length = 4;
	int uniq_sig_num = 3;
	MyMatrix tmpNone;
	Eigen::VectorXf tmpNoneVec;
	std::cout << "tmpNone : " << tmpNone.rows() << " " << tmpNone.cols() << std::endl;
	std::cout << "tmpNoneVec : " << tmpNoneVec.rows() << " " << tmpNoneVec.cols() << std::endl;
	MyMatrix tmpP1 = MyMatrix::Identity(orig_sigmas_length, orig_sigmas_length + 1);
	tmpP1.setZero();
	std::cout << tmpP1 << std::endl;
	std::cout << "#########################################" << std::endl;
	tmpP1.conservativeResize(orig_sigmas_length+1, orig_sigmas_length + 1);
	std::cout << tmpP1 << std::endl;
	system("pause");
	//P1 = np.insert(P1, uniq_sig_num, np.array((0.0, )* (orig_sigmas_length)+(1.0, )), axis = 0)
	MyMatrix P1(orig_sigmas_length + 1, orig_sigmas_length + 1);
	P1.block(0, 0, uniq_sig_num, orig_sigmas_length + 1) = tmpP1.block(0, 0, uniq_sig_num, orig_sigmas_length + 1);
	P1.block(uniq_sig_num, 0, 1, orig_sigmas_length + 1).setZero();
	P1.block(uniq_sig_num, 0, 1, orig_sigmas_length + 1).coeffRef(0, orig_sigmas_length) = 1.0;
	P1.block(uniq_sig_num + 1, 0, orig_sigmas_length - uniq_sig_num, orig_sigmas_length + 1) = tmpP1.block(uniq_sig_num, 0, orig_sigmas_length - uniq_sig_num, orig_sigmas_length + 1);

	std::cout << P1 << std::endl;
	Eigen::VectorXi tmpV, tmpVV;
	tmpV.resize(5);
	tmpV.setZero();

	tmpV[0] = 1;
	tmpV[1] = 2;
	tmpV[2] = 3;
	tmpV[3] = 4;
	tmpV[4] = 5;

	tmpVV.resize(5);
	tmpVV.setOnes();
	std::cout << tmpV.rows() << "  " << tmpV.cols() << std::endl;
	std::cout << tmpV.colwise().sum().value() << std::endl;
	std::cout << tmpVV.colwise().sum().value() << std::endl;
	
	tmpVV *= 5;
	std::cout << tmpV * tmpVV << std::endl;
	std::cout << tmpVV*5 << std::endl;
	std::cout << tmpVV - tmpV << std::endl;
	std::vector<int> v(5);
	std::generate(v.begin(), v.end(), std::rand); // Using the C function rand()

	std::cout << "v: ";
	for (auto iv : v) {
		std::cout << iv << " ";
	}
	std::cout << "\n";

	

	// Initialize with default values 0,1,2,3,4 from a lambda function
	// Equivalent to std::iota(v.begin(), v.end(), 0);
	int n = { 5 };
	std::generate(v.begin(), v.end(), [&n]{ return n++; });

	std::cout << "v: ";
	for (auto iv : v) {
		std::cout << iv << " ";
	}
	std::cout << "\n";
	system("pause");
}
