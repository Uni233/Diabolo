#ifndef _MyISVD_H_
#define _MyISVD_H_

#include "VR_Global_Define.h"
#include <boost/timer.hpp>
/*
class SVD_updater(object):
	def __init__(self, U,S,Vh, update_V = False, reorth_step=100):
	def add_column(self, new_col, same_subspace=None):
	def get_current_svd(self):
	def _reorthogonalize(self):
	def reorth_was_pereviously(self):
func_calls = []
it_count = []
def one_root(func, a_i, b_i, ii,sigmas,m_vec):
def find_roots(sigmas, m_vec, method=1):
def _SVD_upd_diag_equal_sigmas( sigmas, m_vec , new_col):
def _arrays_merge(a1, a2, decreasing=True , func=None):
def _SVD_upd_diag(sigmas, m_vec, new_col=True,method=2):
def update_SVD(U, S, Vh, a_col, a_col_col=True):
def test_SVD_updater():
def test_SVD_update_reorth(n_rows,start_n_col, n_max_cols, prob_same_subspace,                           reorth_step,file_name):
def test_root_finder():
def test_root_finder_lapack():
def test_update_svd(n_rows,start_n_col, n_max_cols, step_n_col):
def test_array_merge():
def test_equal_sigmas():
def test_SVD_comp_complexity(n_rows,start_n_col, n_max_cols, step_n_col):
if __name__ == '__main__':
*/
#define VARNAME(v) (#v)

template< class T >
void printInfo(std::ofstream& out, const char* lpszVarName, const T& var)
{

	out << std::endl << "********************************************" << std::endl;
	out << "**             " << lpszVarName << "  start   ****" << std::endl;
	out << "********************************************" << std::endl;
	out << var << std::endl;
	out << std::endl << "********************************************" << std::endl;
	out << "**             " << lpszVarName << "  end   ****" << std::endl;
	out << "********************************************" << std::endl;
}
namespace YC
{
	class MyTimer
	{
	public:
		MyTimer(){}
		~MyTimer(){}
		double elapsed(){ return ter.elapsed(); }
	private:
		boost::timer ter;
	};
#if SingleDomainISVD
	struct TerminationCondition  {
		bool operator() (double min, double max)  {
			return abs(min - max) <= 0.000001;
		}
	};

	struct FunctionToApproximate  {
		FunctionToApproximate(const MyVector& m_vec2, const MyVector& sigma2)
			:vec2(m_vec2), sig2(sigma2)
		{}
		MyFloat operator() (MyFloat l)  {
			const MyFloat l_2 = l*l;
			MyFloat retVal = 1.;
			Q_ASSERT(vec2.size() == sig2.size());
			for (size_t i = 0; i < vec2.size(); i++)
			{
				retVal += vec2[i] / (sig2[i] - l_2);
			}
			return retVal;
		}
	private:
		const MyVector vec2;
		const MyVector sig2;
	};

	typedef std::pair< MyInt, MyFloat > enumeratePair;

	class SVD_updater
	{
	public:
		
		SVD_updater(std::ofstream& outfile);
		~SVD_updater();
		void initial(const MyMatrix& paramU, const MyVector& paramS, const MyMatrix& paramVh, bool update_V, const MyInt reorthStep);
		void add_column(const MyVector& new_col);
		void get_current_svd(MyMatrix& Ur,MyVector& S, MyMatrix& Vr);
		bool reorth_was_pereviously();
		static void find_roots(const MyVector& sigmas, const MyVector& m_vec_addCols, MyVector& vecRoots);
		static void update_SVD(MyMatrix& U, MyVector& S, MyMatrix& Vh, const MyVector& a_col, bool a_col_col = true);
		template< class T >
		void printInfo(const char* lpszVarName, const T& var);
	private:
		void extend_matrix_by_one(MyMatrix& M);
		void concatenateVec(MyVector& vec, MyFloat v);
		static void concatenateMat_OneColumn(MyMatrix& mat, MyVector col);
		//void _SVD_upd_diag(const MyVector& matS, const MyVector& m_vec, bool new_col, MyMatrix& U1, MyVector& S1, MyMatrix& V1);
		static void _SVD_upd_diag(MyVector sigmas, MyVector& m_vec, bool new_col, MyMatrix& U1, MyVector& S1, MyMatrix& V1, int method = 2);
		void _reorthogonalize();
		
		static MyFloat secular_equation(const MyVector& m_vec2, const MyVector& sig2, const MyFloat&  l);
		static MyFloat one_root(const MyFloat a_i, const MyFloat b_i, /*const MyInt ii, const MyVector& sigmas, const MyVector& m_vec,*/ const MyVector& sigmas2, const MyVector& m_vec2);
		static void _SVD_upd_diag_equal_sigmas(MyVector& sigmas, const MyVector& vec_m, bool new_col, bool & is_equal_sigmas, MyInt& uniq_length, MyMatrix& matU, MyVector& m_vec_transformed);//equal_sigmas, uniq_sig_num, U_eq, m_vec_transformed
		static std::vector< enumeratePair > python_enumerate(const MyVector& vec);
		static void _arrays_merge(const MyVector& a1, const MyVector& a2, std::vector< MyInt >& perm, MyVector& srt, bool decreasing = true, bool func = false);
		static bool function_comp(MyFloat a, MyFloat b, bool decreasing);
		
	public:
		static const MyMatrix& my_block_diag(const MyMatrix& mat1, const MyMatrix& mat2);
		static const MyMatrix& my_insert_row(const MyMatrix& tmpP1, const MyInt uniq_sig_num, const MyVector& tmpRow);
		static const MyMatrix my_hstack(const MyMatrix& tmpMat, const MyVector& tmpCol);
		static const MyMatrix my_vstack(const MyMatrix& tmpMat, const MyVector& tmpRow);
		static MyVector my_random_sample(const MyInt nSize);
		static MyFloat  my_random_sample();
		static MyMatrix my_random_sample(const MyInt nRows, const MyInt nCols);
		static void my_sort_vector(MyVector& vec);
	protected:
		static const std::vector<bool>& isinf(const MyVector& tmpVec);
		static const std::vector<bool>& isnan(const MyVector& tmpVec);
		static const MyMatrix& my_eye(const MyInt nRows, const MyInt nCols);
		static const MyMatrix& my_eye(const MyInt nSize);
		static const MyIntVector& nonzero(const std::vector<bool>& tmpVec);
		static bool ifany(const std::vector<bool>& tmpIntVec);
		static bool ifany(const MyIntVector& tmpIntVec);
		static void setArrayValueWithSpecialIndexes(MyVector& tmpVec, const std::vector<bool>& tmpIntVec, const MyFloat val);
		static void setArrayValueWithSpecialIndexes(MyVector& tmpVec, const MyIntVector& tmpIntVec, const MyFloat val);

		
		static bool  isNone(const MyMatrix& mat);
		static bool  isNone(const MyVector& vec);
		static void  setNone(MyMatrix& mat);
		static void  setNone(MyVector& mat);
	private:
		enum{ rounding_ndigits =14};
		static MyFloat epsilon1;
		MyFloat zero_epsilon;

		MyMatrix outer_U;
		MyMatrix outer_Vh;

		MyMatrix inner_U; 
		//bool is_inner_U;
		MyMatrix inner_Vh; 
		//bool is_inner_Vh;
		MyMatrix inner_V_new_pseudo_inverse;
		//bool is_inner_V_new_pseudo_inverse;

		MyVector self_S;
		MyInt m_outer_rows;
		MyInt n_rank_count;

		bool is_update_Vh;

		MyInt reorth_step;
		MyInt update_count;
		MyInt reorth_count;

		MyVector m_vec;
		MyVector new_u;
		MyFloat mu;
		bool rank_increases;
		MyMatrix U1, V1;

		std::ofstream& m_outfile;
	};
	void sp_linalg_svd(const MyMatrix& A, MyMatrix& U, MyVector& S, MyMatrix& Vh);
	void test_SVD_updater();
	void test_root_finder();
	void test_SVD_update_reorth(const MyInt n_rows, const MyInt start_n_col, const MyInt n_max_cols, const MyFloat prob_same_subspace, const MyInt reorth_step);
	void test_update_svd(const MyInt n_rows, const MyInt start_n_col, const MyInt n_max_cols, const MyInt step_n_col);
	template< class T >
	void printVector(const char* lpszStr, const std::vector< T >& vecs)
	{
		std::cout << lpszStr << " : " << std::endl;
		std::copy(vecs.begin(), vecs.end(), std::ostream_iterator< T>(std::cout, " "));
		std::cout << std::endl;
	}
#endif//SingleDomainISVD
}
#endif//_MyISVD_H_