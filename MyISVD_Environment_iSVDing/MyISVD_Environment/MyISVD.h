#ifndef _MyISVD_H_
#define _MyISVD_H_

#include "VR_Global_Define.h"

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

namespace YC
{
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
		
		SVD_updater();
		~SVD_updater();
		void initial(const MyInt nDofs, bool update_V, const MyInt reorthStep);
		void add_column(const MyVector& new_col);
		void get_current_svd(MyMatrix& Ur,MyVector& S, MyMatrix& Vr);
		void find_roots(const MyVector& sigmas, const MyVector& m_vec_addCols, MyVector& vecRoots);
		void update_SVD(MyMatrix& U, MyVector& S, MyMatrix& Vh, const MyVector& a_col, bool a_col_col = true);
		
	private:
		void extend_matrix_by_one(MyMatrix& M);
		void concatenateVec(MyVector& vec, MyFloat v);
		void concatenateMat_OneColumn(MyMatrix& mat, MyVector col);
		//void _SVD_upd_diag(const MyVector& matS, const MyVector& m_vec, bool new_col, MyMatrix& U1, MyVector& S1, MyMatrix& V1);
		void _SVD_upd_diag(MyVector& sigmas, MyVector& m_vec, bool new_col, MyMatrix& U1, MyVector& S1, MyMatrix& V1, int method = 2);
		void _reorthogonalize();
		bool reorth_was_pereviously();
		MyFloat secular_equation(const MyVector& m_vec2, const MyVector& sig2, const MyFloat&  l);
		MyFloat one_root(const MyFloat a_i, const MyFloat b_i, const MyInt ii, const MyVector& sigmas, const MyVector& m_vec, const MyVector& sigmas2, const MyVector& m_vec2);
		void _SVD_upd_diag_equal_sigmas(MyVector& sigmas, const MyVector& vec_m, bool new_col, bool & is_equal_sigmas, MyInt& uniq_length, MyMatrix& matU, MyVector& m_vec_transformed);//equal_sigmas, uniq_sig_num, U_eq, m_vec_transformed
		std::vector< enumeratePair > python_enumerate(const MyVector& vec);
		void _arrays_merge(const MyVector& a1, const MyVector& a2, std::vector< MyInt >& perm, MyVector& srt, bool decreasing = true, bool func = false);
		bool function_comp(MyFloat a, MyFloat b, bool decreasing);
		
	public:
		static const MyMatrix& my_block_diag(const MyMatrix& mat1, const MyMatrix& mat2);
		static const MyMatrix& my_insert_row(const MyMatrix& tmpP1, const MyInt uniq_sig_num, const MyVector& tmpRow);
		static const MyMatrix& my_hstack(const MyMatrix& tmpMat, const MyVector& tmpCol);
		static const MyMatrix& my_vstack(const MyMatrix& tmpMat, const MyVector& tmpRow);
	protected:
		const std::vector<bool>& isinf(const MyVector& tmpVec);
		const std::vector<bool>& isnan(const MyVector& tmpVec);
		const MyMatrix& my_eye(const MyInt nRows, const MyInt nCols);
		const MyMatrix& my_eye(const MyInt nSize);
		const MyIntVector& nonzero(const std::vector<bool>& tmpVec);
		bool ifany(const std::vector<bool>& tmpIntVec);
		bool ifany(const MyIntVector& tmpIntVec);
		void setArrayValueWithSpecialIndexes(MyVector& tmpVec, const std::vector<bool>& tmpIntVec, const MyFloat val);
		void setArrayValueWithSpecialIndexes(MyVector& tmpVec, const MyIntVector& tmpIntVec, const MyFloat val);

		
		bool  isNone(const MyMatrix& mat);
		bool  isNone(const MyVector& vec);
		void  setNone(MyMatrix& mat);
		void  setNone(MyVector& mat);
	private:
		enum{ rounding_ndigits =14};
		MyFloat epsilon1;
		MyFloat zero_epsilon;

		MyMatrix outer_U;
		MyMatrix outer_Vh;

		MyMatrix inner_U; 
		//bool is_inner_U;
		MyMatrix inner_Vh; 
		//bool is_inner_Vh;
		MyMatrix inner_V_new_pseudo_inverse;
		//bool is_inner_V_new_pseudo_inverse;

		MyVector S;
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
	};
	void sp_linalg_svd(const MyMatrix& A, MyMatrix& U, MyVector& S, MyMatrix& Vh);
	void test_SVD_updater();
#endif//SingleDomainISVD
}
#endif//_MyISVD_H_