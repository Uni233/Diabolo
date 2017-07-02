#include "MyISVD.h"
#include <iostream>
#include <math.h>       /* pow */
#include <boost/typeof/typeof.hpp>
#include <cmath>
#include <utility>
#include <iostream>
#include <fstream>
#include <boost/math/tools/roots.hpp>
#include "MyIterMacro.h"
#include "constant_numbers.h"
#include <algorithm>// stl::generate
#include <math.h>//std::isinf
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <time.h>
#include <algorithm>
#define MAT_INFO(M) std::cout << "#M :" << M.rows() << ", " << M.cols() << std::endl



namespace YC
{
#if SingleDomainISVD
	
	MyFloat SVD_updater::epsilon1;

	template< class T >
	void SVD_updater::printInfo(const char* lpszVarName, const T& var)
	{

		m_outfile << std::endl << "********************************************" << std::endl;
		m_outfile << "**             " << lpszVarName << "  start   ****" << std::endl;
		m_outfile << "********************************************" << std::endl;
		m_outfile << var << std::endl;
		m_outfile << std::endl << "********************************************" << std::endl;
		m_outfile << "**             " << lpszVarName << "  end   ****" << std::endl;
		m_outfile << "********************************************" << std::endl;
	}
	/*
	"""
	Class constructor.

	Input:
	U,S,Vh - current SVD
	update_V - whether or not update matrix V
	reorth_step - how often to perform orthogonalization step

	Output:
	None
	"""
	*/
	void SVD_updater::initial(const MyMatrix& paramU, const MyVector& paramS, const MyMatrix& paramVh, bool update_V, const MyInt reorthStep)
	{
		outer_U = paramU;
		outer_Vh = paramVh;
		self_S = paramS;

		m_outer_rows = (outer_U.rows());
		n_rank_count = (outer_Vh.cols());
		is_update_Vh = (update_V);
		/*is_inner_U = (false);
		is_inner_Vh = (false);
		is_inner_V_new_pseudo_inverse = (false);*/
		reorth_step = (reorthStep);
		epsilon1 = std::pow(10, -1 * rounding_ndigits);
		update_count = 0;
		reorth_count = 0;
	}
	SVD_updater::SVD_updater(std::ofstream& outfile)
		:m_outfile(outfile)
	{
		epsilon1 = std::pow(10, -1 * rounding_ndigits);
		update_count = 0;
		reorth_count = 0;
	}

	SVD_updater::~SVD_updater()
	{}

	void SVD_updater::extend_matrix_by_one(MyMatrix& M)
	{
		const MyInt nRows = M.rows();
		const MyInt nCols = M.cols();
		M.conservativeResize(nRows+1,nCols+1);
		M.row(nRows).setZero();
		M.col(nCols).setZero();
		M.coeffRef(nRows,nCols) = 1.f;
	}

	void SVD_updater::concatenateVec(MyVector& vec, MyFloat v)
	{
		vec.conservativeResize(vec.rows() + 1);
		vec[vec.rows() - 1] = v;
	}

	void SVD_updater::concatenateMat_OneColumn(MyMatrix& mat, MyVector col)
	{
		mat.conservativeResize(Eigen::NoChange, mat.cols() + 1);
		mat.col(mat.cols()-1) = col;
	}

	bool  SVD_updater::isNone(const MyMatrix& mat)
	{
		return (mat.rows() == 0) && (mat.cols() == 0);
	}

	bool  SVD_updater::isNone(const MyVector& vec)
	{
		return (vec.rows() == 0);
	}

	void  SVD_updater::setNone(MyMatrix& mat)
	{
		mat.setZero(0, 0);
	}

	void  SVD_updater::setNone(MyVector& vec)
	{
		vec.setZero(0,1);
	}
	/*
	Add column to the current SVD.

	Input:
	new_col - one or two dimensional vector ( if two dimensional one it must have the shape (len,1) )
	same_subspace - parameter provided for testing. During the test we know whether the new column
	is generated from the same subspace or not. Then it is possible to provide this
	parameter for debuging.
	Output:
	None
	*/
	void SVD_updater::add_column(const MyVector& new_col)
	{
		zero_epsilon = std::sqrt(std::numeric_limits< MyFloat >::epsilon() * std::pow(m_outer_rows, 2) * n_rank_count * 10 * 2);

		m_vec = outer_U.transpose() * new_col;//  np.dot(self.outer_U.T, new_col)
		
		if (!isNone(inner_U))
		{
			m_vec = inner_U.transpose() * m_vec;
		}

		Q_ASSERT(n_rank_count < m_outer_rows);
		if (n_rank_count < m_outer_rows)
		{
			if (!isNone(inner_U))
			{
				new_u = new_col - outer_U * (inner_U * m_vec); //new_u = new_col - np.dot(self.outer_U, np.dot(self.inner_U, m_vec))
			}
			else
			{
				new_u = new_col - outer_U * m_vec; // new_u = new_col - np.dot(self.outer_U, m_vec)
			}

			mu = new_u.norm();	//mu = sp.linalg.norm(new_u, ord = 2)
			
			//printInfo(VARNAME(new_u), new_u);
			if (mu < zero_epsilon) 
			{
				// rank is not increased.
				// new column is from the same subspace as the old column
				rank_increases = false;
				_SVD_upd_diag(self_S, m_vec, false/*new_col = False*/, U1, self_S, V1);
				/*printInfo(VARNAME(U1), U1);
				printInfo(VARNAME(self_S), self_S);
				printInfo(VARNAME(V1), V1);*/
			}
			else
			{
				//rank is increased
				rank_increases = true;
				concatenateVec(self_S,0.f);
				//printInfo(VARNAME(self_S), self_S);
				concatenateVec(m_vec, mu);
				//printInfo(VARNAME(new_u), new_u);
				_SVD_upd_diag(self_S, m_vec, true/*new_col = True*/, U1,self_S,V1);
				// Update outer matrix
				concatenateMat_OneColumn(outer_U, new_u / mu);//update outer matrix in case of rank increase
				//printInfo(VARNAME(outer_U), outer_U);
			}

			if (isNone(inner_U))
			{
				inner_U = U1;
			}
			else
			{
				if (rank_increases)
				{
					extend_matrix_by_one(inner_U);
				}
				inner_U = inner_U * U1;
			}

			if (is_update_Vh)
			{
				if (rank_increases)
				{
					extend_matrix_by_one(outer_Vh);
					if (!isNone(inner_Vh))
					{
						extend_matrix_by_one(inner_Vh);
						inner_Vh = V1.transpose() * inner_Vh;
					}
					else
					{
						inner_Vh = V1.transpose();
					}

					if (!isNone(inner_V_new_pseudo_inverse))
					{
						extend_matrix_by_one(inner_V_new_pseudo_inverse);
						inner_V_new_pseudo_inverse = V1.transpose() * inner_V_new_pseudo_inverse;						
					}
					else
					{
						inner_V_new_pseudo_inverse = V1.transpose();
					}
				} 
				else // rank not increase
				{
					const MyInt r = V1.cols();	//r = V1.shape[1]
					MyMatrix W_BIG = V1.block(0, 0, r, r);// W = V1[0:r,:]
					MyMatrix w_tmp = V1.row(r); //  w = V1[r,:]; w.shape= (1, w.shape[0]) # row

					const MyFloat w_norm2 = std::pow(w_tmp.row(0).norm(), 2);// w_norm2 = sp.linalg.norm(w, ord=2)**2
					MyMatrix W_pseudo_inverse = W_BIG.transpose() + ((1 / (1 - w_norm2))*w_tmp.transpose()) * (w_tmp * W_BIG.transpose()); //W_pseudo_inverse = W.T + np.dot( (1/(1-w_norm2))*w.T, np.dot(w,W.T) )
                   
                    
                    //del r, w_norm2

					if (!isNone(inner_Vh))
					{
						inner_Vh = W_BIG.transpose() * inner_Vh;//self.inner_Vh = np.dot(W.T, self.inner_Vh)
					}
					else
					{
						inner_Vh = W_BIG.transpose();//self.inner_Vh = W.T
					}

					if (!isNone(inner_V_new_pseudo_inverse))
					{
						inner_V_new_pseudo_inverse = W_pseudo_inverse * inner_V_new_pseudo_inverse;//self.inner_V_new_pseudo_inverse = np.dot(W_pseudo_inverse, self.inner_V_new_pseudo_inverse)
					}
					else
					{
						inner_V_new_pseudo_inverse = W_pseudo_inverse;//self.inner_V_new_pseudo_inverse = W_pseudo_inverse
					}

					concatenateMat_OneColumn(outer_Vh, (inner_V_new_pseudo_inverse.transpose() * w_tmp.transpose()));//self.outer_Vh = np.hstack( (self.outer_Vh, np.dot( self.inner_V_new_pseudo_inverse.T, w.T)) )

                    //del w, W_pseudo_inverse, W
				}
			}

		}//if (n_rank_count < m_outer_rows)
		else
		{
			MyError("n_rank_count < m_outer_rows");
		}
		n_rank_count = outer_Vh.cols();// self.n = self.outer_Vh.shape[1]
        //new_col.shape  = old_shape

		update_count++;//self.update_count += 1 # update update step counter
		reorth_count++;// self.reorth_count += 1

		if (reorth_count >= reorth_step)
		{
			_reorthogonalize();
			reorth_count = 0;
		}
        
        //# Temp return, return afeter tests:
        //#return np.dot( self.outer_U, self.inner_U), self.S, np.dot( self.inner_Vh, self.outer_Vh)
	}//add_column(const MyVector& new_col)

	void SVD_updater::get_current_svd(MyMatrix& Ur, MyVector& SS, MyMatrix& Vhr)
	{
		if (!isNone(inner_U))
		{
			Ur = outer_U * inner_U;
		}
		else
		{
			Ur = outer_U;
		}

		if (is_update_Vh)
		{
			if (!isNone(inner_Vh))
			{
				Vhr = inner_Vh * outer_Vh;
			}
			else
			{
				Vhr = outer_Vh;
			}
		}
		else
		{
			setNone(Vhr);//	Vhr.setZero();
		}
		
		SS = self_S;
	}

	/*
	Uses orthogonalization method mentioned in:
	Brand, M. "Fast low-rank modifications of the thin singular value
	decomposition Linear Algebra and its Applications" , 2006, 415, 20 - 30.

	Actually the usefulness of this method is not wel justified. But it is
	implemeted here. This function is called from "add_column" method.
	*/
	void SVD_updater::_reorthogonalize()
	{
		if (!isNone(inner_U))
		{
			//US = np.dot(self.inner_U, np.diag(self.S))
			Q_ASSERT(inner_U.cols() == self_S.size());
			MyMatrix US = inner_U * self_S.asDiagonal();
			//(Utmp, Stmp, Vhtmp) = sp.linalg.svd(US, full_matrices = False, compute_uv = True, overwrite_a = False, check_finite = False)
			Eigen::JacobiSVD< MyMatrix > svds_thin(US, Eigen::ComputeThinU | Eigen::ComputeThinV);
			MyMatrix Utmp = svds_thin.matrixU();
			MyVector Stmp = svds_thin.singularValues();
			MyMatrix Vhtmp = svds_thin.matrixV().transpose();
			
			//self.inner_U = Utmp
			inner_U = Utmp;
			//self.S = Stmp
			self_S = Stmp;

			//# update outer matrix ->
			//self.outer_U = np.dot(self.outer_U, self.inner_U)
			outer_U = outer_U * inner_U;
			//self.inner_U = None
			setNone(inner_U);
			//# update outer matrix < -
			if (is_update_Vh)//if self.update_Vh:
			{
				//self.inner_Vh = np.dot(Vhtmp, self.inner_Vh)
				inner_Vh = Vhtmp *inner_Vh;

				//# update outer matrix ->
				//self.outer_Vh = np.dot(self.inner_Vh, self.outer_Vh)
				outer_Vh = inner_Vh * outer_Vh;
				//self.inner_Vh = None
				setNone(inner_Vh);
				//self.inner_V_new_pseudo_inverse = None
				setNone(inner_V_new_pseudo_inverse);
				//# update outer matrix < -

				if (is_update_Vh)
				{
					//return self.outer_U, self.S, self.outer_Vh
				} 
				else
				{
					//return self.outer_U, self.S, None
				}
			}
		}
		else
		{
			MyError("_reorthogonalize : isNone(inner_U)");
		}
	}

	/*
	Function for external testing.

	Functions returns the boolean value which tells
	whether reorthogonalization has performed on the previous step.
	*/
	bool SVD_updater::reorth_was_pereviously()
	{
		return (reorth_count == 0);
	}

	MyFloat SVD_updater::secular_equation(const MyVector& m_vec2, const MyVector& sig2, const MyFloat&  l)
	{
		//func = lambda l: 1 + np.sum(m_vec2 / (sig2 - l**2 ))
		const MyFloat l_2 = l*l;
		MyFloat retVal = 1.;
		Q_ASSERT(m_vec2.size() == sig2.size());
		for (size_t i = 0; i < m_vec2.size(); i++)
		{
			retVal += m_vec2[i] / (sig2[i] - l_2);
		}
		return retVal;
	}

	

	/*
	Function is used to find one root on the interval [a_i,b_i] a_i < b_i. It must be
	so that f(a_i) < 0, f(b_i) > 0. The reason for this function is that for new singular
	values values at a_i and b_i f is infinity. So that procedure is needed to find correct
	intervals.

	Derivative free method brentq is used internally to find root. Maybe some
	other method would work faster.

	Inputs:
	func - function which can be called
	a_i - left interval value
	b_i - right interval value

	Output:
	Root on this interval
	*/
	MyFloat SVD_updater::one_root(const MyFloat a_i, const MyFloat b_i, /*const MyInt ii, const MyVector& sigmas, const MyVector& m_vec,*/ const MyVector& sigmas2, const MyVector& m_vec2)
	{
		/*std::cout << "a_i " << a_i << std::endl;
		std::cout << "b_i " << b_i << std::endl;
		std::cout << "sigmas2 " << sigmas2.transpose() << std::endl;
		std::cout << "m_vec2 " << m_vec2.transpose() << std::endl;*/
		MyFloat shift = 0.01;//shift = 0.01
		const MyFloat delta = b_i - a_i;
		const MyFloat tmp = shift*delta;

		const MyFloat eps = std::numeric_limits< MyFloat >::epsilon();// eps = np.finfo(float).eps

		MyFloat a_new = a_i + tmp;
		MyFloat b_new = b_i - tmp;

		//# a_i * eps  - absolute error
		MyFloat a_max_it = std::ceil(std::log10(tmp / (a_i * eps))) + 2; //= np.ceil(np.log10(tmp / (a_i * eps))) + 2
		MyFloat b_max_it = std::ceil(std::log10(tmp / (b_i * eps))) + 2; //np.ceil(np.log10(tmp / (b_i * eps))) + 2

		if (std::isnan(a_max_it) || a_max_it > 20 )
		{
			a_max_it = 17;
		}
		/*if np.isnan(a_max_it) or a_max_it > 20:
			a_max_it = 17*/
		if (std::isnan(b_max_it) || b_max_it > 20)
		{
			b_max_it = 17;
		}
		/*if np.isnan(b_max_it) or b_max_it > 20:
			b_max_it = 17*/

		bool a_found = false; 
		bool b_found = false; 
		MyInt it = 0;

		while (!(a_found && b_found))
		{
			shift /= 10.0;
			if (!a_found)
			{
				if (secular_equation(m_vec2,sigmas2,a_new) >= 0.0)
				{
					a_new = a_i + shift*delta;
					if (it >= a_max_it)
					{
						return a_new;
					}
				}
				else
				{
					a_found = true;
				}
			}

			if (!b_found)
			{
				if (secular_equation(m_vec2, sigmas2, b_new) <= 0.0)
				{
					b_new = b_i - shift*delta;
					if (it >= b_max_it)
					{
						return b_new;
					}
				}
				else
				{
					b_found = true;
				}
			}
			it += 1;
		}

		using boost::math::tools::bisect;
		std::pair<double, double> result = bisect(FunctionToApproximate(m_vec2, sigmas2), a_new, b_new, TerminationCondition());
		MyFloat root = (result.first + result.second) / 2;  // = 0.381966...
		return root;

	/*res = opt.brentq(func, a_new, b_new, full_output=True, disp=False)
    if res[1].converged == False:
        raise ValueError("Root is not found")

    func_calls.append(res[1].function_calls)
    it_count.append(it)
    return res[0]
		return 0.f;*/
	}

	/*
	Find roots of secular equation of augmented singular values matrix

	Inputs:
	sigmas - (n*n) matrix of singular values, which are on the diagonal.

	m_vec - additional column vector which is attached to the right of the Sigma.
	Must be (m*1)
	method - which method to use to find roots

	There are two ways to find roots for secular equation of augmented s.v.
	matrix. One way is to use again SVD decomposition and the second method
	is to find roots of algebraic function on a certain interval.
	Currently method 2 is used by using imported lapack function.
	*/
	void SVD_updater::find_roots(const MyVector& sigmas, const MyVector& m_vec_addCols, MyVector& vecRoots)
	{
		Q_ASSERT(sigmas.size() >1);
		MyVector m_vec2 = m_vec_addCols.cwiseAbs2();//		m_vec2 = np.power(m_vec,2)
		MyVector sig2 = sigmas.cwiseAbs2();//  sig2 = np.power(sigmas,2)
		MyInt it_len;
		bool append_zero;
        //func = lambda l: 1 + np.sum(m_vec2 / (sig2 - l**2 ))
		if (sigmas[sigmas.size() - 1] < epsilon1 && sigmas[sigmas.size() - 2] < epsilon1)
		{
			//# two zeros at the end
			it_len = sigmas.size() - 1;
			append_zero = true;
		}
		else
		{
			it_len = sigmas.size();
			append_zero = false;
		}
        /*if (sigmas[-1] < epsilon1) and (sigmas[-2] < epsilon1): 
            it_len = len(sigmas) - 1
            append_zero = True
        else:
            it_len = len(sigmas)
            append_zero = False*/

		std::vector< MyFloat > roots;   // roots = [] # roots in increasing order (singular values of new matrix)
										//# It is assumed that the first eigenvalue of Sigma [(n+1)*(n+1)] is zero
		MyFloat root;
		
		for (size_t i = 0; i < it_len; i++)
		{
			if (0 == i)
			{
				//root = one_root(func, sigmas[0], (sigmas[0] + np.sqrt(np.sum(m_vec2)) ),i,sigmas,m_vec )
				root = one_root(sigmas[0], (sigmas[0] + std::sqrt(m_vec2.transpose().rowwise().sum()[0])), /*i, sigmas, m_vec,*/ sig2, m_vec2);
				printf("[%9.7f,%9.7f] root %9.7f\n", sigmas[0], (sigmas[0] + std::sqrt(m_vec2.transpose().rowwise().sum()[0])), root);
			}
			else
			{
				//root = one_root(func, sigmas[i], sigmas[i-1],i,sigmas,m_vec )
				root = one_root(sigmas[i], sigmas[i - 1], /*i, sigmas, m_vec,*/ sig2, m_vec2);
				printf("[%9.7f,%9.7f] root %9.7f\n", sigmas[i], sigmas[i - 1], root);
			}
			roots.push_back(root);
		}
		
		if (append_zero)
		{
			roots.push_back(0.0);
		}

		vecRoots.resize(roots.size());
		for (size_t i = 0; i < roots.size(); i++)
		{
			vecRoots[i] = roots[i];
		}
		

        /*for i in xrange(0, it_len ):

            if (i == 0):
                root = one_root(func, sigmas[0], (sigmas[0] + np.sqrt(np.sum(m_vec2)) ),i,sigmas,m_vec )
            else:
                root = one_root(func, sigmas[i], sigmas[i-1],i,sigmas,m_vec )

            roots.append( root )
        if append_zero:
            roots.append( 0.0 )

        return np.array(roots)*/
	}

	/*
	[s for s in enumerate( sigmas ) ]
	eg: 
		b = np.array([10, 11, 12])
		c=[s for s in enumerate( b ) ]

		c is [(0, 10), (1, 11), (2, 12)]
	*/
	std::vector< enumeratePair > SVD_updater::python_enumerate(const MyVector& vec)
	{
		std::vector< enumeratePair > retVec;
		for (int i = 0; i < vec.size();++i)
		{
			retVec.push_back( std::make_pair(i,vec[i]) );
		}
		return retVec;
	}
	/*
	This is internal function which is call by _SVD_upd_diag. It analyses
	if there are equal sigmas in the set of sigmas and if there are returns
	the appropriate unitary transformation which separates unique and equal
	sigmas. This is needed because the method in _SVD_upd_diag works only for
	unique sigmas. It also detects zeros in the m_vec and performs appropriate
	permutation if these zeros are found. When new column is added the
	original matrix is square and it is mandatory that sigma[-1] = 0 and
	m_vec[-1] != 0. When new row is added original matrix is not square
	and sigmas are arbitary.

	Inputs:
	sigmas - singular values of a square matrix
	m_vec - extra column added to the square diagonal matrix of singular
	vectors.
	new_col - says which task is task is solved. True if originally the
	problem of new column is solved, False if new row is added.
	Output:
	is_equal_sigmas - boolean which is True if there are equal sigmas,
	False otherwise.
	uniq_length - quantity of unique sigmas.
	U - unitary transformation which transform the
	original matrix (S + mm*) into the form
	where equal sigmas are at the end of the diagonal of the new matrix.
	*/
	void SVD_updater::_SVD_upd_diag_equal_sigmas(MyVector& sigmas, const MyVector& vec_m, 
		bool new_col, bool & is_equal_sigmas, MyInt& uniq_length, MyMatrix& matU, MyVector& m_vec_transformed)
	{
		const MyInt orig_sigma_len = sigmas.size();// len(sigmas) # length ofsigma and m_cel vectors
		matU.resize(0,0);
		matU.setZero();// U = None # unitary matrix which turns m_vec into right view in case repeated sigmas and / or zeros in m_vec
		
		MyVector z_col = vec_m;//z_col = m_vec.copy()

		if (new_col)
		{
			/*
			if new_col: # temporary change the last element in order to avoid
						# processing if there are other zero singular values
				sigmas[-1] = -1
			*/
			sigmas[sigmas.size() - 1] = -1;
		}
    
		std::vector< enumeratePair > indexed_sigmas = python_enumerate(sigmas);//indexed_sigmas = [s for s in enumerate(sigmas)] # list of indexed sigmas

		//# detect zero components in m_vec ->
		std::vector< MyInt > zero_inds;//zero_inds = [] # indices of elements where m_col[i] = 0
		std::vector< MyInt > nonzero_inds;// nonzero_inds = [] # indices of elements where m_col[i] != 0

		MyVector m_vec_abs = vec_m.cwiseAbs();
		for (int i = 0; i < m_vec_abs.size();++i)
		{
			if (m_vec_abs[i] < epsilon1)
			{
				zero_inds.push_back(i);
			}
			else
			{
				nonzero_inds.push_back(i);
			}
		}
		MyInt num_nonzero = nonzero_inds.size();// len(nonzero_inds)

		bool U_active = false;
		if (zero_inds.size() > 0)
		{
			U_active = true;
		}

		{
			std::vector< enumeratePair > indexed_sigmas_nonzero;
			for (iterAllOf(iter, nonzero_inds))
			{
				//indexed_sigmas =  [indexed_sigmas[i] for i in nonzero_inds] # permutation according to zeros in z_col
				indexed_sigmas_nonzero.push_back(indexed_sigmas[nonzero_inds[*iter]]); MyNotice
			}
			indexed_sigmas = indexed_sigmas_nonzero;
		}

		{
			//z_col = [z_col[i] for i in nonzero_inds] + [z_col[i] for i in zero_inds] # permutation according to zeros in z_col
			std::vector< MyFloat > z_col_nonzero_zero;			
			for (iterAllOf(iter, nonzero_inds))
			{
				z_col_nonzero_zero.push_back(z_col[nonzero_inds[*iter]]);
			}
			for (iterAllOf(iter, zero_inds))
			{
				z_col_nonzero_zero.push_back(z_col[zero_inds[*iter]]);
			}
			//std::cout << "z_col.size() = " << z_col.size() << std::endl;
			Q_ASSERT(z_col.size() == z_col_nonzero_zero.size());
			for (int i = 0; i < z_col_nonzero_zero.size(); ++i)
			{
				z_col[i] = z_col_nonzero_zero[i];
			}

			/*for (iterAllOf(iter, z_col_nonzero_zero))
			{

				z_col[*iter] = z_col_nonzero_zero[*iter];
			}*/
		}
		
		//# detect zero components in m_vec <-
		std::vector< MyInt > permutation_list;
		std::vector< std::vector< MyInt > > equal_inds_list;// = [] # list of lists where indices of equal sigmas are stored in sublists
		std::vector< MyInt > unique_inds_list;// unique_inds_list = [] # indices of unique sigmas(including the first appears of duplicates).
							  //# Needed for construction of permutation matrix
		bool found_equality_chain = false; 
		std::vector<MyInt> curr_equal_inds;
		enumeratePair prev_val = indexed_sigmas[0];// prev_val = indexed_sigmas[0];
		unique_inds_list.push_back(prev_val.first); //unique_inds_list.append(prev_val[0]); 
		MyInt i = 0;
		MyFloat v = 0.0;// # def of i, v is needed for correct deletion

		for (int iter = 1; iter < indexed_sigmas.size(); ++iter)
		{
			i = indexed_sigmas[iter].first;
			v = indexed_sigmas[iter].second;

			if (numbers::IsEqual(v, prev_val.second))
			{
				if (found_equality_chain)
				{
					curr_equal_inds.push_back(i);
				} 
				else
				{
					curr_equal_inds.push_back(prev_val.first);// curr_equal_inds.append(prev_val[0])
					curr_equal_inds.push_back(i);
					found_equality_chain = true;
				}
			}
			else
			{
				if (found_equality_chain)
				{
					equal_inds_list.push_back(curr_equal_inds);
					
					unique_inds_list.push_back(i);
					curr_equal_inds.clear();
					found_equality_chain = false;
				} 
				else
				{
					unique_inds_list.push_back(i);
				}
			}
			prev_val = std::make_pair(i,v);
		}//for (int iter = 1; iter < indexed_sigmas_nonzero.size();++iter)
		if (curr_equal_inds.size()>0)
		{
			/*
			if curr_equal_inds:
				equal_inds_list.append(curr_equal_inds)
			*/
			equal_inds_list.push_back(curr_equal_inds);
		}
		//del indexed_sigmas, curr_equal_inds, found_equality_chain, prev_val, i, v
		bool equal_sigmas = false;// # Boolean indicator that there are sigmas which are equal
		std::vector< MyInt > permute_indices;//add by yc
		std::vector< MyInt > extra_indices;
		if (equal_inds_list.size() > 0)//# there are equal singular values and we need to do unitary transformation
		{
			bool U_active = true;//	U_active = True
			bool equal_sigmas = true;// # Boolean indicator that there are sigmas which are equal
			matU = MyMatrix::Identity(orig_sigma_len, orig_sigma_len);// U = np.eye(orig_sigma_len) # unitary matrix which is a combination of Givens rotations
			permute_indices.clear();// permute_indices = [] # which indices to put in the end of matrix S and m_col
									//# in m_col this indicas must be zero-valued.
			for (iterAllOf(iter, equal_inds_list))
			{
				std::vector< MyInt >& ll = *iter;
				MyMatrix U_part;//U_part = None
				bool is_U_part = false;
				MyFloat m = vec_m[ll[0]];
				const MyInt len_ll = ll.size();
				for (int i = 1 MyNotice; i < len_ll; ++i)//for i in xrange(1,len(ll)):
				{
					MyMatrix U_tmp = MyMatrix::Identity(len_ll, len_ll);//U_tmp = np.eye( len(ll) )
					permute_indices.push_back(ll[i]);//permute_indices.append( ll[i] )

					MyFloat a = m; MyFloat b = vec_m[ll[i]];//a = m; b = m_vec[ ll[i] ]
					m = std::sqrt(std::pow(a,2) + std::pow(b,2));//m = np.sqrt( a**2 + b**2 )
					MyFloat alpha = a / m; MyFloat beta = b / m;

					U_tmp.coeffRef(0,0) = alpha;//U_tmp[0, 0] = alpha;
					U_tmp.coeffRef(0, i) = beta;//U_tmp[ 0, i ] = beta
					U_tmp.coeffRef(i,0) = -1.0*beta;//U_tmp[i, 0] = -beta;
					U_tmp.coeffRef(i, i) = alpha;//U_tmp[ i, i ] = alpha

					if (!is_U_part)
					{
						//U_part = U_tmp.copy()
						is_U_part = true;
						U_part.resize(U_tmp.rows(), U_tmp.cols());
						U_part = U_tmp;
					} 
					else
					{
						//U_part = np.dot(U_tmp, U_part)
						U_part = U_tmp * U_part; 
					}
				}
				//U[np.array(ll, ndmin = 2).T, np.array(ll, ndmin = 2)] = U_part
				for (int r = 0; r < len_ll;++r)
				{
					for (int c = 0; c < len_ll;++c)
					{
						matU.coeffRef(ll[r], ll[c]) = U_part.coeff(r, c);
					}
				}
				
			}
			//extra_indices = permute_indices + zero_inds
			extra_indices = permute_indices;
			extra_indices.insert(extra_indices.end(), zero_inds.begin(), zero_inds.end());
		} 
		else
		{
			permute_indices.clear();
		}
		
		MyInt unique_num = unique_inds_list.size();// unique_num = len(unique_inds_list)
		const MyInt equal_num = permute_indices.size();//	equal_num = len(permute_indices)
		//assert(orig_sigma_len == unique_num + equal_num + (orig_sigma_len - num_nonzero)), "Length of equal and/or unique indices is wrong"
		Q_ASSERT( orig_sigma_len == (unique_num + equal_num + (orig_sigma_len - num_nonzero)) );
		//extra_indices = permute_indices + zero_inds
		extra_indices = permute_indices;
		extra_indices.insert(extra_indices.end(), zero_inds.begin(), zero_inds.end());

		if (extra_indices.size() > 0)//if extra_indices :
		{
			/*
			# Permute indices are repeated indices moved to the end of array
			# Sigmas corresponding to permute indices are sorted as well as sigmas corresponding
			# to zero ind. Hence we need to merge these two sigma arrays and take it into
			# account in the permutation matrix.
			*/
			if (permute_indices.size() > 0 && zero_inds.size() > 0)//if permute_indices and zero_inds : # need to merge this two arrays
			{
				//permute_sigmas = sigmas[permute_indices]
				MyVector permute_sigmas; permute_sigmas.resize(permute_indices.size());
				for (int m = 0; m < permute_indices.size();++m)
				{
					permute_sigmas[m] = sigmas[permute_indices[m]];
				}
				//zero_sigmas = sigmas[zero_inds]
				MyVector zero_sigmas; zero_sigmas.resize(zero_inds.size());
				for (int m = 0; m < zero_inds.size();++m)
				{
					zero_sigmas[m] = sigmas[zero_inds[m]];
				}
				
				std::vector< MyInt > perm;
				MyVector str;
				_arrays_merge(permute_sigmas, zero_sigmas, perm, str);
				//del permute_sigmas, zero_sigmas, srt

				//permutation_list = unique_inds_list + [extra_indices[i] for i in perm]
				permutation_list = unique_inds_list;
				for (int m = 0; m < perm.size();++m)
				{
					permutation_list.push_back(extra_indices[perm[m]]);
				}
			} 
			else
			{
				//permutation_list = unique_inds_list + extra_indices
				permutation_list = unique_inds_list;
				permutation_list.insert(permutation_list.end(), extra_indices.begin(), extra_indices.end());
			}
			MyMatrix matP(orig_sigma_len, orig_sigma_len); matP.setZero();// P = np.zeros((orig_sigma_len, orig_sigma_len)) # global permutation matrix

			/*for (i, s) in enumerate(permutation_list) :
				P[i, s] = 1*/
			for (int ii = 0; ii < permutation_list.size();++ii)
			{
				matP.coeffRef(ii, permutation_list[ii]) = 1;
			}

			if (equal_sigmas)
			{
				//U = np.dot(P, U) # incorporate permutation matrix into the matrix U
				matU = matP * matU;
			} 
			else
			{
				//U = P
				matU = matP;
			}
			//z_col = np.dot(U, m_vec) # replace z_col accordingly
			z_col = matU * vec_m;
		}
		else
		{
			unique_num = orig_sigma_len;
			matU.resize(0, 0); matU.setZero();//	U = None
			z_col.resize(0); z_col.setZero();//	z_col = None
		}
		if (new_col > 0)
		{
			sigmas[sigmas.size() - 1] = 0.0;
		}
		
		is_equal_sigmas = U_active; uniq_length = unique_num; /*matU; */ m_vec_transformed = z_col;
		//return U_active, unique_num, U, z_col
	}//_SVD_upd_diag_equal_sigmas

	bool SVD_updater::function_comp(MyFloat a, MyFloat b, bool decreasing)
	{
		if (MyNotice !decreasing)
		{
			return a <= b;
		}
		else
		{
			return a >= b;
		}
	}
	/*
	Auxiliary method which merges two SORTED arrays into one sorted array.

	Input:
	a1 - first array
	a2 - second array
	decreasing - a1 and a2 are sorted in decreasing order as well as new array
	func - function which is used to extract numerical value from the
	elemets of arrays. If it is None than indexing is used.

	Output:
	perm - permutation of indices. Indices of the second array starts
	from the length of the first array ( len(a1) )
	str -  sorted array
	*/
	void SVD_updater::_arrays_merge(const MyVector& a1, const MyVector& a2, std::vector< MyInt >& perm, MyVector& srt, bool decreasing, bool func)
	{
		const MyInt len_a1 = a1.size();
		const MyInt len_a2 = a2.size();//	len_a1 = len(a1); len_a2 = len(a2)
		perm.clear();
		srt.resize(0);
		if (0 == len_a1)
		{
			for (int i = 0; i < len_a2;++i)
			{
				perm.push_back(len_a1 + i);
			}
			srt = a2;
			return;
		}
		/*if len_a1 == 0:
			return range(len_a1, len_a1 + len_a2), a2*/

		if (0 == len_a2)
		{
			for (int i = 0; i < len_a1; ++i)
			{
				perm.push_back(i);
			}
			srt = a1;
			return;
		}
		/*if len_a2 == 0 :
			return range(len_a1), a1*/

		std::vector< MyInt > inds_a1(len_a1);
		{
			//inds_a1 = range(len_a1);
			int n = { 0 };
			std::generate(inds_a1.begin(), inds_a1.end(), [&n]{ return n++; });
		}

		std::vector< MyInt > inds_a2(len_a2);
		{
			//inds_a2 = range(len_a1, len_a1 + len_a2) # indices of array elements
			int n = { len_a1 };
			std::generate(inds_a2.begin(), inds_a2.end(), [&n]{ return n++; });
		}
		
		perm.resize(len_a1 + len_a2);
		{
			//perm = [np.nan, ] * (len_a1 + len_a2) # return permutation array
			int n = { MyNan };
			std::generate(perm.begin(), perm.end(), [&n]{ return n; });
		}
		
		srt.resize(len_a1 + len_a2); srt.setZero();//srt = [np.nan, ] * (len_a1 + len_a2) # return sorted array
		
		if (function_comp(a1[a1.size() - 1], a2[0],decreasing))
		{
			//# already sorted
			srt.block(0, 0, len_a1, 1) = a1;
			srt.block(len_a1, 0, len_a2, 1) = a2;

			perm.resize(0);
			perm.insert(perm.end(), inds_a1.begin(), inds_a1.end());
			perm.insert(perm.end(), inds_a2.begin(), inds_a2.end());

			/*
			srt[0:len_a1] = a1; srt[len_a1:] = a2
			return (inds_a1 + inds_a2), srt
			*/
			return;
		}

		if (function_comp(a2[a2.size() - 1], a1[0],decreasing))
		{
			//# also sorted but a2 goes first
			srt.block(0, 0, len_a2, 1) = a2;
			srt.block(len_a2, 0, len_a1, 1) = a1;
			perm.resize(0);
			perm.insert(perm.end(), inds_a2.begin(), inds_a2.end());
			perm.insert(perm.end(), inds_a1.begin(), inds_a1.end());
			/*
			srt[0:len_a2] = a2; srt[len_a2:] = a1
			return (inds_a2 + inds_a1), srt
			*/
			return;
		}

		MyInt a1_ind = 0, a2_ind = 0;// # indices of current elements of a1 and a2 arrays
		MyInt perm_ind = 0;// # current index of output array
		bool exit_crit = false;

		while (!exit_crit)
		{
			//if comp(a1[a1_ind], a2[a2_ind]) :
			if (function_comp(a1[a1_ind], a2[a2_ind],decreasing))
			{
				perm[perm_ind] = inds_a1[a1_ind];
				srt[perm_ind] = a1[a1_ind];
				a1_ind += 1;
			} 
			else
			{
				perm[perm_ind] = inds_a2[a2_ind];
				srt[perm_ind] = a2[a2_ind];
				a2_ind += 1;
			}
			perm_ind += 1;

			if (a1_ind == len_a1)
			{
				for (int a = a2_ind, p = perm_ind; a < len_a2; ++a,++p)
				{
					perm[p] = inds_a2[a];
					srt[p] = a2[a];
				}
				exit_crit = true;
				/*perm[perm_ind:] = inds_a2[a2_ind:]
				srt[perm_ind:] = a2[a2_ind:]
				exit_crit = True*/
			}

			if (a2_ind == len_a2)
			{
				for (int a = a1_ind, p = perm_ind; a < len_a1;++a,++p)
				{
					perm[p] = inds_a1[a];
					srt[p] = a1[a];
				}
				exit_crit = true;
				/*perm[perm_ind:] = inds_a1[a1_ind:]
				srt[perm_ind:] = a1[a1_ind:]
				exit_crit = True*/
			}
		}
	}

//2015-12-15
	/*
	"""
	This is internal function which is called by update_SVD and SVD_update
	class. It returns the SVD of diagonal matrix augmented by one column.
	There are two ways to compose augmented matrix. One way is when
	sigma[-1] = 0 and m_vec[-1] != 0 and column m_vec substitutes zero column
	in diagonal matrix np.diag(sigmas). The resulted matrix is square. This
	case is needed when the rank of the original matrix A increased by 1.
	Parameter for this case is new_col=True.
	The second case is when column m_vec is added to np.diag(sigmas). There
	are no restrictions on value of sigmas and m_vec. This case is used when
	the rank of the of the original matrix A is not increased.
	Parameter for this case is new_col=False.

	Inputs:
	sigmas - SORTED singular values of a square matrix
	m_vec - extra column added to the square diagonal matrix of singular
	vectors.
	new_col - says which task is task is solved. See comments to the
	function. True if originally the problem of new column is
	solved, False if new row is added.
	method: int: which method is used to find roots of secular equation
	Outputs:
	U, sigmas, V - SVD of the diagonal matrix plus one column.

	!!! Note that unlike update_SVD and scipy SVD routines V
	not V transpose is returned.
	"""
	*/
	void SVD_updater::_SVD_upd_diag(MyVector sigmas, MyVector& m_vec, bool new_col, MyMatrix& matU, MyVector& vecS, MyMatrix& matV, int method/* = 2*/)
	{
		const int orig_sigmas_length = sigmas.size();
		bool equal_sigmas;
		MyInt uniq_sig_num;
		MyMatrix U_eq;
		MyVector m_vec_transformed;
		_SVD_upd_diag_equal_sigmas(sigmas, m_vec, new_col, equal_sigmas, uniq_sig_num, U_eq, m_vec_transformed);
		/*std::cout << "equal_sigmas : " << equal_sigmas  << std::endl;
		std::cout << "uniq_sig_num : " << uniq_sig_num  << std::endl;
		std::cout << "U_eq : " << U_eq << std::endl;
		std::cout << "m_vec_transformed : " << m_vec_transformed << std::endl; MyPause;*/

		MyVector old_sigmas;
		MyVector old_mvec;

		MyVector extra_sigmas;
		if (equal_sigmas)
		{
			old_sigmas = sigmas;
			old_mvec = m_vec;

			//sigmas = np.diag(np.dot(np.dot(U_eq, np.diag(sigmas)), U_eq.T))
			MyMatrix matSigmas;
			matSigmas.resize(sigmas.size(), sigmas.size()); matSigmas.setZero();
			matSigmas.diagonal() = sigmas;
			MyVector tmpSigmas = ((U_eq * matSigmas) * U_eq.transpose()).diagonal();
			
			//extra_sigmas = sigmas[uniq_sig_num:]
			//sigmas = sigmas[0:uniq_sig_num]
			extra_sigmas.resize(sigmas.size() - uniq_sig_num); extra_sigmas.setZero();
			sigmas.resize(uniq_sig_num); sigmas.setZero();

			int ii = 0;
			for (; ii < uniq_sig_num MyNotice;++ii)
			{
				sigmas[ii] = tmpSigmas[ii];
			}

			for (int i = 0; i < extra_sigmas.size();++i,++ii)
			{
				extra_sigmas[i] = tmpSigmas[ii];
			}
			//m_vec = m_vec_transformed[0:uniq_sig_num]
			m_vec = m_vec_transformed.block(0, 0, uniq_sig_num,1);
		}

		MyVector& new_sigmas = vecS;
		MyInt new_size = MyNan;
		if (1 == sigmas.size()) //if (len(sigmas) == 1) :
		{			
			//new_sigmas = np.array((np.sqrt(sigmas[0] * *2 + m_vec[0] * *2), ))
			new_sigmas.resize(1);
			new_sigmas[0] = std::sqrt(std::pow(sigmas[0], 2) + std::pow(m_vec[0], 2));			
			new_size = 1;
		}
		else
		{
			MyVector ret;
			//ret = find_roots(sigmas, m_vec, method = method)
			
			find_roots(sigmas, m_vec, ret);
			new_sigmas = ret;
			if (3 == method)
			{
				if (new_col && ((sigmas[-1] < epsilon1) && (sigmas[-2] < epsilon1)))
				{
					/*
					# This check has been written for the case when roots were found by eigh method.So.it should be used
					# when root computing method is 3, if it is 1 this step is done in the function fing_roots.
					# Remind that for the case new_col = True sigmas[-1] = 0 - compulsory.
					# This branch handles the case when there are other zero sigmas.
					*/
					new_sigmas[-1] = 0;
				}
			}
			/*if (method == 3) :
				if new_col and((sigmas[-1] < epsilon1) and(sigmas[-2] < epsilon1)) :
					# This check has been written for the case when roots were found by eigh method.So.it should be used
					# when root computing method is 3, if it is 1 this step is done in the function fing_roots.
					# Remind that for the case new_col = True sigmas[-1] = 0 - compulsory.
					# This branch handles the case when there are other zero sigmas.
					new_sigmas[-1] = 0*/

			//del ret

			//new_size = len(new_sigmas)
			new_size = new_sigmas.size();
		}
		
		//U = np.empty((new_size, new_size))
		matU.resize(new_size, new_size); matU.setZero();
		
		/*if new_col:
			V = np.empty((new_size, new_size))
		else:
			V = np.empty((new_size + 1, new_size))*/
		if (new_col)
		{
			matV.resize(new_size, new_size); matV.setZero();
		} 
		else
		{
			matV.resize(new_size + 1, new_size); matV.setZero();
		}
		
		//for i in xrange(0, len(new_sigmas)) :
		Q_ASSERT(m_vec.size() == sigmas.size());
		for (int i = 0; i < new_sigmas.size();++i)
		{
			//tmp1 = m_vec / ((sigmas - new_sigmas[i]) * (sigmas + new_sigmas[i])) # unnormalized left sv
			MyVector tmp1; tmp1.resize(m_vec.size());
			for (int k = 0; k < m_vec.size();++k)
			{
				/*printInfo(VARNAME(m_vec[k]), m_vec[k]);
				printInfo(VARNAME(sigmas[k]), sigmas[k]);
				printInfo(VARNAME(new_sigmas[k]), new_sigmas[k]);
				printInfo(VARNAME(tmp1[k]), tmp1[k]);*/
				tmp1[k] = m_vec[k] / ((sigmas[k] - new_sigmas[i])*(sigmas[k] + new_sigmas[i]));
			}
			//printInfo(VARNAME(tmp1), tmp1.transpose());
			//if np.any(np.isinf(tmp1)) :
			const  std::vector<bool>& constBoolVec = isinf(tmp1);
			if (ifany(/*isinf(tmp1)*/constBoolVec))
			{
				//#tmp1[:] = 0
				//#tmp1[i] = 1
				if (new_sigmas[i] < epsilon1) //# new singular value is zero
				{
					//tmp1[np.isinf(tmp1)] = 0
					setArrayValueWithSpecialIndexes(tmp1, constBoolVec, 0.0);
					
				}
				else
				{
					//# we can not determine the value to put instead of infinity.Hence,
					//# other property is used to do it.I.e.scalar product of tmp1 and m_vec must equal - 1.
					
					//nonzero_inds = np.nonzero(np.isinf(tmp1))[0]
					const  MyIntVector nonzero_inds = nonzero(constBoolVec);
					//if len(nonzero_inds) == 1:
					if (nonzero_inds.size() == 1)
					{
						//tmp1[nonzero_inds] = 0
						//tmp1[nonzero_inds] = (-1 - np.dot( tmp1, m_vec)) / m_vec[nonzero_inds]
						setArrayValueWithSpecialIndexes(tmp1, nonzero_inds, 0.0);
						tmp1[nonzero_inds[0]] = (-1 - tmp1.dot(m_vec)) / m_vec[nonzero_inds[0]]; // mark! nonzero_inds.size() == 1
					}
					else
					{
						//raise ValueError("Unhandeled case 1")
						MyError("Unhandeled case 1");
					}
				}
			}
			const  std::vector<bool>& constBoolVec_isnan = isnan(tmp1);
			if (ifany(/*isinf(tmp1)*/constBoolVec_isnan)) //if np.any(np.isnan(tmp1)) : # temporary check
			{
				//#pass # For debugging
				//raise ValueError("Unhandeled case 2")
				MyError("Unhandeled case 2");
			}

			//nrm = sp.linalg.norm(tmp1, ord = 2)
			MyFloat nrm = tmp1.norm();
			//U[:, i] = tmp1 / nrm
			matU.col(i) = tmp1 / nrm;

			//tmp2 = tmp1 * sigmas# unnormalized right sv
			MyVector tmp2 = tmp1.cwiseProduct(sigmas);

			if (new_col)
			{
				//tmp2[-1] = -1
				Q_ASSERT(tmp2.size() > 0);
				Q_ASSERT(matV.rows() == tmp2.size());
				tmp2[tmp2.size() - 1] = -1;
				//nrm = sp.linalg.norm(tmp2, ord = 2)
				nrm = tmp2.norm();
				//V[:, i] = tmp2 / nrm
				matV.col(i) = tmp2 / nrm;
			} 
			else
			{
				//MAT_INFO(matV);
				//MAT_INFO(tmp2);
				Q_ASSERT(matV.rows() == (tmp2.size()+1) );
				matV.col(i).block(0, 0, tmp2.size(), 1) = tmp2; //V[0:-1, i] = tmp2
				matV.coeffRef(tmp2.size(), i) = -1; //V[-1, i] = -1
				nrm = matV.col(i).norm();//	nrm = sp.linalg.norm(V[:, i], ord = 2)
				matV.col(i) /= nrm;//	V[:, i] = V[:, i] / nrm
			}//if
		}//for

		//del tmp1, tmp2, nrm
		if (equal_sigmas)//if equal_sigmas:
		{
			const MyInt extra_sigma_size = orig_sigmas_length - uniq_sig_num;
			MyMatrix eye = MyMatrix::Identity(extra_sigma_size, extra_sigma_size);// eye = np.eye(extra_sigma_size)
			U_eq = U_eq.transpose();//U_eq = U_eq.T
			std::cout << "U_eq.size() " << U_eq.rows() << " , " << U_eq.cols() << std::endl;
			std::cout << "orig_sigmas_length = " << orig_sigmas_length << std::endl;
			std::cout << "uniq_sig_num = " << uniq_sig_num << std::endl;
			MyError("");
			//U = np.dot(U_eq, sp.linalg.block_diag(U, eye))
			matU = U_eq * my_block_diag(matU, eye);			
			if (new_col)//if new_col:
			{
				//V = np.dot( U_eq, sp.linalg.block_diag( V, eye ) )
				matV = U_eq * my_block_diag(matV, eye);
			}
			else
			{
				//V = sp.linalg.block_diag(V, eye)
				matV = my_block_diag(matV, eye);
				//P1 = np.eye(orig_sigmas_length, orig_sigmas_length + 1)
				MyMatrix tmpP1 = my_eye(orig_sigmas_length, orig_sigmas_length + 1);
				MyMatrix P1;
				{
					//P1 = np.insert(P1, uniq_sig_num, np.array((0.0, )* (orig_sigmas_length)+(1.0, )), axis = 0)				
					MyVector tmpRow(orig_sigmas_length + 1); tmpRow.setZero(); tmpRow[orig_sigmas_length] = 1.0;
					P1 = my_insert_row(tmpP1, uniq_sig_num, tmpRow);
					/*MyMatrix P1(orig_sigmas_length + 1, orig_sigmas_length + 1);
					P1.block(0, 0, uniq_sig_num, orig_sigmas_length + 1) = tmpP1.block(0, 0, uniq_sig_num, orig_sigmas_length + 1);
					P1.block(uniq_sig_num, 0, 1, orig_sigmas_length + 1).setZero();
					P1.block(uniq_sig_num, 0, 1, orig_sigmas_length + 1).coeffRef(0, orig_sigmas_length) = 1.0;
					P1.block(uniq_sig_num + 1, 0, orig_sigmas_length - uniq_sig_num, orig_sigmas_length + 1) = tmpP1.block(uniq_sig_num, 0, orig_sigmas_length - uniq_sig_num, orig_sigmas_length + 1);*/
				}
				{
					//U_eq = np.hstack((U_eq, np.array((0.0, )*U_eq.shape[0], ndmin = 2).T))
					MyVector tmpCol(U_eq.rows()); tmpCol.setZero();
					U_eq = my_hstack(U_eq, tmpCol);
				}
				{
					//U_eq = np.vstack((U_eq, np.array((0.0, )*U_eq.shape[0] + (1.0, ), ndmin = 2)))
					MyVector tmpRow(U_eq.rows() + 1); tmpRow.setZero(); tmpRow[U_eq.rows()] = 1.0;
					U_eq = my_vstack(U_eq, tmpRow);
				}
				
				//V = np.dot(U_eq, np.dot(P1.T, V))
				matV = U_eq * (P1.transpose() * matV);				
			}
			//perm,new_sigmas = _arrays_merge( new_sigmas, extra_sigmas )
			MyVector new_sigmas_ret;
			std::vector< MyInt > perm;
			_arrays_merge(new_sigmas, extra_sigmas, perm, new_sigmas_ret);

			//new_sigmas = np.array(new_sigmas)
			new_sigmas = new_sigmas_ret;

			//U = U[:, perm] # replace columns
			//V = V[:, perm] # replace columns
			std::vector< MyVector > vecU, vecV;
			for (iterAllOf(itr,perm))
			{
				MyInt nColId = *itr;
				vecU.push_back(matU.col(nColId));
				vecV.push_back(matV.col(nColId));
			}

			MyInt nMatU_Rows = matU.rows();
			MyInt nMatV_Rows = matV.rows();
			matU.setZero(nMatU_Rows, perm.size());
			matV.setZero(nMatV_Rows, perm.size());

			for (int m = 0; m < perm.size();++m)
			{
				matU.col(m) = vecU[m];
				matV.col(m) = vecV[m];
			}
		}
	}

	

	const MyMatrix SVD_updater::my_hstack(const MyMatrix& tmpMat, const MyVector& tmpCol)
	{
		MyMatrix retMat(tmpMat);
		retMat.conservativeResize(Eigen::NoChange, tmpMat.cols() + 1);
		retMat.col(retMat.cols()-1) = tmpCol;
		//retMat.block(0, tmpMat.cols(), tmpMat.rows(),1) = tmpCol;
		return retMat;
	}

	const MyMatrix SVD_updater::my_vstack(const MyMatrix& tmpMat, const MyVector& tmpRow)
	{
		MyMatrix retMat(tmpMat);
		retMat.conservativeResize(tmpMat.rows() + 1, Eigen::NoChange);
		retMat.row(retMat.rows()-1) = tmpRow.transpose();
		//retMat.block(tmpMat.rows(), 0, 1, tmpMat.cols()) = tmpRow.transpose();
		return retMat;
	}

	MyFloat  SVD_updater::my_random_sample()
	{
		boost::mt19937 rng(time(0));

		// 2. uniform_01  
		boost::uniform_01<boost::mt19937&> u01(rng);
		return u01();
	}

	MyVector SVD_updater::my_random_sample(const MyInt nSize)
	{
		MyVector retV;
		retV.resize(nSize);
		boost::mt19937 rng(time(0));

		// 2. uniform_01  
		boost::uniform_01<boost::mt19937&> u01(rng);
		for (int i = 0; i < nSize; ++i)
		{
			retV[i] = u01();
			//std::cout << u01() << std::endl;
		}
		return retV;
	}

	MyMatrix SVD_updater::my_random_sample(const MyInt nRows, const MyInt nCols)
	{
		MyMatrix retM;
		retM.resize(nRows, nCols);
		boost::mt19937 rng(time(0));

		// 2. uniform_01  
		boost::uniform_01<boost::mt19937&> u01(rng);
		for (int r = 0; r < nRows; ++r)
		{
			for (int c = 0; c < nCols;++c)
			{
				retM.coeffRef(r,c) = u01();;
			}
		}
		return retM;
	}

	void SVD_updater::my_sort_vector(MyVector& vec)
	{
		std::sort(vec.derived().data(), vec.derived().data() + vec.derived().size());
	}

	const MyMatrix& SVD_updater::my_insert_row(const MyMatrix& tmpP1, const MyInt uniq_sig_num, const MyVector& tmpRow)
	{		
		const MyInt nCols = tmpP1.cols();
		const MyInt nRows = tmpP1.rows();

		Q_ASSERT(nCols == tmpRow.size());
		MyMatrix P1(nRows + 1, nCols);
		P1.block(0, 0, uniq_sig_num, nCols) = tmpP1.block(0, 0, uniq_sig_num, nCols);
		P1.block(uniq_sig_num, 0, 1, nCols) = tmpRow.transpose();
		P1.block(uniq_sig_num + 1, 0, nRows - uniq_sig_num, nCols) = tmpP1.block(uniq_sig_num, 0, nRows - uniq_sig_num, nCols);
		return P1;
	}

	const MyMatrix& SVD_updater::my_eye(const MyInt nRows, const MyInt nCols)
	{
		return MyMatrix::Identity(nRows, nCols);
	}

	const MyMatrix& SVD_updater::my_eye(const MyInt nSize)
	{
		return MyMatrix::Identity(nSize, nSize);
	}

	const MyMatrix& SVD_updater::my_block_diag(const MyMatrix& mat1, const MyMatrix& mat2)
	{
		MyMatrix block_diag;
		const MyInt block_diag_size = mat1.rows() + mat2.rows();
		block_diag.resize(block_diag_size, block_diag_size);
		block_diag.setZero();
		block_diag.block(0, 0, mat1.rows(), mat1.cols()) = mat1;
		block_diag.block(mat1.rows(), mat1.cols(), mat2.rows(), mat2.cols()) = mat2;
		return block_diag;
	}

	const MyIntVector& SVD_updater::nonzero(const std::vector<bool>& constBoolVec)
	{
		MyIntVector ret;
		ret.resize(constBoolVec.size());
		
		int index = 0;
		for (int i = 0; i < constBoolVec.size(); ++i)
		{
			if (constBoolVec[i])
			{
				ret[index++] = i;
			}
		}
		ret.conservativeResize(index);
		return ret;
	}

	void SVD_updater::setArrayValueWithSpecialIndexes(MyVector& tmpVec, const MyIntVector& constIntVec, const MyFloat val)
	{
		for (int i = 0; i < constIntVec.size(); ++i)
		{
			tmpVec[constIntVec[i]] = val;
		}
	}

	void SVD_updater::setArrayValueWithSpecialIndexes(MyVector& tmpVec, const std::vector<bool>& constBoolVec, const MyFloat val)
	{
		Q_ASSERT(tmpVec.size() == constBoolVec.size());
		for (int i = 0; i < constBoolVec.size(); ++i)
		{
			if (constBoolVec[i])
			{
				tmpVec[i] = val;
			}
		}
	}

	bool SVD_updater::ifany(const std::vector<bool>& tmpBoolVec)
	{
		bool ret = false;
		for (int i = 0; i<tmpBoolVec.size();++i)
		{
			ret = ret || (tmpBoolVec[i]);
		}
		return ret;
	}

	bool SVD_updater::ifany(const MyIntVector& tmpIntVec)
	{
		return tmpIntVec.colwise().sum().value() != 0;
	}

	const std::vector<bool>& SVD_updater::isinf(const MyVector& tmpVec)
	{
		std::vector<bool> ret(tmpVec.size());
		for (int i = 0; i < tmpVec.size();++i)
		{
			//ret[i]  = (std::isinf(tmpVec[i])) ? 1 : 0;
			ret[i] = (std::isinf(tmpVec[i]));
		}
		return ret;
	}

	const std::vector<bool>& SVD_updater::isnan(const MyVector& tmpVec)
	{
		std::vector<bool> ret(tmpVec.size());
		for (int i = 0; i < tmpVec.size(); ++i)
		{
			//ret[i]  = (std::isinf(tmpVec[i])) ? 1 : 0;
			ret[i] = (std::isnan(tmpVec[i]));
		}
		return ret;
	}

	/*
	This is the function which updates SVD decomposition by one column.
	In real situation SVD_updater class is more preferable to use, because
	it is intended for continuous updating and provides some additional
	features.

	Function which updates SVD decomposition of A, when new column a_col is
	added to the matrix. Actually a_col can be a new row as well. The only
	requirement is that if A has size (m*n) then m >= n. Otherwise, error
	is raised.

	Inputs:
	U,S,Vh - thin SVD of A, which is obtained e.g from scipy.linalg.svd
	S - is a vector of singular values.

	a_col - vector with new column (or row of A)
	a_col_col - True if a_col a column, False - if it is row

	Outputs:
	U, new_sigmas, Vh - new thin SVD of [A,a_col]
	*/
	void SVD_updater::update_SVD(MyMatrix& U, MyVector& S, MyMatrix& Vh, const MyVector& a_col, bool a_col_col)
	{
		MyVector s_vec;
		MyFloat s_mu;
		MyMatrix s_U1, s_V1;
		MyVector& new_sigmas = S;
		//U_shape = U.shape
		Eigen::Vector2i U_shape; U_shape << U.rows(), U.cols();
		//Vh_shape = Vh.shape
		Eigen::Vector2i Vh_shape; Vh_shape << Vh.rows(), Vh.cols();
		
		if (Vh_shape[0] < Vh_shape[1])//if Vh_shape[0] < Vh_shape[1]:
		{
			//raise ValueError("Function update_SVD: Number of columns in V - %i is larger than  the number of rows - %i" % Vh_shape[:: - 1])
			MyError("Function update_SVD: Number of columns in V - %i is larger than  the number of rows - %i");
		}
		if (a_col_col && U_shape[0] != a_col.size())//if a_col_col and(U_shape[0] != a_col.size) :
		{
			MyError("Function update_SVD: Matrix column size - %i and new column size %i mismatch.");
		}
		if (a_col_col && U_shape[0] == Vh_shape[1])//if a_col_col and(U_shape[0] == Vh_shape[1]) :
		{
			MyError("Function update_SVD:  Column can't be added to the square matrix set a_col_col=False instead.");
		}
		if (!a_col_col && U_shape[1] != a_col.size()) //if not a_col_col and(U_shape[1] != a_col.size) :
		{
			MyError("Function update_SVD: Matrix row size - %i and new row size %i mismatch.");
		}
		//# Old epsilon :
		//#zero_epsilon = np.sqrt(np.finfo(np.float64).eps * U.shape[0] * *2 * Vh.shape[1] * 10 * 2)
		//# Test epsilon
		//zero_epsilon = np.finfo(np.float64).eps * np.sqrt(U.shape[0]) * Vh.shape[1] * sp.linalg.norm(a_col, ord = 2) * 2 
		//# epsilon to determine when rank not increases
		const MyFloat zero_epsilon = std::numeric_limits< MyFloat >::epsilon() *  std::sqrt(U_shape[0]) * Vh_shape[1] * a_col.norm() * 2;
		Eigen::Vector2i a_col_old_shape; a_col_old_shape << a_col.rows(), a_col.cols();
		//a_col.shape = (a_col.shape[0],) if (len(a_col.shape) == 1) else ( max(a_col.shape), )
		if (a_col_col) // if a_col_col: # new column
		{
			const MyInt old_size = U_shape[1];
			//m_vec = np.dot(U.T, a_col) # m vector in the motes
			s_vec = U.transpose() * a_col;
			//new_u = a_col - np.dot(U, m_vec) # unnormalized new left eigenvector
			MyVector new_u = a_col - U * s_vec;
			//mu = sp.linalg.norm(new_u, ord=2)
			s_mu = new_u.norm();

			//Vh = np.hstack((Vh, np.array((0.0, )*old_size, ndmin = 2).T))
			MyVector tmpVec1(old_size); tmpVec1.setZero();
			Vh = my_hstack(Vh, tmpVec1);
			//Vh = np.vstack((Vh, np.array((0.0, )*old_size + (1.0, ), ndmin = 2)))
			MyVector tmpVec2(old_size + 1); tmpVec2.setZero(); tmpVec2[old_size] = 1.0;
			Vh = my_vstack(Vh, tmpVec2);

			if (s_mu < zero_epsilon) //if (np.abs(mu) < zero_epsilon) : # new column is from the same subspace as the old column
			{
				//# rank is not increased.
				//U1, new_sigmas, V1 = _SVD_upd_diag(S, m_vec, new_col = False)
				
				_SVD_upd_diag(S, s_vec, false/*new_col = False*/, s_U1, new_sigmas, s_V1);
				//# This block adds zero to singular value and modify matrices
				//# accordingly.It can be removed if needed ->

				//new_sigmas = np.concatenate((new_sigmas, (0.0, )))
				new_sigmas.conservativeResize(new_sigmas.size() + 1);
				new_sigmas[new_sigmas.size() - 1] = 0.0;
				
				//U1 = np.hstack((U1, np.zeros((U1.shape[0], 1))))
				MyVector tmpVec3(s_U1.rows()); tmpVec3.setZero();
				s_U1 = my_hstack(s_U1, tmpVec3);
				
				//V1 = np.hstack((V1, np.zeros((V1.shape[0], 1))))
				MyVector tmpVec4(s_V1.rows()); tmpVec4.setZero();
				s_V1 = my_hstack(s_V1, tmpVec4);
				//# <-
			}
			else
			{
				//U = np.hstack((U, new_u[:, np.newaxis] / mu))
				concatenateMat_OneColumn(U, new_u / s_mu);
				//S = np.concatenate((S, (0.0, )))
				S.conservativeResize(S.size() + 1);
				S[S.size() - 1] = 0.0;
				//m_vec = np.concatenate((m_vec, (mu, )))
				s_vec.conservativeResize(s_vec.size() + 1);
				s_vec[s_vec.size() - 1] = s_mu;
				//U1, new_sigmas, V1 = _SVD_upd_diag(S, m_vec, new_col = True)*/
				
				_SVD_upd_diag(S, s_vec, true, s_U1, new_sigmas, s_V1);
			}

			//U = np.dot(U, U1)
			U = U*s_U1;
			//Vh = np.dot(V1.T, Vh) # V matrix.Need to return V.T though
			Vh = s_V1.transpose() * Vh;
		}
		else
		{
			MyError("unsupport add rows!");
		}

		//a_col.shape = a_col_old_shape
		//return U, new_sigmas, Vh
	}

	void sp_linalg_svd(const MyMatrix& A, MyMatrix& U, MyVector& S, MyMatrix& Vh)
	{
		Eigen::JacobiSVD< MyMatrix > svds_thin(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
		U = svds_thin.matrixU();
		S = svds_thin.singularValues();
		Vh = svds_thin.matrixV().transpose();
	}



	
#endif//SingleDomainISVD
}