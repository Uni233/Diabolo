#include "MyISVD.h"
#include <iostream>
#include <fstream>
/************************************************************************/
/* Test SVD Update Function                                             */
/************************************************************************/
extern double randomData[10][100];
namespace YC
{
	const MyMatrix my_create_A(const MyInt nRows, const MyInt nCols)
	{
		MyMatrix retM;
		retM.resize(nRows,nCols);
		retM << 0.9056801, 0.48447475, 0.42011392, 0.30472733, 0.00843324, 0.6522033, 0.19413367, 0.23909341,
			0.3179708, 0.65649403, 0.30839429, 0.71380905, 0.49781292, 0.76306449, 0.64409656, 0.57922037,
			0.42942776, 0.88768028, 0.01815448, 0.97425654, 0.19886164, 0.57157693, 0.85045993, 0.87534483,
			0.00784726, 0.09032871, 0.41602674, 0.64684495, 0.31079271, 0.50564065, 0.3271686, 0.06235824,
			0.09461475, 0.48460725, 0.08556322, 0.01336891, 0.59693139, 0.41763722, 0.35913776, 0.05314399,
			0.73321036, 0.53093696, 0.76043148, 0.19416429, 0.40289542, 0.62120059, 0.07687949, 0.53588116,
			0.37180105, 0.39653784, 0.15216076, 0.36789635, 0.22628061, 0.26534357, 0.40251042, 0.58034583,
			0.50419113, 0.12409339, 0.39753808, 0.79468386, 0.12654629, 0.75420829, 0.33952077, 0.16471485,
			0.1353444, 0.96982062, 0.41970288, 0.0710964, 0.92442615, 0.01142337, 0.13947985, 0.02317687,
			0.45948046, 0.33613392, 0.50287407, 0.99777036, 0.61062965, 0.86019752, 0.7628495, 0.64967171;

		return retM;
	}

	const MyVector my_create_a1(const MyInt nRows)
	{
		MyVector retV;
		retV.resize(nRows);
		retV << 1.26827339, 2.29286328, 2.52623203, 1.28984524, 1.11845847, 1.73974328, 1.39673562, 1.38192792, 1.74378424, 2.58102257;
		return retV;
	}

	const MyVector my_create_a2(const MyInt nRows)
	{
		MyVector retV;
		retV.resize(nRows);
		retV << 0.93164404, 0.45368118, 0.07903227, 0.6739326, 0.17897846, 0.35516663, 0.81410867, 0.36931628, 0.6929429, 0.66749607;
		return retV;
	}

	

	void test_SVD_updater()
	{
		/*
		#    # Test column update. Thin matrix, rank increases
		#    A = rnd.rand(5,3)
		#    (U,S,Vh) = sp.linalg.svd( A , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )
		#
		#    a1 = rnd.rand(5,1)
		#    A = np.hstack( (A,a1) )
		#    (Ut1,St1,Vht1) = sp.linalg.svd( A  , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )
		#
		#    SVD_upd = SVD_updater( U,S,Vh, update_V = True, reorth_step=10)
		#
		#    (Us1, Ss1, Vhs1) = SVD_upd.add_column(a1 )
		#    Ar1 = np.dot( Us1, np.dot(np.diag(Ss1), Vhs1 ) )
		#
		#    diff1 = np.max( np.abs( A - Ar1) )/St1[0]
		#
		#    a2 = rnd.rand(5,1)
		#    A = np.hstack( (A,a2) )
		#
		#    (Us2, Ss2, Vhs2) = SVD_upd.add_column( a2 )
		#
		#    Ar2 = np.dot( Us2, np.dot(np.diag(Ss2), Vhs2 ) )
		#
		#    diff2 = np.max( np.abs( A - Ar2) )/St1[0]
		*/
		std::ofstream outfile("d:/myisvd.txt");
		//#Test column update.Thin matrix, rank not increases
		const MyInt n_rows = 10;
		const MyInt n_cols = 8;
		//MyMatrix A = MyMatrix::Random(n_rows, n_cols);//	A = rnd.rand(n_rows, n_cols)
		MyMatrix A;
		A = my_create_A(n_rows, n_cols);//	A = rnd.rand(n_rows, n_cols)
		MyMatrix U, Vh; MyVector S;
		sp_linalg_svd(A, U, S, Vh);
		/*printInfo(outfile, VARNAME(U),U);
		printInfo(outfile, VARNAME(S), S);
		printInfo(outfile, VARNAME(Vh), Vh);*/
		
		//(U, S, Vh) = sp.linalg.svd(A, full_matrices = False, compute_uv = True, overwrite_a = False, check_finite = False)

		//##a1 = rnd.rand(5,1)
		//MyVector a1 = A * MyVector::Random(n_cols);//	a1 = np.dot(A, rnd.rand(n_cols, 1))
		MyVector a1;
		a1 = my_create_a1(n_rows);
		A = SVD_updater::my_hstack(A, a1);//	A = np.hstack((A, a1))
		MyMatrix Ut, Vht; MyVector St;
		sp_linalg_svd(A, Ut, St, Vht);//	(Ut, St, Vht) = sp.linalg.svd(A, full_matrices = False, compute_uv = True, overwrite_a = False, check_finite = False)
		/*printInfo(outfile, VARNAME(Ut), Ut);
		printInfo(outfile, VARNAME(St), St);
		printInfo(outfile, VARNAME(Vht), Vht);*/
		
		//
		SVD_updater SVD_upd(outfile);
		SVD_upd.initial(U, S, Vh,true,10);//	SVD_upd = SVD_updater(U, S, Vh, update_V = True, reorth_step = 10)

		MyMatrix Us1, Vhs1; MyVector Ss1;
		SVD_upd.add_column(a1);
		SVD_upd.get_current_svd(Us1, Ss1, Vhs1); //(Us1, Ss1, Vhs1) = SVD_upd.add_column(a1)
		printInfo(outfile, VARNAME(Us1), Us1);
		printInfo(outfile, VARNAME(Ss1), Ss1);
		printInfo(outfile, VARNAME(Vhs1), Vhs1);
		
		MyMatrix Ar1 = Us1 * (Ss1.asDiagonal() * Vhs1);	//Ar1 = np.dot(Us1, np.dot(np.diag(Ss1), Vhs1))

		const MyFloat maxCoeff1 = (A - Ar1).cwiseAbs().maxCoeff();
		const MyFloat diff1 = maxCoeff1 / St[0];// np.max(np.abs(A - Ar1)) / St[0]
		std::cout << "maxCoeff1 = " << maxCoeff1 << std::endl;
		std::cout << "diff1 = " << diff1 << std::endl;
		//MyVector a2 = MyVector::Random(n_rows);// rnd.rand(n_rows, 1)
		MyVector a2;
		a2 = my_create_a2(n_rows);
		//#a2 = np.dot(A, np.array([2, 1, 4, -3], ndmin = 2).T)
		A = SVD_updater::my_hstack(A, a2);//	A = np.hstack((A, a1))	A = np.hstack((A, a2))
		sp_linalg_svd(A, Ut, St, Vht);//	(Ut, St, Vht) = sp.linalg.svd(A, full_matrices = False, compute_uv = True, overwrite_a = False, check_finite = False)
		
		MyMatrix Us2, Vhs2; MyVector Ss2;
		SVD_upd.add_column(a2);//	(Us2, Ss2, Vhs2) = SVD_upd.add_column(a2)
		SVD_upd.get_current_svd(Us2, Ss2, Vhs2);
		printInfo(outfile, VARNAME(Us2), Us2);
		printInfo(outfile, VARNAME(Ss2), Ss2);
		printInfo(outfile, VARNAME(Vhs2), Vhs2);
		MyMatrix Ar2 = Us2 * (Ss2.asDiagonal() * Vhs2);	//Ar2 = np.dot(Us2, np.dot(np.diag(Ss2), Vhs2))

		const MyFloat diff2 = (A - Ar2).cwiseAbs().maxCoeff() / St[0];//np.max(np.abs(A - Ar2)) / St[0]

		
		std::cout << "diff2 = " << diff2 << std::endl;

		outfile.close();
	}

	/*
	Function which test root finder function.
	*/
	void test_root_finder()
	{
		/*times_root_1 = []
		times_root_2 = []
		times_root_3 = []

		max_diff_1 = []
		max_diff_2 = []

		sigma_sizes = []*/
		std::vector< double > times_root_1, times_root_2, times_root_3;
		MyVector max_diff_1, max_diff_2;
		std::vector<MyInt> sigma_sizes;

		for (int k = 0; k < 1;++k)
		{
			/*
			print k
			sigma_size = 100 + 100 * k
			sigma_sizes.append(sigma_size)
			mult_factor = 100
			*/
			std::cout << "k = " << k << std::endl;
			MyInt sigma_size = 100 + 100 * k;
			sigma_sizes.push_back(sigma_size);
			MyInt mult_factor = 100;

			/*sigmas = rnd.random_sample(sigma_size) * mult_factor
			m_vec = rnd.random_sample(sigma_size) * mult_factor*/
			//MyVector sigmas = SVD_updater::my_random_sample(sigma_size) * mult_factor;
			MyVector sigmas;
			sigmas.resize(sigma_size);
			sigmas << /*98.2880744, 97.1442958, 96.8464834, 96.3860476, 96.1262223
				, 95.8770117, 95.7409063, 93.7847375, 93.5669708, 93.1420133
				, 92.7171852, 91.1481592, 90.4044675, 89.1193891, 84.5507709
				, 84.4541526, 83.1839751, 81.7021745, 81.5639174, 79.7505152
				, 77.6859768, 75.9878433, 75.9694421, 75.0990273, 73.8706885
				, 73.7367515, 73.3208083, 73.1865427, 71.3268534, 70.5483073
				, 70.4158118, 70.2643088, 67.9238619, 67.1019850, 66.3305113
				, 65.7997611, 65.3755813, 65.2549611, 65.2513975, 62.8801924
				, 62.3930343, 60.3248426, 59.9713075, 59.7096080, 58.4133536
				, 58.0868311, 55.2070359, 54.4791492, 53.5732034, 52.6781692
				, 51.6082451, 49.8642431, 48.3652907, 47.2438931, 43.7463315
				, 42.8584396, 41.6323478, 40.4740606, 39.6170377, 38.0082660
				, 37.6083908, 36.4104976, 35.6256173, 34.6309426, 33.9479688
				, 30.2791012, 27.8031205, 25.6247694, 24.6457723, 24.4078890
				, 23.0880613, 21.4876384, 20.5666168, 18.7040410, 16.2958437
				, 16.0106974, 15.1567213, 14.2345796, 13.7382814, 11.1290483
				, 10.8616881, 10.4031528, 9.98769497, 9.92026600, 9.77219414
				, 8.50470693, 8.48750783, 8.46829927, 8.36015403, 7.85279907
				, 7.81145898, 6.63767275, 6.61695617, 6.50959155, 5.82547603
				, 5.12341546, 4.30182597, 3.36762766, 2.89635290, 0.90699739*/
				99.8082269, 99.4848635, 99.4846131, 97.0484007, 89.0102277
				, 87.7820039, 87.0290665, 86.4176157, 86.4054287, 85.4708987
				, 82.3081512, 81.1205467, 80.2804562, 79.5385221, 78.1606819
				, 77.3606844, 76.4060016, 74.6260955, 73.9304644, 73.1443053
				, 72.1806508, 71.5697897, 70.7464372, 70.2902130, 69.2292878
				, 68.4806801, 68.2169537, 68.1448861, 67.8756066, 67.3696120
				, 66.4557082, 65.8797043, 64.3193958, 63.4318661, 62.6373900
				, 60.7634310, 60.0325111, 59.9653160, 58.5282379, 58.0767718
				, 58.0402851, 57.8523671, 57.3796977, 56.5908738, 55.0878900
				, 54.2919202, 52.8753610, 51.0434421, 49.0355951, 47.8786942
				, 46.7943297, 44.5987708, 44.1461570, 44.0130046, 43.6976245
				, 41.7445795, 41.2105363, 41.0778699, 40.7664479, 40.7056873
				, 40.5607847, 38.6034755, 36.8351618, 35.7342967, 35.5354068
				, 34.9337039, 32.9472981, 32.8569442, 31.4635378, 30.1211937
				, 29.6713512, 28.2272398, 28.0484934, 27.6164631, 26.8820505
				, 23.4592140, 22.3014409, 21.1496200, 19.8395463, 19.7490998
				, 17.7587208, 16.8562351, 16.6119016, 16.4307944, 13.7759935
				, 12.8006643, 12.2355944, 11.6582895, 11.5892011, 10.9680432
				, 10.3930190, 9.70427276, 8.22101144, 7.91073840, 7.47631989
				, 5.89514806, 5.58442716, 3.81370706, 2.68193061, 0.34659889;

			//MyVector m_vec = SVD_updater::my_random_sample(sigma_size) * mult_factor;
			MyVector m_vec;
			m_vec.resize(sigma_size);
			m_vec << /*43.3046333, 83.0880392, 5.94821087, 68.0995818, 37.4778469
				, 80.7476220, 62.7396150, 13.6658981, 91.8961501, 31.7669015
				, 66.6799599, 52.3273701, 56.7000854, 9.69181281, 47.9118910
				, 33.7163064, 36.9934362, 65.8741128, 91.5695755, 46.6713912
				, 19.4441218, 4.85834789, 3.24627800, 32.8703582, 25.4485176
				, 27.4756595, 76.4655190, 91.0869932, 89.1995355, 8.46623558
				, 32.7444092, 96.0027814, 3.93820535, 43.9847335, 87.8832238
				, 86.7089406, 23.5383858, 60.4223898, 26.7127823, 52.3914933
				, 13.1045632, 18.5248031, 46.1620860, 13.2896289, 8.23388170
				, 39.5619300, 55.6169302, 1.06425410, 83.5709253, 31.4598593
				, 59.1592760, 48.5621509, 86.5078015, 44.3979814, 23.9540253
				, 94.0136297, 36.1607226, 23.0058720, 28.0775063, 91.1361513
				, 9.69656249, 67.3192931, 71.5057634, 10.3081576, 3.16085965
				, 11.0070465, 69.0700229, 24.9530156, 8.05659981, 13.2580279
				, 7.38026908, 85.5550283, 27.0427612, 21.3923259, 30.2566923
				, 95.6270786, 48.8190725, 98.2287180, 89.4975773, 69.1143267
				, 51.4727873, 70.7336752, 30.5650896, 12.7031417, 0.94473049
				, 94.4744785, 85.0307736, 54.5119876, 1.94125792, 73.9272139
				, 45.8024923, 92.0975745, 5.52857282, 9.14396986, 68.4689814
				, 83.9134861, 31.0116722, 20.8640918, 54.5836449, 44.8920897*/
				82.2825563, 50.4249790, 66.2449477, 21.2139303, 21.9807626
				, 34.2845939, 95.3286964, 4.62493571, 77.7614150, 54.0195951
				, 23.4075400, 80.2826053, 40.9973746, 60.2871710, 4.12360768
				, 74.0269796, 2.21338334, 17.1586796, 8.01220290, 39.6990914
				, 86.5220729, 55.9435577, 41.1463938, 22.7431005, 81.5362352
				, 51.5285772, 81.1074991, 56.3722089, 21.3021116, 24.8004418
				, 67.9944373, 2.10031133, 46.0317260, 21.4659368, 25.7629270
				, 7.42512012, 59.1188465, 11.8353133, 79.7845993, 2.84773683
				, 74.7549343, 28.5049016, 14.9752719, 32.0487558, 60.0573142
				, 33.0530688, 53.7306110, 17.9775675, 91.9155888, 24.1804964
				, 53.2323849, 64.1800127, 29.3802915, 10.4051983, 80.5478140
				, 43.8942370, 11.0501651, 44.9121560, 43.9523638, 96.8405603
				, 7.63459126, 69.9744506, 38.7927445, 51.5237228, 87.6205303
				, 10.9744422, 38.9096586, 18.3683113, 60.4163647, 77.5882977
				, 81.5023297, 24.6998733, 15.2152245, 95.4353747, 54.9134151
				, 36.5284082, 98.6259492, 19.1306341, 45.0189803, 62.5319340
				, 91.3804874, 86.2535846, 4.17708774, 11.7211210, 2.23867564
				, 1.54742490, 18.4330188, 19.0498652, 75.7632282, 52.0708836
				, 36.5848751, 71.1429596, 56.2646329, 76.8106353, 68.8313066
				, 8.38021418, 39.6540953, 87.3741620, 27.0233113, 52.7657773;
			
			//#dk = sp.io.loadmat('root_find.mat')
			//#sigmas = dk['sigmas'].squeeze(); m_vec = dk['m_vec'].squeeze(); mu = dk['mu'][0]

			/*
			sigmas.sort()
			sigmas = sigmas[:: - 1]
			*/
			//SVD_updater::my_sort_vector(sigmas);
			//sigmas.reverseInPlace();

			/*with Timer() as t :
			roots1 = find_roots(sigmas, m_vec, method = 1)
			times_root_1.append(t.msecs) # roots by root finder*/
			MyVector roots1;
			{
				MyTimer t;
				SVD_updater::find_roots(sigmas, m_vec, roots1);
				times_root_1.push_back(t.elapsed());
			}
			std::cout << roots1.transpose() << std::endl;
		}
	}

	MyMatrix create_test_update_svd_data(const MyInt nIdx,const MyInt nRows, const MyInt nCols)
	{
		MyMatrix retM;
		retM.resize(nRows,nCols);
		if (1 == nIdx)
		{
			retM << 0.71965209, 0.10394377, 0.88086911, 0.82791074,
				0.63857792, 0.62124960, 0.46705838, 0.42050837,
				0.73374524, 0.43255784, 0.68638908, 0.43796660,
				0.15679188, 0.04476867, 0.74813049, 0.77997415,
				0.83885228, 0.53440803, 0.62440534, 0.38784760,
				0.00820651, 0.12026052, 0.74845083, 0.84675189,
				0.39083661, 0.62694114, 0.95087870, 0.99519200,
				0.46465162, 0.40674976, 0.79492681, 0.44919616,
				0.63373716, 0.82881898, 0.09403227, 0.70558578,
				0.63488707, 0.33284217, 0.71333861, 0.87266502,
				0.27826899, 0.31519030, 0.56345760, 0.50789607,
				0.30213376, 0.93559629, 0.99176407, 0.40260898,
				0.91452533, 0.34413869, 0.40817676, 0.92010464,
				0.14271376, 0.94762318, 0.53327804, 0.22086292,
				0.53903429, 0.83166506, 0.79240531, 0.91880181,
				0.46987234, 0.25129831, 0.49778708, 0.58563091,
				0.31572162, 0.86077692, 0.82396090, 0.75115188,
				0.23195010, 0.53486891, 0.54342924, 0.96245520,
				0.30450759, 0.10792104, 0.09553078, 0.64006642,
				0.05544778, 0.70773258, 0.02659925, 0.96948704;
		}
		else if (2 == nIdx)
		{
			retM << 0.90082042, 0.17081390, 0.94452281, 0.05422025, 0.74136480, 0.41927233, 0.00560871, 0.31846994, 0.65605938,
				0.33031394, 0.67160694, 0.77026873, 0.07547127, 0.96576604, 0.05336753, 0.62136159, 0.51688762, 0.72565831,
				0.97161003, 0.57356368, 0.60165495, 0.22493581, 0.68193657, 0.89401414, 0.02566363, 0.12831408, 0.73684449,
				0.36405218, 0.91513426, 0.39983060, 0.51412133, 0.05713298, 0.37765258, 0.72611466, 0.09842389, 0.05956541,
				0.87700976, 0.04838465, 0.60398718, 0.24821352, 0.53330711, 0.67519316, 0.43490997, 0.25220287, 0.37084888,
				0.44448756, 0.25421245, 0.48291743, 0.24625648, 0.66055068, 0.43648703, 0.18978911, 0.75950936, 0.16075723,
				0.58764903, 0.24962558, 0.69005727, 0.37472823, 0.91998376, 0.40480702, 0.04783547, 0.68667920, 0.68884991,
				0.72300277, 0.07656111, 0.15455384, 0.65568812, 0.92101915, 0.94223191, 0.30337258, 0.57901473, 0.00462090,
				0.57784045, 0.69324175, 0.34077294, 0.62697038, 0.44212460, 0.19898438, 0.78771975, 0.08769219, 0.64014214,
				0.65776032, 0.35966214, 0.05922354, 0.27878649, 0.86449540, 0.23754843, 0.01465197, 0.57285620, 0.44105562,
				0.78798387, 0.04954493, 0.50759931, 0.52687980, 0.45766193, 0.82607485, 0.96271377, 0.39273863, 0.68171236,
				0.13567813, 0.19473379, 0.72492957, 0.67372028, 0.32596748, 0.68868863, 0.15410596, 0.32186968, 0.52455684,
				0.75949299, 0.12981846, 0.98064110, 0.29556692, 0.95991444, 0.70191588, 0.64980372, 0.50582635, 0.93078056,
				0.48902985, 0.42445199, 0.02804902, 0.12204404, 0.76399666, 0.52806051, 0.04646705, 0.75683033, 0.18954409,
				0.51359509, 0.94741264, 0.39309703, 0.35909299, 0.92225753, 0.33517443, 0.50834815, 0.69874211, 0.71800244,
				0.30208566, 0.41238534, 0.77152994, 0.14075719, 0.38852076, 0.17288888, 0.02248815, 0.62611401, 0.09153190,
				0.19544746, 0.49209724, 0.25507068, 0.62021150, 0.07456781, 0.25573064, 0.15524513, 0.97811068, 0.01880327,
				0.04065320, 0.97175864, 0.21282540, 0.90883196, 0.68485195, 0.60050391, 0.50080395, 0.94265096, 0.00910681,
				0.58443365, 0.40415618, 0.02540060, 0.83749399, 0.66273653, 0.34637821, 0.60185735, 0.16334707, 0.71753336,
				0.78769216, 0.69432741, 0.22377304, 0.40940736, 0.57727517, 0.56679442, 0.35835413, 0.65398842, 0.79017684;
		} 
		else
		{
			MyError("create_test_update_svd_data Error!");
		}
		return retM;
	}

	MyVector create_test_update_svd_data(const MyInt nIdx, const MyInt nSize)
	{
		MyVector retV;
		retV.resize(nSize);
		if (1 == nIdx)
		{
			retV << 0.83515757, 0.19217845, 0.71657799, 0.53269153, 0.63328524,
				0.17910017, 0.66718438, 0.23185261, 0.07687934, 0.59879315,
				0.16041055, 0.93518715, 0.45175547, 0.65649621, 0.25263371,
				0.09190515, 0.23078254, 0.28304635, 0.79878967, 0.25087127;
		}
		else if (2 == nIdx)
		{
			retV << 0.68742843, 0.41961509, 0.59002201, 0.04174215, 0.38987400,
				0.46464132, 0.36735948, 0.88972388, 0.61896473, 0.25286243,
				0.39542374, 0.04998226, 0.38844746, 0.67633425, 0.86208829,
				0.68184973, 0.26127471, 0.96376092, 0.42262458, 0.21213216;
		}
		else
		{
			MyError("create_test_update_svd_data Error!");
		}
		return retV;
	}
	/*
	Test SVD update
	*/
	void test_update_svd(const MyInt n_rows, const MyInt start_n_col, const MyInt n_max_cols, const MyInt step_n_col)
	{
		/*
		update_time = []
		new_svd_time = []
		sing_val_diff = []
		left_sv_diff = []
		right_sv_diff = []
		left_orig_ort = []
		left_upd_ort = []
		right_orig_ort = []
		right_upd_ort = []
		*/
		std::vector< double > update_time, new_svd_time;
		std::vector< double > sing_val_diff, left_sv_diff, right_sv_diff;
		std::vector< double >left_orig_ort, left_upd_ort, right_orig_ort, right_upd_ort;

		//column_sizes = []; k = 0
		std::vector< MyInt > column_sizes; MyInt k=0;

		for (MyInt column_num = start_n_col; column_num < (n_max_cols + 1); column_num += step_n_col)
		{
			/*
			k += 1
			print k
			column_sizes.append(column_num)
			mult_factor = 1
			*/
			k += 1;
			std::cout << "k = " << k << std::endl;
			column_sizes.push_back(column_num);
			MyInt mult_factor = 1;

			/*
			matrix = rnd.random_sample((n_rows, column_num - 1)) * mult_factor
			new_col = rnd.random_sample(n_rows) * mult_factor
			*/
#if 1
			MyMatrix matrix = SVD_updater::my_random_sample(n_rows, column_num - 1) * mult_factor;
			MyVector new_col = SVD_updater::my_random_sample(n_rows) * mult_factor;
#else
			MyMatrix matrix = create_test_update_svd_data(k, n_rows, column_num - 1);
			MyVector new_col = create_test_update_svd_data(k, n_rows);
#endif
			
			//(um, sm, vm) = sp.linalg.svd(matrix, full_matrices = False, compute_uv = True, overwrite_a = False, check_finite = False)
			MyMatrix um, vm;
			MyVector sm;
			sp_linalg_svd(matrix, um, sm, vm);

			MyMatrix uu(um), vu(vm);
			MyVector su(sm);
			{
				MyTimer t;				
				//(uu, su, vu) = update_SVD(um, sm, vm, new_col, a_col_col = True)
				SVD_updater::update_SVD(uu, su, vu, new_col, true);
				update_time.push_back(t.elapsed());

				/*std::cout << "uu " << uu << std::endl;
				std::cout << "su " << su.transpose() << std::endl;
				std::cout << "vu " << vu << std::endl;
				MyPause;*/
			}

			MyMatrix uf, vf;
			MyVector sf;
			{
				MyTimer t;
				MyMatrix newMatrix = SVD_updater::my_hstack(matrix, new_col);
				sp_linalg_svd(newMatrix, uf, sf, vf);
				new_svd_time.push_back(t.elapsed());
			}

			//sing_val_diff.append(np.max(np.abs(su - sf)))
			sing_val_diff.push_back((su - sf).cwiseAbs().maxCoeff());
						
			//left_sv_diff.append(np.max(np.abs(uu) - np.abs(uf)))
			left_sv_diff.push_back((uu.cwiseAbs() - uf.cwiseAbs()).maxCoeff());

			//right_sv_diff.append(np.max(np.abs(vu) - np.abs(vf)))
			right_sv_diff.push_back((vu.cwiseAbs() - vf.cwiseAbs()).maxCoeff());

			//left_orig_ort.append(np.max(np.abs(np.dot(uf.T, uf) - np.eye(uf.shape[1]))))			
			left_orig_ort.push_back(((uf.transpose() * uf).cwiseAbs() - MyMatrix::Identity(uf.cols(), uf.cols())).cwiseAbs().maxCoeff());
			//left_upd_ort.append( np.max( np.abs( np.dot( uu.T, uu) - np.eye( uu.shape[1] ) ) ) )
			left_upd_ort.push_back(((uu.transpose() * uu).cwiseAbs() - MyMatrix::Identity(uu.cols(), uu.cols())).cwiseAbs().maxCoeff());

			//right_orig_ort.append(np.max(np.abs(np.dot(vf.T, vf) - np.eye(vf.shape[1]))))			
			right_orig_ort.push_back(((vf.transpose() * vf).cwiseAbs() - MyMatrix::Identity(vf.cols(), vf.cols())).cwiseAbs().maxCoeff());

			//right_upd_ort.append(np.max(np.abs(np.dot(vu.T, vu) - np.eye(vu.shape[1]))))			
			right_upd_ort.push_back(((vu.transpose() * vu).cwiseAbs() - MyMatrix::Identity(vu.cols(), vu.cols())).cwiseAbs().maxCoeff());
		}

		YC::printVector(VARNAME(sing_val_diff), sing_val_diff);
		YC::printVector(VARNAME(left_sv_diff), left_sv_diff);
		YC::printVector(VARNAME(right_sv_diff), right_sv_diff);
		YC::printVector(VARNAME(left_orig_ort), left_orig_ort);
		YC::printVector(VARNAME(left_upd_ort), left_upd_ort);
		YC::printVector(VARNAME(right_orig_ort), right_orig_ort);
		YC::printVector(VARNAME(right_upd_ort), right_upd_ort);

	}

	/*
	Test how the orthogonality property changes of updated SVD.

	Inputs:
	n_rows - how many rows in the initial matrix (this value is constant during iterations)
	start_n_col - how many columns in the initial matrix (columns are added sequentially)
	n_max_cols - numbers of columns to add.
	prob_same_subspace - probability that the column is from the current column subspace
	reorth_step - how often to do reorthogonalization.
	*/
	MyMatrix create_test_SVD_update_reorth_A()
	{
		MyMatrix retA;
		retA.resize(100, 1);
		retA << 0.20504751, 0.62461702, 0.24533112, 0.62166542, 0.08243618,
			0.90790515, 0.19928764, 0.77546985, 0.31632110, 0.99132378,
			0.51537571, 0.15631456, 0.38463115, 0.10108232, 0.42741865,
			0.69776152, 0.69389884, 0.76109079, 0.02059112, 0.53334490,
			0.37568794, 0.09814213, 0.51247313, 0.59663187, 0.90076234,
			0.16974630, 0.52689625, 0.75114047, 0.64562051, 0.52082982,
			0.92850371, 0.26756891, 0.60045564, 0.02082458, 0.21236487,
			0.23050767, 0.87649072, 0.37836699, 0.16731725, 0.18733381,
			0.70206472, 0.84600670, 0.70004331, 0.82887190, 0.05733586,
			0.40639001, 0.58748996, 0.34547096, 0.61686319, 0.68002707,
			0.16258392, 0.12445550, 0.45641318, 0.91735973, 0.43000519,
			0.91697308, 0.84943528, 0.52701441, 0.46665051, 0.52163081,
			0.61983114, 0.19170129, 0.74890275, 0.65616029, 0.82602449,
			0.74141743, 0.50651077, 0.54140798, 0.84114749, 0.77902512,
			0.97560365, 0.99675110, 0.11925482, 0.10223444, 0.54189706,
			0.42712467, 0.52909272, 0.11479223, 0.10212392, 0.54183666,
			0.21361821, 0.08711362, 0.10351850, 0.76925798, 0.06494626,
			0.41314377, 0.33037962, 0.87569492, 0.71138806, 0.45953190,
			0.60057219, 0.07529552, 0.85707864, 0.07447058, 0.20387754,
			0.66378801, 0.05097381, 0.74762842, 0.63626767, 0.96012246;
		return retA;
	}

	MyVector create_test_SVD_update_reorth_col(const MyInt idx)
	{
		MyVector retV;
		retV.resize(100);
		for (int i = 0; i < 100;++i)
		{
			retV[i] = randomData[idx][i];
		}
		return retV;
	}
	
	void test_SVD_update_reorth(const MyInt n_rows, const MyInt start_n_col, const MyInt n_max_cols, const MyFloat prob_same_subspace, const MyInt reorth_step)
	{
#define test_SVD_update_reorth_DEBUG (0)
		std::ofstream outfile("d:/test_SVD_update_reorth.txt");
		bool test_update_SVD_function = false;

		
#if test_SVD_update_reorth_DEBUG		
		MyMatrix A; A.resize(n_rows, start_n_col); 
		A = create_test_SVD_update_reorth_A();
#else
		MyMatrix A =  create_test_SVD_update_reorth_A();//SVD_updater::my_random_sample(n_rows, start_n_col);//A = rnd.rand(n_rows, start_n_col)
#endif
		MyMatrix U, Vh; MyVector S;
		sp_linalg_svd(A,U,S,Vh); //(U, S, Vh) = sp.linalg.svd(A, full_matrices = False, compute_uv = True, overwrite_a = False, check_finite = False)
#if test_SVD_update_reorth_DEBUG
		std::cout << "U : " << U << std::endl;
		std::cout << "S : " << S << std::endl;
		std::cout << "Vh : " << Vh << std::endl;
		MyPause;
#endif
		SVD_updater svd_upd(outfile); 
		svd_upd.initial(U, S, Vh, true, reorth_step);

		MyMatrix svd_comp, times;
		const MyInt nIdxStep = -3;
		if (test_update_SVD_function)
		{
			svd_comp.resize(n_max_cols, 10 + nIdxStep);
			times.resize(n_max_cols, 5);
		} 
		else
		{
			svd_comp.resize(n_max_cols, 9 + nIdxStep);
			times.resize(n_max_cols, 4);
		}
		
		bool same_sub = false;

		for (int ii = 0; ii < n_max_cols; ++ii)
		{
			MyVector a1;
			if (false/*prob_same_subspace  >= (SVD_updater::my_random_sample())*/) //# column from the same subspace
			{
				
				a1 = SVD_updater::my_random_sample(A.cols());
				//std::cout << "2" << std::endl;
				a1 = A * a1;
				a1 = a1 / a1.maxCoeff();
				same_sub = true;
			} 
			else
			{
				//std::cout << "3" << std::endl;
				//a1 = SVD_updater::my_random_sample(n_rows); //a1 = rnd.rand(n_rows, 1)
				a1 = create_test_SVD_update_reorth_col(ii);
				//std::cout << "4" << std::endl;
				same_sub = false;
			}

			A = SVD_updater::my_hstack(A, a1);// np.hstack((A, a1))

			{
				MyTimer t;
				svd_upd.add_column(a1);
				times.coeffRef(ii, 0) = t.elapsed();
			}
			MyMatrix Ut, Vht;
			MyVector St;
			{
				MyTimer t;
				svd_upd.get_current_svd(Ut, St, Vht);
				//(Ut, St, Vht) = svd_upd.get_current_svd()
				std::cout << "Ut : " << Ut << std::endl;
				std::cout << "St : " << St << std::endl;
				std::cout << "Vht : " << Vht << std::endl;
#if test_SVD_update_reorth_DEBUG
				std::cout << "Ut : " << Ut << std::endl;
				std::cout << "St : " << St << std::endl;
				std::cout << "Vht : " << Vht << std::endl;
				MyPause;
#endif
				times.coeffRef(ii, 1) = t.elapsed();
				times.coeffRef(ii, 2) = times.coeff(ii, 0) + times.coeff(ii, 1);//	times[ii, 2] = times[ii, 0] + times[ii, 1]
			}
			MyMatrix Us, Vhs;
			MyVector Ss;
			{
				MyTimer t;
#if test_SVD_update_reorth_DEBUG
				std::cout << "A : " << A << std::endl;
#endif
				sp_linalg_svd(A,Us,Ss,Vhs);

				std::cout << "Us : " << Us << std::endl;
				std::cout << "Ss : " << Ss << std::endl;
				std::cout << "Vhs : " << Vhs << std::endl;
#if test_SVD_update_reorth_DEBUG
				std::cout << "Us : " << Us << std::endl;
				std::cout << "Ss : " << Ss << std::endl;
				std::cout << "Vhs : " << Vhs << std::endl;
				MyPause;
#endif
				//(Us, Ss, Vhs) = sp.linalg.svd(A, full_matrices = False, compute_uv = True, overwrite_a = False, check_finite = False)
				times.coeffRef(ii, 3) = t.elapsed();//	times[ii, 3] = t.msecs
			}

			if (test_update_SVD_function)
			{
				MyTimer t;
				SVD_updater::update_SVD(U, S, Vh, a1, true);
				times.coeffRef(ii,4) = t.elapsed();
			}

			MyVector tmp; tmp.resize(St.size());
			MyFloat maxValue = std::numeric_limits<MyFloat>::min();
			MyInt   maxIdx = Invalid_Id;
			for (int m = 0; m < St.size(); ++m)
			{
				tmp[m] = std::abs(Ss[m] - St[m]) / Ss[m];
				if (tmp[m] > maxValue)
				{
					maxIdx = m;
					maxValue = tmp[m];
				}
			}
			std::cout << "tmp : " << tmp << std::endl; //MyPause;
			MyFloat t1 = tmp.maxCoeff();
			MyInt t_pos = maxIdx;
#if test_SVD_update_reorth_DEBUG
			std::cout << "t1 " << t1 << "  t_pos " << t_pos << std::endl; MyPause;
#endif
			svd_comp.coeffRef(ii, 0) = t1;
			svd_comp.coeffRef(ii, 1) = t_pos;
			//svd_comp.coeffRef(ii, 2) = std::sqrt(A.rows()*A.cols());// np.sqrt(A.shape[0] * A.shape[1]);
			//svd_comp.coeffRef(ii, 3) = A.rows();
			//svd_comp.coeffRef(ii, 4) = A.cols();
			
			svd_comp.coeffRef(ii, 5 + nIdxStep) = MyInt(svd_upd.reorth_was_pereviously());
			
			MyMatrix Ar1 = Ut * (St.asDiagonal()*Vht);//Ar1 = np.dot(Ut, np.dot(np.diag(St), Vht))

			svd_comp.coeffRef(ii, 6 + nIdxStep) = (Ar1 - A).cwiseAbs().maxCoeff() / Ss[0]; //svd_comp[ii, 6] = np.max(np.abs(Ar1 - A) / Ss[0])
			//svd_comp[ii, 7] = np.max(np.abs(np.dot(Ut.T, Ut) - np.eye(Ut.shape[1])))
			svd_comp.coeffRef(ii, 7 + nIdxStep) = ((Ut.transpose() * Ut) - MyMatrix::Identity(Ut.cols(), Ut.cols())).cwiseAbs().maxCoeff();
			//svd_comp[ii, 8] = np.max(np.abs(np.dot(Vht, Vht.T) - np.eye(Vht.shape[0])))
			svd_comp.coeffRef(ii, 8 + nIdxStep) = ((Vht * Vht.transpose()) - MyMatrix::Identity(Vht.rows(), Vht.rows())).cwiseAbs().maxCoeff();
			
			if (test_update_SVD_function)
			{
				tmp.resize(Ss.size());
				for (int m = 0; m < Ss.size();++m)
				{
					tmp[m] = std::abs(Ss[m] - S[m]) / Ss[m];
				}
				svd_comp.coeffRef(ii, 9 + nIdxStep) = tmp.maxCoeff();
			}
		}

		std::cout << "svd_comp : " << std::endl << svd_comp << std::endl; 
	}
	
}