#ifndef _VR_NUMERICAL_NEWMARKCONTANT_H
#define _VR_NUMERICAL_NEWMARKCONTANT_H

namespace YC
{
	namespace Numerical
	{
		template< typename T >
		class NewmarkContant
		{
		public:
			NewmarkContant()
			{
				T g_NewMark_alpha    = .25;
				T g_NewMark_delta    = .5;
				T g_NewMark_timestep = 1./64.0;
				T _default_delta = g_NewMark_delta;
				T _default_alpha = g_NewMark_alpha;
				T _default_timestep = g_NewMark_timestep;
				m_db_NewMarkConstant[0] = (1.0f/(_default_alpha*_default_timestep*_default_timestep));
				m_db_NewMarkConstant[1] = (_default_delta/(_default_alpha*_default_timestep));
				m_db_NewMarkConstant[2] = (1.0f/(_default_alpha*_default_timestep));
				m_db_NewMarkConstant[3] = (1.0f/(2.0f*_default_alpha)-1.0f);
				m_db_NewMarkConstant[4] = (_default_delta/_default_alpha -1.0f);
				m_db_NewMarkConstant[5] = (_default_timestep/2.0f*(_default_delta/_default_alpha-2.0f));
				m_db_NewMarkConstant[6] = (_default_timestep*(1.0f-_default_delta));
				m_db_NewMarkConstant[7] = (_default_delta*_default_timestep);
			}
			T operator[](int idx){return m_db_NewMarkConstant[idx];}
		private:
			T m_db_NewMarkConstant[8];
		};
	}//namespace Numerical
}//namespace YC
#endif//_VR_NUMERICAL_NEWMARKCONTANT_H

