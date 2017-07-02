/*   Gaussian Elimination.
*    
*    Copyright (C) 2012-2013 Orange Owl Solutions.  
*
*    This file is part of Bluebird Library.
*    Gaussian Elimination is free software: you can redistribute it and/or modify
*    it under the terms of the Lesser GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.
*
*    Gaussian Elimination is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    Lesser GNU General Public License for more details.
*
*    You should have received a copy of the GNU General Public License
*    along with Gaussian Elimination.  If not, see <http://www.gnu.org/licenses/>.
*
*
*    For any request, question or bug reporting please visit http://www.orangeowlsolutions.com/
*    or send an e-mail to: info@orangeowlsolutions.com
*
*
*/


/**************/
/* TIMING CPU */
/**************/

#include <windows.h>
#include <iostream>

#include "TimingCPU.h"

struct PrivateTimingCPU {
	double	PCFreq;
	__int64 CounterStart;
};
			
// --- Default constructor
TimingCPU::TimingCPU() { privateTimingCPU = new PrivateTimingCPU; (*privateTimingCPU).PCFreq = 0.0; (*privateTimingCPU).CounterStart = 0; }

// --- Default destructor
TimingCPU::~TimingCPU() { }

// --- Starts the timing
void TimingCPU::StartCounter()
{
	LARGE_INTEGER li;
	if(!QueryPerformanceFrequency(&li)) std::cout << "QueryPerformanceFrequency failed!\n";

	(*privateTimingCPU).PCFreq = double(li.QuadPart)/1000.0;

	QueryPerformanceCounter(&li);
	(*privateTimingCPU).CounterStart = li.QuadPart;
}

// --- Gets the timing counter in ms
double TimingCPU::GetCounter()
{
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return double(li.QuadPart-(*privateTimingCPU).CounterStart)/(*privateTimingCPU).PCFreq;
}


