#include "VR_Global_Define.h"

#include <stdio.h>

unsigned long inKB(unsigned long bytes)

{ return bytes/1024; }

unsigned long inMB(unsigned long bytes)

{ return bytes/(1024*1024); }

void printStats(unsigned long free, unsigned long total)
{
	printf("^^^^ Free : %lu bytes (%lu KB) (%lu MB)\n", free, inKB(free), inMB(free));
	printf("^^^^ Total: %lu bytes (%lu KB) (%lu MB)\n", total, inKB(total), inMB(total));
	printf("^^^^ %f%% free, %f%% used\n", 100.0*free/(double)total, 100.0*(total - free)/(double)total);
}

namespace YC
{
	MyFloat NormInf(const cgVector& v)
	{
		MyFloat max = 0;
		for (int i = 0; i != v.rows(); i++) {
			if (std::fabsf(v[i]/* this->data[0][i]*/) > max ) {
				max = std::fabsf( v[i] /*this->data[0][i]*/);
			}
		}
		return max;
	}
}
