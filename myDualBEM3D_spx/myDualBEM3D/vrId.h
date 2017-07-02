#ifndef _vrId_H_
#define _vrId_H_

namespace VR
{
	class vrId
	{
	public:
		vrId() :nCount(0)
		{}
		int value()const{ return nCount; }
		void increase(){ nCount++; }
	private:
		int nCount;
	};
}
#endif//_vrId_H_