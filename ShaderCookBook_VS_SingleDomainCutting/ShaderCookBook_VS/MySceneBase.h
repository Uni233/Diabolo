#pragma once
#include "scene.h"
class MySceneBase :
	public Scene
{
public:
	MySceneBase(void);
	virtual ~MySceneBase(void);

	virtual void mouse(int x,int y)=0;
	virtual void motion(int x,int y)=0;
};

