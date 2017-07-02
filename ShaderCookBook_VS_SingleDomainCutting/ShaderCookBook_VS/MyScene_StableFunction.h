
namespace YC
{

	void MyScene::update( float t )
	{

	}

	void MyScene::mouse(int x,int y)
	{
		if (GLUT_ACTIVE_ALT == glutGetModifiers())
		{
#if 0
			GLint viewport[4];
			GLdouble mvmatrix[16], projmatrix[16];
			GLfloat winx, winy, winz;
			GLdouble posx, posy, posz;

			view = lightFrustum->getViewMatrix();
			projection = lightFrustum->getProjectionMatrix();

			model = mat4(1.0f);

			model *= glm::scale(0.5f,0.5f,0.5f);
			model *= mat4(MyTransform.s.M00,MyTransform.s.M10,MyTransform.s.M20,MyTransform.s.M30,
				MyTransform.s.M01,MyTransform.s.M11,MyTransform.s.M21,MyTransform.s.M31,
				MyTransform.s.M02,MyTransform.s.M12,MyTransform.s.M22,MyTransform.s.M32,
				MyTransform.s.M03,MyTransform.s.M13,MyTransform.s.M23,MyTransform.s.M33);

			mat4 mv = view * model;

			//glScalef(0.1, 0.1, 0.1);
			glGetIntegerv(GL_VIEWPORT, viewport);			/* 获取三个矩阵 */

			winx = x;
			winy = windowHeight - y;

			float * mvPtr = glm::value_ptr(mv);
			float * projPtr = glm::value_ptr(projection);
			for (int i=0;i<16;++i)
			{
				mvmatrix[i] = mvPtr[i];projmatrix[i]=projPtr[i];
			}

			glReadPixels((int)winx, (int)winy, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winz);			/* 获取深度 */
			gluUnProject(winx, winy, winz, mvmatrix, projmatrix, viewport, &posx, &posy, &posz);	/* 获取三维坐标 */
			m_physicalSimulation.addBlade(MyPoint(posx,posy,-0.5f));
			LogInfo("%f,%f,%f\n",posx,posy,posz);
			//cout << posx << ' ' << posy << ' ' << posz << endl;
#endif
			double mvmatrix[16];
			double projmatrix[16];
			int viewport[4];
			double dX, dY, dZ, dClickY; // glUnProject uses doubles, but I'm using floats for these 3D vectors

			glGetIntegerv(GL_VIEWPORT, viewport);	

			model = mat4(1.0f);

			model *= glm::scale(0.5f,0.5f,0.5f);
			model *= mat4(MyTransform.s.M00,MyTransform.s.M10,MyTransform.s.M20,MyTransform.s.M30,
				MyTransform.s.M01,MyTransform.s.M11,MyTransform.s.M21,MyTransform.s.M31,
				MyTransform.s.M02,MyTransform.s.M12,MyTransform.s.M22,MyTransform.s.M32,
				MyTransform.s.M03,MyTransform.s.M13,MyTransform.s.M23,MyTransform.s.M33);

			mat4 mv = view * model; //glGetDoublev (GL_MODELVIEW_MATRIX, mvmatrix);
			projection;//glGetDoublev (GL_PROJECTION_MATRIX, projmatrix);
			dClickY = double (windowHeight - y); 
			// OpenGL renders with (0,0) on bottom, mouse reports with (0,0) on top
			float * mvPtr = glm::value_ptr(mv);
			float * projPtr = glm::value_ptr(projection);
			for (int i=0;i<16;++i)
			{
				mvmatrix[i] = mvPtr[i];projmatrix[i]=projPtr[i];
			}


			gluUnProject ((double) x, dClickY, 0.0, mvmatrix, projmatrix, viewport, &dX, &dY, &dZ);
			MyPoint p0(dX,dY,dZ);
			
			gluUnProject ((double) x, dClickY, 1.0, mvmatrix, projmatrix, viewport, &dX, &dY, &dZ);
			MyPoint p1(dX,dY,dZ);
			m_physicalSimulation.addBlade(p0,p1);
			//LogInfo("%f,%f,%f\n",dX,dY,dZ);
			/*L1 = CVec3( (float) dX, (float) dY, (float) dZ );
			gluUnProject ((double) x, dClickY, 1.0, mvmatrix, projmatrix, viewport, &dX, &dY, &dZ);
			L2 = CVec3( (float) dX, (float) dY, (float) dZ );*/
		}
		else
		{
			MousePt.s.X = (MyFloat)x;
			MousePt.s.Y = (MyFloat)y;
			LastRot = ThisRot;	
			ArcBall.click(&MousePt);
		}	
	}

	void MyScene::motion(int x,int y)
	{
		if (GLUT_ACTIVE_ALT == glutGetModifiers())
		{
			double mvmatrix[16];
			double projmatrix[16];
			int viewport[4];
			double dX, dY, dZ, dClickY; // glUnProject uses doubles, but I'm using floats for these 3D vectors

			glGetIntegerv(GL_VIEWPORT, viewport);	

			model = mat4(1.0f);

			model *= glm::scale(0.5f,0.5f,0.5f);
			model *= mat4(MyTransform.s.M00,MyTransform.s.M10,MyTransform.s.M20,MyTransform.s.M30,
				MyTransform.s.M01,MyTransform.s.M11,MyTransform.s.M21,MyTransform.s.M31,
				MyTransform.s.M02,MyTransform.s.M12,MyTransform.s.M22,MyTransform.s.M32,
				MyTransform.s.M03,MyTransform.s.M13,MyTransform.s.M23,MyTransform.s.M33);

			mat4 mv = view * model; //glGetDoublev (GL_MODELVIEW_MATRIX, mvmatrix);
			projection;//glGetDoublev (GL_PROJECTION_MATRIX, projmatrix);
			dClickY = double (windowHeight - y); 
			// OpenGL renders with (0,0) on bottom, mouse reports with (0,0) on top
			float * mvPtr = glm::value_ptr(mv);
			float * projPtr = glm::value_ptr(projection);
			for (int i=0;i<16;++i)
			{
				mvmatrix[i] = mvPtr[i];projmatrix[i]=projPtr[i];
			}

			gluUnProject ((double) x, dClickY, 0.0, mvmatrix, projmatrix, viewport, &dX, &dY, &dZ);
			MyPoint p0(dX,dY,dZ);

			gluUnProject ((double) x, dClickY, 1.0, mvmatrix, projmatrix, viewport, &dX, &dY, &dZ);
			MyPoint p1(dX,dY,dZ);
			m_physicalSimulation.addBlade(p0,p1);
		} 
		else
		{
			MousePt.s.X = (MyFloat)x;
			MousePt.s.Y = (MyFloat)y;
			//g_trackball.drag(glm::vec2(x,y));
			Quat4fT     ThisQuat;

			ArcBall.drag(&MousePt, &ThisQuat);						// Update End Vector And Get Rotation As Quaternion
			Matrix3fSetRotationFromQuat4f(&ThisRot, &ThisQuat);		// Convert Quaternion Into Matrix3fT
			Matrix3fMulMatrix3f(&ThisRot, &LastRot);				// Accumulate Last Rotation Into This One
			Matrix4fSetRotationFromMatrix3f(&MyTransform, &ThisRot);	// Set Our Final Transform's Rotation From This One
		}
	}
#if SHOWFPS
	void MyScene::computeFPS()
	{
		using namespace YC;
		static int fpsCount=0;
		static int fpsLimit=50;
		fpsCount++;

		if (fpsCount == fpsLimit) {

			float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
			//sprintf(VR_FEM::TexRender::fps, "Cuda GL Interop Wrapper: %3.1f fps ", ifps);  
			sprintf(TexRender::fps, "%3.1f fps ", ifps);  

			//glutSetWindowTitle(TexRender::fps);
			fpsCount = 0; 

			cutilCheckError(sdkResetTimer(&timer));  
		}

		m_MyGLSLFont.printTextOnGLSL(TexRender::fps);
	}
#endif

#if SHOW_SHADOWMAP

#if USE_PCF
	void MyScene::setupFBO()
	{
		GLfloat border[] = {1.0f, 0.0f,0.0f,0.0f };
		// The depth buffer texture
		
		glGenTextures(1, &m_depthTex);
		glBindTexture(GL_TEXTURE_2D, m_depthTex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadowMapWidth,
			shadowMapHeight, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LESS );
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);

		// Assign the depth buffer texture to texture channel 0
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, m_depthTex);

		// Create and set up the FBO
		glGenFramebuffers(1, &shadowFBO);
		glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
			GL_TEXTURE_2D, m_depthTex, 0);

		GLenum drawBuffers[] = {GL_NONE};
		glDrawBuffers(1, drawBuffers);

		GLenum result = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if( result == GL_FRAMEBUFFER_COMPLETE) {
			printf("Framebuffer is complete.\n");
		} else {
			printf("Framebuffer is not complete.\n");
		}

		glBindFramebuffer(GL_FRAMEBUFFER,0);
	}
#else
	void MyScene::setupFBO()
	{
		GLfloat border[] = {1.0f, 0.0f,0.0f,0.0f };
		// The depth buffer texture
		GLuint depthTex;
		glGenTextures(1, &depthTex);
		glBindTexture(GL_TEXTURE_2D, depthTex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadowMapWidth,
			shadowMapHeight, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LESS);    

		// Assign the depth buffer texture to texture channel 0
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, depthTex);

		// Create and set up the FBO
		glGenFramebuffers(1, &shadowFBO);
		glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
			GL_TEXTURE_2D, depthTex, 0);

		GLenum drawBuffers[] = {GL_NONE};
		glDrawBuffers(1, drawBuffers);

		GLenum result = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if( result == GL_FRAMEBUFFER_COMPLETE) {
			printf("Framebuffer is complete.\n");
		} else {
			printf("Framebuffer is not complete.\n");
		}

		glBindFramebuffer(GL_FRAMEBUFFER,0);
	}
#endif

	void MyScene::spitOutDepthBuffer() 
	{
		int size = shadowMapWidth * shadowMapHeight;
		float * buffer = new float[size];
		unsigned char * imgBuffer = new unsigned char[size * 4];

		glGetTexImage(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT,GL_FLOAT,buffer);

		for( int i = 0; i < shadowMapHeight; i++ )
			for( int j = 0; j < shadowMapWidth; j++ )
			{
				int imgIdx = 4 * ((i*shadowMapWidth) + j);
				int bufIdx = ((shadowMapHeight - i - 1) * shadowMapWidth) + j;

				// This is just to make a more visible image.  Scale so that
				// the range (minVal, 1.0) maps to (0.0, 1.0).  This probably should
				// be tweaked for different light configurations.
				float minVal = 0.88f;
				float scale = (buffer[bufIdx] - minVal) / (1.0f - minVal);
				unsigned char val = (unsigned char)(scale * 255);
				imgBuffer[imgIdx] = val;
				imgBuffer[imgIdx+1] = val;
				imgBuffer[imgIdx+2] = val;
				imgBuffer[imgIdx+3] = 0xff;
			}

			//    QImage img(imgBuffer, shadowMapWidth, shadowMapHeight, QImage::Format_RGB32);
			//    img.save("depth.png", "PNG");

			SOIL_save_image("texture.bmp",SOIL_SAVE_TYPE_BMP, shadowMapWidth,shadowMapHeight,SOIL_LOAD_RGBA,imgBuffer) ;

			delete [] buffer;
			delete [] imgBuffer;
			exit(1);
	}
#endif
}