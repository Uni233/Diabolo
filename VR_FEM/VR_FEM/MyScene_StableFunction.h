
namespace YC
{

	void MyScene::update( float t )
	{

	}

	void MyScene::mouse(int x,int y)
	{
		MousePt.s.X = (MyFloat)x;
		MousePt.s.Y = (MyFloat)y;
		LastRot = ThisRot;	
		ArcBall.click(&MousePt);	
	}

	void MyScene::motion(int x,int y)
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