#ifndef OPENCV_320_PUTCHINESETEXT_H
#define OPENCV_320_PUTCHINESETEXT_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include <ft2build.h>
#include FT_FREETYPE_H 

#include <string>
#include <assert.h>  
#include <locale.h>  
#include <ctype.h>  

class cv320PutChText
{
public:
	cv320PutChText() {}
	cv320PutChText(const char *freeType);
	~cv320PutChText();
    int Loadttc(const char* ttc_path);
    void release();
	void getFont(int *type, cv::Scalar *size = NULL, bool *underline = NULL, float *diaphaneity = NULL);
	void setFont(int *type, cv::Scalar *size = NULL, bool *underline = NULL, float *diaphaneity = NULL);
	void restoreFont(const int frontSize);
	int putText(cv::Mat &frame, std::string& text, cv::Point pos, cv::Scalar color,const int frontSize);
    int putText(cv::Mat &frame, const char    *text, cv::Point pos, const int frontSize);
	int putText(cv::Mat &frame, const char    *text, cv::Point pos, cv::Scalar color, const int frontSize);
	int putText(cv::Mat &frame, const wchar_t *text, cv::Point pos, cv::Scalar color, const int frontSize);

    static std::wstring stows(const std::string& s);
private:
	void putWChar(cv::Mat&frame, wchar_t wc, cv::Point &pos, cv::Scalar color);
private:
	FT_Library  m_library;
	FT_Face     m_face;
	int         m_fontType;
	cv::Scalar  m_fontSize;
	bool        m_fontUnderline;
	float       m_fontDiaphaneity;
};

#endif
