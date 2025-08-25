#pragma once
#include <opencv2/opencv.hpp>
#include<opencv2/video/background_segm.hpp>
#include <vector>
#include <string>
#include <bitset> 


class  GaussBgModeling
{
public:
 	 GaussBgModeling(cv::Ptr<cv::BackgroundSubtractorMOG2>& _mod);
    ~GaussBgModeling() {}
private:
	cv::Mat _rawMat;//.背景建模-原图
	cv::Mat _foregroundMat;//.背景建模-前景图
public:
	cv::Mat GetForegroundImg(cv::Mat origin_img,cv::Ptr<cv::BackgroundSubtractorMOG2> _mod);
};