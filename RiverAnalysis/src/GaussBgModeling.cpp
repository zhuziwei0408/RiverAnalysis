#include "GaussBgModeling.h"

#include <string>
#include <vector>
#include <bitset> 
#include <queue>  
#include <glog/logging.h>

GaussBgModeling::GaussBgModeling(cv::Ptr<cv::BackgroundSubtractorMOG2>& _mod)
{
	_mod = cv::createBackgroundSubtractorMOG2();
	_mod->setHistory(500);
	// 模型匹配阈值  
	_mod->setVarThreshold(50);
	// 阴影阈值  
	_mod->setShadowThreshold(0.7);
	// 前景中模型参数，设置为0表示背景，255为前景，默认值127  
	_mod->setShadowValue(127);//相当于对比度
							  // 背景阈值设定 backgroundRatio*history  
							  //_mod->setBackgroundRatio(2);
							  // 设置阈值的降低的复杂性  
	_mod->setComplexityReductionThreshold(0.02);
	// 高斯混合模型组件数量  
	_mod->setNMixtures(100);
	// 设置每个高斯组件的初始方差  
	_mod->setVarInit(0.5);
	// 新模型匹配阈值  
	_mod->setVarThresholdGen(9);
}

cv::Mat GaussBgModeling::GetForegroundImg(cv::Mat origin_img,cv::Ptr<cv::BackgroundSubtractorMOG2> _mod){
		if(origin_img.empty()){
			usleep(200);
			return origin_img;
		}
		if (origin_img.channels() == 1){
			_rawMat = origin_img;
		}
		else if (origin_img.channels() == 3){
			cv::cvtColor(origin_img, _rawMat, cv::COLOR_BGR2GRAY);
		}
		else{
			return origin_img; 
		}
		if (_rawMat.rows* _rawMat.cols> 280*20){
			cv::Size dsize = cv::Size(640, 360);
			cv::resize(_rawMat, _rawMat, dsize);
		}
		_mod->apply(_rawMat, _foregroundMat);
		/* imshow("_foregroundMat", _foregroundMat);
		cv::waitKey(10);
		_foregroundMat = _foregroundMat(_detectRegion); */
		return _foregroundMat;
}

