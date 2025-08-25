
#pragma once

#include <vector>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include<vector>

struct AlarmData {
	std::string camera_id;
	std::string current_time;
	int scene_type;
	cv::Mat img;
	std::vector<cv::Rect>rectangle_array_vect;
	float TotalArea;
	float Speed;
	float DraftValue;
	int DraftValueType;
	bool IsActive;
	std::string Color;
	AlarmData() {
		clear();
	}
	void clear() {
		camera_id.clear();
		current_time.clear();
		scene_type = -1;
		img.release();
		rectangle_array_vect.clear();
		TotalArea = 0.0;
		Speed = 0.0;
		DraftValue = 0;
		DraftValueType = 1;
		IsActive = false;
		Color = "";
	}
};

class AlarmMsgQueue
{
public:
	AlarmMsgQueue(int size, int timeout);
	~AlarmMsgQueue();

	AlarmData* get_head_to_write();
	AlarmData* get_tail_to_read();

	void head_next();
	void tail_next();
private:
	 std::vector<AlarmData*>  _data;

	 int _size;
	 int _element_size;

	int _head;
	int _tail;

	int _timeout;
	std::mutex _mutex;

	 std::condition_variable _not_full_cond;
	 std::condition_variable _not_empty_cond;
};





