#include "AlarmMsgQueue.h"
#include <iostream>

#include <glog/logging.h>

AlarmMsgQueue::AlarmMsgQueue(int size, int timeout)
{
	_data = std::vector<AlarmData*>(size, NULL);
	for (int i = 0; i < _data.size(); ++i) {
		_data[i] = new AlarmData();
	}
	_head = 0;
	_tail = 0;
	_size = size;            
	_element_size = 0;
	_timeout = timeout;
}

AlarmMsgQueue::~AlarmMsgQueue()
{
	std::lock_guard<std::mutex> lk(_mutex);
	_head = -1;
	_tail = -1;
	_size = 0;
	_element_size = 0;

	for (int i = 0; i < _data.size(); ++i) {
		_data[i]->clear();
		delete _data[i];
		_data[i] = NULL;
	}
	_data.clear();
}

AlarmData* AlarmMsgQueue::get_head_to_write() {
	std::unique_lock<std::mutex> lk(_mutex);

	if (_timeout < 0) {
		_not_full_cond.wait(lk, [this]() { return _element_size < _size - 1; });
	}
	else if (_element_size >= _size - 1/*容量已满*/ &&
		_not_full_cond.wait_for(lk, std::chrono::milliseconds(_timeout)) == std::cv_status::timeout) {
		return NULL;
	}
    LOG(INFO) << "get_head_to_write: head: "<< _head << " tail: " << _tail << " size: " << _element_size;
	// printf("get_head_to_write:\t head:%d\t tail:%d\t element_size:%d \n", _head, _tail, _element_size);
	return _data[_head];
}

void AlarmMsgQueue::head_next() {
	std::lock_guard<std::mutex> lk(_mutex);
	_element_size++;
	if (++_head == _size - 1)
		_head = 0;
	LOG(INFO) << "head_next: head: "<< _head << " tail: " << _tail << " size: " << _element_size;
	_not_empty_cond.notify_one();
}

AlarmData* AlarmMsgQueue::get_tail_to_read() {
	std::unique_lock<std::mutex> lk(_mutex);

	if (_timeout < 0) {
		_not_empty_cond.wait(lk, [this]() { return _element_size > 0; });
		//printf("get_tail_to_read:\t head:%d\t tail:%d\t element_size:%d \n", _head, _tail, _element_size);
	}
	else if (_element_size <= 0) {
		std::cv_status status = _not_empty_cond.wait_for(lk, std::chrono::milliseconds(_timeout));
		if (status == std::cv_status::timeout) {
			return NULL;//如果为空或等待一会仍然为空则返回NULL，等待一会有值了通过上面的_not_empty_cond.notify_one();从阻塞状态被唤醒
		}
		else {
            LOG(INFO) << "status is not timeout. element_size: " << _element_size;
		}
	}
	LOG(INFO) << "get_tail_to_write: head: "<< _head << " tail: " << _tail << " size: " << _element_size;
    return _data[_tail];
}
void AlarmMsgQueue::tail_next() {
	std::lock_guard<std::mutex> lk(_mutex);
	_element_size--;
	if (++_tail == _size - 1)
		_tail = 0;
	LOG(INFO) << "tail_next: head: "<< _head << " tail: " << _tail << " size: " << _element_size;
	_not_full_cond.notify_one();
}
