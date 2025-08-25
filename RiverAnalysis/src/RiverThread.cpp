#include "RiverThread.h"

#include <chrono>
RiverThread::RiverThread()
    :_is_run(false), _thread(NULL) {
}

void RiverThread::Start() {
    if (_thread != NULL) {
        if (!_thread->joinable()) {
            return;
        }
        delete _thread;
        _thread = NULL;
    }
    _thread = new std::thread(std::mem_fn(&RiverThread::Run), this);
}

void RiverThread::Stop() {
    if (_thread == NULL || !_thread->joinable())
        return;
    _is_run = false;
    _thread->join();
    delete _thread;
    _thread = NULL;
}

void RiverThread::CallStop() {
    _wait_stop_cond.notify_all();
}
void RiverThread::WaitUtilDie() {
    std::unique_lock<std::mutex> lk(_mutex);
    _wait_stop_cond.wait(lk);
}

void RiverThread::WaitFor(uint32_t ms) {
    std::unique_lock<std::mutex> lk(_mutex);
    _wait_stop_cond.wait_for(lk, std::chrono::milliseconds(ms));
}

