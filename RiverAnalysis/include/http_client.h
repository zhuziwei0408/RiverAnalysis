#pragma once
#include <string>
#include <functional>
#include "RiverThread.h"
#include "AlarmMsgQueue.h"
extern "C"
{
#include "mongoose.h"
};

class HttpClient : public RiverThread {
public:
	HttpClient(AlarmMsgQueue* msg_queue, std::string url);
	~HttpClient();
	void SendReq(const std::string &url, std::string body);
	
    static void OnHttpEvent(mg_connection *connection, int event_type, void *event_data);
    
    int s_exit_flag;
	void Run();
	AlarmMsgQueue* _out_queue;
private:
	std::string serverURL;
};
