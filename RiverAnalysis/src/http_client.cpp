#include "http_client.h"
#include "cJSON.h"
#include <iostream>
#ifdef LOG
#undef LOG
#endif
#include <glog/logging.h>
// 初始化client静态变量
//
static std::string base64Encode(const unsigned char* Data, int DataByte);
static std::string Mat2Base64(const cv::Mat img, std::string imgType);
HttpClient::HttpClient(AlarmMsgQueue* msg_queue, std::string url) 
    : _out_queue(msg_queue), serverURL(url){
    LOG(INFO) << "SendUrl" << serverURL;
}
HttpClient::~HttpClient() {
}
// 客户端的网络请求响应
void HttpClient::OnHttpEvent(mg_connection *connection, int event_type, void *event_data)
{
	int connect_status;
    http_message *hm = (struct http_message *)event_data; 
    HttpClient* inst = (HttpClient*)connection->mgr->user_data;
    int& s_exit_flag = inst->s_exit_flag;
	
    switch (event_type) 
	{
	case MG_EV_CONNECT:
        mg_set_timer(connection, 0);
		connect_status = *(int *)event_data;
		if (connect_status != 0) 
		{
            LOG(ERROR) << "Error connecting to server, error code: " << connect_status;
			s_exit_flag = 1;
		}
		break;
	case MG_EV_HTTP_REPLY:
	{
		connection->flags |= MG_F_SEND_AND_CLOSE;
		s_exit_flag = 1; // 每次收到请求后关闭本次连接，重置标记
        LOG(INFO) << "Send a message";
                    std::string rsp = std::string(hm->body.p, hm->body.len);
                std::cout<<"rsp "<<rsp<<std::endl;
	}
		break;
	case MG_EV_CLOSE:
		if (s_exit_flag == 0) 
		{
			s_exit_flag = 1;
            LOG(ERROR) << "Server closed connection";
		};
		break;
    case MG_EV_TIMER:
        LOG(ERROR) << "Server Timeout";
        connection->flags |= MG_F_CLOSE_IMMEDIATELY;
        break;
	default:
		break;
	}
}

// 发送一次请求，并回调处理，然后关闭本次连接
void HttpClient::SendReq(const std::string &url, std::string body)
{

	// 给回调函数赋值
	mg_mgr mgr;
	mg_mgr_init(&mgr, this);
	const char* extra_headers = "Content-Type:application/json;charset=utf-8\r\n";
	auto connection = mg_connect_http(&mgr, OnHttpEvent, url.c_str(), extra_headers, body.c_str());
	mg_set_protocol_http_websocket(connection);
    mg_set_timer(connection, mg_time() + 0.5);
    s_exit_flag = 0;
	while (s_exit_flag == 0) {
		mg_mgr_poll(&mgr, 50);
    }

	mg_mgr_free(&mgr);
}
void HttpClient::Run()
{
	_is_run = true;
	while (_is_run)
	{
		AlarmData* alarmDataPtr = _out_queue->get_tail_to_read();
		if (alarmDataPtr == NULL)
		{
                   usleep(100*1000);
	           continue;
		}
		cJSON *root = cJSON_CreateObject();
		cJSON_AddStringToObject(root, "VideoId", alarmDataPtr->camera_id.c_str());
		cJSON_AddStringToObject(root, "StartTime", alarmDataPtr->current_time.c_str());
		cJSON_AddNumberToObject(root, "SceneType", alarmDataPtr->scene_type);
		cJSON *ExtendData = cJSON_CreateObject();
		char* ExtendDatastream;
		switch (alarmDataPtr->scene_type)
		{
		case WATERGAUGE:   // 水尺
                       cJSON_AddNumberToObject(ExtendData, "Value", alarmDataPtr->DraftValue);
                       cJSON_AddNumberToObject(ExtendData, "Type", 0);
		       break;
		case WATERCOLOR:// 水色
			cJSON_AddStringToObject(ExtendData, "Color", alarmDataPtr->Color.c_str());
			break;
		case INVADE:  // 入侵
                        cJSON_AddNumberToObject(ExtendData, "IsActive", alarmDataPtr->IsActive);
                        break;
		case FLOATER:  // 漂浮物
                        cJSON_AddNumberToObject(ExtendData, "TotalArea", alarmDataPtr->TotalArea);
			cJSON_AddNumberToObject(ExtendData, "Speed", alarmDataPtr->Speed);
                        break;
		case FISHING: // 钓鱼
                        cJSON_AddNumberToObject(ExtendData, "IsActive", alarmDataPtr->IsActive);
			break;
		case LITTER: // 倾倒垃圾
                        cJSON_AddNumberToObject(ExtendData, "IsActive", alarmDataPtr->IsActive);
			break;
		case SWIMING: // 游泳
			cJSON_AddNumberToObject(ExtendData, "IsActive", alarmDataPtr->IsActive);
			break;
		default:break;
		}
		ExtendDatastream = cJSON_PrintUnformatted(ExtendData);
		cJSON_AddStringToObject(root, "ExtendData", ExtendDatastream);
		cJSON * rectangle_array = cJSON_CreateArray();
		cJSON *pointObj = cJSON_CreateObject();
		for (size_t i = 0; i < alarmDataPtr->rectangle_array_vect.size(); i++)
		{
		  cJSON *pointObj = cJSON_CreateObject();//要是局部变量
		  cJSON_AddNumberToObject(pointObj, "X", alarmDataPtr->rectangle_array_vect[i].x);
		  cJSON_AddNumberToObject(pointObj, "Y", alarmDataPtr->rectangle_array_vect[i].y);
		  cJSON_AddNumberToObject(pointObj, "Width", alarmDataPtr->rectangle_array_vect[i].width);
		  cJSON_AddNumberToObject(pointObj, "Height", alarmDataPtr->rectangle_array_vect[i].height);
		  cJSON_AddItemToArray(rectangle_array, pointObj);
		}
		cJSON_AddItemToObject(root, "Locations",rectangle_array);
		std::string Snapshot = Mat2Base64(alarmDataPtr->img, "jpg");
		//cJSON_AddStringToObject(root, "Snapshot", Snapshot.c_str());
		std::string body = cJSON_PrintUnformatted(root);
                std::cout<<body<<std::endl;
		cJSON_Delete(root);
		cJSON_Delete(ExtendData);
		_out_queue->tail_next();
		SendReq(serverURL, body);

	}
    _is_run =false;
    CallStop();
}


std::string base64Encode(const unsigned char* Data, int DataByte)
{
	//编码表
	const char EncodeTable[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	//返回值
	std::string strEncode;
	unsigned char Tmp[4] = { 0 };
	int LineLength = 0;
	for (int i = 0; i < (int)(DataByte / 3); i++)
	{
		Tmp[1] = *Data++;
		Tmp[2] = *Data++;
		Tmp[3] = *Data++;
		strEncode += EncodeTable[Tmp[1] >> 2];
		strEncode += EncodeTable[((Tmp[1] << 4) | (Tmp[2] >> 4)) & 0x3F];
		strEncode += EncodeTable[((Tmp[2] << 2) | (Tmp[3] >> 6)) & 0x3F];
		strEncode += EncodeTable[Tmp[3] & 0x3F];
		if (LineLength += 4, LineLength == 76) { strEncode += "\r\n"; LineLength = 0; }
	}
	//对剩余数据进行编码
	int Mod = DataByte % 3;
	if (Mod == 1)
	{
		Tmp[1] = *Data++;
		strEncode += EncodeTable[(Tmp[1] & 0xFC) >> 2];
		strEncode += EncodeTable[((Tmp[1] & 0x03) << 4)];
		strEncode += "==";
	}
	else if (Mod == 2)
	{
		Tmp[1] = *Data++;
		Tmp[2] = *Data++;
		strEncode += EncodeTable[(Tmp[1] & 0xFC) >> 2];
		strEncode += EncodeTable[((Tmp[1] & 0x03) << 4) | ((Tmp[2] & 0xF0) >> 4)];
		strEncode += EncodeTable[((Tmp[2] & 0x0F) << 2)];
		strEncode += "=";
	}
	return strEncode;
}

std::string Mat2Base64(const cv::Mat img, std::string imgType)
{
	//Mat转base64
	std::string img_data;
	std::vector<uchar> vecImg;
	std::vector<int> vecCompression_params;
	vecCompression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	vecCompression_params.push_back(90);
	imgType = "." + imgType;
	cv::imencode(imgType, img, vecImg, vecCompression_params);
	img_data = base64Encode(vecImg.data(), vecImg.size());
	return img_data;
}

