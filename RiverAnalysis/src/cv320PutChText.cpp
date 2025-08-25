#include"cv320PutChText.h"

cv320PutChText::cv320PutChText(const char *freeType)
{
	assert(freeType != NULL);
	if (FT_Init_FreeType(&m_library))
		throw;
	if (FT_New_Face(m_library, freeType, 0, &m_face))
		throw;
	setlocale(LC_ALL, "");
}
cv320PutChText::~cv320PutChText()
{
}

int cv320PutChText::Loadttc(const char* ttc_path) {
    if (ttc_path == NULL)
        return -1;
    if (FT_Init_FreeType(&m_library))
        return -1;
    if (FT_New_Face(m_library, ttc_path, 0, &m_face))
        return -1;
    setlocale(LC_ALL, "");
    return 0;
}

void cv320PutChText::release() {
	FT_Done_Face(m_face);
    FT_Done_FreeType(m_library);
}

/********************************
*函数名：getfont
*描述：获取字体类型
*参数：
size:字体大小/空白比例/间隔比例/旋转角度
underline:下划线
diaphaneity:透明度
**********************************/
void cv320PutChText::getFont(int *type,cv::Scalar *size, bool *underline, float *diaphaneity)
{
	if (type) *type = m_fontType;
	if (size) *size = m_fontSize;
	if (underline) *underline = m_fontUnderline;
	if (diaphaneity) *diaphaneity = m_fontDiaphaneity;
}

/********************************
*函数名：setfont
*描述：设置字体类型
*参数：
size:字体大小/空白比例/间隔比例/旋转角度
underline:下划线
diaphaneity:透明度
**********************************/
void cv320PutChText::setFont(int *type, cv::Scalar *size, bool *underline, float *diaphaneity)
{
	// 参数合法性检查 
	if (type)
	{
		if (type >= 0) m_fontType = *type;
	}
	if (size)
	{
		m_fontSize.val[0] = fabs(size->val[0]);
		m_fontSize.val[1] = fabs(size->val[1]);
		m_fontSize.val[2] = fabs(size->val[2]);
		m_fontSize.val[3] = fabs(size->val[3]);
	}
	if (underline)
	{
		m_fontUnderline = *underline;
	}
	if (diaphaneity)
	{
		m_fontDiaphaneity = *diaphaneity;
	}
	FT_Set_Pixel_Sizes(m_face, (int)m_fontSize.val[0], 0);
}
/********************************
*函数名：restorefont
*描述：恢复原始的字体类型
*参数：
**********************************/
void cv320PutChText::restoreFont(const int frontSize)
{
	m_fontType = 0;            // 字体类型(不支持)  
	m_fontSize.val[0] = frontSize;      // 字体大小  
	m_fontSize.val[1] = 0.5;   // 空白字符大小比例  
	m_fontSize.val[2] = 0.2;   // 间隔大小比例  
	m_fontSize.val[3] = 0;      // 旋转角度(不支持)  
	m_fontUnderline = false;   // 下画线(不支持)  
	m_fontDiaphaneity = 1.0;   // 色彩比例(可产生透明效果)  
							   // 设置字符大小  
	FT_Set_Pixel_Sizes(m_face, (int)m_fontSize.val[0], 0);
}

/********************************
*函数名：putfont
*描述：输出汉字，遇到不能输出的字符将停止
*参数：
frame:输出的图像
text：文本内容
pos:文本位置
返回值：返回成功输出的字符长度，失败返回-1
**********************************/
int cv320PutChText::putText(cv::Mat &frame, const char* text, cv::Point pos,const int frontSize)
{
	return putText(frame, text, pos, CV_RGB(255, 255, 255),frontSize);
}

/********************************
*函数名：putText
*描述：输出汉字，遇到不能输出的字符将停止
*参数：
frame:输出的图像
text：文本内容
pos:文本位置
color:文本颜色
返回值：返回成功输出的字符长度，失败返回-1
**********************************/
int cv320PutChText::putText(cv::Mat &frame, std::string& text, cv::Point pos, cv::Scalar color,const int frontSize)
{
    return putText(frame, stows(text).c_str(), pos, color, frontSize);
}


int cv320PutChText::putText(cv::Mat &frame, const char *text, cv::Point pos, cv::Scalar color,const int frontSize)
 {
	/*
    
    int i = 0;
	restoreFont(frontSize);
	if (frame.empty()) return -1;
	if (text == NULL) return -1;
	pos.y += frontSize;

	for ( i = 0; text[i] != '\0'; ++i)
	{
		wchar_t wc = text[i];
		if (!isascii(wc))
		{
			// 解析双字节符号
			mbtowc(&wc, &text[i++], 2);
			
		}
		// 输出当前的字符 
		putWChar(frame, wc, pos, color);
	}
	return i;
    */
    std::string input = text;
    return putText(frame, stows(input).c_str(), pos, color, frontSize);
}

int cv320PutChText::putText(cv::Mat &frame, const wchar_t *text, cv::Point pos, cv::Scalar color,const int frontSize)
{
	int i = 0;
	restoreFont(frontSize);
	if (frame.empty()) return -1;
	if (text == NULL) return -1;
	pos.y += frontSize;
	for ( i = 0; text[i] != '\0'; ++i)
	{
		putWChar(frame, text[i], pos, color);
	}
	return i;
}

std::wstring cv320PutChText::stows(const std::string& s) {
    if (s.empty()) {
        return L"";
    }
    unsigned len = s.size() + 1;
    setlocale(LC_CTYPE, "en_US.UTF-8");
    wchar_t *p = new wchar_t[len];
    mbstowcs(p, s.c_str(), len);
    std::wstring w_str(p);
    delete[] p;
    return w_str;
}

void cv320PutChText::putWChar(cv::Mat &frame, wchar_t wc, cv::Point &pos, cv::Scalar color)
{
	// 根据unicode生成字体的二值位图  
	IplImage img = frame;
	FT_UInt glyph_index = FT_Get_Char_Index(m_face, wc);
	FT_Load_Glyph(m_face, glyph_index, FT_LOAD_DEFAULT);
	FT_Render_Glyph(m_face->glyph, FT_RENDER_MODE_MONO);
	FT_GlyphSlot slot = m_face->glyph;
	// 行列数  
	int rows = slot->bitmap.rows;
	int cols = slot->bitmap.width;
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			int off = ((img.origin == 0) ? i : (rows - 1 - i))* slot->bitmap.pitch + j / 8;
			if (slot->bitmap.buffer[off] & (0xC0 >> (j % 8)))
			{
				int r = (img.origin == 0) ? pos.y - (rows - 1 - i) : pos.y + i;;
				int c = pos.x + j;
				if (r >= 0 && r < img.height && c >= 0 && c < img.width)
				{
					cv::Scalar scalar = cvGet2D(&img, r, c);
					float p = m_fontDiaphaneity;
					for (int k = 0; k < 4; ++k)// 进行色彩融合  
					{
						scalar.val[k] = scalar.val[k] * (1 - p) + color.val[k] * p;
					}
					cvSet2D(&img, r, c, scalar);
				}
			}
		}  
	}
	  // 修改下一个字的输出位置 
	double space = m_fontSize.val[0] * m_fontSize.val[1];
	double sep = m_fontSize.val[0] * m_fontSize.val[2];
	pos.x += (int)((cols ? cols : space) + sep);
}
