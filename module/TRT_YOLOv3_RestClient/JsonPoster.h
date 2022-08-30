#pragma once

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4244)
#endif

#include <string>
#include <codecvt> 
#include <cpprest/http_client.h>

#ifdef _WIN32
#pragma warning(pop)
#endif

using namespace web;
using namespace web::http;
using namespace web::http::client;

class JsonPoster
{
private:
	std::wstring wstr_url;
	std::string str_url;
public:
	JsonPoster(const std::string post_url = "")
	{
		// for Linux
		str_url = post_url;

		// for Windows
		std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> conv;
		wstr_url = conv.from_bytes(post_url);
	}

	void PostDetectedResults(std::vector<std::vector<float> >& detection_result)
	{
		try
		{
			Post(wstr_url, str_url, detection_result).wait();
		}
		catch (...)
		{
			std::cout << "Post Error!" << std::endl;
		}
	}

	pplx::task<int> Post(std::wstring wstr_url, const std::string str_url, std::vector<std::vector<float> >& detection_result)
	{
		return pplx::create_task([wstr_url, str_url, detection_result]
			{
				json::value postData;

				postData[U("num")] = json::value::number((int)detection_result.size());

				for (int idx = 0; idx < (int)detection_result.size(); idx++)
				{
					postData[U("objects")][idx][U("x")] = json::value::number(detection_result[idx][0]);
					postData[U("objects")][idx][U("y")] = json::value::number(detection_result[idx][1]);
					postData[U("objects")][idx][U("w")] = json::value::number(detection_result[idx][2]);
					postData[U("objects")][idx][U("h")] = json::value::number(detection_result[idx][3]);
					postData[U("objects")][idx][U("class_id")] = json::value::number(detection_result[idx][4]);
					postData[U("objects")][idx][U("score")] = json::value::number(detection_result[idx][5]);
				}

#ifdef _WIN32
				http_client client(wstr_url);
#else
				http_client client(U(str_url));
#endif // _WIN32

				return client.request(methods::POST, U(""), postData.serialize(), U("application/json"));
			}).then([](http_response response)
				{
					if (response.status_code() == status_codes::OK)
					{
						return response.extract_json();
					}
					pplx::task<json::value> nullData;
					return nullData;
				}).then([](json::value json)
					{
						// リザルトコードを返す
						return /*json[U("success")].as_bool() ? 0 : -1*/0;
					});
	}
};
