#ifndef __LINK_OPENCV_H__
#define __LINK_OPENCV_H__

#ifdef _WIN32

#include <opencv2/core/version.hpp>

// 使用するモジュールのコメントアウトを適宜有効にしてください

#define OPENCV_VER_STR CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define OPENCV_EXT_STR "d.lib"
#else
#define OPENCV_EXT_STR ".lib"
#endif

#define OPENCV_VERSION(a,b,c) (((a) << 16) + ((b) << 8) + (c))
#define OPENCV_VERSION_CODE OPENCV_VERSION(CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION)

//#pragma comment(lib, "opencv_calib3d" OPENCV_VER_STR OPENCV_EXT_STR) // opencv_calib3d
//#pragma comment(lib, "opencv_contrib" OPENCV_VER_STR OPENCV_EXT_STR) // opencv_contrib
#pragma comment(lib, "opencv_core" OPENCV_VER_STR OPENCV_EXT_STR) // opencv_core
//#pragma comment(lib, "opencv_dnn" OPENCV_VER_STR OPENCV_EXT_STR) // opencv_dnn
//#pragma comment(lib, "opencv_features2d" OPENCV_VER_STR OPENCV_EXT_STR) // opencv_features2d
//#pragma comment(lib, "opencv_flann" OPENCV_VER_STR OPENCV_EXT_STR) // opencv_flann
//#pragma comment(lib, "opencv_gpu" OPENCV_VER_STR OPENCV_EXT_STR) // opencv_gpu
//#pragma comment(lib, "opencv_haartraining_engine" "" OPENCV_EXT_STR) // opencv_haartraining
#pragma comment(lib, "opencv_highgui" OPENCV_VER_STR OPENCV_EXT_STR) // opencv_highgui
#pragma comment(lib, "opencv_imgcodecs" OPENCV_VER_STR OPENCV_EXT_STR) // opencv_imgproc
#pragma comment(lib, "opencv_imgproc" OPENCV_VER_STR OPENCV_EXT_STR) // opencv_imgproc
//#pragma comment(lib, "opencv_legacy" OPENCV_VER_STR OPENCV_EXT_STR) // opencv_legacy
//#pragma comment(lib, "opencv_ml" OPENCV_VER_STR OPENCV_EXT_STR) // opencv_ml
//#pragma comment(lib, "opencv_nonfree" OPENCV_VER_STR OPENCV_EXT_STR) // opencv_nonfree
//#pragma comment(lib, "opencv_objdetect" OPENCV_VER_STR OPENCV_EXT_STR) // opencv_objdetect
//#pragma comment(lib, "opencv_photo" OPENCV_VER_STR OPENCV_EXT_STR) // opencv_photo
//#pragma comment(lib, "opencv_stitching" OPENCV_VER_STR OPENCV_EXT_STR) // opencv_stitching
//#pragma comment(lib, "opencv_ts" OPENCV_VER_STR OPENCV_EXT_STR) // opencv_ts
//#pragma comment(lib, "opencv_video" OPENCV_VER_STR OPENCV_EXT_STR) // opencv_video
#pragma comment(lib, "opencv_videoio" OPENCV_VER_STR OPENCV_EXT_STR) // opencv_videoio
//#pragma comment(lib, "opencv_videostab" OPENCV_VER_STR OPENCV_EXT_STR) // opencv_videostab

#endif /* _WIN32 */

#endif /* __LINK_OPENCV_H__ */