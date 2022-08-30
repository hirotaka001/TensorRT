#ifndef __TRT_COMMON_H__
#define __TRT_COMMON_H__

#define TRT_VERSION_MAJOR 1
#define TRT_VERSION_MINOR 4

#if __has_include(<kros/trt/TRT_YOLOv4/TRT_YOLOv4.h>)
#include <kros/trt/TRT_YOLOv4/TRT_YOLOv4.h>
#endif

#if __has_include(<kros/trt/TRT_YOLOv3/TRT_YOLOv3.h>)
#include <kros/trt/TRT_YOLOv3/TRT_YOLOv3.h>
#endif

#if __has_include(<kros/trt/TRT_CPN/TRT_CPN.h>)
#include <kros/trt/TRT_CPN/TRT_CPN.h>
#endif

#if __has_include(<kros/trt/TRT_ResNet50/TRT_ResNet50.h>)
#include <kros/trt/TRT_ResNet50/TRT_ResNet50.h>
#endif

#if __has_include(<kros/trt/TRT_ResNet50_Feature/TRT_ResNet50_Feature.h>)
#include <kros/trt/TRT_ResNet50_Feature/TRT_ResNet50_Feature.h>
#endif

#endif
