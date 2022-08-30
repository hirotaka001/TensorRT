#!/bin/bash
ARCH=`arch`
../../bin/gcc_${ARCH}_Release/TRT_YOLOv4 -gpu=0 -b=1 -sb=1 -t=1 -m=yolov4 -w=512 -h=512 -ppc=1 -c=80 -sc=0.5 -n=0.5 -d=1
