#!/bin/bash
ARCH=`arch`
../../bin/gcc_${ARCH}_Release/TRT_CPN -gpu=0 -b=1 -sb=1 -t=1 -m=cpn -w=192 -h=256 -d=1
