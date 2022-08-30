#!/bin/bash
bash uninstall.sh
ARCH=`arch`

if type sudo > /dev/null 2>&1; then
    SUDO_COM=sudo
else
    SUDO_COM=""
fi

$SUDO_COM mkdir -p /usr/local/include/kros/trt/TRT
$SUDO_COM mkdir -p /usr/local/include/kros/trt/##MODEL##
$SUDO_COM cp ../../include/kros/trt.h /usr/local/include/kros
$SUDO_COM cp -r ../../include/kros/trt/TRT /usr/local/include/kros/trt
$SUDO_COM cp -r ../../include/kros/trt/##MODEL## /usr/local/include/kros/trt
$SUDO_COM cp ../../lib/${ARCH}/##MODEL_LIB## /usr/local/lib/
$SUDO_COM ldconfig
