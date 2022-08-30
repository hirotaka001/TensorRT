#!/bin/bash
if type sudo > /dev/null 2>&1; then
    SUDO_COM=sudo
else
    SUDO_COM=""
fi

$SUDO_COM rm -f /usr/local/include/kros/trt.h
$SUDO_COM rm -rf /usr/local/include/kros/trt/TRT
$SUDO_COM rm -rf /usr/local/include/kros/trt/##MODEL##
$SUDO_COM rm -f /usr/local/lib/##MODEL_LIB##
$SUDO_COM ldconfig
