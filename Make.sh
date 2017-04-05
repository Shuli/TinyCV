#!/bin/bash

export Qt5Core_DIR=/usr/local/Qt-5.8.0/
export Qt5Gui_DIR=/usr/local/Qt-5.8.0/
export Qt5Widgets_DIR=/usr/local/Qt-5.8.0/
export Qt5Test_DIR=/usr/local/Qt-5.8.0/
export Qt5Concurrent_DIR=/usr/local/Qt-5.8.0/
export Qt5OpenGL_DIR=/usr/local/Qt-5.8.0/

cmake . -DBUILD_CPU=ON -DBUILD_CUDA=ON -DBUILD_CL=ON -DBUILD_X5=ON -DBUILD_MEASURE=ON -DBUILD_VIDEO=ON
make


