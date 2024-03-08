#! /bin/bash

cd 3rd/FBoW
mkdir build
mkdir install
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
make -j4 install