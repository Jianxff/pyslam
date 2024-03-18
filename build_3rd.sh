#! /bin/bash

cd 3rd

cd eigen-3.4.0
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../../install
make -j4 install
cd ../..


cd FBoW
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../../install
make -j4 install
cd ../..

cd g2o
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../../install
make -j4 install
cd ../..