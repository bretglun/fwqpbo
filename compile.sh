#!/bin/bash

#EIGEN_PATH=/path/to/eigen/
EIGEN_PATH=~/Kod/recon_ks/include/eigen/

g++ -shared cpp/FW.cpp -o cpp/bin/FW.so -I $EIGEN_PATH -stdlib=libstdc++