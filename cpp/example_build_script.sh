# example build script:
# adjust the path for Eigen in the cmake step below and then compile using these commands

mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../bin/Release -DEIGEN_INCLUDE_DIR=/home/lee8rx/anaconda/envs/pyir/include/eigen3 ..
make -j4
make install
