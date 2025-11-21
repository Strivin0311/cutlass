#!/bin/bash

BUILD_ROOT=build

# NOTE: don't forget 'a' at the end, according to https://github.com/NVIDIA/cutlass?tab=readme-ov-file#target-architecture
ARCH=90a
# ARCH=100a 

OPTIONS=""
# OPTIONS="-DCUTLASS_LIBRARY_KERNELS=all"
# OPTIONS="-DGOOGLETEST_DIR=/path/to/googletest"

rm -rf $BUILD_ROOT && mkdir -p $BUILD_ROOT && cd $BUILD_ROOT

cmake --debug-output .. -DCUTLASS_NVCC_ARCHS=$ARCH $OPTIONS