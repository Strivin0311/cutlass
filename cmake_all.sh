#!/bin/bash

BUILD_ROOT=build
ARCH=90a # NOTE: don't forget 'a' at the end, according to https://github.com/NVIDIA/cutlass?tab=readme-ov-file#target-architecture

OPTIONS=""
# OPTIONS="-DCUTLASS_LIBRARY_KERNELS=all"

rm -rf $BUILD_ROOT && mkdir -p $BUILD_ROOT && cd $BUILD_ROOT

cmake .. -DCUTLASS_NVCC_ARCHS=$ARCH $OPTIONS