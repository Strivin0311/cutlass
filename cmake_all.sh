#!/bin/bash

BUILD_ROOT=build
ARCH=90

rm -rf $BUILD_ROOT && mkdir -p $BUILD_ROOT && cd $BUILD_ROOT

cmake .. -DCUTLASS_NVCC_ARCHS=$ARCH