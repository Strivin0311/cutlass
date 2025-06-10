
SEP="--------------------------------------------------------------------"

BUILD_ROOT=./build
SRC_ROOT=.
COMMON_INCLUDE=../common

TARGET=basic_gemm


echo "$SEP"
echo "Building ${TARGET}"
echo "$SEP"

mkdir -p $BUILD_ROOT

NVCC_GENCODE="arch=compute_90,code=sm_90"

nvcc -ccbin g++ -gencode=$NVCC_GENCODE \
-lcuda -lcudart \
-I${COMMON_INCLUDE} \
-o $BUILD_ROOT/$TARGET $SRC_ROOT/$TARGET.cu


echo "$SEP"
echo "Running ${TARGET}"
echo "$SEP"

CMD=$BUILD_ROOT/$TARGET

$CMD


echo "$SEP"
echo "Profiling ${TARGET}"
echo "$SEP"

nsys profile \
    --force-overwrite true \
    -o ${TARGET}.nsys-rep \
    --capture-range=cudaProfilerApi \
    $CMD


echo "$SEP"
echo "Done"
echo "$SEP"

