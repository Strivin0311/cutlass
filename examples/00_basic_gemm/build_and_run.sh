
#!/bin/bash

SEP="--------------------------------------------------------------------"

BUILD_ROOT=build
SRC_ROOT=.
COMMON_INCLUDE=../common

TARGET=basic_gemm

# default not skip any step except profiling
SKIP_BUILD=false
SKIP_RUN=false
SKIP_PROFILE=true

# parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            ;;
        --skip-run)
            SKIP_RUN=true
            ;;
        --skip-profile)
            SKIP_PROFILE=true
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# build
if [ "$SKIP_BUILD" = false ]; then
    echo "$SEP"
    echo "Building ${TARGET}"
    echo "$SEP"

    rm -rf $BUILD_ROOT && mkdir -p $BUILD_ROOT || exit

    NVCC_GENCODE="arch=compute_90,code=sm_90"

    nvcc -ccbin g++ -gencode=$NVCC_GENCODE \
    -lcuda -lcudart \
    -I${COMMON_INCLUDE} \
    -o $BUILD_ROOT/$TARGET $SRC_ROOT/$TARGET.cu
else
    echo "$SEP"
    echo "Skipping build process"
    echo "$SEP"
fi

# run

CMD=$BUILD_ROOT/$TARGET

if [ "$SKIP_RUN" = false ]; then
    echo "$SEP"
    echo "Running ${TARGET}"
    echo "$SEP"
    $CMD
else
    echo "$SEP"
    echo "Skipping run process"
    echo "$SEP"
fi

# profile
if [ "$SKIP_PROFILE" = false ]; then
    echo "$SEP"
    echo "Profiling ${TARGET}"
    echo "$SEP"

    nsys profile \
        --force-overwrite true \
        -o ${TARGET}.nsys-rep \
        --capture-range=cudaProfilerApi \
        $CMD
else
    echo "$SEP"
    echo "Skipping profiling process"
    echo "$SEP"
fi


echo "$SEP"
echo "Done"
echo "$SEP"

