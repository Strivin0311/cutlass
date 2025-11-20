
#!/bin/bash

# NOTE: you should run `cmake_all.sh` in the root directory first before running this script

SEP="--------------------------------------------------------------------"

BUILD_ROOT=../../build
SRC_ROOT=examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling

BUILD_TARGET=68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling_all

RUN_TARGET=68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling
# RUN_TARGET=68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling_with_sparse_groups

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
    echo "Building ${BUILD_TARGET}"
    echo "$SEP"

    cd $BUILD_ROOT || exit
    make $BUILD_TARGET || exit
    cd -
else
    echo "$SEP"
    echo "Skipping build process"
    echo "$SEP"
fi

# run

CMD=$BUILD_ROOT/$SRC_ROOT/$RUN_TARGET

if [ "$SKIP_RUN" = false ]; then
    echo "$SEP"
    echo "Running ${RUN_TARGET}"
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
    echo "Profiling ${RUN_TARGET}"
    echo "$SEP"

    nsys profile \
        --force-overwrite true \
        -o ${RUN_TARGET}.nsys-rep \
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

