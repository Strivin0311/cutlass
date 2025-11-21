
#!/bin/bash

# NOTE: you should run `cmake_all.sh` in the root directory first before running this script

SEP="--------------------------------------------------------------------"

BUILD_ROOT=../../../../build
SRC_ROOT=examples/cute/tutorial/blackwell

# TARGET=cute_tutorial_01_mma_sm100
TARGET=cute_tutorial_02_mma_tma_sm100
# TARGET=cute_tutorial_03_mma_tma_multicast_sm100
# TARGET=cute_tutorial_04_mma_tma_2sm_sm100
# TARGET=cute_tutorial_05_mma_tma_epi_sm100

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

    cd $BUILD_ROOT || exit
    make $TARGET || exit
    cd -
else
    echo "$SEP"
    echo "Skipping build process"
    echo "$SEP"
fi

# run

CMD=$BUILD_ROOT/$SRC_ROOT/$TARGET

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

