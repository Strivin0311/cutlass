
#!/bin/bash

# NOTE: you should run `cmake_all.sh` in the root directory first before running this script

SEP="--------------------------------------------------------------------"

BUILD_ROOT=../../build
SRC_ROOT=examples/83_blackwell_sparse_gemm

TARGET=83_blackwell_sparse_gemm

# default not skip any step except profiling
SKIP_BUILD=true
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

k=1024
nk=8
M=$(( nk * k ))
N=$(( nk * k ))
K=$(( nk * k ))

if [[ $K -lt 1024 ]]; then
    nk_k=$K
else
    nk_k="$(( K / k ))k"
fi

CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET --m=$M --n=$N --k=$K"

LOG_ROOT=logs
mkdir -p $LOG_ROOT
LOG_PATH=$LOG_ROOT/${TARGET}_M${nk}k_N${nk}k_K${nk_k}.log
echo "Log path: $LOG_PATH"

if [ "$SKIP_RUN" = false ]; then
    echo "$SEP"
    echo "Running ${TARGET}"
    echo "$SEP"
    $CMD > $LOG_PATH 2>&1
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

