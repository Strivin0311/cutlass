
#!/bin/bash

# NOTE: you should run `cmake_all.sh` in the root directory first before running this script

SEP="--------------------------------------------------------------------"

BUILD_ROOT=../../build
SRC_ROOT=examples/03_visualize_layout

TARGET=03_visualize_layout

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

CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET RowMajor --extent=16,8 --output-shape=8 --vectorize=1"
# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET ColumnMajor --extent=16,8 --output-shape=16 --vectorize=1"

# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET RowMajorInterleaved<1> --extent=16,8 --output-shape=8 --vectorize=1"
# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET RowMajorInterleaved<1> --extent=16,8 --output-shape=8 --vectorize=4"
# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET RowMajorInterleaved<4> --extent=16,8 --output-shape=8 --vectorize=1"
# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET RowMajorInterleaved<8> --extent=16,8 --output-shape=8 --vectorize=1"

# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET ColumnMajorInterleaved<1> --extent=16,8 --output-shape=16 --vectorize=1"
# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET ColumnMajorInterleaved<1> --extent=16,8 --output-shape=16 --vectorize=4"
# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET ColumnMajorInterleaved<4> --extent=16,8 --output-shape=16 --vectorize=1"
# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET ColumnMajorInterleaved<4> --extent=16,8 --output-shape=16 --vectorize=4"

# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET TensorOpMultiplicand<4,64> --extent=64,64 --vectorize=32 --output-shape=256,4"
# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET TensorOpMultiplicand<4,128> --extent=128,32 --vectorize=32 --output-shape=256,4"
# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET TensorOpMultiplicand<4,256> --extent=256,16 --vectorize=32 --output-shape=256,4"
# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET TensorOpMultiplicand<8,32> --extent=32,64 --vectorize=16 --output-shape=128,4"
# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET TensorOpMultiplicand<8,64> --extent=64,32 --vectorize=16 --output-shape=128,4"
# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET TensorOpMultiplicand<8,64> --extent=64,32 --vectorize=16 --output-shape=128,4"
# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET TensorOpMultiplicand<8,128> --extent=128,16 --vectorize=16 --output-shape=128,4"
# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET TensorOpMultiplicand<16,32> --extent=32,32 --vectorize=8 --output-shape=64,4"
# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET TensorOpMultiplicand<16,64> --extent=64,16 --vectorize=8 --output-shape=64,4"
# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET TensorOpMultiplicand<32,16> --extent=16,32 --vectorize=4 --output-shape=32,4"
# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET TensorOpMultiplicand<32,32> --extent=32,16 --vectorize=4 --output-shape=32,4"
# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET TensorOpMultiplicandCongruous<32,32> --extent=32,16 --vectorize=4 --output-shape=32,4"
# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET TensorOpMultiplicandCongruous<64, 16> --extent=16,16 --vectorize=2 --output-shape=16,4"

# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET VoltaTensorOpMultiplicandCrosswise<16,32> --extent=32,64 --vectorize=4 --output-shape=64,4"
# CMD="$BUILD_ROOT/$SRC_ROOT/$TARGET VoltaTensorOpMultiplicandCongruous<16> --extent=64,32 --vectorize=8 --output-shape=64,4"

if [ "$SKIP_RUN" = false ]; then
    echo "$SEP"
    echo "Running ${TARGET}"
    echo "$SEP"
    $CMD > $TARGET.log 2>&1
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

