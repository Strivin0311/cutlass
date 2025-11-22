
#!/bin/bash

# NOTE: you should run `cmake_all.sh` in the root directory first before running this script

SEP="--------------------------------------------------------------------"

BUILD_ROOT=../../build
SRC_ROOT=examples/77_blackwell_fmha

BUILD_TARGET=77_blackwell_fmha_all

# RUN_TARGET=77_blackwell_fmha_fp8
RUN_TARGET=77_blackwell_fmha_fp16
# RUN_TARGET=77_blackwell_fmha_gen_fp8
# RUN_TARGET=77_blackwell_fmha_gen_fp16
# RUN_TARGET=77_blackwell_mla_2sm_fp8
# RUN_TARGET=77_blackwell_mla_2sm_fp16
# RUN_TARGET=77_blackwell_mla_2sm_cpasync_fp8 # FIXME: cpasync page size pow2
# RUN_TARGET=77_blackwell_mla_2sm_cpasync_fp16 # FIXME: cpasync page size pow2
# RUN_TARGET=77_blackwell_fmha_bwd_fp8
# RUN_TARGET=77_blackwell_fmha_bwd_fp16
# RUN_TARGET=77_blackwell_mla_fwd_fp8
# RUN_TARGET=77_blackwell_mla_fwd_fp16

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

mask_type="full"
# mask_type="causal"
# mask_type="varlen-full"
# mask_type="varlen-causal"

k=1024
nk=8
seqlen=$(( nk * k ))

if [[ $RUN_TARGET == *"bwd"* ]]; then
    wd="bwd"
else
    wd="fwd"
fi

# uneven
# FIXME: not well supported by current implementation
# varlen="256:64:512:128:64"
# varlen_batch_size=5

# even
if [[ $seqlen -le 4096 ]] then
    varlen_unit=512
elif [[ $seqlen -le 16384 ]]; then
    varlen_unit=1024
else
    varlen_unit=2048
fi

varlen_q="${varlen_unit}"
varlen_k="${varlen_unit}"
varlen_batch_size=$(( seqlen / varlen_unit ))
for (( i=1; i<$varlen_batch_size; i++ )); do
    varlen_q="${varlen_q}:${varlen_unit}"
    varlen_k="${varlen_k}:${varlen_unit}"
done

echo "Final varlen_q: $varlen_q"
echo "Final varlen_k: $varlen_k"
echo "Final varlen_batch_size: $varlen_batch_size"


if [[ $mask_type == "full" ]]; then
    CMD="$BUILD_ROOT/$SRC_ROOT/$RUN_TARGET --q=${seqlen} --k=${seqlen} --h=8 --h_k=8 --d=128 --b=1 --mask=no"
elif [[ $mask_type == "causal" ]]; then
    CMD="$BUILD_ROOT/$SRC_ROOT/$RUN_TARGET --q=${seqlen} --k=${seqlen} --h=8 --h_k=8 --d=128 --b=1 --mask=causal"
elif [[ $mask_type == "varlen-full" ]]; then
    CMD="$BUILD_ROOT/$SRC_ROOT/$RUN_TARGET --varlen --varlen-q=${varlen_q} --varlen-k=${varlen_k} --h=8 --h_k=8 --d=128 --b=${varlen_batch_size} --mask=no"
elif [[ $mask_type == "varlen-causal" ]]; then
    CMD="$BUILD_ROOT/$SRC_ROOT/$RUN_TARGET --varlen --varlen-q=${varlen_q} --varlen-k=${varlen_k} --h=8 --h_k=8 --d=128 --b=${varlen_batch_size} --mask=causal"
else
    echo "Unknown mask type: $mask_type"
    exit 1
fi

LOG_ROOT=logs/${mask_type}/${wd}/
mkdir -p $LOG_ROOT
LOG_PATH=$LOG_ROOT/${RUN_TARGET}_${nk}k.log
echo "Log path: $LOG_PATH"

if [ "$SKIP_RUN" = false ]; then
    echo "$SEP"
    echo "Running ${RUN_TARGET}"
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

