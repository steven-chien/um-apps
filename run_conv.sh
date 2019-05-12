#!/bin/bash -x

if [[ $# -ne 7 ]]; then
	echo "[conv0 size in-memory] [conv0 size out-of-core] [conv1 size in-memory] [conv1 size out-of-core] [conv2 size in-memory] [conv2 size out-of-core] [iterations]"
	exit 1
fi

conv0_size_in_memory=$1
conv0_size_oversubscribe=$2
conv1_size_in_memory=$3
conv1_size_oversubscribe=$4
conv2_size_in_memory=$5
conv2_size_oversubscribe=$6
iter=$7

[ -z "$base_path" ] && base_path=/home/steven/Programs/UnifiedMemory

cpupower frequency-set --governor performance

cd ${base_path}/convolutionFFT2D/convolutionFFT2D
size=$conv0_size_in_memory
./convolutionFFT2D -size=${size} -fft0Iter=${iter} > run_normal_conv0.txt 2>&1
size=$conv1_size_in_memory
./convolutionFFT2D -size=${size} -fft1Iter=${iter} > run_normal_conv1.txt 2>&1
size=$conv2_size_in_memory
./convolutionFFT2D -size=${size} -fft2Iter=${iter} > run_normal_conv2.txt 2>&1

for version in convolutionFFT2DUM convolutionFFT2DUM_hint convolutionFFT2DUM_prefetch convolutionFFT2DUM_hint_prefetch; do
	cd ${base_path}/convolutionFFT2D/${version}
	size=$conv0_size_in_memory
	./convolutionFFT2D -size=${size} -fft0Iter=${iter} > run_normal_conv0.txt 2>&1
	size=$conv1_size_in_memory
	./convolutionFFT2D -size=${size} -fft1Iter=${iter} > run_normal_conv1.txt 2>&1
	size=$conv2_size_in_memory
	./convolutionFFT2D -size=${size} -fft2Iter=${iter} > run_normal_conv2.txt 2>&1

	size=$conv0_size_oversubscribe
	./convolutionFFT2D -size=${size} -fft0Iter=${iter} > run_oversub_conv0.txt 2>&1
	size=$conv1_size_oversubscribe
	./convolutionFFT2D -size=${size} -fft1Iter=${iter} > run_oversub_conv1.txt 2>&1
	size=$conv2_size_oversubscribe
	./convolutionFFT2D -size=${size} -fft2Iter=${iter} > run_oversub_conv2.txt 2>&1
done
