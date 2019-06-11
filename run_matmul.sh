#!/bin/bash -x

if [[ $# -ne 3 ]]; then
	echo "[size in-memory] [size out-of-core] [iterations]"
	exit 1
fi

size_in_memory=$1
size_oversubscribe=$2
iter=$3

[ -z "$base_path" ] && base_path=/home/steven/Programs/UnifiedMemory

cpupower frequency-set --governor performance

base_path=/home/steven/Programs/UnifiedMemory

cd ${base_path}/matrixMul/matrixMul
size=$size_in_memory
for i in `seq 0 $iter`; do
	./matrixMul -wA=${size} -hA=${size} -wB=${size} -hB=${size} -iterations=1 > run_normal_${i}.txt 2>&1
done

for version in matrixMulUM matrixMulUM_hint matrixMulUM_prefetch matrixMulUM_hint_prefetch; do
#for version in matrixMulUM_hint matrixMulUM_prefetch matrixMulUM_hint_prefetch; do
	cd ${base_path}/matrixMul/${version}
	size=$size_in_memory

	for i in `seq 0 $iter`; do
		./matrixMul -wA=${size} -hA=${size} -wB=${size} -hB=${size} -iterations=1 > run_normal_${i}.txt 2>&1
	done

#	size=32768
#	for i in `seq 0 5`; do
#		./matrixMul -wA=${size} -hA=${size} -wB=${size} -hB=${size} -iterations=${iter} > run_oversubscribe_${i}.txt 2>&1
#	done
#	srun --mem-bind=local -u  -n 1 --accel-bind=g nvprof -f -o prof_oversubscribe.nvvp ./matrixMul -wA=${size} -hA=${size} -wB=${size} -hB=${size} -iterations=${iter}
done
