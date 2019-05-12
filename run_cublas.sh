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

cd ${base_path}/simpleCUBLAS/simpleCUBLAS
size=$size_in_memory
for i in `seq 0 ${iter}`; do
	./simpleCUBLAS -iterations=1 -size=${size} > run_normal_${i}.txt 2>&1
done

for version in simpleCUBLASUM simpleCUBLASUM_hint simpleCUBLASUM_hint_prefetch simpleCUBLASUM_prefetch; do
	cd ${base_path}/simpleCUBLAS/${version}
	size=$size_in_memory
	for i in `seq 0 ${iter}`; do
		./simpleCUBLAS -iterations=1 -size=${size} > run_normal_${i}.txt 2>&1
	done

	size=$size_oversubscribe
	for i in `seq 0 ${iter}`; do
		./simpleCUBLAS -iterations=1 -size=${size} > run_oversubscribe_${i}.txt 2>&1
	done
done