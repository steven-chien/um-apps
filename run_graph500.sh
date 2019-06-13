#!/bin/bash -x

if [[ $# -ne 4 ]]; then
	echo "[scale in memory] [edge in memory] [scale out-of-core] [edge out-of-core]"
	exit 1
fi

#24 28 25 28

iter=0
scale_in_memory=$1
edge_in_memory=$2
scale_oversubscribe=$3
edge_oversubscribe=$4

[ -z "$base_path" ] && base_path=/home/steven/Programs/UnifiedMemory

cpupower frequency-set --governor performance

cd ${base_path}/Graph500_GPU/gpu-csr
scale=$scale_in_memory
edge=$edge_in_memory
for i in `seq 0 ${iter}`; do
	./romeo -s ${scale} -e ${edge} > run_normal_explicit_${i}.txt 2>&1
done

for version in romeo_um romeo_um_advise romeo_um_prefetch romeo_um_both; do
	scale=$scale_in_memory
	edge=$edge_in_memory

	for i in `seq 0 ${iter}`; do
		./${version} -s ${scale} -e ${edge} > run_normal_${version}_${i}.txt 2>&1
	done

#	scale=$scale_oversubscribe
#	edge=$edge_oversubscribe
#
#	for i in `seq 0 ${iter}`; do
#		./${version} -s ${scale} -e ${edge} > run_oversubscribe_${version}_${i}.txt 2>&1
#	done
done
