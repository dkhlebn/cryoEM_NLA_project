#!/bin/bash


DATASET=$1
PROJECT_UID=$2
PROJECT_WD=$3
PROJECT_NAME=$4
WORKSPACE=$5
META=$6
RANKS=$7

HOST_NAME=`hostname`
mrc=("" "--mrc_ttsvd_job" "--mrc_tucker_job")
cls=("" "--cls_ttsvd_job" "--cls_tucker_job")

./fetch_EMPIAR.sh ${DATASET}

for mrc_mode in "${mrc[@]}"; do
    for cls_mode in "${cls[@]}"; do
        python3 pipeline.py \
            --project ${PROJECT_UID} \
            --workspace ${WORKSPACE} \
            --hostname ${HOST_NAME}\
            --tsv ${META} \
            --dataset_dir ${DATASET} \
            ${mrc_mode} --mrc_ranks ${RANKS} \
            ${cls_mode} --cls_ranks ${RANKS};
    done;
done

rm -rf ./${PROJECT_WD}/${PROJECT_NAME}
rm -rf ./${DATASET}
