#!/usr/bin/env bash

REGION="asia-east1"
TIER="BASIC_GPU"
BUCKET="ml-specialized"

BASE_NAME="rossmann" # change to your model name

PACKAGE_PATH=../trainer # this can be a gcs location to a zipped and uploaded package
TRAIN_FILES=gs://${BUCKET}/rossmann/data/processed/transformed/tr.pkl
VALID_FILES=gs://${BUCKET}/rossmann/data/processed/transformed/vl.pkl
MODEL_DIR=gs://${BUCKET}/${BASE_NAME}/models

CURRENT_DATE=`date +%Y%m%d_%H%M%S`
# JOB_NAME=train_${BASE_NAME}_${TIER}_${CURRENT_DATE}
JOB_NAME=tune_${BASE_NAME}_${CURRENT_DATE} # for hyper-parameter tuning jobs

# train configs
MODEL_NAME=neu_mf # dnn or neu_mf
TRAIN_STEPS=2308
VALID_STEPS=989
VERBOSITY=INFO
SAVE_CHECKPOINTS_STEPS=2308
THROTTLE_SECS=60

gcloud ml-engine jobs submit training ${JOB_NAME} \
    --job-dir=${MODEL_DIR} \
    --runtime-version=1.10 \
    --region=${REGION} \
    --scale-tier=${TIER} \
    --module-name=trainer.ctrl \
    --package-path=${PACKAGE_PATH}  \
    --config=../config.yaml \
    -- \
    --method=train \
    --model-name=${MODEL_NAME} \
    --train-files=${TRAIN_FILES} \
    --valid-files=${VALID_FILES} \
    --train-steps=${TRAIN_STEPS} \
    --valid-steps=${VALID_STEPS} \
    --verbosity=${VERBOSITY} \
    --save-checkpoints-steps=${SAVE_CHECKPOINTS_STEPS} \
    --throttle-secs=${THROTTLE_SECS} \

    # below tuned by ml engine, see config.yaml
    # --embedding-size=16 \
    # --first-mlp-layer-size= 512 \
    # --first-factor-layer-size= 32 \
    # --scale_factor=0.7 \
    # --drop-rate=0.3 \
    # --num-layers=3 \
    # --learning-rate:=0.001
