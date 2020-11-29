#!/usr/bin/env sh

set

cd /tf_research

python object_detection/model_main_tf2.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --checkpoint_dir=${MODEL_DIR} \
    --eval_timeout=10000 \
    --alsologtostderr
