#!/usr/bin/env sh

set
# protoc --version

python patch_config.py

cd /tf_research

python object_detection/model_main_tf2.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --checkpoint_every_n=500 \
    --checkpoint_max_to_keep=25 \
    --alsologtostderr
