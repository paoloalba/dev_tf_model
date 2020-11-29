#!/usr/bin/env sh

set

cd /tf_research

python object_detection/exporter_main_v2.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --input_type image_tensor \
    --trained_checkpoint_dir=${MODEL_DIR} \
    --output_directory ${OUTPUT_EXTRACT_MODEL_DIR}
