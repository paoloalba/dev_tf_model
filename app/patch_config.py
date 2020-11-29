import os

from helpers.pipeline_patcher import MLPipelinePatcher

prm_stg = '/mnt/mounted_volume/permanent_storage/'
src_dir = os.getenv("SAMPLE_DIR")
model_base_name = os.getenv("MODEL_BASE_NAME")

if src_dir:
    if not os.path.isdir(src_dir):
        raise Exception("Not a valid SAMPLE_DIR: {0}".format(src_dir))
else:
    raise Exception("Not a valid SAMPLE_DIR: {0}".format(src_dir))

pip_patcher = MLPipelinePatcher(prm_stg, src_dir)

pip_patcher.patch_pipeline_config(model_base_name)