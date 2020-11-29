@REM GLOBAL SETTINGS
@set registry=<docker_registry_address>
@set registryName=<docker_registry_name>

@set dockerfile=docker-compose.yml
@set versionNumber=0.1.0

@REM TASK SPECIFIC SETTINGS
@REM @set taskType=training
@REM @set taskType=evaluation
@set taskType=extraction
@set repositoryName=object_detection_%taskType%
@set dockerfile_src=<Docker_file_path>%taskType%

@REM ENV VARIABLES
@set IS_DOCKER=y
@set permanent_storage=<path_to_permanent_storage_on_host>
@set PIPELINE_CONFIG_PATH=/mnt/mounted_volume/permanent_storage/patched_config/pipeline.config
@set MODEL_DIR=/mnt/mounted_volume/permanent_storage/models/<your_model_name>/

@REM ENV VAR FOR TRAINING
@REM @set SAMPLE_DIR=
@REM @set MODEL_BASE_NAME=efficientdet_d2_coco17_tpu-32
@REM @set BATCH_SIZE=5
@REM @set NUM_CLASSES=40
@REM @set CHECKPOINT_EVERY_N=500
@REM @set CHECKPOINT_MAX_TO_KEEP=25

@REM ENV VAR FOR EVALUATION
@REM @set EVAL_TIMEOUT=10000

@REM ENV VAR FOR EXTRACTION
@set OUTPUT_EXTRACT_MODEL_DIR=/mnt/mounted_volume/permanent_storage/models/<your_model_name>/extracted_model/

@REM GENERATE DOCKER COMPOSE FILE
call docker-compose -f %dockerfile% build

call docker-compose -f %dockerfile% up
call docker-compose -f %dockerfile% down
