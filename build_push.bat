@REM GLOBAL SETTINGS
@set registry=<docker_registry_address>
@set registryName=<docker_registry_name>

@set mainRepositoryname=<model_>
@set versionNumber=<version>

@set taskType=training
@set dockerfile=Dockerfile_%taskType%
@set repositoryName=%mainRepositoryname%%taskType%
@set imageFullTagTraining=%registry%/%repositoryName%:%versionNumber%

call docker build -f %dockerfile% -t %imageFullTagTraining% .

@set taskType=evaluation
@set dockerfile=Dockerfile_%taskType%
@set repositoryName=%mainRepositoryname%%taskType%
@set imageFullTagEvaluation=%registry%/%repositoryName%:%versionNumber%

call docker build -f %dockerfile% -t %imageFullTagEvaluation% .

call az acr login --name %registryName%

call docker push %imageFullTagTraining%
call docker push %imageFullTagEvaluation%

