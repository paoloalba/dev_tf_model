# TF Model development

This project is intended to produce docker images ready to be employed for further model development from the available TF models (https://github.com/tensorflow/models).

In particular I offer a wrap around the available modules from the object detection project, in order to train, evaluate and extract a novel model from the available checkpoints.
In the specific I concentrate on TF 2.x versions (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).

This repository also offers a convenient runtime patcher for ```pipeline.config```, and a class to generate TFRecord files from Pascal-annotated images.

### Build and run

Modify the ```build_run.bat``` file for your needs.

This will first set all the environment variables, which will be transported and evaluated from the ```docker-compose.yml``` file for build, run and destruction of docker components.
The desired Docker file (training, evaluation, extraction) will be extracted and processed.
For development purposes I set the base image to ```tensorflow/tensorflow:latest-gpu```, but in principle any other TF 2.x release should be compatible.

The use is intended for windows hosts.

### Build and run

Modify the ```build_push.bat``` file for your needs.
This will first build the training and evaluation Docker images, and then push them to your specified Docker Registry.
The script is already set for the use of Azure Container Registry, but any other technology can be employed.

### Run on a K8s cluster

For run on a K8s cluster see https://github.com/paoloalba/k8s_deployer
