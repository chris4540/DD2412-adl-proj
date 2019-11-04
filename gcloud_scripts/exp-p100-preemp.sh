#!/bin/bash
#
# For creating google cloud preemptible instance with GPU nvidia-tesla-p100
# THis instance mainly for developement
#
# Reference:
# https://cloud.google.com/ai-platform/deep-learning-vm/docs/tensorflow_start_instance
# https://cloud.google.com/compute/docs/gpus/
# For using different GPU cards:
# change the gpu config. See the doc above
# =================================================

export ZONE="europe-west1-b"
export IMAGE_NAME="tf-1-14-cu100-20191004"
export INSTANCE_NAME="tf-exp-p100-preemp3"
export INSTANCE_TYPE="n1-highmem-2"
export GPU_CONFIG="type=nvidia-tesla-p100,count=1"
export BOOT_DISK_SIZE="100GB"

gcloud compute instances create $INSTANCE_NAME \
  --machine-type=$INSTANCE_TYPE \
  --zone=$ZONE \
  --image=$IMAGE_NAME \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=$BOOT_DISK_SIZE \
  --accelerator=$GPU_CONFIG \
  --metadata="install-nvidia-driver=True" \
  --preemptible
