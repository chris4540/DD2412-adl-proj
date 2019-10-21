#!/bin/bash
# 
# For preemptible GPU nvidia-tesla-p100
# Reference:
# https://cloud.google.com/ai-platform/deep-learning-vm/docs/tensorflow_start_instance
# =================================================
export IMAGE_FAMILY="tf2-latest-gpu"
export ZONE="europe-west1-b"
# export ZONE="europe-west1-d"
# export ZONE="europe-west4-a"
export INSTANCE_NAME="tf-dd2412-proj-exp-p100"
export BOOT_DISK_SIZE="200GB"

gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=$BOOT_DISK_SIZE \
  --accelerator="type=nvidia-tesla-p100,count=1" \
  --metadata="install-nvidia-driver=True"