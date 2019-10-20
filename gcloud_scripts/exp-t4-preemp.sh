#!/bin/bash
#
# For preemptible GPU nvidia-tesla-p100
# Reference:
# https://cloud.google.com/ai-platform/deep-learning-vm/docs/tensorflow_start_instance
# https://course.fast.ai/start_gcp.html#step-3-create-an-instance
# =================================================
export IMAGE_FAMILY="tf-latest-gpu"
# export ZONE="europe-west1-b"
# export ZONE="europe-west1-d"
export ZONE="europe-west4-b"
export INSTANCE_NAME="tf-t4-preemp2"
export INSTANCE_TYPE="n1-highmem-2"
# export BOOT_DISK_SIZE="200GB"

gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --machine-type=$INSTANCE_TYPE \
  --accelerator="type=nvidia-tesla-p4,count=1" \
  --metadata="install-nvidia-driver=True" \
  --preemptible
