#!/bin/bash
#
# For preemptible GPU nvidia-tesla-p100
# Reference:
# https://cloud.google.com/ai-platform/deep-learning-vm/docs/tensorflow_start_instance
# =================================================
export IMAGE_FAMILY="tf-latest-gpu"
# export ZONE="europe-west1-b"
# export ZONE="europe-west1-d"
export ZONE="europe-west4-a"
export INSTANCE_NAME="tf-chris-p100-preemp"
export INSTANCE_TYPE="n1-highmem-2"

gcloud compute instances create $INSTANCE_NAME \
  --machine-type=$INSTANCE_TYPE \
  --zone=$ZONE \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --accelerator="type=nvidia-tesla-p100,count=1" \
  --metadata="install-nvidia-driver=True" \
  --preemptible
